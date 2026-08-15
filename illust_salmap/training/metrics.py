import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics import KLDivergence, Metric
from torchmetrics.functional.classification import binary_auroc

EPSILON = 1e-8

# A prediction whose spread is this small relative to its magnitude carries no ranking
# information -- what is left is float noise, which z-scoring would amplify to O(1).
FLAT_PREDICTION_TOLERANCE = 1e-6


def normalized(target: Tensor, range_floor: float = EPSILON) -> Tensor:
    """
    Min-max normalizes every image in the batch to [0, 1], independently of the others.

    Saliency metrics are defined per image, so the statistics must not be shared across
    the batch: a dim image next to a bright one would otherwise be squashed.

    Differentiable on purpose -- `training.losses` builds the training objective out of
    this and `to_distribution`. The metrics that use it disable gradients in their own
    `update`, so nothing accumulates a graph here.

    Args:
        target (Tensor): A batch shaped (B, ...).
        range_floor (float): Smallest spread the division will use. The default only has
            to keep a constant image finite. A caller that differentiates through this
            needs a much larger floor -- see `losses.SPREAD_FLOOR`.

    Returns:
        Tensor: The same shape, each image scaled to [0, 1]. Constant images become 0.
    """
    flat = torch.flatten(target, start_dim=1)

    min_value = flat.min(dim=1, keepdim=True).values
    max_value = flat.max(dim=1, keepdim=True).values

    scaled = (flat - min_value) / (max_value - min_value).clamp_min(range_floor)

    return scaled.reshape(target.shape)


def to_distribution(target: Tensor, epsilon: float = EPSILON, range_floor: float = EPSILON) -> Tensor:
    """
    Turns each saliency map into a probability distribution over its pixels.

    Normalizing by the sum -- not by a softmax -- is what the saliency literature means
    by a saliency distribution. A softmax would push the values through exp() and change
    the shape of the distribution, making KL and SIM incomparable with published numbers.

    Args:
        target (Tensor): A batch shaped (B, ...), in any value range.
        epsilon (float): Floor added before normalizing, so empty maps stay finite.
        range_floor (float): Passed to `normalized`.

    Returns:
        Tensor: A (B, N) tensor whose rows sum to 1.
    """
    flat = torch.flatten(normalized(target, range_floor), start_dim=1) + epsilon

    return flat / flat.sum(dim=1, keepdim=True)


@torch.no_grad()
def convert_kl_div(predict_img: Tensor, target_img: Tensor, epsilon: float = EPSILON) -> tuple[Tensor, Tensor]:
    """
    Prepares a prediction/ground truth pair for `torchmetrics.KLDivergence`.

    Note the returned order: KLDivergence computes D_KL(P||Q) from `update(p, q)`, and
    the saliency convention is D_KL(ground truth || prediction). The ground truth
    therefore comes first, and callers should splat the result:

        self.kl_div(*convert_kl_div(predict, ground_truth))

    Returns:
        tuple[Tensor, Tensor]: (ground truth distribution, prediction distribution).
    """
    return to_distribution(target_img, epsilon), to_distribution(predict_img, epsilon)


@torch.no_grad()
def convert_auroc(predict_img: Tensor, target_img: Tensor, threshold: float = 0.5) -> tuple[Tensor, Tensor]:
    """
    Prepares a prediction/ground truth pair for `torchmetrics.AUROC(task="binary")`.

    The prediction stays continuous -- AUROC ranks it, so binarizing it would collapse
    the ROC curve to a single point. Only the ground truth is thresholded, standing in
    for the fixation points that a continuous fixation map no longer carries.

    Returns:
        tuple[Tensor, Tensor]: (continuous scores, binary labels), both shaped (B, N).
    """
    predict_flat = torch.flatten(normalized(predict_img), start_dim=1)
    target_flat = (torch.flatten(normalized(target_img), start_dim=1) > threshold).long()

    return predict_flat, target_flat


def correlation_coefficient(predict_img: Tensor, target_img: Tensor, epsilon: float = EPSILON) -> Tensor:
    """
    CC: the Pearson correlation of each image with its ground truth.

    No normalization is needed first -- correlation is already invariant to the scale and
    offset of either map, which is exactly why the saliency literature reports it.

    Differentiable, and shared with `training.losses`.

    Returns:
        Tensor: One correlation per image, shaped (B,). A constant map scores 0.
    """
    predict_flat = torch.flatten(predict_img, start_dim=1)
    target_flat = torch.flatten(target_img, start_dim=1)

    predict_centered = predict_flat - predict_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)

    covariance = (predict_centered * target_centered).sum(dim=1)
    # Each norm is floored on its own rather than their product. Flooring the product
    # lets a large ground-truth norm carry a collapsed prediction straight through, so
    # the gradient the prediction sees still goes as 1 / its own vanishing norm.
    scale = predict_centered.norm(dim=1).clamp_min(epsilon) * target_centered.norm(dim=1).clamp_min(epsilon)

    return covariance / scale


class SaliencySimilarity(Metric):
    """
    SIM: the histogram intersection of two saliency distributions.

    Both maps are normalized to sum to 1, and the metric sums the elementwise minimum.
    1.0 means the distributions coincide, 0.0 means they share no mass.

    torchmetrics has no equivalent -- `CosineSimilarity` measures the angle between the
    raw vectors, which is a different quantity and not the SIM reported in the
    saliency literature.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, epsilon: float = EPSILON, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, preds: Tensor, target: Tensor) -> None:
        predict_dist = to_distribution(preds, self.epsilon)
        target_dist = to_distribution(target, self.epsilon)

        self.total += torch.minimum(predict_dist, target_dist).sum(dim=1).sum()
        self.count += preds.size(0)

    def compute(self) -> Tensor:
        if self.count == 0:
            return torch.tensor(float("nan"), device=self.total.device)

        return self.total / self.count


class NormalizedScanpathSaliency(Metric):
    """
    NSS: the mean of the z-scored prediction, sampled at the fixated pixels.

    The prediction is standardized per image (so the score is invariant to its scale and
    offset) and averaged over the locations the ground truth marks as fixated. 0.0 means
    chance; higher is better, and values above ~1 indicate the prediction concentrates
    mass where people actually looked.

    The reference definition samples discrete fixation points. This dataset only ships
    continuous fixation maps, so `threshold` binarizes the normalized ground truth to
    stand in for them -- the same approximation `convert_auroc` makes. Images whose
    ground truth has no pixel above the threshold contribute nothing.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, preds: Tensor, target: Tensor) -> None:
        saliency = torch.flatten(preds, start_dim=1)
        fixations = (torch.flatten(normalized(target), start_dim=1) > self.threshold).to(saliency.dtype)

        mean = saliency.mean(dim=1, keepdim=True)
        std = saliency.std(dim=1, unbiased=False, keepdim=True)

        # A flat prediction scores 0 (chance) rather than an amplified rounding error.
        scale = saliency.abs().amax(dim=1, keepdim=True).clamp_min(EPSILON)
        is_flat = std <= FLAT_PREDICTION_TOLERANCE * scale

        standardized = torch.where(is_flat, torch.zeros_like(saliency), (saliency - mean) / std.clamp_min(EPSILON))

        fixation_counts = fixations.sum(dim=1)
        scorable = fixation_counts > 0

        if not bool(scorable.any()):
            return

        scores = (standardized * fixations).sum(dim=1)[scorable] / fixation_counts[scorable]

        self.total += scores.sum()
        self.count += int(scorable.sum())

    def compute(self) -> Tensor:
        if self.count == 0:
            return torch.tensor(float("nan"), device=self.total.device)

        return self.total / self.count


class CorrelationCoefficient(Metric):
    """
    CC: the Pearson correlation between prediction and ground truth, averaged over images.

    `torchmetrics.image.SpatialCorrelationCoefficient` is a different metric despite the
    name -- it high-pass filters both maps and correlates them inside a local window, so
    it is dominated by fine detail. A prediction equal to the ground truth plus mild pixel
    noise scores 0.98 as CC and 0.01 as SCC, which is backwards for saliency: the maps are
    smooth by construction and the high frequencies are the part that does not matter.
    """

    is_differentiable = True
    higher_is_better = True
    full_state_update = False

    def __init__(self, epsilon: float = EPSILON, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, preds: Tensor, target: Tensor) -> None:
        self.total += correlation_coefficient(preds, target, self.epsilon).sum()
        self.count += preds.size(0)

    def compute(self) -> Tensor:
        if self.count == 0:
            return torch.tensor(float("nan"), device=self.total.device)

        return self.total / self.count


class AreaUnderROC(Metric):
    """
    AUC: how well the prediction ranks fixated pixels above the rest, per image.

    `torchmetrics.AUROC` pools everything handed to it into a single ranking, so passing
    it a (B, N) batch scores the batch as though it were one image -- two images sitting
    at different overall levels then interfere with each other's ranking. Saliency AUC is
    defined per image, so each one is scored separately and the results averaged.

    The ground truth is thresholded to stand in for fixation points, the same
    approximation `NormalizedScanpathSaliency` makes. An image whose ground truth is
    entirely fixation or entirely background has no ROC curve and is skipped.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, preds: Tensor, target: Tensor) -> None:
        predict_flat, target_flat = convert_auroc(preds, target, self.threshold)

        for prediction, fixations in zip(predict_flat, target_flat):
            positives = int(fixations.sum())
            if positives == 0 or positives == fixations.numel():
                continue

            self.total += binary_auroc(prediction, fixations)
            self.count += 1

    def compute(self) -> Tensor:
        if self.count == 0:
            return torch.tensor(float("nan"), device=self.total.device)

        return self.total / self.count


# Short aliases matching how the metrics are named in the literature.
SIM = SaliencySimilarity
NSS = NormalizedScanpathSaliency
CC = CorrelationCoefficient
AUC = AreaUnderROC


class SaliencyMetrics(Module):
    """
    The saliency metric suite, wired to raw model output and ground truth.

    Owns one set of metrics for a single stage. Registering them here keeps the three
    LightningModules from repeating -- and drifting apart on -- the conversion calls.

    Usage:
        self.val_metrics = SaliencyMetrics("val_")
        ...
        self.log_dict(self.val_metrics.update(predict, ground_truth), on_epoch=True)
    """

    def __init__(self, prefix: str = ""):
        super().__init__()
        self.prefix = prefix

        self.kl_div = KLDivergence()
        self.sim = SaliencySimilarity()
        self.cc = CorrelationCoefficient()
        self.nss = NormalizedScanpathSaliency()
        self.auroc = AreaUnderROC()

    def update(self, predict: Tensor, ground_truth: Tensor) -> dict[str, Metric]:
        """
        Feeds one batch to every metric.

        Returns:
            dict[str, Metric]: The metric objects keyed by log name, ready for
                `log_dict` -- Lightning aggregates them over the epoch itself.
        """
        # update(), not the metric's forward(): forward additionally computes the value for
        # this batch alone, which is thrown away here -- Lightning calls compute() itself at
        # the end of the epoch. For AUROC that saved computation is a full sort per image.
        self.kl_div.update(*convert_kl_div(predict, ground_truth))
        self.sim.update(predict, ground_truth)
        self.cc.update(predict, ground_truth)
        self.nss.update(predict, ground_truth)
        self.auroc.update(predict, ground_truth)

        return {
            f"{self.prefix}kl_div": self.kl_div,
            f"{self.prefix}sim": self.sim,
            f"{self.prefix}cc": self.cc,
            f"{self.prefix}nss": self.nss,
            f"{self.prefix}auroc": self.auroc,
        }
