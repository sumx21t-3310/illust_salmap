"""
Differentiable versions of the saliency metrics, for use as a training objective.

`MSELoss` treats every pixel as an independent regression target. Saliency maps are
mostly background, so the MSE-optimal prediction drifts towards the dataset's average
map -- which scores badly on every metric in `metrics.py`, none of which MSE optimizes.
The losses here are those metrics turned around: KL is minimized, CC and NSS are
negated and minimized.

The primitives are shared with `metrics.py` rather than reimplemented, so the quantity
being optimized and the quantity being reported cannot drift apart.
"""

import torch
from torch import Tensor
from torch.nn import Module

from illust_salmap.training.metrics import EPSILON, correlation_coefficient, normalized, to_distribution

# Floor on the per-image spread that the CC and NSS terms divide by.
#
# The metrics get away with EPSILON (1e-8) because they only need a constant map to come
# out finite. A loss also has to keep the *derivative* finite, and that goes as 1 / spread:
# an exactly constant prediction produced gradients around 1e11 before this floor, and
# 6.9 after it. Predictions and targets both live in [-1, 1], so 1e-3 sits far below any
# real map's spread -- a healthy step comes out bit-for-bit identical either way.
#
# This is not protection against corrupted weights: past 65504 a gradient is inf under
# `16-mixed`, and the GradScaler responds by skipping the optimizer step, so the damage
# is a stalled run rather than a poisoned one. bf16 and fp32 have the range to absorb it
# outright. The floor is here because 1e11 is not a number the optimizer should ever be
# handed, not because the alternative is silent corruption.
#
# It does not bound the KL term, whose gradient also carries
# target_probability / prediction_probability and grows without limit wherever the
# prediction puts no mass at all. `--grad-clip` is the backstop for that one.
SPREAD_FLOOR = 1e-3


def kl_divergence_loss(
        predict: Tensor,
        target: Tensor,
        epsilon: float = EPSILON,
        spread_floor: float = SPREAD_FLOOR,
) -> Tensor:
    """
    D_KL(ground truth || prediction), averaged over the batch.

    The saliency convention puts the ground truth first: the penalty is for missing mass
    the ground truth has, not for extra mass the prediction invents. This is the term
    that actually drives the map into shape, which is why it carries the most weight.
    """
    target_dist = to_distribution(target, epsilon, spread_floor)
    predict_dist = to_distribution(predict, epsilon, spread_floor)

    ratio = target_dist / (predict_dist + epsilon)

    return (target_dist * torch.log(ratio + epsilon)).sum(dim=1).mean()


def correlation_loss(predict: Tensor, target: Tensor, epsilon: float = SPREAD_FLOOR) -> Tensor:
    """Negated CC, so that perfect correlation is -1 and anticorrelation is +1."""
    return -correlation_coefficient(predict, target, epsilon).mean()


def nss_loss(predict: Tensor, target: Tensor, threshold: float = 0.5, epsilon: float = SPREAD_FLOOR) -> Tensor:
    """
    Negated NSS: how high the standardized prediction sits at the fixated pixels.

    Mirrors `metrics.NormalizedScanpathSaliency`, including the threshold that stands in
    for discrete fixation points. Images whose ground truth has nothing above the
    threshold contribute nothing; if that is the whole batch the loss is zero, but still
    attached to the graph so the optimizer step stays well defined.
    """
    saliency = torch.flatten(predict, start_dim=1)
    fixations = (torch.flatten(normalized(target), start_dim=1) > threshold).to(saliency.dtype)

    mean = saliency.mean(dim=1, keepdim=True)
    std = saliency.std(dim=1, unbiased=False, keepdim=True)
    standardized = (saliency - mean) / std.clamp_min(epsilon)

    fixation_counts = fixations.sum(dim=1)
    scorable = fixation_counts > 0

    if not bool(scorable.any()):
        return saliency.sum() * 0.0

    scores = (standardized * fixations).sum(dim=1)[scorable] / fixation_counts[scorable]

    return -scores.mean()


class SaliencyLoss(Module):
    """
    KL + CC + NSS, the combination saliency models are normally trained with.

    Every term is invariant to the scale and offset of the prediction, exactly as the
    metrics are -- the model is asked to get the *shape* of the map right, not its
    absolute level. That also means the head's output range is unconstrained by the loss,
    which is fine here because `normalized()` rescales before anything is reported.

    The terms live on different scales (KL is typically 0.5-2, CC is bounded by 1, NSS
    runs to about 3), so the weights below are a starting point rather than a tuned
    result. KL leads because it is the term that penalizes missing a fixated region;
    NSS is damped because it is the noisiest of the three early in training.
    """

    def __init__(
            self,
            kl_weight: float = 1.0,
            cc_weight: float = 1.0,
            nss_weight: float = 0.1,
            threshold: float = 0.5,
            epsilon: float = EPSILON,
            spread_floor: float = SPREAD_FLOOR,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.cc_weight = cc_weight
        self.nss_weight = nss_weight
        self.threshold = threshold
        # Two different jobs, so two different constants: `epsilon` keeps a probability
        # away from zero inside the logarithm, `spread_floor` keeps the gradient finite
        # when a prediction collapses. Sharing one value would force the second to be as
        # small as the first.
        self.epsilon = epsilon
        self.spread_floor = spread_floor

    def forward(self, predict: Tensor, target: Tensor) -> Tensor:
        loss = self.kl_weight * kl_divergence_loss(predict, target, self.epsilon, self.spread_floor)
        loss = loss + self.cc_weight * correlation_loss(predict, target, self.spread_floor)
        loss = loss + self.nss_weight * nss_loss(predict, target, self.threshold, self.spread_floor)

        return loss

    def extra_repr(self) -> str:
        return f"kl_weight={self.kl_weight}, cc_weight={self.cc_weight}, nss_weight={self.nss_weight}"
