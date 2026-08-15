import pytest
import torch
from torchmetrics import AUROC, KLDivergence

from illust_salmap.training.metrics import (
    AreaUnderROC,
    CorrelationCoefficient,
    NormalizedScanpathSaliency,
    SaliencyMetrics,
    SaliencySimilarity,
    convert_auroc,
    convert_kl_div,
    normalized,
    to_distribution,
)


@pytest.fixture
def ground_truth():
    torch.manual_seed(0)
    return torch.rand(4, 1, 32, 32)


def test_normalized_scales_each_image_independently():
    """A dim image next to a bright one must still span the full range."""
    batch = torch.stack([torch.linspace(0.0, 0.1, 16), torch.linspace(0.0, 10.0, 16)]).reshape(2, 1, 4, 4)

    result = normalized(batch)

    for image in result:
        assert image.min().item() == pytest.approx(0.0)
        assert image.max().item() == pytest.approx(1.0)


def test_normalized_handles_a_constant_image():
    result = normalized(torch.full((2, 1, 4, 4), 0.7))

    assert torch.isfinite(result).all()
    assert result.abs().max().item() == pytest.approx(0.0)


def test_to_distribution_sums_to_one(ground_truth):
    distribution = to_distribution(ground_truth)

    assert torch.allclose(distribution.sum(dim=1), torch.ones(ground_truth.size(0)), atol=1e-6)
    assert (distribution >= 0).all()


def test_kl_divergence_is_zero_for_identical_maps(ground_truth):
    kl_div = KLDivergence()

    assert kl_div(*convert_kl_div(ground_truth.clone(), ground_truth)).item() == pytest.approx(0.0, abs=1e-6)


def test_kl_divergence_puts_the_ground_truth_first(ground_truth):
    """
    Regression: the reference distribution used to be passed second, so the metric
    reported D_KL(prediction || ground truth) instead of the saliency convention.
    """
    predict = torch.rand_like(ground_truth)

    reference, approximation = convert_kl_div(predict, ground_truth)

    assert torch.allclose(reference, to_distribution(ground_truth))
    assert torch.allclose(approximation, to_distribution(predict))


def test_kl_divergence_is_invariant_to_prediction_scale(ground_truth):
    """Per-image normalization means a rescaled prediction is the same prediction."""
    predict = torch.rand_like(ground_truth)
    kl_div = KLDivergence()

    plain = kl_div(*convert_kl_div(predict, ground_truth)).item()
    kl_div.reset()
    rescaled = kl_div(*convert_kl_div(predict * 3.0 - 1.0, ground_truth)).item()

    assert plain == pytest.approx(rescaled, abs=1e-5)


def test_auroc_keeps_the_prediction_continuous(ground_truth):
    """Regression: the prediction used to be thresholded into {0, 1}."""
    preds, target = convert_auroc(torch.rand_like(ground_truth), ground_truth)

    assert preds.unique().numel() > 2
    assert set(target.unique().tolist()) <= {0, 1}


def test_auroc_is_perfect_for_a_monotone_prediction(ground_truth):
    """
    Regression: a prediction that ranks every pixel correctly must score 1.0.
    Binarizing the prediction used to report ~0.7 here.
    """
    monotone = ground_truth ** 3

    score = AUROC(task="binary")(*convert_auroc(monotone, ground_truth)).item()

    assert score == pytest.approx(1.0, abs=1e-6)


def test_auroc_is_chance_for_an_unrelated_prediction(ground_truth):
    score = AUROC(task="binary")(*convert_auroc(torch.rand_like(ground_truth), ground_truth)).item()

    assert score == pytest.approx(0.5, abs=0.1)


def test_similarity_is_one_for_identical_maps(ground_truth):
    assert SaliencySimilarity()(ground_truth.clone(), ground_truth).item() == pytest.approx(1.0, abs=1e-5)


def test_similarity_is_near_zero_for_disjoint_maps():
    left = torch.zeros(1, 1, 4, 4)
    left[0, 0, 0, :] = 1.0
    right = torch.zeros(1, 1, 4, 4)
    right[0, 0, 3, :] = 1.0

    assert SaliencySimilarity()(left, right).item() == pytest.approx(0.0, abs=1e-3)


def test_similarity_is_symmetric(ground_truth):
    predict = torch.rand_like(ground_truth)

    forward = SaliencySimilarity()(predict, ground_truth).item()
    backward = SaliencySimilarity()(ground_truth, predict).item()

    assert forward == pytest.approx(backward, abs=1e-6)


def test_nss_rewards_a_matching_prediction(ground_truth):
    matching = NormalizedScanpathSaliency()(ground_truth.clone(), ground_truth).item()
    unrelated = NormalizedScanpathSaliency()(torch.rand_like(ground_truth), ground_truth).item()

    assert matching > 0
    assert matching > unrelated


def test_nss_is_negative_for_an_inverted_prediction(ground_truth):
    assert NormalizedScanpathSaliency()(-ground_truth, ground_truth).item() < 0


def test_nss_is_invariant_to_prediction_scale_and_offset(ground_truth):
    predict = torch.rand_like(ground_truth)

    plain = NormalizedScanpathSaliency()(predict, ground_truth).item()
    shifted = NormalizedScanpathSaliency()(predict * 5.0 + 2.0, ground_truth).item()

    assert plain == pytest.approx(shifted, abs=1e-5)


def test_nss_is_zero_for_a_flat_prediction(ground_truth):
    flat = torch.full_like(ground_truth, 0.3)

    assert NormalizedScanpathSaliency()(flat, ground_truth).item() == pytest.approx(0.0, abs=1e-6)


def test_nss_skips_images_without_fixations():
    """A ground truth with nothing above the threshold must not divide by zero."""
    metric = NormalizedScanpathSaliency()
    metric.update(torch.rand(2, 1, 8, 8), torch.zeros(2, 1, 8, 8))

    assert metric.count == 0
    assert torch.isnan(metric.compute())


def test_correlation_is_one_for_identical_maps(ground_truth):
    assert CorrelationCoefficient()(ground_truth.clone(), ground_truth).item() == pytest.approx(1.0, abs=1e-5)


def test_correlation_is_minus_one_for_an_inverted_prediction(ground_truth):
    assert CorrelationCoefficient()(-ground_truth, ground_truth).item() == pytest.approx(-1.0, abs=1e-5)


def test_correlation_is_invariant_to_scale_and_offset(ground_truth):
    plain = CorrelationCoefficient()(ground_truth.clone(), ground_truth).item()
    rescaled = CorrelationCoefficient()(ground_truth * 7.0 - 3.0, ground_truth).item()

    assert plain == pytest.approx(rescaled, abs=1e-5)


def test_correlation_is_zero_for_a_flat_prediction(ground_truth):
    flat = torch.full_like(ground_truth, 0.3)

    assert CorrelationCoefficient()(flat, ground_truth).item() == pytest.approx(0.0, abs=1e-6)


def test_correlation_survives_pixel_noise():
    """
    Regression: this used to be `SpatialCorrelationCoefficient`, which high-pass filters
    both maps. A ground truth plus mild pixel noise scored 0.01 there, and ~0.98 here.
    """
    grid = torch.linspace(-3, 3, 64)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    blob = torch.exp(-(xx ** 2 + yy ** 2)).view(1, 1, 64, 64)
    noisy = (blob + 0.05 * torch.randn_like(blob)).clamp(0.0, 1.0)

    assert CorrelationCoefficient()(noisy, blob).item() > 0.9


def test_auroc_scores_each_image_separately():
    """
    Regression: the batch used to be pooled into one ranking, which weights an image by
    its number of fixated pixels. One perfect and one inverted image must average to 0.5
    however much of each image is fixated.
    """
    torch.manual_seed(0)
    wide = torch.rand(1, 1, 64, 64) ** 0.2
    narrow = torch.rand(1, 1, 64, 64) ** 8

    target = torch.cat([wide, narrow])
    predict = torch.cat([wide, -narrow])

    assert AreaUnderROC()(predict, target).item() == pytest.approx(0.5, abs=1e-6)


def test_auroc_skips_images_without_a_roc_curve():
    """A ground truth that is all background has no positives, so it has no AUC."""
    metric = AreaUnderROC()
    metric.update(torch.rand(2, 1, 8, 8), torch.zeros(2, 1, 8, 8))

    assert metric.count == 0
    assert torch.isnan(metric.compute())


def test_saliency_metrics_reports_every_metric(ground_truth):
    metrics = SaliencyMetrics("val_")

    logged = metrics.update(torch.rand_like(ground_truth), ground_truth)

    assert set(logged) == {"val_kl_div", "val_sim", "val_cc", "val_nss", "val_auroc"}
    for name, metric in logged.items():
        assert torch.isfinite(metric.compute()), f"{name} is not finite"
