import pytest
import torch

from illust_salmap.training.losses import (
    SaliencyLoss,
    correlation_loss,
    kl_divergence_loss,
    nss_loss,
)


@pytest.fixture
def ground_truth():
    """Two gaussian blobs in different corners, in [-1, 1] as the datamodules deliver them."""
    grid = torch.linspace(-3.0, 3.0, 64)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    blobs = torch.stack([torch.exp(-((xx - c) ** 2 + (yy - c) ** 2)) for c in (-1.0, 1.0)])

    return (blobs.unsqueeze(1) - 0.5) / 0.5


def test_perfect_prediction_beats_every_alternative(ground_truth):
    loss = SaliencyLoss()
    torch.manual_seed(0)

    perfect = loss(ground_truth.clone(), ground_truth).item()

    assert perfect < loss(torch.zeros_like(ground_truth), ground_truth).item()
    assert perfect < loss(torch.rand_like(ground_truth) * 2 - 1, ground_truth).item()
    assert perfect < loss(-ground_truth, ground_truth).item()


def test_inverted_prediction_is_worse_than_a_flat_one(ground_truth):
    """Being confidently wrong must cost more than saying nothing."""
    loss = SaliencyLoss()

    assert loss(-ground_truth, ground_truth).item() > loss(torch.zeros_like(ground_truth), ground_truth).item()


def test_loss_is_invariant_to_prediction_scale_and_offset(ground_truth):
    """
    Every term is normalized per image, so the loss constrains the shape of the map and
    not its level -- the same property the metrics have.
    """
    loss = SaliencyLoss()
    predict = torch.rand_like(ground_truth)

    assert loss(predict, ground_truth).item() == pytest.approx(loss(predict * 5.0 - 2.0, ground_truth).item(), abs=1e-5)


def test_gradients_reach_the_prediction(ground_truth):
    predict = (torch.rand_like(ground_truth) * 2 - 1).requires_grad_(True)

    SaliencyLoss()(predict, ground_truth).backward()

    assert predict.grad is not None
    assert torch.isfinite(predict.grad).all()
    assert predict.grad.abs().sum() > 0


def test_a_batch_without_fixations_stays_finite_and_differentiable():
    """
    NSS has nothing to score when no pixel clears the threshold. The step must still be
    well defined rather than producing a nan that poisons every weight.
    """
    predict = torch.rand(2, 1, 8, 8, requires_grad=True)

    loss = SaliencyLoss()(predict, torch.zeros(2, 1, 8, 8))
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(predict.grad).all()


def test_a_collapsed_prediction_does_not_explode_the_cc_and_nss_gradients(ground_truth):
    """
    Regression: every term divides by the prediction's own spread, so a prediction that
    collapses towards constant sent the gradient to ~1e11. The floor caps CC and NSS.
    KL is not bounded by it -- gradient clipping is the backstop there.
    """
    flat = torch.zeros_like(ground_truth).requires_grad_(True)

    correlation_loss(flat, ground_truth).backward()
    assert flat.grad.abs().max() < 1e3

    flat.grad = None
    nss_loss(flat, ground_truth).backward()
    assert flat.grad.abs().max() < 1e3


def test_the_spread_floor_leaves_a_healthy_prediction_untouched(ground_truth):
    """The floor must be a guard rail, not something that shifts the objective."""
    torch.manual_seed(0)
    predict = torch.rand_like(ground_truth) * 2 - 1

    guarded = SaliencyLoss().forward(predict, ground_truth).item()
    unguarded = SaliencyLoss(spread_floor=1e-8).forward(predict, ground_truth).item()

    assert guarded == pytest.approx(unguarded, abs=1e-9)


def test_kl_divergence_is_zero_for_an_identical_map(ground_truth):
    assert kl_divergence_loss(ground_truth.clone(), ground_truth).item() == pytest.approx(0.0, abs=1e-4)


def test_correlation_loss_is_minus_one_for_an_identical_map(ground_truth):
    assert correlation_loss(ground_truth.clone(), ground_truth).item() == pytest.approx(-1.0, abs=1e-5)


def test_nss_loss_is_negative_for_a_matching_prediction(ground_truth):
    assert nss_loss(ground_truth.clone(), ground_truth).item() < 0


def test_weights_switch_terms_off():
    """Each term must be reachable on its own, so the weights can be tuned per dataset."""
    torch.manual_seed(0)
    target = torch.rand(2, 1, 16, 16) * 2 - 1
    predict = torch.rand(2, 1, 16, 16) * 2 - 1

    only_cc = SaliencyLoss(kl_weight=0.0, cc_weight=1.0, nss_weight=0.0)

    assert only_cc(predict, target).item() == pytest.approx(correlation_loss(predict, target).item(), abs=1e-6)
