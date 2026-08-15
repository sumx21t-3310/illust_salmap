import pytest
import torch
from torch.nn import Conv2d, Module

from illust_salmap.models.checkpoint import load_weights


class TinyNet(Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv2d(3, 1, 1)


class WrapperLikeNet(Module):
    """A network that keeps its own `model` attribute, the way a torchvision wrapper does."""

    def __init__(self):
        super().__init__()
        self.model = Conv2d(3, 1, 1)


def save_lightning_checkpoint(path, module: Module):
    """Mimics what pytorch-lightning writes: the network nested under `model.`."""
    state_dict = {f"model.{key}": value for key, value in module.state_dict().items()}
    torch.save({"state_dict": state_dict, "epoch": 3, "global_step": 42}, path)


def test_loads_a_lightning_checkpoint_into_a_bare_module(tmp_path):
    trained = TinyNet()
    torch.nn.init.constant_(trained.conv.weight, 0.25)
    ckpt = tmp_path / "epoch=3.ckpt"
    save_lightning_checkpoint(ckpt, trained)

    restored = load_weights(TinyNet(), ckpt)

    assert torch.equal(restored.conv.weight, trained.conv.weight)


def test_returns_the_same_instance(tmp_path):
    ckpt = tmp_path / "model.ckpt"
    save_lightning_checkpoint(ckpt, TinyNet())

    model = TinyNet()

    assert load_weights(model, ckpt) is model


def test_loads_a_plain_state_dict(tmp_path):
    trained = TinyNet()
    torch.nn.init.constant_(trained.conv.weight, 0.5)
    ckpt = tmp_path / "weights.pt"
    torch.save(trained.state_dict(), ckpt)

    restored = load_weights(TinyNet(), ckpt)

    assert torch.equal(restored.conv.weight, trained.conv.weight)


def test_loads_a_network_that_has_its_own_model_attribute(tmp_path):
    """
    A network that holds another one as `self.model` has a bare checkpoint that is
    already `model.`-prefixed. Stripping the prefix must not corrupt that case.
    """
    trained = WrapperLikeNet()
    torch.nn.init.constant_(trained.model.weight, 0.75)
    ckpt = tmp_path / "wrapper_like.pt"
    torch.save(trained.state_dict(), ckpt)

    restored = load_weights(WrapperLikeNet(), ckpt)

    assert torch.equal(restored.model.weight, trained.model.weight)


def test_nested_lightning_checkpoint_of_such_a_network(tmp_path):
    trained = WrapperLikeNet()
    torch.nn.init.constant_(trained.model.weight, 0.125)
    ckpt = tmp_path / "wrapper_like.ckpt"
    save_lightning_checkpoint(ckpt, trained)

    restored = load_weights(WrapperLikeNet(), ckpt)

    assert torch.equal(restored.model.weight, trained.model.weight)


def test_rejects_a_checkpoint_from_a_different_architecture(tmp_path):
    ckpt = tmp_path / "other.ckpt"
    save_lightning_checkpoint(ckpt, WrapperLikeNet())

    with pytest.raises(RuntimeError, match="does not hold weights for TinyNet"):
        load_weights(TinyNet(), ckpt)


def test_factories_do_not_import_the_training_stack():
    """Loading a model must not drag pytorch-lightning in behind it."""
    import illust_salmap.models.checkpoint as checkpoint
    import illust_salmap.models.unet.unet as unet

    for module in (checkpoint, unet):
        assert not hasattr(module, "SaliencyModel")

    assert unet.unet().__class__.__name__ == "UNet"
