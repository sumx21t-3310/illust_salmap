from pathlib import Path

import torch
from torch.nn import Module

LIGHTNING_PREFIX = "model."


def load_weights(model: Module, ckpt_path: str | Path, prefix: str = LIGHTNING_PREFIX) -> Module:
    """
    Restores `model`'s weights from a checkpoint, in place.

    Training checkpoints are written by a LightningModule that holds the network as
    `self.model`, so every key carries that prefix. Stripping it here lets a bare
    nn.Module be restored without building the training wrapper -- inference and export
    therefore do not depend on `illust_salmap.training`, pytorch-lightning, or the
    metric stack.

    Both a Lightning checkpoint and a plain `state_dict` file are accepted. The prefixed
    reading is tried first, then the checkpoint as-is, so a network that happens to have
    its own `model` attribute still loads either way.

    Args:
        model (Module): The network to load into.
        ckpt_path (str | Path): Path to a `.ckpt` or `.pt` file.
        prefix (str): The attribute path the training wrapper stored the network under.

    Returns:
        Module: The same instance, with the weights loaded.

    Raises:
        RuntimeError: If the checkpoint does not match `model` under either reading.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    unwrapped = {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}

    failures = []
    for reading, candidate in (("wrapped", unwrapped), ("bare", state_dict)):
        if not candidate:
            continue
        try:
            model.load_state_dict(candidate)
            return model
        except RuntimeError as err:
            failures.append(f"as a {reading} checkpoint -> {err}")

    raise RuntimeError(
        f"{ckpt_path} does not hold weights for {type(model).__name__}. Tried: " + " | ".join(failures)
    )
