from pathlib import Path

from torch.nn import Module


def get_class_name(obj: object):
    return type(obj).__name__


def get_save_path(root: str | Path, datamodule, model: Module):

    model_name = get_class_name(model)
    module_name = get_class_name(datamodule)
    return Path(f"{root}/{module_name}/{model_name}")


def get_log_path(root: str | Path, datamodule, model: Module):
    return get_save_path(root, datamodule, model) / "logs"


def get_checkpoint_path(root: str | Path, datamodule, model: Module):
    return get_save_path(root, datamodule, model) / "checkpoints"
