import inspect

import pytest
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from illust_salmap import cli
from illust_salmap.training.metrics import SaliencyMetrics


class FakeDataModule(LightningDataModule):
    """Stands in for a real dataset: same constructor, random tensors, no download."""

    def __init__(self, root: str, batch_size: int, num_workers: int, seed: int, img_size=(32, 32)):
        super().__init__()
        self.batch_size = batch_size
        height, width = img_size
        self.images = torch.rand(8, 3, height, width) * 2 - 1
        self.maps = torch.rand(8, 1, height, width) * 2 - 1

    def _loader(self) -> DataLoader:
        return DataLoader(TensorDataset(self.images, self.maps), batch_size=self.batch_size)

    train_dataloader = _loader
    val_dataloader = _loader
    test_dataloader = _loader


@pytest.fixture
def fake_dataset(monkeypatch):
    monkeypatch.setitem(cli.DATASETS, "salicon", FakeDataModule)


def test_every_dataset_takes_the_same_constructor():
    """cli.py builds all three the same way, so their signatures must agree."""
    expected = {"root", "batch_size", "num_workers", "img_size", "seed"}

    for name, datamodule in cli.DATASETS.items():
        parameters = set(inspect.signature(datamodule.__init__).parameters)
        assert expected <= parameters, f"{name} is missing {expected - parameters}"


def test_every_model_factory_accepts_a_checkpoint_argument():
    for name, factory in cli.MODELS.items():
        parameters = list(inspect.signature(factory).parameters)
        assert parameters == ["ckpt_path"], f"{name} has an unexpected signature: {parameters}"


def test_every_monitor_is_a_quantity_the_model_actually_logs():
    """
    ModelCheckpoint fails at runtime, an epoch in, if it monitors a name nobody logs.
    `val_loss` comes from the LightningModule; the rest come from the metric suite.
    """
    logged = set(SaliencyMetrics("val_").update(torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8)))

    assert set(cli.MONITORS) - {"val_loss"} <= logged


@pytest.mark.parametrize("loss", sorted(cli.LOSSES))
def test_smoke_run_works_for_every_loss(monkeypatch, tmp_path, fake_dataset, loss):
    monkeypatch.setattr("sys.argv", [
        "illust-salmap", "--model", "dummy", "--dataset", "salicon", "--smoke",
        "--num-workers", "0", "--batch-size", "4", "--img-size", "32", "32",
        "--accelerator", "cpu", "--output-root", str(tmp_path), "--loss", loss,
    ])

    cli.main()

    assert list(tmp_path.rglob("*.ckpt")), f"--loss {loss} wrote no checkpoint"


def test_checkpoints_can_be_selected_on_a_saliency_metric(monkeypatch, tmp_path, fake_dataset):
    """`--monitor val_cc` has to maximise, not minimise, and name the file after it."""
    monkeypatch.setattr("sys.argv", [
        "illust-salmap", "--model", "dummy", "--dataset", "salicon", "--smoke",
        "--num-workers", "0", "--batch-size", "4", "--img-size", "32", "32",
        "--accelerator", "cpu", "--output-root", str(tmp_path), "--monitor", "val_cc",
    ])

    cli.main()

    assert [path for path in tmp_path.rglob("*.ckpt") if path.name != "last.ckpt"]


def test_smoke_run_trains_validates_and_tests(monkeypatch, tmp_path, fake_dataset, capsys):
    """Exercises the whole entrypoint: wiring, metrics, image logging, checkpointing."""
    monkeypatch.setattr("sys.argv", [
        "illust-salmap",
        "--model", "dummy",
        "--dataset", "salicon",
        "--smoke",
        "--num-workers", "0",
        "--batch-size", "4",
        "--img-size", "32", "32",
        "--accelerator", "cpu",
        "--output-root", str(tmp_path),
    ])

    cli.main()

    output = capsys.readouterr().out
    assert "best checkpoint" in output

    checkpoints = list(tmp_path.rglob("*.ckpt"))
    assert checkpoints, "no checkpoint was written"
    assert list(tmp_path.rglob("events.out.tfevents*")), "nothing was logged to TensorBoard"


def test_smoke_run_can_be_resumed_from_its_weights(monkeypatch, tmp_path, fake_dataset):
    """--weights must load a finished run's checkpoint into a freshly built network."""
    argv = [
        "illust-salmap", "--model", "dummy", "--dataset", "salicon", "--smoke",
        "--num-workers", "0", "--batch-size", "4", "--img-size", "32", "32",
        "--accelerator", "cpu", "--output-root", str(tmp_path),
    ]
    monkeypatch.setattr("sys.argv", argv)
    cli.main()

    checkpoint = next(path for path in tmp_path.rglob("*.ckpt") if path.name == "last.ckpt")
    monkeypatch.setattr("sys.argv", argv + ["--weights", str(checkpoint)])

    cli.main()
