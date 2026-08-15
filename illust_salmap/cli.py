"""
Training entrypoint.

    illust-salmap --model unet --dataset salicon
    illust-salmap --model unet_v2 --dataset cat2000 --epochs 100 --batch-size 8

Datasets download themselves on first use. Outputs (TensorBoard logs and checkpoints)
are written under `--output-root`, keyed by dataset and model so runs do not overwrite
each other.

Use `--smoke` for a two-batch dry run: it exercises download, training, validation,
test and image logging in about a minute, which is worth doing before renting a GPU.
"""

import argparse
import os
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import MSELoss
from torch.optim import Adam

from illust_salmap.dataset.cat2000 import Cat2000
from illust_salmap.dataset.imp1k import Imp1k
from illust_salmap.dataset.salicon import SALICON
from illust_salmap.models.dummy_net import dummy_net
from illust_salmap.models.unet.unet import unet
from illust_salmap.models.unet.unet_lite import unet_lite
from illust_salmap.models.unet.unet_v2 import unet_v2
from illust_salmap.models.unet.unet_v3 import unet_v3
from illust_salmap.training.losses import SaliencyLoss
from illust_salmap.training.saliency_model import SaliencyModel
from illust_salmap.training.paths import get_checkpoint_path, get_log_path

DATASETS = {
    "salicon": SALICON,
    "cat2000": Cat2000,
    "imp1k": Imp1k,
}

MODELS = {
    "unet": unet,
    "unet_lite": unet_lite,
    "unet_v2": unet_v2,
    "unet_v3": unet_v3,
    "dummy": dummy_net,
}

LOSSES = {
    "saliency": SaliencyLoss,
    "mse": MSELoss,
}

# What each logged quantity wants: a loss and a divergence go down, a similarity goes up.
# `val_loss` is whatever `--loss` selected, so selecting on it only means "the best model"
# when that loss is the saliency one.
MONITORS = {
    "val_loss": "min",
    "val_kl_div": "min",
    "val_cc": "max",
    "val_sim": "max",
    "val_nss": "max",
    "val_auroc": "max",
}


def parse_args():
    # prog is pinned so `python -m illust_salmap` does not report itself as `__main__.py`.
    parser = argparse.ArgumentParser(prog="illust-salmap", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", default="saliency", choices=sorted(LOSSES),
                        help="saliency: KL + CC + NSS, the metrics themselves. mse: pixelwise, "
                             "optimizes none of them and drifts towards the average map.")
    parser.add_argument("--img-size", type=int, nargs=2, default=None, metavar=("HEIGHT", "WIDTH"),
                        help="Defaults to each dataset's own size, which matches its aspect ratio.")
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--grad-clip", type=float, default=10.0,
                        help="Clip gradients to this global norm. 0 disables it. A healthy step "
                             "measures around 9 at initialization and falls from there, so this is "
                             "a safety net for a collapsing prediction rather than a regularizer.")

    parser.add_argument("--data-root", type=Path, default=Path("./data"),
                        help="Where datasets are downloaded and extracted.")
    parser.add_argument("--output-root", type=Path, default=Path("./output"),
                        help="Where logs and checkpoints are written.")
    parser.add_argument("--weights", type=Path, default=None,
                        help="Initialize the network from a checkpoint (weights only, no optimizer state).")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume a run from a Lightning checkpoint (restores optimizer and epoch).")

    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default=None,
                        help="Lightning precision string. Defaults to bf16/fp16 mixed on CUDA, fp32 otherwise.")
    parser.add_argument("--monitor", default="val_loss", choices=sorted(MONITORS),
                        help="Quantity that selects the best checkpoint and drives early stopping.")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience on --monitor. 0 disables it.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true",
                        help="Two batches per stage for one epoch, to verify the pipeline end to end.")

    return parser.parse_args()


def default_precision() -> str:
    """Picks the fastest precision the accelerator supports without hurting stability."""
    if not torch.cuda.is_available():
        return "32-true"

    return "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"


def build_optimization(lr: float):
    def build(params):
        return Adam(params, lr=lr)

    return build


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)

    if torch.cuda.is_available():
        # Lets Ampere and newer use TF32 for matmuls; a large speedup at no practical cost here.
        torch.set_float32_matmul_precision("high")

    network = MODELS[args.model](args.weights) if args.weights else MODELS[args.model]()

    dataset_options = dict(
        root=str(args.data_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    if args.img_size:
        dataset_options["img_size"] = tuple(args.img_size)

    datamodule = DATASETS[args.dataset](**dataset_options)
    criterion = LOSSES[args.loss]()
    module = SaliencyModel(network, criterion=criterion, optimization_builder=build_optimization(args.lr))

    log_path = get_log_path(args.output_root, datamodule, network)
    checkpoint_path = get_checkpoint_path(args.output_root, datamodule, network)

    mode = MONITORS[args.monitor]
    logger = TensorBoardLogger(save_dir=str(log_path), name="")
    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_path),
            filename="{epoch:03d}-{" + args.monitor + ":.4f}",
            monitor=args.monitor,
            mode=mode,
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if args.patience > 0 and not args.smoke:
        callbacks.append(EarlyStopping(monitor=args.monitor, mode=mode, patience=args.patience))

    limit_batches = 2 if args.smoke else 1.0
    trainer = Trainer(
        max_epochs=1 if args.smoke else args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision or default_precision(),
        logger=logger,
        callbacks=callbacks,
        limit_train_batches=limit_batches,
        limit_val_batches=limit_batches,
        limit_test_batches=limit_batches,
        log_every_n_steps=10,
        gradient_clip_val=args.grad_clip or None,
    )

    parameters = sum(p.numel() for p in network.parameters()) / 1e6
    print(f"model      : {type(network).__name__} ({parameters:.2f} M params)")
    print(f"dataset    : {type(datamodule).__name__} at {args.data_root}")
    print(f"loss       : {criterion}")
    print(f"monitor    : {args.monitor} ({mode})")
    print(f"precision  : {trainer.precision}")
    print(f"logs       : {log_path}")
    print(f"checkpoints: {checkpoint_path}")

    trainer.fit(module, datamodule=datamodule, ckpt_path=str(args.resume) if args.resume else None)
    trainer.test(module, datamodule=datamodule, ckpt_path="best")

    best = trainer.checkpoint_callback.best_model_path
    if best:
        print(f"best checkpoint: {best}")


if __name__ == "__main__":
    main()
