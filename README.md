# illust_salmap

A machine learning project for generating saliency maps of illustrations. The project utilizes various deep learning techniques to analyze images and produce attention maps that highlight key features and areas of interest.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). `uv.lock` pins every
package, so the same versions are installed on the dev machine and on the training box.

```bash
uv sync
```

`torch` / `torchvision` are resolved per-platform: Windows gets the CPU wheels, Linux
gets the CUDA 12.4 build. Run anything through `uv run`, e.g.:

```bash
uv run python -m illust_salmap.models.unet
```

Tests need no network access:

```bash
uv run pytest
```

## Training

```bash
uv run illust-salmap --model unet --dataset salicon
```

Models: `unet`, `unet_lite`, `unet_v2`, `unet_v3`, and `dummy` for smoke runs.
Datasets: `salicon`, `cat2000`, `imp1k` — each downloads itself on first use.

Logs and checkpoints land under `--output-root` (default `./output`), keyed by dataset
and model. `--weights` starts from a trained network, `--resume` continues a run.
See `--help` for the rest.

Before spending GPU time, verify the pipeline end to end with two batches:

```bash
uv run illust-salmap --model dummy --dataset salicon --smoke
```

### RunPod

Clone onto the volume and run the setup script — it installs uv, points the venv and the
uv cache at `/workspace`, syncs, and finishes with a smoke run:

```bash
git clone https://github.com/sumx21t-3310/illust_salmap.git /workspace/illust_salmap
```

```bash
cd /workspace/illust_salmap && ./scripts/runpod_setup.sh
```

It is idempotent, so run it again after every pod restart. `--no-smoke` skips the dry run.

What it is working around: the container disk is wiped when the pod stops, so anything
outside `/workspace` — including the default `.venv` — is gone on the next boot.

```bash
export UV_CACHE_DIR=/workspace/.uv-cache UV_PROJECT_ENVIRONMENT=/workspace/.venv UV_LINK_MODE=copy
```

Note that torch 2.5.1 / CUDA 12.4 supports up to sm_90 — pick an A100, H100, L40S or 4090
pod. Blackwell cards (5090, B200) would need torch 2.7+ on cu128; the script warns when it
finds one.

## Directory Structure

Everything lives under `illust_salmap/`. The four layers below are stacked, and no layer
imports from one above it.

- `installer/`: Downloads and extracts the dataset archives, and records what is installed.
- `dataset/`: One LightningDataModule per dataset, on top of the installer. `split.py`
  holds the stratified split, `stats.py` the mean/std pass over a dataset.
- `models/`: The UNet-family networks, plus `dummy_net` for smoke runs. `checkpoint.py`
  loads weights into a bare `nn.Module`, `ez_bench.py` times a forward pass.
- `training/`: The LightningModule wrapper, the saliency metrics, the output-path helpers
  (`paths.py`) and the TensorBoard plot (`visualize.py`).
- `cli.py`: The `illust-salmap` entrypoint — the dataset and model registries, and the
  Trainer wiring.