#!/usr/bin/env bash
# RunPod のポッドを学習できる状態にする。冪等なので、再接続のたびに実行してよい。
#
#   git clone https://github.com/sumx21t-3310/illust_salmap.git /workspace/illust_salmap
#   cd /workspace/illust_salmap
#   ./scripts/runpod_setup.sh
#
# uv の導入 → venv とキャッシュを永続ボリュームへ退避 → uv sync → スモークラン、まで行う。
# スモークは SALICON を実際にダウンロードするので、不要なら --no-smoke。
#
# 注意: 環境変数は子プロセスにしか効かないので、このシェルに引き継ぎたい場合は
# `source scripts/runpod_setup.sh` するか、~/.bashrc に書いた分を読み直すこと（下で追記する）。
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SMOKE=1

for arg in "$@"; do
    case "$arg" in
        --no-smoke) SMOKE=0 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

# ポッド停止でコンテナディスクは消える。ボリュームが無いなら、そこに何を置いても失われる。
if [ ! -d "$WORKSPACE" ]; then
    echo "!! $WORKSPACE が無い。ボリュームを付けたポッドで実行すること" >&2
    echo "   （別の場所に置くなら WORKSPACE=/path $0）" >&2
    exit 1
fi

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA_ROOT="$WORKSPACE/data"
OUTPUT_ROOT="$WORKSPACE/output"

case "$REPO_ROOT" in
    "$WORKSPACE"/*) ;;
    *) echo "!! リポジトリが $WORKSPACE の外にある（$REPO_ROOT）。ポッド停止で消える" >&2 ;;
esac

# torch 2.5.1 / cu124 は sm_90 まで。Blackwell（5090, B200 = sm_120）ではカーネルが無い。
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)
    echo "gpu        : ${gpu:-unknown}"
    major=${gpu##*, }
    major=${major%%.*}
    case "$major" in ''|*[!0-9]*) major="" ;; esac
    if [ -n "$major" ] && [ "$major" -ge 10 ]; then
        echo "!! この GPU は torch 2.5.1 / cu124 の対応外（sm_90 まで）。" >&2
        echo "   A100 / H100 / L40S / 4090 のポッドを取るか、torch 2.7+ / cu128 へ上げること" >&2
    fi
else
    echo "gpu        : nvidia-smi が無い（CPU ポッド？）"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "==> uv を導入"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# インストーラは ~/.local/bin に置くだけで、実行中のシェルの PATH は通さない。
if [ -f "$HOME/.local/bin/env" ]; then
    # shellcheck disable=SC1091
    . "$HOME/.local/bin/env"
fi

# venv もキャッシュもボリュームへ。UV_LINK_MODE=copy は、キャッシュと venv が
# 別ファイルシステムに載るためのハードリンク不可への対処。
export UV_CACHE_DIR="$WORKSPACE/.uv-cache"
export UV_PROJECT_ENVIRONMENT="$WORKSPACE/.venv"
export UV_LINK_MODE=copy

# ~/.bashrc 自体がコンテナディスクにあるなら停止で消えるが、その場合はこれを再実行すれば戻る。
MARKER="# illust_salmap runpod setup"
if ! grep -qF "$MARKER" "$HOME/.bashrc" 2>/dev/null; then
    echo "==> ~/.bashrc に uv の参照先を追記"
    cat >> "$HOME/.bashrc" <<EOF

$MARKER
export UV_CACHE_DIR="$WORKSPACE/.uv-cache"
export UV_PROJECT_ENVIRONMENT="$WORKSPACE/.venv"
export UV_LINK_MODE=copy
EOF
fi

echo "==> uv sync"
cd "$REPO_ROOT"
uv sync

if [ "$SMOKE" -eq 1 ]; then
    echo "==> スモークラン（SALICON をダウンロードするので初回は時間がかかる）"
    uv run illust-salmap --model dummy --dataset salicon --smoke \
        --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT"
fi

cat <<EOF

準備完了。学習は tmux の中で:

    tmux new -s train
    uv run illust-salmap --model unet_v3 --dataset salicon \\
        --data-root $DATA_ROOT --output-root $OUTPUT_ROOT

TensorBoard（ポッドの 6006 を expose しておく）:

    uv run tensorboard --logdir $OUTPUT_ROOT --host 0.0.0.0 --port 6006

中断した学習の再開:

    uv run illust-salmap --model unet_v3 --dataset salicon \\
        --data-root $DATA_ROOT --output-root $OUTPUT_ROOT \\
        --resume $OUTPUT_ROOT/SALICON/UNetV3/checkpoints/last.ckpt
EOF
