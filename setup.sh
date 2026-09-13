#!/usr/bin/env bash
# One-command setup for macOS / Linux.  Run from the repo folder:
#   ./setup.sh              (auto GPU/CPU)
#   ./setup.sh --cpu        (force CPU builds)
#   ./setup.sh --skip-deep
#
# Creates two separate environments (they need different PyTorch versions):
#   .venv                                         fast model, export, server, tests
#   vmax_model2/Track_limit_detection/.venv_bench deep model
set -euo pipefail
cd "$(dirname "$0")"
# Copy instead of hardlinking from uv's cache: hardlinks fail inside cloud-synced
# folders and across filesystems.
export UV_LINK_MODE=copy

CPU=0; SKIP_DEEP=0
for arg in "$@"; do
  case "$arg" in
    --cpu) CPU=1 ;;
    --skip-deep) SKIP_DEEP=1 ;;
    *) echo "Unknown option: $arg"; exit 2 ;;
  esac
done
step() { printf '\n==> %s\n' "$1"; }
warn() { printf 'WARNING: %s\n' "$1"; }

step 'Checking for uv'
if ! command -v uv >/dev/null 2>&1; then
  command -v python3 >/dev/null 2>&1 || { echo 'Neither uv nor python3 found. Install uv: https://docs.astral.sh/uv/'; exit 1; }
  echo 'uv not found; installing it with pip'
  python3 -m pip install --user --quiet uv
  export PATH="$(python3 -m site --user-base)/bin:$PATH"
  command -v uv >/dev/null 2>&1 || { echo 'uv installed but not on PATH; add ~/.local/bin to PATH and re-run.'; exit 1; }
fi
uv --version

step 'Checking the deep model checkpoint (Git LFS)'
CKPT=vmax_model2/Track_limit_detection/training_runs/boundary_heatmaps_20260912_191710_183913/best_model/boundary_heatmaps.pt
size() { wc -c <"$1" | tr -d ' '; }
if [ -f "$CKPT" ] && [ "$(size "$CKPT")" -lt 1048576 ]; then
  if command -v git-lfs >/dev/null 2>&1; then git lfs install --local && git lfs pull
  else warn 'Git LFS is not installed, so the deep model checkpoint is only a placeholder. Install it (https://git-lfs.com), then: git lfs pull'; fi
fi
if [ -f "$CKPT" ] && [ "$(size "$CKPT")" -ge 1048576 ]; then echo "Checkpoint OK"; fi

step 'Choosing PyTorch builds'
# torch 2.14: CUDA 13.0 (driver >= 580) or CPU.  torch 2.11: CUDA 12.8 (driver >= 570) or CPU.
# macOS uses the default PyPI builds (Apple Silicon GPU via MPS).
DRIVER=0
if [ "$CPU" = 0 ] && command -v nvidia-smi >/dev/null 2>&1; then
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || echo 0)
  DRIVER=${DRIVER:-0}
fi
if [ "$(uname)" = Darwin ]; then
  FAST_INDEX=(); DEEP_INDEX=(); echo 'macOS: using default PyPI PyTorch builds'
else
  if [ "$DRIVER" -ge 580 ]; then FAST=cu130; else FAST=cpu; fi
  if [ "$DRIVER" -ge 570 ]; then DEEP=cu128; else DEEP=cpu; fi
  FAST_INDEX=(--index-url "https://download.pytorch.org/whl/$FAST")
  DEEP_INDEX=(--index-url "https://download.pytorch.org/whl/$DEEP")
  [ "$DRIVER" = 0 ] && echo 'No NVIDIA GPU detected (or --cpu): installing CPU builds. Everything works, inference is slower.'
  [ "$DRIVER" != 0 ] && [ "$DRIVER" -lt 580 ] && warn "NVIDIA driver $DRIVER < 580: the fast model uses the CPU build."
  echo "Fast model: torch 2.14.0 ($FAST)   Deep model: torch 2.11.0 ($DEEP)"
fi

step 'Creating .venv (fast model, server, tests)'
uv venv .venv --python 3.12 --allow-existing
FAST_PY=.venv/bin/python
uv pip install --python "$FAST_PY" torch==2.14.0 torchvision==0.29.0 ${FAST_INDEX[@]+"${FAST_INDEX[@]}"}
uv pip install --python "$FAST_PY" -r requirements-fast.txt
"$FAST_PY" -c "import torch, ultralytics, cv2, numpy; print('fast env OK: torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

if [ "$SKIP_DEEP" = 0 ]; then
  step 'Creating .venv_bench (deep model)'
  DEEP_DIR=vmax_model2/Track_limit_detection
  uv venv "$DEEP_DIR/.venv_bench" --python 3.12 --allow-existing
  DEEP_PY="$DEEP_DIR/.venv_bench/bin/python"
  uv pip install --python "$DEEP_PY" torch==2.11.0 torchvision==0.26.0 ${DEEP_INDEX[@]+"${DEEP_INDEX[@]}"}
  uv pip install --python "$DEEP_PY" -r "$DEEP_DIR/requirements-deep.txt"
  "$DEEP_PY" -c "import torch, transformers; print('deep env OK: torch', torch.__version__, '| transformers', transformers.__version__, '| CUDA available:', torch.cuda.is_available())"
fi

step 'Checking FFmpeg'
if command -v ffmpeg >/dev/null 2>&1; then echo 'FFmpeg found'
elif [ -n "${FFMPEG:-}" ]; then echo "Using FFMPEG=$FFMPEG"
else warn 'FFmpeg not found. Only needed to render new simulator clips.'; fi

step 'Done'
echo 'Start the demo:  .venv/bin/python vmax_live_server.py   then open http://127.0.0.1:8010'
