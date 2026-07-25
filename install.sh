#!/usr/bin/env bash
# One-shot environment setup for this repository on macOS (Apple Silicon or
# Intel). Idempotent: safe to re-run.
#
# What it does:
#   1. Verifies Xcode Command Line Tools (mmcv is compiled from source).
#   2. Installs uv if missing.
#   3. Creates the virtualenv and installs all pinned dependencies
#      (mmengine / mmcv / mmpretrain / mmdet / mmsegmentation / mmdet3d).
#   4. Optionally downloads MNIST with --with-data.
set -euo pipefail
cd "$(dirname "$0")"

log() { printf '\033[1;34m[install.sh]\033[0m %s\n' "$*"; }

if [[ "$(uname -s)" != "Darwin" ]]; then
  log "WARNING: this script targets macOS; continuing anyway."
fi

# 1. Xcode Command Line Tools (provides clang for the mmcv source build)
if ! xcode-select -p >/dev/null 2>&1; then
  log "Xcode Command Line Tools are missing. Starting installer..."
  xcode-select --install
  log "Re-run ./install.sh after the installation finishes."
  exit 1
fi
log "Xcode Command Line Tools: $(xcode-select -p)"

# 2. uv
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
log "uv: $(uv --version)"

# 3. Python + dependencies (mmcv has no macOS wheel and is built from
#    source here — expect several minutes on the first run).
#    -Wno-invalid-specialization: torch 2.1 headers specialize
#    std::is_arithmetic, which libc++ >= 17 (new Xcode/clang) rejects by
#    default; older clang silently ignores the flag.
export CFLAGS="${CFLAGS:+$CFLAGS }-Wno-invalid-specialization"
export CXXFLAGS="${CXXFLAGS:+$CXXFLAGS }-Wno-invalid-specialization"
log "Syncing environment (first run compiles mmcv, please be patient)..."
MAX_JOBS="${MAX_JOBS:-$(sysctl -n hw.ncpu)}" uv sync --group dev

log "Environment report:"
uv run python -m mmmac.utils.env

# 4. Optional: MNIST for the sample configs / overfit test
if [[ "${1:-}" == "--with-data" ]]; then
  log "Downloading MNIST to data/mnist ..."
  uv run python tools/misc/download_mnist.py
fi

log "Done. Try: uv run python tools/train.py configs/mnist/lenet5_mnist_overfit.py"
