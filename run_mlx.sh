#!/usr/bin/env bash
# Launch the MLX / Apple-silicon visualizer (app_mlx.py) natively on the Mac.
#
# MLX uses Metal, so this runs on the Mac host. It installs the MLX deps into
# the local .venv on first run, then starts the Gradio app on
# http://localhost:7860.
#
# Usage:
#   ./run_mlx.sh                 # real model (downloads ~16.6GB MLX quant once)
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "warning: MLX requires Apple silicon (arm64 macOS); this host is $(uname -s)/$(uname -m)." >&2
fi

VENV=".venv"
if [[ ! -d "$VENV" ]]; then
  echo "Creating virtualenv in $VENV ..."
  python3 -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Install deps only when mlx_vlm is missing (skip the slow path on every run).
if ! python -c "import mlx_vlm" >/dev/null 2>&1; then
  echo "Installing MLX dependencies (first run only) ..."
  pip install -r requirements-mlx.txt
fi

exec python app_mlx.py
