#!/usr/bin/env bash
set -euo pipefail

# Assumes your conda environment is already activated and dependencies are installed.
# Pass all training arguments after this script.
# Examples:
#   scripts/run_training.sh --model resnet
#   scripts/run_training.sh --all
#   scripts/run_training.sh --bagging 3 --models resnet vit efficientnet

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "Using Python: $(which python)"
echo "Launching training with args: $*"
python -m training.train "$@"
