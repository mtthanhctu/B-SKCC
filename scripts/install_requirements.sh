#!/usr/bin/env bash
set -euo pipefail

# Assumes your desired conda environment is already activated.
# Installs Python dependencies for the project.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "Installing project requirements into current environment: $(which python)"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
