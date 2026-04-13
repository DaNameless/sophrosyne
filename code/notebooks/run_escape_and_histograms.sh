#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$HOME/miniforge3/envs/dynsys-mpi/bin/python"
N_CORES="${1:-16}"

echo "========================================"
echo "  Escape Maps + Histograms — $(date)"
echo "  cores : $N_CORES"
echo "  script: $SCRIPT_DIR/escape_and_histograms.py"
echo "========================================"

mpirun -n "$N_CORES" "$PYTHON" "$SCRIPT_DIR/escape_and_histograms.py"

echo "========================================"
echo "  Done — $(date)"
echo "========================================"
