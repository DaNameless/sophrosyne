#!/bin/bash
#SBATCH --partition=LocalQ
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:30:00
#SBATCH --output=/home/gisc1/rssg/sophrosyne/benchmark/%x_%j.out
#SBATCH --error=/home/gisc1/rssg/sophrosyne/benchmark/%x_%j.err
#SBATCH --chdir=/home/gisc1/rssg/sophrosyne/benchmark

set -e   # stop on first error

BENCHMARK=/home/gisc1/rssg/sophrosyne/benchmark/benchmark.py
PYTHON=$(which python)

for i in 1 2 4 8 16 ; do
    echo ">>> Running with $i ranks..."
    mpirun -n $i $PYTHON $BENCHMARK
done

$PYTHON $BENCHMARK --plot
