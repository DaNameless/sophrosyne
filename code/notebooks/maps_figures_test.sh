#!/bin/bash
#SBATCH --partition=LocalQ
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --output=/home/gisc1/rssg/sophrosyne/sophrosyne/code/notebooks/%x_%j.out
#SBATCH --error=/home/gisc1/rssg/sophrosyne/sophrosyne/code/notebooks/%x_%j.err
#SBATCH --chdir=/home/gisc1/rssg/sophrosyne/sophrosyne/code/notebooks

set -e

PYTHON=$(which python)

mpirun -n 16 $PYTHON maps_figures_test.py
