#!/bin/bash
#SBATCH --partition=LocalQ
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --time=72:00:00
#SBATCH --output=/home/gisc1/rssg/sophrosyne/paper_figures/all_%j.out
#SBATCH --error=/home/gisc1/rssg/sophrosyne/paper_figures/all_%j.err
#SBATCH --chdir=/home/gisc1/rssg/sophrosyne/paper_figures/

set -e

PYTHON=$(which python)

mpirun -n 16 $PYTHON all_figures.py
