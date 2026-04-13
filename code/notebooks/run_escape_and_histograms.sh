#!/bin/bash
#SBATCH --partition=LocalQ
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --output=/home/gisc1/rssg/sophrosyne/paper_figures/escape_%j.out
#SBATCH --error=/home/gisc1/rssg/sophrosyne/paper_figures/escape_%j.err
#SBATCH --chdir=/home/gisc1/rssg/sophrosyne/paper_figures/

set -e

PYTHON=$(which python)

mpirun -n 16 $PYTHON escape_and_histograms.py
