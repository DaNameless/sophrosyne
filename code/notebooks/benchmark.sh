#!/bin/bash
set -e   # stop on first error

cd "$(dirname "$0")"

for i in 1 2 ; do
    echo ">>> Running with $i ranks..."
    mpirun -n $i python benchmark.py
done

python benchmark.py --plot
