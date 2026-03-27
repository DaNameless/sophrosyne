"""
MPI Scaling Benchmark — discrete_sophrosyne
============================================
Measures wall time for analysis_phase_diagram and analysis_min_N_escape
as a function of the number of MPI ranks.

Usage
-----
Run at different process counts and collect results:

    mpirun -n 1  python benchmark.py
    mpirun -n 2  python benchmark.py
    mpirun -n 4  python benchmark.py
    mpirun -n 8  python benchmark.py
    mpirun -n 16 python benchmark.py

Each run appends one row to benchmark_results.csv.
After all runs, call:

    python benchmark.py --plot

to read the CSV and produce the scaling plot.
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no windows, safe on clusters

from modules.discrete_sophrosyne import CoupledMapLattice, tent

# ── benchmark parameters (keep small so each run is fast) ─────────────────────
EPS_VALUES  = np.arange(0.01, 0.5,  0.04)   # 13 values  — phase diagram
N_VALUES    = (100, 1000, 10000)             # 3 sizes    — phase diagram
PARAM_RANGE = (2, 3.4)                   # map param sweep — min_N
N_PARAM     = 20                            # points along the sweep — min_N
EPS_FIXED   = 0.1
RESULTS_CSV = "../../outputs/benchmark_results.csv"


def make_system():
    return CoupledMapLattice(
        lambda x: tent(x, a=1.99), name="tent_benchmark", output_dir="../../outputs/tent_benchmark"
    )


def bench_phase_diagram(sys):
    t0 = time.perf_counter()
    sys.analysis_phase_diagram(eps_values=EPS_VALUES, N_values=N_VALUES)
    return time.perf_counter() - t0


def bench_min_N(sys):
    factory = lambda a: CoupledMapLattice(
        lambda x: tent(x, a=a), name=f"tent_a{a:.2f}"
    )
    t0 = time.perf_counter()
    sys.analysis_min_N_escape(
        sweep='map_param',
        map_factory=factory,
        param_range=PARAM_RANGE,
        n_param=N_PARAM,
        eps_fixed=EPS_FIXED,
        sweep_label='a',
    )
    return time.perf_counter() - t0


def run_benchmark():
    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
    except ImportError:
        rank, size = 0, 1

    sys_obj = make_system()

    # synchronise all ranks before timing
    if size > 1:
        from mpi4py import MPI
        MPI.COMM_WORLD.Barrier()

    t_phase = bench_phase_diagram(sys_obj)

    if size > 1:
        MPI.COMM_WORLD.Barrier()

    t_minn  = bench_min_N(sys_obj)

    # only rank 0 writes results
    if rank == 0:
        csv_path = Path(RESULTS_CSV)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        header = not csv_path.exists()
        with open(csv_path, 'a') as f:
            if header:
                f.write("n_ranks,analysis,wall_time_s\n")
            f.write(f"{size},phase_diagram,{t_phase:.4f}\n")
            f.write(f"{size},min_N_escape,{t_minn:.4f}\n")

        print(f"\n{'='*50}")
        print(f"  Ranks              : {size}")
        print(f"  phase_diagram      : {t_phase:.2f} s")
        print(f"  min_N_escape       : {t_minn:.2f} s")
        print(f"  Results appended to: {csv_path}")
        print(f"{'='*50}\n")


def plot_scaling():
    import matplotlib.pyplot as plt
    import csv

    csv_path = Path(RESULTS_CSV)
    if not csv_path.exists():
        print(f"No results file found at {csv_path}")
        return

    data = {}   # {analysis: {n_ranks: wall_time}}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            analysis = row['analysis']
            n        = int(row['n_ranks'])
            t        = float(row['wall_time_s'])
            data.setdefault(analysis, {})[n] = t

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (analysis, timings) in zip(axes, data.items()):
        ranks  = sorted(timings)
        times  = [timings[n] for n in ranks]
        t1     = timings.get(1, times[0])
        speedup    = [t1 / t for t in times]
        efficiency = [s / n for s, n in zip(speedup, ranks)]

        ax2 = ax.twinx()
        ax.plot(ranks, speedup,    'ko-',  ms=5, lw=1.2, label='Speedup')
        ax.plot(ranks, ranks,      'k--',  lw=0.8, alpha=0.4, label='Ideal')
        ax2.plot(ranks, efficiency, 's--', ms=4, lw=0.9, color='gray',
                 label='Efficiency')

        ax.set_xlabel('MPI ranks')
        ax.set_ylabel('Speedup')
        ax2.set_ylabel('Efficiency', color='gray')
        ax2.set_ylim(0, 1.2)
        ax.set_title(analysis.replace('_', ' '))
        ax.legend(loc='upper left')
        ax2.legend(loc='lower right')

    plt.suptitle('MPI Scaling — discrete_sophrosyne', fontsize=12)
    plt.tight_layout()
    out = Path(RESULTS_CSV).parent / "benchmark_scaling.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true',
                        help='Plot results from benchmark_results.csv')
    args = parser.parse_args()

    if args.plot:
        plot_scaling()
    else:
        run_benchmark()
