"""
linz_escape_min_N.py
====================
Computes the minimum-N escape map for the Linz-Sprott coupled system
over a 2-D grid of (a, ε) values.

For each grid point, binary-searches the smallest ensemble size N that
keeps the system bounded.  Results are saved as a compressed .npz file
and plotted as a heatmap (log₁₀ N_c colorbar; grey = always escaping).

MPI-aware: run with   mpirun -n <N>  python linz_escape_min_N.py
or plain:             python linz_escape_min_N.py
"""

import matplotlib
matplotlib.use('Agg')

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.sophrosyne import (
    LinzSprott,
    EscapeMapAnalyzer,
)

# ── MPI setup ─────────────────────────────────────────────────────────────────
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    comm, rank, size = None, 0, 1

# ── Output directory ──────────────────────────────────────────────────────────
OUT = "./linz_escape_min_N"
if rank == 0:
    os.makedirs(OUT, exist_ok=True)
if size > 1:
    comm.Barrier()

# ── Grid ──────────────────────────────────────────────────────────────────────
# 'a' is the Linz-Sprott damping parameter; chaotic attractor lives near 0.5–0.6.
# ε is the mean-field coupling strength.
A_VALUES   = np.linspace(0.50, 0.55, 200)
EPS_VALUES = np.linspace(0.00, 0.4, 200)

# ── Integration & search settings ─────────────────────────────────────────────
DT        = 0.05        # time step (same as all_figures.py)
STEPS     = 50_000      # steps per trial
WINDOW    = 5_000       # rolling tail buffer (not stored during transient)
THRESHOLD = 100.0       # |state| > THRESHOLD → escaped
N_MIN     = 1           # lower bound for binary search
N_MAX     = 100_000     # upper bound; nan in output means always escaping here
N_TRIALS  = 3           # independent trials per grid point (different seeds)

if rank == 0:
    print("=" * 60)
    print("Linz-Sprott  —  minimum-N escape map")
    print(f"  grid    : {len(A_VALUES)} × {len(EPS_VALUES)} = "
          f"{len(A_VALUES) * len(EPS_VALUES)} points")
    print(f"  N range : [{N_MIN}, {N_MAX}]")
    print(f"  dt={DT}, steps={STEPS}, threshold={THRESHOLD}, trials={N_TRIALS}")
    print(f"  backend : {'MPI ×' + str(size) if size > 1 else str(os.cpu_count()) + ' workers'}")
    print("=" * 60)

# ── Analyzer ──────────────────────────────────────────────────────────────────
analyzer = EscapeMapAnalyzer(
    system_cls  = LinzSprott,
    base_params = {},
    epsilon     = 0.0,      # ε is swept via param2_name="epsilon"
    dt          = DT,
    steps       = STEPS,
    window      = WINDOW,
    threshold   = THRESHOLD,
    n_jobs      = os.cpu_count() or 1,
)

# ── Compute ───────────────────────────────────────────────────────────────────
result = analyzer.compute_min_N(
    param1_name   = "a",
    param1_values = A_VALUES,
    param2_name   = "epsilon",
    param2_values = EPS_VALUES,
    N_min         = N_MIN,
    N_max         = N_MAX,
    n_trials      = N_TRIALS,
    seed          = 42,
)

# ── Save & plot (root only) ───────────────────────────────────────────────────
if rank == 0 and result is not None:
    # Save raw data
    out_npz = f"{OUT}/escape_map_min_N.npz"
    np.savez_compressed(
        out_npz,
        a_values   = result.param1_values,
        eps_values = result.param2_values,
        min_N      = result.min_N,
        N_max      = np.array(result.N_max),
    )
    print(f"  -> Saved {out_npz}")

    print(f"Done → {OUT}")
