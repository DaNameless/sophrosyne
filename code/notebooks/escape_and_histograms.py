import matplotlib
matplotlib.use('Agg')

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.sophrosyne import (
    LinzSprott, SimulationRunner,
    EscapeMapAnalyzer, EscapeMapPlotter,
    HistogramPlotter,
)
from modules.discrete_sophrosyne import CoupledMapLattice

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
except ImportError:
    rank = 0

OUT = "../../outputs/papers_figures"

# ══════════════════════════════════════════════════════════════════════════════
# LINZ-SPROTT  (continuous)
# ══════════════════════════════════════════════════════════════════════════════
# Note: EscapeMapAnalyzer uses mp.Pool internally — run on a single MPI rank.

LINZ_OUT = f"{OUT}/linz"
os.makedirs(LINZ_OUT, exist_ok=True)

if rank == 0:

    # ── Escape map: min N to prevent escape, sweep a × epsilon ───────────────
    analyzer = EscapeMapAnalyzer(
        system_cls  = LinzSprott,
        base_params = {},
        epsilon     = 0.0,          # fixed; swept as param2
        dt          = 0.005,
        steps       = 50_000,
        window      = 5_000,
        threshold   = 100.0,
        n_jobs      = -1,
    )

    result_min_N = analyzer.compute_min_N(
        param1_name   = "a",
        param1_values = np.linspace(0.3, 0.8, 60),
        param2_name   = "epsilon",
        param2_values = np.linspace(0.0, 0.15, 60),
        N_min         = 1,
        N_max         = 500,
        n_trials      = 3,
    )
    EscapeMapPlotter.plot_min_N(
        result_min_N,
        output = f"{LINZ_OUT}/escape_map_min_N.png",
        cmap   = "plasma",
    )
    np.savez_compressed(
        f"{LINZ_OUT}/escape_map_min_N.npz",
        a_values   = result_min_N.param1_values,
        eps_values = result_min_N.param2_values,
        min_N      = result_min_N.min_N,
    )

    # ── Fixed-N escape map: colour by escape time ─────────────────────────────
    result_fixed = analyzer.compute(
        param1_name   = "a",
        param1_values = np.linspace(0.3, 0.8, 80),
        param2_name   = "epsilon",
        param2_values = np.linspace(0.0, 0.15, 80),
        seed          = 42,
    )
    EscapeMapPlotter.plot(
        result_fixed,
        output = f"{LINZ_OUT}/escape_map_fixed_N.png",
        cmap   = "viridis",
    )

    # ── Histograms at a representative bounded point ──────────────────────────
    runner = SimulationRunner(
        system_cls   = LinzSprott,
        system_params = {"a": 0.54},
        N            = 100,
        epsilon      = 0.01,
        dt           = 0.005,
        steps        = 200_000,
        window       = 10_000,
        verbose      = False,
    )
    sim = runner.run()
    HistogramPlotter.plot(
        sim,
        output       = f"{LINZ_OUT}/histograms_a0.54_eps0.01.png",
        gaussian_fit = True,
    )
    HistogramPlotter.plot(
        sim,
        output       = f"{LINZ_OUT}/histograms_a0.54_eps0.01_nofit.png",
        gaussian_fit = False,
    )
    print(f"Linz-Sprott done → {LINZ_OUT}")


# ══════════════════════════════════════════════════════════════════════════════
# TENT MAP  (discrete, MPI-aware)
# ══════════════════════════════════════════════════════════════════════════════

TENT_OUT = f"{OUT}/tent_escape"
os.makedirs(TENT_OUT, exist_ok=True) if rank == 0 else None

sys_tent = CoupledMapLattice.from_tent(a=1.99, output_dir=TENT_OUT)

result_tent = sys_tent.analysis_escape_phase_diagram(
    map_factory = lambda a: CoupledMapLattice.from_tent(a=a),
    a_range     = (1.0, 3.0),
    eps_range   = (0.0, 0.6),
    n_a         = 80,
    n_eps       = 80,
    N_min       = 1,
    N_max       = 20_000,
    T_total     = 4_000,
    x_bound     = 20,
    n_trials    = 3,
    n_jobs      = 1,           # MPI handles parallelism
    sweep_label = 'a',
    cmap        = 'plasma',
    output      = f"{TENT_OUT}/escape_map_a.png",
    show        = False,
)
if rank == 0 and result_tent is not None:
    sys_tent.save_data(result_tent, 'escape_map_a.npz')

    # ── Element histogram ─────────────────────────────────────────────────────
    sys_tent.analysis_element_histogram(
        N           = 10_000,
        eps         = 0.3,
        T_total     = 200_000,
        T_transient = 100_000,
        output      = f"{TENT_OUT}/histogram_N10000_eps0.3.png",
        show        = False,
    )
    print(f"Tent done → {TENT_OUT}")


# ══════════════════════════════════════════════════════════════════════════════
# LOZI MAP  (discrete, MPI-aware)
# ══════════════════════════════════════════════════════════════════════════════

LOZI_OUT = f"{OUT}/lozi_escape"
os.makedirs(LOZI_OUT, exist_ok=True) if rank == 0 else None

sys_lozi = CoupledMapLattice.from_lozi(a=1.7, b=0.5, output_dir=LOZI_OUT)

# ── Escape map: sweep a (b fixed) ────────────────────────────────────────────
result_lozi_a = sys_lozi.analysis_escape_phase_diagram(
    map_factory = lambda a: CoupledMapLattice.from_lozi(a=a, b=0.5),
    a_range     = (1.0, 2.5),
    eps_range   = (0.0, 0.6),
    n_a         = 80,
    n_eps       = 80,
    N_min       = 1,
    N_max       = 20_000,
    T_total     = 4_000,
    x_bound     = 20,
    n_trials    = 3,
    n_jobs      = 1,
    sweep_label = 'a',
    cmap        = 'plasma',
    output      = f"{LOZI_OUT}/escape_map_a.png",
    show        = False,
)
if rank == 0 and result_lozi_a is not None:
    sys_lozi.save_data(result_lozi_a, 'escape_map_a.npz')

# ── Escape map: sweep b (a fixed) ────────────────────────────────────────────
result_lozi_b = sys_lozi.analysis_escape_phase_diagram(
    map_factory = lambda b: CoupledMapLattice.from_lozi(a=1.7, b=b),
    a_range     = (0.1, 0.9),
    eps_range   = (0.0, 0.6),
    n_a         = 80,
    n_eps       = 80,
    N_min       = 1,
    N_max       = 20_000,
    T_total     = 4_000,
    x_bound     = 20,
    n_trials    = 3,
    n_jobs      = 1,
    sweep_label = 'b',
    cmap        = 'plasma',
    output      = f"{LOZI_OUT}/escape_map_b.png",
    show        = False,
)
if rank == 0 and result_lozi_b is not None:
    sys_lozi.save_data(result_lozi_b, 'escape_map_b.npz')

    # ── Element histogram (x and y panels) ───────────────────────────────────
    sys_lozi.analysis_element_histogram(
        N           = 10_000,
        eps         = 0.32,
        T_total     = 200_000,
        T_transient = 100_000,
        output      = f"{LOZI_OUT}/histogram_N10000_eps0.32.png",
        show        = False,
    )
    print(f"Lozi done → {LOZI_OUT}")
