"""
all_figures.py
==============
Runs every available analysis for:
  - Discrete maps : Tent, Logistic, Lozi
  - Continuous    : Linz-Sprott

MPI-aware: run with   mpirun -n <N>  python all_figures.py
or plain:             python all_figures.py
"""

import matplotlib
matplotlib.use('Agg')

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.sophrosyne import (
    LinzSprott, SimulationRunner,
    BifurcationAnalyzer, BifurcationPlotter,
    EscapeMapAnalyzer, EscapeMapPlotter,
    HistogramPlotter, Plotter,
)
from modules.discrete_sophrosyne import CoupledMapLattice

# ── MPI setup ────────────────────────────────────────────────────────────────
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    comm, rank, size = None, 0, 1

OUT = "../../outputs/papers_figures"

def mkdir(path):
    if rank == 0:
        os.makedirs(path, exist_ok=True)
    if size > 1:
        comm.Barrier()

# ─────────────────────────────────────────────────────────────────────────────
# Shared grid settings
# ─────────────────────────────────────────────────────────────────────────────
GRID = dict(n_a=80, n_eps=80, N_min=1, N_max=50_000,
            T_total=4_000, x_bound=20, n_trials=3, n_jobs=1)

N_VALUES  = (1_000, 10_000, 100_000)
EPS_FIXED = 0.4          # representative coupling for timeseries / histogram
CMAP      = 'plasma'


# ══════════════════════════════════════════════════════════════════════════════
# TENT MAP
# ══════════════════════════════════════════════════════════════════════════════
TENT_OUT = f"{OUT}/tent"
mkdir(TENT_OUT)

sys_tent = CoupledMapLattice.from_tent(a=3.0, output_dir=TENT_OUT)

# 1. Time series
if rank == 0:
    sys_tent.analysis_timeseries(
        eps=EPS_FIXED, N_values=N_VALUES,
        T_show=300, T_transient=10_000,
    )

# 2. Fluctuation scaling
if rank == 0:
    sys_tent.analysis_fluctuation_scaling(
        eps=EPS_FIXED, N_values=N_VALUES,
        T_show=2_000, T_transient=10_000,
    )

# 3. Distribution
if rank == 0:
    sys_tent.analysis_distribution(
        eps=EPS_FIXED, N_values=N_VALUES,
        T_show=2_000, T_transient=10_000,
    )

# 4. Self-consistency
if rank == 0:
    sys_tent.analysis_self_consistency(
        eps=EPS_FIXED, h_range=(0.0, 1.0),
        N_sc=5_000, T_sc=50_000, T_trans_sc=10_000,
        T_transient=10_000, T_show=500,
    )

# 5. Phase diagram α(ε)
sys_tent.analysis_phase_diagram(
    eps_values=np.linspace(0.01, 0.9, 60),
    N_values=N_VALUES, n_jobs=1,
)

# 6. Mean-field bifurcation — sweep ε
sys_tent.analysis_meanfield_bifurcation(
    sweep='eps', eps_range=(0.0, 1.0),
    N=5_000, n_param=300,
    T_total=20_000, T_transient=18_000,
)

# 7. Mean-field bifurcation — sweep a
sys_tent.analysis_meanfield_bifurcation(
    sweep='map_param',
    map_factory=lambda a: CoupledMapLattice.from_tent(a=a),
    param_range=(2.0, 4.0), sweep_label='a',
    eps_fixed=EPS_FIXED, N=5_000, n_param=300,
    T_total=20_000, T_transient=18_000,
)

# 8. Escape phase diagram — sweep a
result_tent = sys_tent.analysis_escape_phase_diagram(
    map_factory=lambda a: CoupledMapLattice.from_tent(a=a),
    a_range=(2.0, 4.0), eps_range=(0.0, 1.0),
    sweep_label='a', cmap=CMAP,
    output=f"{TENT_OUT}/escape_map_a.png", show=False,
    **GRID,
)
if rank == 0 and result_tent is not None:
    sys_tent.save_data(result_tent, 'escape_map_a.npz')

# 9. Element histogram
if rank == 0:
    sys_tent.analysis_element_histogram(
        N=10_000, eps=EPS_FIXED,
        T_total=50_000, T_transient=10_000,
        plot_elements=True, plot_mean_field=True,
        gaussian_fit=None,
        output=f"{TENT_OUT}/histogram_N10000_eps{EPS_FIXED}.png", show=False,
    )

if rank == 0:
    print(f"Tent done → {TENT_OUT}")


# ══════════════════════════════════════════════════════════════════════════════
# LOGISTIC MAP
# ══════════════════════════════════════════════════════════════════════════════
LOG_OUT = f"{OUT}/logistic"
mkdir(LOG_OUT)

EPS_FIXED = 0.2
sys_log = CoupledMapLattice.from_logistic(r=4.2, output_dir=LOG_OUT)

# 1–4. Standard analyses
if rank == 0:
    sys_log.analysis_timeseries(
        eps=EPS_FIXED, N_values=N_VALUES,
        T_show=300, T_transient=10_000,
    )
    sys_log.analysis_fluctuation_scaling(
        eps=EPS_FIXED, N_values=N_VALUES,
        T_show=2_000, T_transient=10_000,
    )
    sys_log.analysis_distribution(
        eps=EPS_FIXED, N_values=N_VALUES,
        T_show=2_000, T_transient=10_000,
    )
    sys_log.analysis_self_consistency(
        eps=EPS_FIXED, h_range=(0.0, 1.0),
        N_sc=5_000, T_sc=50_000, T_trans_sc=10_000,
        T_transient=10_000, T_show=500,
    )

# 5. Phase diagram
sys_log.analysis_phase_diagram(
    eps_values=np.linspace(0.01, 0.9, 60),
    N_values=N_VALUES, n_jobs=1,
)

# 6–7. Bifurcation diagrams
sys_log.analysis_meanfield_bifurcation(
    sweep='eps', eps_range=(0.0, 1.0),
    N=5_000, n_param=300,
    T_total=20_000, T_transient=18_000,
)
sys_log.analysis_meanfield_bifurcation(
    sweep='map_param',
    map_factory=lambda r: CoupledMapLattice.from_logistic(r=r),
    param_range=(3.9, 5.0), sweep_label='r',
    eps_fixed=EPS_FIXED, N=5_000, n_param=300,
    T_total=20_000, T_transient=18_000,
)

# 8. Escape phase diagram — sweep r
result_log = sys_log.analysis_escape_phase_diagram(
    map_factory=lambda r: CoupledMapLattice.from_logistic(r=r),
    a_range=(3.9, 5.0), eps_range=(0.0, 1.0),
    sweep_label='r', cmap=CMAP,
    output=f"{LOG_OUT}/escape_map_r.png", show=False,
    **GRID,
)
if rank == 0 and result_log is not None:
    sys_log.save_data(result_log, 'escape_map_r.npz')

# 9. Element histogram
if rank == 0:
    sys_log.analysis_element_histogram(
        N=10_000, eps=EPS_FIXED,
        T_total=50_000, T_transient=10_000,
        plot_elements=True, plot_mean_field=True,
        gaussian_fit=None,
        output=f"{LOG_OUT}/histogram_N10000_eps{EPS_FIXED}.png", show=False,
    )

if rank == 0:
    print(f"Logistic done → {LOG_OUT}")


# ══════════════════════════════════════════════════════════════════════════════
# LOZI MAP
# ══════════════════════════════════════════════════════════════════════════════
LOZI_OUT = f"{OUT}/lozi"
mkdir(LOZI_OUT)

EPS_LOZI = 0.25

sys_lozi = CoupledMapLattice.from_lozi(a=2, b=0.5, output_dir=LOZI_OUT)

# 1–4. Standard analyses
if rank == 0:
    sys_lozi.analysis_timeseries(
        eps=EPS_LOZI, N_values=N_VALUES,
        T_show=300, T_transient=10_000,
    )
    sys_lozi.analysis_fluctuation_scaling(
        eps=EPS_LOZI, N_values=N_VALUES,
        T_show=2_000, T_transient=10_000,
    )
    sys_lozi.analysis_distribution(
        eps=EPS_LOZI, N_values=N_VALUES,
        T_show=2_000, T_transient=10_000,
    )
    sys_lozi.analysis_self_consistency(
        eps=EPS_LOZI, h_range=(-0.5, 1.5),
        N_sc=5_000, T_sc=50_000, T_trans_sc=10_000,
        T_transient=10_000, T_show=500,
    )

# 5. Phase diagram
sys_lozi.analysis_phase_diagram(
    eps_values=np.linspace(0.01, 0.9, 60),
    N_values=N_VALUES, n_jobs=1,
)

# 6–7. Bifurcation diagrams
sys_lozi.analysis_meanfield_bifurcation(
    sweep='eps', eps_range=(0.0, 1.0),
    N=5_000, n_param=300,
    T_total=20_000, T_transient=18_000,
)
sys_lozi.analysis_meanfield_bifurcation(
    sweep='map_param',
    map_factory=lambda a: CoupledMapLattice.from_lozi(a=a, b=0.5),
    param_range=(1.75, 3.0), sweep_label='a',
    eps_fixed=EPS_LOZI, N=5_000, n_param=300,
    T_total=20_000, T_transient=18_000,
)

# 8a. Escape phase diagram — sweep a (b fixed)
result_lozi_a = sys_lozi.analysis_escape_phase_diagram(
    map_factory=lambda a: CoupledMapLattice.from_lozi(a=a, b=0.5),
    a_range=(1.75, 3.0), eps_range=(0.0, 1.0),
    sweep_label='a', cmap=CMAP,
    output=f"{LOZI_OUT}/escape_map_a.png", show=False,
    **GRID,
)
if rank == 0 and result_lozi_a is not None:
    sys_lozi.save_data(result_lozi_a, 'escape_map_a.npz')

# 8b. Escape phase diagram — sweep b (a fixed)
result_lozi_b = sys_lozi.analysis_escape_phase_diagram(
    map_factory=lambda b: CoupledMapLattice.from_lozi(a=1.8, b=b),
    a_range=(0., 1), eps_range=(0.0, 1.0),
    sweep_label='b', cmap=CMAP,
    output=f"{LOZI_OUT}/escape_map_b.png", show=False,
    **GRID,
)
if rank == 0 and result_lozi_b is not None:
    sys_lozi.save_data(result_lozi_b, 'escape_map_b.npz')

# 9. Element histogram (x and y panels)
if rank == 0:
    sys_lozi.analysis_element_histogram(
        N=10_000, eps=EPS_LOZI,
        T_total=50_000, T_transient=10_000,
        plot_elements=True, plot_mean_field=True,
        gaussian_fit=None,
        output=f"{LOZI_OUT}/histogram_N10000_eps{EPS_LOZI}.png", show=False,
    )

if rank == 0:
    print(f"Lozi done → {LOZI_OUT}")


# ══════════════════════════════════════════════════════════════════════════════
# LINZ-SPROTT  (continuous)
# ══════════════════════════════════════════════════════════════════════════════
LINZ_OUT = f"{OUT}/linz"
mkdir(LINZ_OUT)

A_LINZ   = 0.53
EPS_LINZ = 0.1
DT       = 0.05

# ── 1. Time series + attractor + sigma (single representative run) ────────────
if rank == 0:
    runner = SimulationRunner(
        system_cls=LinzSprott, system_params={"a": A_LINZ},
        N=100, epsilon=EPS_LINZ, dt=DT,
        steps=200_000, window=10_000, verbose=False,
    )
    sim = runner.run()
    Plotter.plot_tail_timeseries(sim, output_dir=LINZ_OUT, show=False)
    Plotter.plot_attractor_3d(sim, output_dir=LINZ_OUT, show=False)
    Plotter.plot_sigma_evolution(sim, output_dir=LINZ_OUT, show=False)

# ── 2. Bifurcation diagram — sweep a ─────────────────────────────────────────
if rank == 0:
    bif = BifurcationAnalyzer(
        LinzSprott, base_params={},
        t_transient=1000, t_steady=1000, dt=DT,
        threshold=200.0,
    )
    result_bif = bif.compute("a", np.linspace(0.5, 0.6, 400))
    BifurcationPlotter.plot(
        result_bif,
        output=f"{LINZ_OUT}/bifurcation_a.png", show=False,
    )

# ── 3. Escape map (fixed N) — ε vs a ─────────────────────────────────────────
analyzer = EscapeMapAnalyzer(
    system_cls=LinzSprott, base_params={},
    epsilon=0.0, dt=DT,
    steps=50_000, window=5_000, threshold=100.0,
    n_jobs=os.cpu_count() or 1,
)

result_fixed = analyzer.compute(
    param1_name="a",   param1_values=np.linspace(0.5, 0.6, 80),
    param2_name="epsilon", param2_values=np.linspace(0.0, 1.0, 80),
    seed=42,
)
if rank == 0 and result_fixed is not None:
    EscapeMapPlotter.plot(
        result_fixed,
        output=f"{LINZ_OUT}/escape_map_fixed_N.png", cmap="viridis", show=False,
    )

# ── 4. Escape map (min N) — ε vs a ───────────────────────────────────────────
result_min_N = analyzer.compute_min_N(
    param1_name="a",   param1_values=np.linspace(0.5, 0.6, 80),
    param2_name="epsilon", param2_values=np.linspace(0.0, 1.0, 80),
    N_min=1, N_max=10_000, n_trials=3,
)
if rank == 0 and result_min_N is not None:
    EscapeMapPlotter.plot_min_N(
        result_min_N,
        output=f"{LINZ_OUT}/escape_map_min_N.png", cmap=CMAP, show=False,
    )
    np.savez_compressed(
        f"{LINZ_OUT}/escape_map_min_N.npz",
        a_values=result_min_N.param1_values,
        eps_values=result_min_N.param2_values,
        min_N=result_min_N.min_N,
    )

# ── 5. Histograms ─────────────────────────────────────────────────────────────
if rank == 0:
    HistogramPlotter.plot(
        sim,
        plot_elements=True, plot_mean_field=True,
        gaussian_fit="elements",
        output=f"{LINZ_OUT}/histogram_elements_and_mf.png", show=False,
    )
    HistogramPlotter.plot(
        sim,
        plot_elements=True, plot_mean_field=False,
        gaussian_fit=None,
        output=f"{LINZ_OUT}/histogram_elements_nofit.png", show=False,
    )

if rank == 0:
    print(f"Linz-Sprott done → {LINZ_OUT}")
