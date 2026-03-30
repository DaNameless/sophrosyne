import matplotlib
matplotlib.use('Agg')

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.discrete_sophrosyne import CoupledMapLattice

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
except ImportError:
    rank = 0


# ══════════════════════════════════════════════════════════════════════════════
# TENT MAP   a = 3, eps = 0.5
# ══════════════════════════════════════════════════════════════════════════════

sys_tent = CoupledMapLattice.from_tent(
    a=3, output_dir="../../outputs/papers_figures/tent"
)


# ── Trajectory plots ──────────────────────────────────────────────────────
sys_tent.plot_trajectories(
    10000, eps=0, n_show=5, T_total=15, T_transient=0,
    save=True, y_lim=(-5, 2)
)
sys_tent.plot_trajectories(
    10000, eps=0.25, n_show=5, T_total=15, T_transient=0,
    save=True, y_lim=(-5, 2)
)
sys_tent.plot_trajectories(
    10000, eps=0.5, n_show=5,
    T_total=int(1e6) + 15, T_transient=int(1e6),
    save=True, y_lim=(0, 1.5)
)
sys_tent.plot_trajectories(
    10000, eps=0.75, n_show=5, T_total=15, T_transient=0,
    save=True, y_lim=(0, 1.5)
)

# ── Core statistical analyses ─────────────────────────────────────────────
sys_tent.analysis_timeseries(eps=0.5, N_values=[1000, 10000, 100000, 1000000])
sys_tent.analysis_fluctuation_scaling(eps=0.5, N_values=[1000, 10000, 100000, 1000000])
sys_tent.analysis_distribution(eps=0.5, N_values=[1000, 10000, 100000, 1000000])
sys_tent.analysis_self_consistency(eps=0.5)

# ── Bifurcation diagrams ──────────────────────────────────────────────────
sys_tent.bifurcation_diagram(bifurcation_param="eps", h_bar=0.684, y_lim=(-0.5, 1.5))
sys_tent.bifurcation_diagram(
    bifurcation_param='map_param',
    map_factory=lambda a: CoupledMapLattice.from_tent(a=a),
    param_range=(0, 3), eps_fixed=0.5, h_bar=0.684
)

# ── Mean-field bifurcation (MPI-aware) ────────────────────────────────────────
sys_tent.analysis_meanfield_bifurcation(N=100000, n_param=2000)

sys_tent.analysis_meanfield_bifurcation(
    sweep='map_param',
    map_factory=lambda a: CoupledMapLattice.from_tent(a=a),
    param_range=(2, 3.4), eps_fixed=0.5,
    sweep_label="a", n_param=2000, N=100000
)

# ── Minimum N to prevent escape (MPI-aware) ───────────────────────────────────
sys_tent.analysis_min_N_escape(
    sweep='map_param',
    map_factory=lambda a: CoupledMapLattice.from_tent(a=a),
    param_range=(2, 3.4),
    eps_fixed=0.5,
    n_param=1000,
    N_max=1000000,
    sweep_label="a"
)

sys_tent.analysis_min_N_escape(
    sweep='map_param',
    map_factory=lambda a: CoupledMapLattice.from_tent(a=a),
    param_range=(2, 3.4),
    eps_fixed=0.4,
    n_param=1000,
    N_max=1000000,
    sweep_label="a"
)


# ══════════════════════════════════════════════════════════════════════════════
# LOGISTIC MAP   r = 4.2, eps = 0.35
# ══════════════════════════════════════════════════════════════════════════════

sys_logistic = CoupledMapLattice.from_logistic(
    r=4.2, output_dir="../../outputs/papers_figures/logistic"
)

# ── Trajectory plots ──────────────────────────────────────────────────────
sys_logistic.plot_trajectories(
    10000, eps=0, n_show=10, T_total=15, T_transient=0,
    save=True, y_lim=(-0.5, 1.25)
)
sys_logistic.plot_trajectories(
    10000, eps=0.05, n_show=10, T_total=15, T_transient=0,
    save=True, y_lim=(-0.5, 1.25)
)
sys_logistic.plot_trajectories(
    10000, eps=0.35, n_show=5,
    T_total=int(1e6) + 15, T_transient=int(1e6),
    save=True, y_lim=(0.3, 1.0)
)
sys_logistic.plot_trajectories(
    10000, eps=0.75, n_show=5, T_total=15, T_transient=0,
    save=True, y_lim=(-0.5, 1.25)
)

# ── Core statistical analyses ─────────────────────────────────────────────
sys_logistic.analysis_timeseries(eps=0.35, N_values=[1000, 10000, 100000, 1000000])
sys_logistic.analysis_fluctuation_scaling(eps=0.35, N_values=[1000, 10000, 100000, 1000000])
sys_logistic.analysis_distribution(eps=0.35, N_values=[1000, 10000, 100000, 1000000])
sys_logistic.analysis_self_consistency(eps=0.35)

# ── Bifurcation diagrams ──────────────────────────────────────────────────
sys_logistic.bifurcation_diagram(bifurcation_param="eps", h_bar=0.685, y_lim=(-0.5, 1.5))
sys_logistic.bifurcation_diagram(
    bifurcation_param='map_param',
    map_factory=lambda r: CoupledMapLattice.from_logistic(r=r),
    param_range=(3.0, 5.0), eps_fixed=0.35,
    h_bar=0.685, sweep_label='r'
)

# ── Mean-field bifurcation (MPI-aware) ────────────────────────────────────────
sys_logistic.analysis_meanfield_bifurcation(N=100000, n_param=2000)

sys_logistic.analysis_meanfield_bifurcation(
    sweep='map_param',
    map_factory=lambda r: CoupledMapLattice.from_logistic(r=r),
    param_range=(3.0, 5.0), eps_fixed=0.35,
    sweep_label='r', n_param=2000, N=100000
)

# ── Minimum N to prevent escape (MPI-aware) ───────────────────────────────────
sys_logistic.analysis_min_N_escape(
    sweep='map_param',
    map_factory=lambda r: CoupledMapLattice.from_logistic(r=r),
    param_range=(4, 5),
    eps_fixed=0.35,
    n_param=1000,
    N_max=1000000,
    sweep_label="r"
)

sys_logistic.analysis_min_N_escape(
    sweep='map_param',
    map_factory=lambda r: CoupledMapLattice.from_logistic(r=r),
    param_range=(4, 5),
    eps_fixed=0.2,
    n_param=1000,
    N_max=1000000,
    sweep_label='r'
)


# ══════════════════════════════════════════════════════════════════════════════
# LOZI MAP   a = 2, b = 0.5, eps = 0.32
# ══════════════════════════════════════════════════════════════════════════════

sys_lozi = CoupledMapLattice.from_lozi(
    a=2, b=0.5, output_dir="../../outputs/papers_figures/lozi"
)

# ── Trajectory plots (x and y components) ────────────────────────────────
sys_lozi.plot_trajectories(
    10000, eps=0, n_show=5, T_total=60, T_transient=0,
    save=True, y_lim=(-5, 1.5)
)
sys_lozi.plot_trajectories(
    10000, eps=0, n_show=5, T_total=60, T_transient=0,
    save=True, y_lim=(-5, 1), component=1
)
sys_lozi.plot_trajectories(
    10000, eps=0.1, n_show=5, T_total=60, T_transient=0,
    save=True, y_lim=(-5, 1.5)
)
sys_lozi.plot_trajectories(
    10000, eps=0.1, n_show=5, T_total=60, T_transient=0,
    save=True, y_lim=(-5, 1.5), component=1
)
sys_lozi.plot_trajectories(
    10000, eps=0.32, n_show=5,
    T_total=int(1e5) + 60, T_transient=int(1e5),
    save=True, y_lim=(-0.75, 1.25)
)
sys_lozi.plot_trajectories(
    10000, eps=0.32, n_show=5,
    T_total=int(1e5) + 60, T_transient=int(1e5),
    save=True, y_lim=(-0.5, 0.75), component=1
)
sys_lozi.plot_trajectories(
    10000, eps=0.5, n_show=5, T_total=60, T_transient=0,
    save=True, y_lim=(-5, 1.5)
)
sys_lozi.plot_trajectories(
    10000, eps=0.5, n_show=5, T_total=60, T_transient=0,
    save=True, y_lim=(-5, 1.5), component=1
)

# ── Core statistical analyses ─────────────────────────────────────────────
sys_lozi.analysis_timeseries(eps=0.32, N_values=[1000, 10000, 100000, 1000000])
sys_lozi.analysis_fluctuation_scaling(eps=0.32, N_values=[1000, 10000, 100000, 1000000])
sys_lozi.analysis_distribution(eps=0.32, N_values=[1000, 10000, 100000, 1000000])
sys_lozi.analysis_self_consistency(eps=0.32)

# ── Bifurcation diagrams (x and y components) ─────────────────────────────
sys_lozi.bifurcation_diagram(bifurcation_param="eps", h_bar=0.218)
sys_lozi.bifurcation_diagram(bifurcation_param="eps", h_bar=0.218, component=1)
sys_lozi.bifurcation_diagram(
    bifurcation_param='map_param',
    map_factory=lambda a: CoupledMapLattice.from_lozi(a=a, b=0.5),
    param_range=(1.4, 3), eps_fixed=0.32,
    h_bar=0.218, sweep_label='a'
)
sys_lozi.bifurcation_diagram(
    bifurcation_param='map_param',
    map_factory=lambda a: CoupledMapLattice.from_lozi(a=a, b=0.5),
    param_range=(1.4, 3), eps_fixed=0.32,
    h_bar=0.218, sweep_label='a', component=1
)

# ── Mean-field bifurcation (MPI-aware) ────────────────────────────────────────
sys_lozi.analysis_meanfield_bifurcation(N=100000, n_param=2000)

sys_lozi.analysis_meanfield_bifurcation(
    sweep='map_param',
    map_factory=lambda a: CoupledMapLattice.from_lozi(a=a, b=0.5),
    param_range=(1.4, 3), eps_fixed=0.32,
    sweep_label='a', n_param=2000, N=100000
)

# ── Minimum N to prevent escape (MPI-aware) ───────────────────────────────────
sys_lozi.analysis_min_N_escape(
    sweep='map_param',
    map_factory=lambda a: CoupledMapLattice.from_lozi(a=a, b=0.5),
    param_range=(1.4, 3),
    eps_fixed=0.32,
    n_param=1000,
    N_max=1000000,
    sweep_label='a'
)

sys_lozi.analysis_min_N_escape(
    sweep='map_param',
    map_factory=lambda a: CoupledMapLattice.from_lozi(a=a, b=0.5),
    param_range=(1.4, 3),
    eps_fixed=0.2,
    n_param=1000,
    N_max=1000000,
    sweep_label='a'
)
