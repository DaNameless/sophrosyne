"""
Figures for all discrete maps (Logistic, Lozi).
Mirrors the 'Tent Figures' section of tent.ipynb.

Map physics notes
─────────────────
Logistic  f(x) = r·x·(1−x),  r = 3.9
  · Chaotic for r ≳ 3.57; fully chaotic at r = 4.
  · Unstable fixed point  x* = 1 − 1/r ≈ 0.744
  · |f'(x*)| = |r − 2| = 1.9  →  ε* = 1 − 1/1.9 ≈ 0.47
  · h̄ ≈ 0.5  (mean of invariant measure, exact at r = 4)
  · Escape region: r > 4 (map leaves [0, 1] without coupling)

Lozi  f(x,y) = 1 − a|x| + y,  a = 1.7, b = 0.5
  · Chaotic strange attractor for a ≈ 1.4–1.9, b = 0.5
  · h̄ ≈ 0.32  (estimated; verify with a short run_all call)
  · Escape region: a ≳ 1.8 where the attractor grows/destabilises
  · x-component tracked in trajectory plots; attractor spans ≈ [−0.8, 0.8]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.discrete_sophrosyne import CoupledMapLattice


# ══════════════════════════════════════════════════════════════════════════════
# LOGISTIC MAP   r = 4.2
# ══════════════════════════════════════════════════════════════════════════════

sys_logistic = CoupledMapLattice.from_logistic(
    r=4.2, output_dir="../../outputs/papers_figures/logistic"
)

# ── Trajectory plots ──────────────────────────────────────────────────────────
# ε = 0
sys_logistic.plot_trajectories(
    10000, eps=0, n_show=10, T_total=15, T_transient=0,
    save=True, y_lim=(-0.5, 1.25)
)
# ε = 0.05
sys_logistic.plot_trajectories(
    10000, eps=0.05, n_show=10, T_total=15, T_transient=0,
    save=True, y_lim=(-0.5, 1.25)
)
# ε = 0.35: above ε* ≈ 0.47, wait for convergence then show last 15 steps
sys_logistic.plot_trajectories(
    10000, eps=0.5, n_show=5,
    T_total=int(1e6) + 15, T_transient=int(1e6),
    save=True, y_lim=(0.3, 1.0)
)
# ε = 0.75: well above ε*, tight synchronisation
sys_logistic.plot_trajectories(
    10000, eps=0.75, n_show=5, T_total=15, T_transient=0,
    save=True, y_lim=(-0.5, 1.25)
)

# ── Core statistical analyses at ε = 0.35 ─────────────────────────────────────
sys_logistic.analysis_timeseries(eps=0.35, N_values=[1000, 10000, 100000, 1000000])

sys_logistic.analysis_fluctuation_scaling(eps=0.35, N_values=[1000, 10000, 100000, 1000000])

sys_logistic.analysis_distribution(eps=0.35, N_values=[1000, 10000, 100000, 1000000])

sys_logistic.analysis_self_consistency(eps=0.35)

# ── Bifurcation diagrams ──────────────────────────────────────────────────────
# h̄ ≈ 0.5 (mean of logistic invariant measure at r = 3.9)
sys_logistic.bifurcation_diagram(
    bifurcation_param="eps",
    h_bar=0.5,
    y_lim=(-0.5, 1.5)
)

sys_logistic.bifurcation_diagram(
    bifurcation_param='map_param',
    map_factory=lambda r: CoupledMapLattice.from_logistic(r=r),
    param_range=(2.0, 4.0),
    eps_fixed=0.5,
    h_bar=0.5,
    sweep_label='r'
)

# ── Mean-field bifurcation ────────────────────────────────────────────────────
sys_logistic.analysis_meanfield_bifurcation(N=1000, n_jobs=4, n_param=1000)

# ── Minimum N to prevent escape ───────────────────────────────────────────────
# Escape occurs for r ≳ 4 where the uncoupled map leaves [0, 1];
# coupling can stabilise the system if N is large enough.
sys_logistic.analysis_min_N_escape(
    sweep='map_param',
    map_factory=lambda r: CoupledMapLattice.from_logistic(r=r),
    param_range=(3.7, 4.5),
    eps_fixed=0.4,
    n_param=100,
    n_jobs=4
)

sys_logistic.analysis_min_N_escape(
    sweep='map_param',
    map_factory=lambda r: CoupledMapLattice.from_logistic(r=r),
    param_range=(3.7, 4.5),
    eps_fixed=0.4,
    n_param=10,
    n_jobs=2,
    sweep_label='r'
)


# ══════════════════════════════════════════════════════════════════════════════
# LOZI MAP   a = 1.7, b = 0.5
# ══════════════════════════════════════════════════════════════════════════════

sys_lozi = CoupledMapLattice.from_lozi(
    a=1.7, b=0.5, output_dir="../../outputs/papers_figures/lozi"
)

# ── Trajectory plots ──────────────────────────────────────────────────────────
# ε = 0: uncoupled, strange attractor, x ∈ [−0.8, 0.8] approximately
sys_lozi.plot_trajectories(
    10000, eps=0, n_show=5, T_total=15, T_transient=0,
    save=True, y_lim=(-1.5, 1.5)
)
sys_lozi.plot_trajectories(
    10000, eps=0.25, n_show=5, T_total=15, T_transient=0,
    save=True, y_lim=(-1.5, 1.5)
)
# ε = 0.5: wait for transient, then show last 15 steps
sys_lozi.plot_trajectories(
    10000, eps=0.5, n_show=5,
    T_total=int(1e6) + 15, T_transient=int(1e6),
    save=True, y_lim=(-0.5, 1.0)
)
sys_lozi.plot_trajectories(
    10000, eps=0.75, n_show=5, T_total=15, T_transient=0,
    save=True, y_lim=(-0.5, 1.0)
)

# ── Core statistical analyses at ε = 0.5 ─────────────────────────────────────
sys_lozi.analysis_timeseries(eps=0.5, N_values=[1000, 10000, 100000, 1000000])

sys_lozi.analysis_fluctuation_scaling(eps=0.5, N_values=[1000, 10000, 100000, 1000000])

sys_lozi.analysis_distribution(eps=0.5, N_values=[1000, 10000, 100000, 1000000])

sys_lozi.analysis_self_consistency(eps=0.5)

# ── Bifurcation diagrams ──────────────────────────────────────────────────────
# h̄ ≈ 0.32  (1 − 1.7·E[|x|] + E[y]; E[y] ≈ 0 by symmetry, E[|x|] ≈ 0.4)
sys_lozi.bifurcation_diagram(
    bifurcation_param="eps",
    h_bar=0.32,
    y_lim=(-0.5, 1.5)
)

sys_lozi.bifurcation_diagram(
    bifurcation_param='map_param',
    map_factory=lambda a: CoupledMapLattice.from_lozi(a=a),
    param_range=(1.0, 2.0),
    eps_fixed=0.5,
    h_bar=0.32,
    sweep_label='a'
)

# ── Mean-field bifurcation ────────────────────────────────────────────────────
sys_lozi.analysis_meanfield_bifurcation(N=1000, n_jobs=4, n_param=1000)

# ── Minimum N to prevent escape ───────────────────────────────────────────────
# Lozi attractor destabilises and escapes for a ≳ 1.8–2.0 at moderate ε.
sys_lozi.analysis_min_N_escape(
    sweep='map_param',
    map_factory=lambda a: CoupledMapLattice.from_lozi(a=a),
    param_range=(1.7, 2.5),
    eps_fixed=0.4,
    n_param=100,
    n_jobs=4
)

sys_lozi.analysis_min_N_escape(
    sweep='map_param',
    map_factory=lambda a: CoupledMapLattice.from_lozi(a=a),
    param_range=(1.7, 2.5),
    eps_fixed=0.4,
    n_param=10,
    n_jobs=2,
    sweep_label='a'
)
