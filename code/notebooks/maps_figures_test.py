"""
maps_figures_test.py
====================
Quick sanity-check / timing script that mirrors maps_figures.py exactly
but uses drastically reduced parameters so it finishes in minutes locally.

Usage
-----
    python maps_figures_test.py          # single process
    mpirun -n 4 python maps_figures_test.py  # MPI

Differences from maps_figures.py
---------------------------------
    N_values      : [100, 500, 1000]       (vs 1k–1M)
    T_transient   : 2_000                  (vs 1M)
    T_show        : 200                    (vs 2000)
    n_param       : 30                     (vs 500–2000)
    N (meanfield) : 500                    (vs 10k–100k)
    N_max (min_N) : 5_000                  (vs 100k–1M)

Everything else — analysis types, save filenames, MPI guards — is
identical so the output can be compared 1-to-1 against the full run.
"""

import matplotlib
matplotlib.use('Agg')

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.discrete_sophrosyne import CoupledMapLattice

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
except ImportError:
    rank = 0

# ── Reduced parameters ────────────────────────────────────────────────────────
N_VALUES    = [100, 500, 1000]
T_TRANSIENT = 2_000
T_SHOW      = 200
N_PARAM     = 30
N_MF        = 500
N_MAX       = 5_000

OUTPUT_ROOT = "../../outputs/test_figures"

_timings = {}

def _timed(label, fn):
    t0 = time.perf_counter()
    result = fn()
    _timings[label] = time.perf_counter() - t0
    return result


# ══════════════════════════════════════════════════════════════════════════════
# TENT MAP   a = 3
# ══════════════════════════════════════════════════════════════════════════════

sys_tent = CoupledMapLattice.from_tent(
    a=3, output_dir=f"{OUTPUT_ROOT}/tent_a3"
)

if rank == 0:
    # ── Trajectory plots ──────────────────────────────────────────────────────
    data = _timed('tent/trajectories_eps0', lambda: sys_tent.plot_trajectories(
        1000, eps=0, n_show=5, T_total=20, T_transient=0,
        save=True, y_lim=(-1, 1.5)
    ))
    sys_tent.save_data(data, 'trajectories_eps0.npz')

    data = _timed('tent/trajectories_eps0.25', lambda: sys_tent.plot_trajectories(
        1000, eps=0.25, n_show=5, T_total=20, T_transient=0,
        save=True, y_lim=(-1, 1.5)
    ))
    sys_tent.save_data(data, 'trajectories_eps0.25.npz')

    data = _timed('tent/trajectories_eps0.5', lambda: sys_tent.plot_trajectories(
        1000, eps=0.5, n_show=5,
        T_total=T_TRANSIENT + 20, T_transient=T_TRANSIENT,
        save=True, y_lim=(0, 1.5)
    ))
    sys_tent.save_data(data, 'trajectories_eps0.5.npz')

    data = _timed('tent/trajectories_eps0.75', lambda: sys_tent.plot_trajectories(
        1000, eps=0.75, n_show=5, T_total=20, T_transient=0,
        save=True, y_lim=(-1, 1.5)
    ))
    sys_tent.save_data(data, 'trajectories_eps0.75.npz')

    # ── Core statistical analyses ─────────────────────────────────────────────
    data = _timed('tent/timeseries', lambda: sys_tent.analysis_timeseries(
        eps=0.5, N_values=N_VALUES, T_show=T_SHOW, T_transient=T_TRANSIENT
    ))
    sys_tent.save_data(data, 'timeseries_eps0.5.pkl')

    data = _timed('tent/scaling', lambda: sys_tent.analysis_fluctuation_scaling(
        eps=0.5, N_values=N_VALUES, T_show=T_SHOW, T_transient=T_TRANSIENT
    ))
    sys_tent.save_data(data, 'scaling_eps0.5.npz')

    data = _timed('tent/distribution', lambda: sys_tent.analysis_distribution(
        eps=0.5, N_values=N_VALUES, T_show=T_SHOW, T_transient=T_TRANSIENT
    ))
    sys_tent.save_data(data, 'distribution_eps0.5.pkl')

    data = _timed('tent/self_consistency', lambda: sys_tent.analysis_self_consistency(
        eps=0.5, T_sc=5_000, T_trans_sc=1_000, T_transient=T_TRANSIENT, T_show=T_SHOW
    ))
    sys_tent.save_data(data, 'self_consistency_eps0.5.npz')

    # ── Bifurcation diagrams ──────────────────────────────────────────────────
    data = _timed('tent/bifurcation_eps', lambda: sys_tent.bifurcation_diagram(
        bifurcation_param="eps", h_bar=0.684, y_lim=(-0.5, 1.5),
        n_param=N_PARAM, T_iter=500, T_trans=400
    ))
    sys_tent.save_data(data, 'bifurcation_eps.npz')

    data = _timed('tent/bifurcation_map_param', lambda: sys_tent.bifurcation_diagram(
        bifurcation_param='map_param',
        map_factory=lambda a: CoupledMapLattice.from_tent(a=a),
        param_range=(1, 3), eps_fixed=0.5, h_bar=0.684, sweep_label="a",
        n_param=N_PARAM, T_iter=500, T_trans=400, y_lim=(0, 1.5)
    ))
    sys_tent.save_data(data, 'bifurcation_map_param_a.npz')

# ── Mean-field bifurcation (MPI-aware) ────────────────────────────────────────
data = _timed('tent/meanfield_bif_eps_N500', lambda: sys_tent.analysis_meanfield_bifurcation(
    N=N_MF, n_param=N_PARAM, T_total=T_TRANSIENT + T_SHOW, T_transient=T_TRANSIENT
))
if rank == 0:
    sys_tent.save_data(data, 'meanfield_bif_eps_N500.npz')

data = _timed('tent/meanfield_bif_map_param', lambda: sys_tent.analysis_meanfield_bifurcation(
    sweep='map_param',
    map_factory=lambda a: CoupledMapLattice.from_tent(a=a),
    param_range=(2, 3.4), eps_fixed=0.5,
    sweep_label="a", n_param=N_PARAM, N=N_MF,
    T_total=T_TRANSIENT + T_SHOW, T_transient=T_TRANSIENT
))
if rank == 0:
    sys_tent.save_data(data, 'meanfield_bif_map_param_a_eps0.5_N500.npz')

# ── Minimum N to prevent escape (MPI-aware) ───────────────────────────────────
params, N_vals = _timed('tent/min_N_eps0.5', lambda: sys_tent.analysis_min_N_escape(
    sweep='map_param',
    map_factory=lambda a: CoupledMapLattice.from_tent(a=a),
    param_range=(2, 3.4), eps_fixed=0.5,
    n_param=N_PARAM, N_max=N_MAX, sweep_label="a"
))
if rank == 0:
    sys_tent.save_data({'param': params, 'N_min': N_vals}, 'min_N_map_param_a_eps0.5.npz')

params, N_vals = _timed('tent/min_N_eps0.4', lambda: sys_tent.analysis_min_N_escape(
    sweep='map_param',
    map_factory=lambda a: CoupledMapLattice.from_tent(a=a),
    param_range=(2, 3.4), eps_fixed=0.4,
    n_param=N_PARAM, N_max=N_MAX, sweep_label="a"
))
if rank == 0:
    sys_tent.save_data({'param': params, 'N_min': N_vals}, 'min_N_map_param_a_eps0.4.npz')


# ══════════════════════════════════════════════════════════════════════════════
# LOGISTIC MAP   r = 4.2
# ══════════════════════════════════════════════════════════════════════════════

sys_logistic = CoupledMapLattice.from_logistic(
    r=4.2, output_dir=f"{OUTPUT_ROOT}/logistic"
)

if rank == 0:
    # ── Trajectory plots ──────────────────────────────────────────────────────
    data = _timed('logistic/trajectories_eps0', lambda: sys_logistic.plot_trajectories(
        1000, eps=0, n_show=10, T_total=15, T_transient=0,
        save=True, y_lim=(-0.5, 1.25)
    ))
    sys_logistic.save_data(data, 'trajectories_eps0.npz')

    data = _timed('logistic/trajectories_eps0.05', lambda: sys_logistic.plot_trajectories(
        1000, eps=0.05, n_show=10, T_total=15, T_transient=0,
        save=True, y_lim=(-0.5, 1.25)
    ))
    sys_logistic.save_data(data, 'trajectories_eps0.05.npz')

    data = _timed('logistic/trajectories_eps0.35', lambda: sys_logistic.plot_trajectories(
        1000, eps=0.35, n_show=5,
        T_total=T_TRANSIENT + 15, T_transient=T_TRANSIENT,
        save=True, y_lim=(0.3, 1.0)
    ))
    sys_logistic.save_data(data, 'trajectories_eps0.35.npz')

    data = _timed('logistic/trajectories_eps0.75', lambda: sys_logistic.plot_trajectories(
        1000, eps=0.75, n_show=5, T_total=15, T_transient=0,
        save=True, y_lim=(-0.5, 1.25)
    ))
    sys_logistic.save_data(data, 'trajectories_eps0.75.npz')

    # ── Core statistical analyses ─────────────────────────────────────────────
    data = _timed('logistic/timeseries', lambda: sys_logistic.analysis_timeseries(
        eps=0.35, N_values=N_VALUES, T_show=T_SHOW, T_transient=T_TRANSIENT
    ))
    sys_logistic.save_data(data, 'timeseries_eps0.35.pkl')

    data = _timed('logistic/scaling', lambda: sys_logistic.analysis_fluctuation_scaling(
        eps=0.35, N_values=N_VALUES, T_show=T_SHOW, T_transient=T_TRANSIENT
    ))
    sys_logistic.save_data(data, 'scaling_eps0.35.npz')

    data = _timed('logistic/distribution', lambda: sys_logistic.analysis_distribution(
        eps=0.35, N_values=N_VALUES, T_show=T_SHOW, T_transient=T_TRANSIENT
    ))
    sys_logistic.save_data(data, 'distribution_eps0.35.pkl')

    data = _timed('logistic/self_consistency', lambda: sys_logistic.analysis_self_consistency(
        eps=0.35, T_sc=5_000, T_trans_sc=1_000, T_transient=T_TRANSIENT, T_show=T_SHOW
    ))
    sys_logistic.save_data(data, 'self_consistency_eps0.35.npz')

    # ── Bifurcation diagrams ──────────────────────────────────────────────────
    data = _timed('logistic/bifurcation_eps', lambda: sys_logistic.bifurcation_diagram(
        bifurcation_param="eps", h_bar=0.685, y_lim=(-0.5, 1.5),
        n_param=N_PARAM, T_iter=500, T_trans=400
    ))
    sys_logistic.save_data(data, 'bifurcation_eps.npz')

    data = _timed('logistic/bifurcation_map_param', lambda: sys_logistic.bifurcation_diagram(
        bifurcation_param='map_param',
        map_factory=lambda r: CoupledMapLattice.from_logistic(r=r),
        param_range=(3.0, 5.0), eps_fixed=0.35, h_bar=0.685, sweep_label='r',
        n_param=N_PARAM, T_iter=500, T_trans=400
    ))
    sys_logistic.save_data(data, 'bifurcation_map_param_r.npz')

# ── Mean-field bifurcation (MPI-aware) ────────────────────────────────────────
data = _timed('logistic/meanfield_bif_eps', lambda: sys_logistic.analysis_meanfield_bifurcation(
    N=N_MF, n_param=N_PARAM, T_total=T_TRANSIENT + T_SHOW, T_transient=T_TRANSIENT
))
if rank == 0:
    sys_logistic.save_data(data, 'meanfield_bif_eps_N500.npz')

data = _timed('logistic/meanfield_bif_map_param', lambda: sys_logistic.analysis_meanfield_bifurcation(
    sweep='map_param',
    map_factory=lambda r: CoupledMapLattice.from_logistic(r=r),
    param_range=(3.0, 5.0), eps_fixed=0.35,
    sweep_label='r', n_param=N_PARAM, N=N_MF,
    T_total=T_TRANSIENT + T_SHOW, T_transient=T_TRANSIENT
))
if rank == 0:
    sys_logistic.save_data(data, 'meanfield_bif_map_param_r_eps0.35_N500.npz')

# ── Minimum N to prevent escape (MPI-aware) ───────────────────────────────────
params, N_vals = _timed('logistic/min_N_eps0.35', lambda: sys_logistic.analysis_min_N_escape(
    sweep='map_param',
    map_factory=lambda r: CoupledMapLattice.from_logistic(r=r),
    param_range=(4, 5), eps_fixed=0.35,
    n_param=N_PARAM, N_max=N_MAX, sweep_label="r"
))
if rank == 0:
    sys_logistic.save_data({'param': params, 'N_min': N_vals}, 'min_N_map_param_r_eps0.35.npz')

params, N_vals = _timed('logistic/min_N_eps0.2', lambda: sys_logistic.analysis_min_N_escape(
    sweep='map_param',
    map_factory=lambda r: CoupledMapLattice.from_logistic(r=r),
    param_range=(4, 5), eps_fixed=0.2,
    n_param=N_PARAM, N_max=N_MAX, sweep_label='r'
))
if rank == 0:
    sys_logistic.save_data({'param': params, 'N_min': N_vals}, 'min_N_map_param_r_eps0.2.npz')


# ══════════════════════════════════════════════════════════════════════════════
# LOZI MAP   a = 2, b = 0.5
# ══════════════════════════════════════════════════════════════════════════════

sys_lozi = CoupledMapLattice.from_lozi(
    a=2, b=0.5, output_dir=f"{OUTPUT_ROOT}/lozi"
)

if rank == 0:
    # ── Trajectory plots ──────────────────────────────────────────────────────
    data = _timed('lozi/trajectories_eps0_c0', lambda: sys_lozi.plot_trajectories(
        1000, eps=0, n_show=5, T_total=60, T_transient=0,
        save=True, y_lim=(-5, 1.5)
    ))
    sys_lozi.save_data(data, 'trajectories_eps0_c0.npz')

    data = _timed('lozi/trajectories_eps0_c1', lambda: sys_lozi.plot_trajectories(
        1000, eps=0, n_show=5, T_total=60, T_transient=0,
        save=True, y_lim=(-5, 1), component=1
    ))
    sys_lozi.save_data(data, 'trajectories_eps0_c1.npz')

    data = _timed('lozi/trajectories_eps0.32_c0', lambda: sys_lozi.plot_trajectories(
        1000, eps=0.32, n_show=5,
        T_total=T_TRANSIENT + 60, T_transient=T_TRANSIENT,
        save=True, y_lim=(-0.75, 1.25)
    ))
    sys_lozi.save_data(data, 'trajectories_eps0.32_c0.npz')

    data = _timed('lozi/trajectories_eps0.32_c1', lambda: sys_lozi.plot_trajectories(
        1000, eps=0.32, n_show=5,
        T_total=T_TRANSIENT + 60, T_transient=T_TRANSIENT,
        save=True, y_lim=(-0.5, 0.75), component=1
    ))
    sys_lozi.save_data(data, 'trajectories_eps0.32_c1.npz')

    # ── Core statistical analyses ─────────────────────────────────────────────
    data = _timed('lozi/timeseries', lambda: sys_lozi.analysis_timeseries(
        eps=0.32, N_values=N_VALUES, T_show=T_SHOW, T_transient=T_TRANSIENT
    ))
    sys_lozi.save_data(data, 'timeseries_eps0.32.pkl')

    data = _timed('lozi/scaling', lambda: sys_lozi.analysis_fluctuation_scaling(
        eps=0.32, N_values=N_VALUES, T_show=T_SHOW, T_transient=T_TRANSIENT
    ))
    sys_lozi.save_data(data, 'scaling_eps0.32.npz')

    data = _timed('lozi/distribution', lambda: sys_lozi.analysis_distribution(
        eps=0.32, N_values=N_VALUES, T_show=T_SHOW, T_transient=T_TRANSIENT
    ))
    sys_lozi.save_data(data, 'distribution_eps0.32.pkl')

    data = _timed('lozi/self_consistency', lambda: sys_lozi.analysis_self_consistency(
        eps=0.32, h_range=(-0.5, 1.5),
        T_sc=5_000, T_trans_sc=1_000, T_transient=T_TRANSIENT, T_show=T_SHOW
    ))
    sys_lozi.save_data(data, 'self_consistency_eps0.32.npz')

    # ── Bifurcation diagrams ──────────────────────────────────────────────────
    data = _timed('lozi/bifurcation_eps_c0', lambda: sys_lozi.bifurcation_diagram(
        bifurcation_param="eps", h_bar=0.218,
        n_param=N_PARAM, T_iter=500, T_trans=400
    ))
    sys_lozi.save_data(data, 'bifurcation_eps_c0.npz')

    data = _timed('lozi/bifurcation_eps_c1', lambda: sys_lozi.bifurcation_diagram(
        bifurcation_param="eps", h_bar=0.218, component=1,
        n_param=N_PARAM, T_iter=500, T_trans=400
    ))
    sys_lozi.save_data(data, 'bifurcation_eps_c1.npz')

    data = _timed('lozi/bifurcation_map_param_c0', lambda: sys_lozi.bifurcation_diagram(
        bifurcation_param='map_param',
        map_factory=lambda a: CoupledMapLattice.from_lozi(a=a, b=0.5),
        param_range=(1.4, 3), eps_fixed=0.32, h_bar=0.218, sweep_label='a',
        n_param=N_PARAM, T_iter=500, T_trans=400
    ))
    sys_lozi.save_data(data, 'bifurcation_map_param_a_c0.npz')

    data = _timed('lozi/bifurcation_map_param_c1', lambda: sys_lozi.bifurcation_diagram(
        bifurcation_param='map_param',
        map_factory=lambda a: CoupledMapLattice.from_lozi(a=a, b=0.5),
        param_range=(1.4, 3), eps_fixed=0.32, h_bar=0.218, sweep_label='a', component=1,
        n_param=N_PARAM, T_iter=500, T_trans=400
    ))
    sys_lozi.save_data(data, 'bifurcation_map_param_a_c1.npz')

# ── Mean-field bifurcation (MPI-aware) ────────────────────────────────────────
data = _timed('lozi/meanfield_bif_eps', lambda: sys_lozi.analysis_meanfield_bifurcation(
    N=N_MF, n_param=N_PARAM, T_total=T_TRANSIENT + T_SHOW, T_transient=T_TRANSIENT
))
if rank == 0:
    sys_lozi.save_data(data, 'meanfield_bif_eps_N500.npz')

data = _timed('lozi/meanfield_bif_map_param', lambda: sys_lozi.analysis_meanfield_bifurcation(
    sweep='map_param',
    map_factory=lambda a: CoupledMapLattice.from_lozi(a=a, b=0.5),
    param_range=(1.4, 3), eps_fixed=0.32,
    sweep_label='a', n_param=N_PARAM, N=N_MF,
    T_total=T_TRANSIENT + T_SHOW, T_transient=T_TRANSIENT
))
if rank == 0:
    sys_lozi.save_data(data, 'meanfield_bif_map_param_a_eps0.32_N500.npz')

# ── Minimum N to prevent escape (MPI-aware) ───────────────────────────────────
params, N_vals = _timed('lozi/min_N_eps0.32', lambda: sys_lozi.analysis_min_N_escape(
    sweep='map_param',
    map_factory=lambda a: CoupledMapLattice.from_lozi(a=a, b=0.5),
    param_range=(1.4, 3), eps_fixed=0.32,
    n_param=N_PARAM, N_max=N_MAX, sweep_label='a'
))
if rank == 0:
    sys_lozi.save_data({'param': params, 'N_min': N_vals}, 'min_N_map_param_a_eps0.32.npz')

params, N_vals = _timed('lozi/min_N_eps0.2', lambda: sys_lozi.analysis_min_N_escape(
    sweep='map_param',
    map_factory=lambda a: CoupledMapLattice.from_lozi(a=a, b=0.5),
    param_range=(1.4, 3), eps_fixed=0.2,
    n_param=N_PARAM, N_max=N_MAX, sweep_label='a'
))
if rank == 0:
    sys_lozi.save_data({'param': params, 'N_min': N_vals}, 'min_N_map_param_a_eps0.2.npz')


# ── Timing summary ────────────────────────────────────────────────────────────
if rank == 0:
    total = sum(_timings.values())
    print("\n" + "=" * 55)
    print("  TIMING SUMMARY")
    print("=" * 55)
    for label, t in _timings.items():
        print(f"  {label:<45s} {t:6.1f} s")
    print("-" * 55)
    print(f"  {'TOTAL':<45s} {total:6.1f} s")
    print("=" * 55)