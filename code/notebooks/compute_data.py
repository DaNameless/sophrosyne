"""
compute_data.py
===============
Runs all analyses for Tent, Logistic, Lozi (discrete) and Linz-Sprott
(continuous) without producing any plots.  All results are saved as
.npz files so they can be plotted independently later.

MPI-aware: run with   mpirun -n <N>  python compute_data.py
or plain:             python compute_data.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.sophrosyne import (
    LinzSprott, SimulationRunner,
    BifurcationAnalyzer,
    EscapeMapAnalyzer,
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

OUT = "../../outputs/data"

def mkdir(path):
    if rank == 0:
        os.makedirs(path, exist_ok=True)
    if size > 1:
        comm.Barrier()

def save(path, **arrays):
    """Save a dict of arrays to a compressed npz file."""
    np.savez_compressed(path, **arrays)
    print(f"  -> Saved {path}.npz")

# ─────────────────────────────────────────────────────────────────────────────
# Shared settings
# ─────────────────────────────────────────────────────────────────────────────
GRID     = dict(n_a=80, n_eps=80, N_min=1, N_max=50_000,
                T_total=4_000, x_bound=20, n_trials=3, n_jobs=1)
N_VALUES = (1_000, 10_000, 100_000)


# ══════════════════════════════════════════════════════════════════════════════
# TENT MAP
# ══════════════════════════════════════════════════════════════════════════════
TENT_OUT = f"{OUT}/tent"
mkdir(TENT_OUT)

EPS_TENT = 0.4
sys_tent = CoupledMapLattice.from_tent(a=3.0, output_dir=TENT_OUT)

if rank == 0:
    # 1. Time series
    d = sys_tent.analysis_timeseries(
        eps=EPS_TENT, N_values=N_VALUES, T_show=300, T_transient=10_000)
    save(f"{TENT_OUT}/timeseries_eps{EPS_TENT}",
         **{f"h_N{N}": d[N] for N in N_VALUES})

    # 2. Fluctuation scaling
    d = sys_tent.analysis_fluctuation_scaling(
        eps=EPS_TENT, N_values=N_VALUES, T_show=2_000, T_transient=10_000)
    save(f"{TENT_OUT}/scaling_eps{EPS_TENT}",
         N_values=d['N_values'], sigmas=d['sigmas'], means=d['means'],
         slope=np.array(d['slope']), intercept=np.array(d['intercept']),
         r2=np.array(d['r2']))

    # 3. Distribution
    d = sys_tent.analysis_distribution(
        eps=EPS_TENT, N_values=N_VALUES, T_show=2_000, T_transient=10_000)
    save(f"{TENT_OUT}/distribution_eps{EPS_TENT}",
         **{f"h_N{N}": d['h_series'][N] for N in d['h_series']},
         **{f"rescaled_N{N}": d['all_rescaled'][N] for N in d['all_rescaled']})

    # 4. Self-consistency
    d = sys_tent.analysis_self_consistency(
        eps=EPS_TENT, h_range=(0.0, 1.0),
        N_sc=5_000, T_sc=50_000, T_trans_sc=10_000,
        T_transient=10_000, T_show=500)
    save(f"{TENT_OUT}/self_consistency_eps{EPS_TENT}",
         h_candidates=d['h_candidates'], f_averages=d['f_averages'],
         residuals=d['residuals'],
         h_star_list=np.array(d['h_star_list']),
         h_sim_mean=np.array(d['h_sim_mean']))

# 5. Phase diagram (MPI-aware)
d = sys_tent.analysis_phase_diagram(
    eps_values=np.linspace(0.01, 0.9, 60), N_values=N_VALUES, n_jobs=1)
if rank == 0 and d is not None:
    save(f"{TENT_OUT}/phase_diagram",
         eps_values=d['eps_values'], alphas=d['alphas'],
         alpha_errors=d['alpha_errors'])

# 6. Mean-field bifurcation — sweep ε (MPI-aware)
d = sys_tent.analysis_meanfield_bifurcation(
    sweep='eps', eps_range=(0.0, 1.0),
    N=5_000, n_param=300, T_total=20_000, T_transient=18_000)
if rank == 0 and d is not None:
    save(f"{TENT_OUT}/mf_bifurcation_eps", param=d['param'], h=d['h'])

# 7. Mean-field bifurcation — sweep a (MPI-aware)
d = sys_tent.analysis_meanfield_bifurcation(
    sweep='map_param',
    map_factory=lambda a: CoupledMapLattice.from_tent(a=a),
    param_range=(2.0, 4.0), sweep_label='a', eps_fixed=EPS_TENT,
    N=5_000, n_param=300, T_total=20_000, T_transient=18_000)
if rank == 0 and d is not None:
    save(f"{TENT_OUT}/mf_bifurcation_a", param=d['param'], h=d['h'])

# 8. Escape phase diagram (MPI-aware)
d = sys_tent.analysis_escape_phase_diagram(
    map_factory=lambda a: CoupledMapLattice.from_tent(a=a),
    a_range=(2.0, 4.0), eps_range=(0.0, 1.0), sweep_label='a', **GRID)
if rank == 0 and d is not None:
    sys_tent.save_data(d, 'escape_map_a.npz')

# 9. Element histogram
if rank == 0:
    d = sys_tent.analysis_element_histogram(
        N=10_000, eps=EPS_TENT, T_total=50_000, T_transient=10_000)
    save(f"{TENT_OUT}/histogram_N10000_eps{EPS_TENT}",
         x=d['components'][0], h_series=d['h_series'])

if rank == 0:
    print(f"Tent done → {TENT_OUT}")


# ══════════════════════════════════════════════════════════════════════════════
# LOGISTIC MAP
# ══════════════════════════════════════════════════════════════════════════════
LOG_OUT = f"{OUT}/logistic"
mkdir(LOG_OUT)

EPS_LOG = 0.2
sys_log = CoupledMapLattice.from_logistic(r=4.2, output_dir=LOG_OUT)

if rank == 0:
    d = sys_log.analysis_timeseries(
        eps=EPS_LOG, N_values=N_VALUES, T_show=300, T_transient=10_000)
    save(f"{LOG_OUT}/timeseries_eps{EPS_LOG}",
         **{f"h_N{N}": d[N] for N in N_VALUES})

    d = sys_log.analysis_fluctuation_scaling(
        eps=EPS_LOG, N_values=N_VALUES, T_show=2_000, T_transient=10_000)
    save(f"{LOG_OUT}/scaling_eps{EPS_LOG}",
         N_values=d['N_values'], sigmas=d['sigmas'], means=d['means'],
         slope=np.array(d['slope']), intercept=np.array(d['intercept']),
         r2=np.array(d['r2']))

    d = sys_log.analysis_distribution(
        eps=EPS_LOG, N_values=N_VALUES, T_show=2_000, T_transient=10_000)
    save(f"{LOG_OUT}/distribution_eps{EPS_LOG}",
         **{f"h_N{N}": d['h_series'][N] for N in d['h_series']},
         **{f"rescaled_N{N}": d['all_rescaled'][N] for N in d['all_rescaled']})

    d = sys_log.analysis_self_consistency(
        eps=EPS_LOG, h_range=(0.0, 1.0),
        N_sc=5_000, T_sc=50_000, T_trans_sc=10_000,
        T_transient=10_000, T_show=500)
    save(f"{LOG_OUT}/self_consistency_eps{EPS_LOG}",
         h_candidates=d['h_candidates'], f_averages=d['f_averages'],
         residuals=d['residuals'],
         h_star_list=np.array(d['h_star_list']),
         h_sim_mean=np.array(d['h_sim_mean']))

d = sys_log.analysis_phase_diagram(
    eps_values=np.linspace(0.01, 0.9, 60), N_values=N_VALUES, n_jobs=1)
if rank == 0 and d is not None:
    save(f"{LOG_OUT}/phase_diagram",
         eps_values=d['eps_values'], alphas=d['alphas'],
         alpha_errors=d['alpha_errors'])

d = sys_log.analysis_meanfield_bifurcation(
    sweep='eps', eps_range=(0.0, 1.0),
    N=5_000, n_param=300, T_total=20_000, T_transient=18_000)
if rank == 0 and d is not None:
    save(f"{LOG_OUT}/mf_bifurcation_eps", param=d['param'], h=d['h'])

d = sys_log.analysis_meanfield_bifurcation(
    sweep='map_param',
    map_factory=lambda r: CoupledMapLattice.from_logistic(r=r),
    param_range=(3.9, 5.0), sweep_label='r', eps_fixed=EPS_LOG,
    N=5_000, n_param=300, T_total=20_000, T_transient=18_000)
if rank == 0 and d is not None:
    save(f"{LOG_OUT}/mf_bifurcation_r", param=d['param'], h=d['h'])

d = sys_log.analysis_escape_phase_diagram(
    map_factory=lambda r: CoupledMapLattice.from_logistic(r=r),
    a_range=(3.9, 5.0), eps_range=(0.0, 1.0), sweep_label='r', **GRID)
if rank == 0 and d is not None:
    sys_log.save_data(d, 'escape_map_r.npz')

if rank == 0:
    d = sys_log.analysis_element_histogram(
        N=10_000, eps=EPS_LOG, T_total=50_000, T_transient=10_000)
    save(f"{LOG_OUT}/histogram_N10000_eps{EPS_LOG}",
         x=d['components'][0], h_series=d['h_series'])

if rank == 0:
    print(f"Logistic done → {LOG_OUT}")


# ══════════════════════════════════════════════════════════════════════════════
# LOZI MAP
# ══════════════════════════════════════════════════════════════════════════════
LOZI_OUT = f"{OUT}/lozi"
mkdir(LOZI_OUT)

EPS_LOZI = 0.25
sys_lozi = CoupledMapLattice.from_lozi(a=2.0, b=0.5, output_dir=LOZI_OUT)

if rank == 0:
    d = sys_lozi.analysis_timeseries(
        eps=EPS_LOZI, N_values=N_VALUES, T_show=300, T_transient=10_000)
    save(f"{LOZI_OUT}/timeseries_eps{EPS_LOZI}",
         **{f"h_N{N}": d[N] for N in N_VALUES})

    d = sys_lozi.analysis_fluctuation_scaling(
        eps=EPS_LOZI, N_values=N_VALUES, T_show=2_000, T_transient=10_000)
    save(f"{LOZI_OUT}/scaling_eps{EPS_LOZI}",
         N_values=d['N_values'], sigmas=d['sigmas'], means=d['means'],
         slope=np.array(d['slope']), intercept=np.array(d['intercept']),
         r2=np.array(d['r2']))

    d = sys_lozi.analysis_distribution(
        eps=EPS_LOZI, N_values=N_VALUES, T_show=2_000, T_transient=10_000)
    save(f"{LOZI_OUT}/distribution_eps{EPS_LOZI}",
         **{f"h_N{N}": d['h_series'][N] for N in d['h_series']},
         **{f"rescaled_N{N}": d['all_rescaled'][N] for N in d['all_rescaled']})

    d = sys_lozi.analysis_self_consistency(
        eps=EPS_LOZI, h_range=(-0.5, 1.5),
        N_sc=5_000, T_sc=50_000, T_trans_sc=10_000,
        T_transient=10_000, T_show=500)
    save(f"{LOZI_OUT}/self_consistency_eps{EPS_LOZI}",
         h_candidates=d['h_candidates'], f_averages=d['f_averages'],
         residuals=d['residuals'],
         h_star_list=np.array(d['h_star_list']),
         h_sim_mean=np.array(d['h_sim_mean']))

d = sys_lozi.analysis_phase_diagram(
    eps_values=np.linspace(0.01, 0.9, 60), N_values=N_VALUES, n_jobs=1)
if rank == 0 and d is not None:
    save(f"{LOZI_OUT}/phase_diagram",
         eps_values=d['eps_values'], alphas=d['alphas'],
         alpha_errors=d['alpha_errors'])

d = sys_lozi.analysis_meanfield_bifurcation(
    sweep='eps', eps_range=(0.0, 1.0),
    N=5_000, n_param=300, T_total=20_000, T_transient=18_000)
if rank == 0 and d is not None:
    save(f"{LOZI_OUT}/mf_bifurcation_eps", param=d['param'], h=d['h'])

d = sys_lozi.analysis_meanfield_bifurcation(
    sweep='map_param',
    map_factory=lambda a: CoupledMapLattice.from_lozi(a=a, b=0.5),
    param_range=(1.75, 3.0), sweep_label='a', eps_fixed=EPS_LOZI,
    N=5_000, n_param=300, T_total=20_000, T_transient=18_000)
if rank == 0 and d is not None:
    save(f"{LOZI_OUT}/mf_bifurcation_a", param=d['param'], h=d['h'])

# Escape maps (sweep a and sweep b)
d = sys_lozi.analysis_escape_phase_diagram(
    map_factory=lambda a: CoupledMapLattice.from_lozi(a=a, b=0.5),
    a_range=(1.75, 3.0), eps_range=(0.0, 1.0), sweep_label='a', **GRID)
if rank == 0 and d is not None:
    sys_lozi.save_data(d, 'escape_map_a.npz')

d = sys_lozi.analysis_escape_phase_diagram(
    map_factory=lambda b: CoupledMapLattice.from_lozi(a=1.8, b=b),
    a_range=(0.1, 0.9), eps_range=(0.0, 1.0), sweep_label='b', **GRID)
if rank == 0 and d is not None:
    sys_lozi.save_data(d, 'escape_map_b.npz')

if rank == 0:
    d = sys_lozi.analysis_element_histogram(
        N=10_000, eps=EPS_LOZI, T_total=50_000, T_transient=10_000)
    save(f"{LOZI_OUT}/histogram_N10000_eps{EPS_LOZI}",
         x=d['components'][0], y=d['components'][1], h_series=d['h_series'])

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

# Simulation tail (for histogram and attractor)
if rank == 0:
    runner = SimulationRunner(
        system_cls=LinzSprott, system_params={"a": A_LINZ},
        N=1_000, epsilon=EPS_LINZ, dt=DT,
        steps=200_000, window=10_000, verbose=False,
    )
    sim = runner.run()
    tail = sim.tail            # (window, N, dim)
    save(f"{LINZ_OUT}/tail_a{A_LINZ}_eps{EPS_LINZ}",
         x=tail[:, :, 0], y=tail[:, :, 1], z=tail[:, :, 2],
         t=sim.t_tail,
         means_x=sim.means_tail[:, 0],
         means_y=sim.means_tail[:, 1],
         means_z=sim.means_tail[:, 2])

# Bifurcation — sweep a
if rank == 0:
    bif = BifurcationAnalyzer(
        LinzSprott, base_params={},
        t_transient=1_000, t_steady=1_000, dt=DT, threshold=200.0,
    )
    result_bif = bif.compute("a", np.linspace(0.5, 0.55, 400))
    # peaks is a dict {label: list of arrays}, one per param value
    # flatten to (param, peak_value) pairs for easy storage
    for lab in result_bif.labels:
        params_flat, peaks_flat = [], []
        for pv, pk in zip(result_bif.param_values, result_bif.peaks[lab]):
            if pk.size:
                params_flat.extend([pv] * len(pk))
                peaks_flat.extend(pk.tolist())
        save(f"{LINZ_OUT}/bifurcation_a_{lab}",
             param=np.array(params_flat),
             peaks=np.array(peaks_flat),
             fixed_param=result_bif.param_values,
             fixed_val=result_bif.fixed[lab])

# Escape maps (MPI-aware)
analyzer = EscapeMapAnalyzer(
    system_cls=LinzSprott, base_params={},
    epsilon=0.0, dt=DT, N=1_000,
    steps=50_000, window=5_000, threshold=100.0,
    n_jobs=os.cpu_count() or 1,
)

result_fixed = analyzer.compute(
    param1_name="a",       param1_values=np.linspace(0.5, 0.55, 100),
    param2_name="epsilon", param2_values=np.linspace(0.0, 1.0, 100),
    seed=42,
)
if rank == 0 and result_fixed is not None:
    save(f"{LINZ_OUT}/escape_map_fixed_N",
         a_values=result_fixed.param1_values,
         eps_values=result_fixed.param2_values,
         escaped=result_fixed.escaped.astype(np.int8),
         escape_time=result_fixed.escape_time)

result_min_N = analyzer.compute_min_N(
    param1_name="a",       param1_values=np.linspace(0.5, 0.6, 100),
    param2_name="epsilon", param2_values=np.linspace(0.0, 1.0, 100),
    N_min=1, N_max=100_000, n_trials=3,
)
if rank == 0 and result_min_N is not None:
    save(f"{LINZ_OUT}/escape_map_min_N",
         a_values=result_min_N.param1_values,
         eps_values=result_min_N.param2_values,
         min_N=result_min_N.min_N)

if rank == 0:
    print(f"Linz-Sprott done → {LINZ_OUT}")
