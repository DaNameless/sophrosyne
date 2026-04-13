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

# Shared grid settings
GRID    = dict(n_a=80, n_eps=80, N_min=1, N_max=20_000,
               T_total=4_000, x_bound=20, n_trials=3, n_jobs=1)
CMAP    = 'plasma'

OUT = "../../outputs/papers_figures"


# ══════════════════════════════════════════════════════════════════════════════
# TENT MAP — sweep a
# ══════════════════════════════════════════════════════════════════════════════

sys_tent = CoupledMapLattice.from_tent(
    a=1.99, output_dir=f"{OUT}/tent"
)

data = sys_tent.analysis_escape_phase_diagram(
    map_factory = lambda a: CoupledMapLattice.from_tent(a=a),
    a_range     = (1.0, 3.0),
    eps_range   = (0.0, 0.6),
    sweep_label = 'a',
    cmap        = CMAP,
    output      = f"{OUT}/tent/escape_map_a.png",
    show        = False,
    **GRID,
)
if rank == 0:
    sys_tent.save_data(data, 'escape_map_a.npz')


# ══════════════════════════════════════════════════════════════════════════════
# LOGISTIC MAP — sweep r
# ══════════════════════════════════════════════════════════════════════════════

sys_logistic = CoupledMapLattice.from_logistic(
    r=3.9, output_dir=f"{OUT}/logistic"
)

data = sys_logistic.analysis_escape_phase_diagram(
    map_factory = lambda r: CoupledMapLattice.from_logistic(r=r),
    a_range     = (3.0, 5.0),
    eps_range   = (0.0, 0.6),
    sweep_label = 'r',
    cmap        = CMAP,
    output      = f"{OUT}/logistic/escape_map_r.png",
    show        = False,
    **GRID,
)
if rank == 0:
    sys_logistic.save_data(data, 'escape_map_r.npz')


# ══════════════════════════════════════════════════════════════════════════════
# LOZI MAP — sweep a  (b fixed at 0.5)
# ══════════════════════════════════════════════════════════════════════════════

sys_lozi = CoupledMapLattice.from_lozi(
    a=1.7, b=0.5, output_dir=f"{OUT}/lozi"
)

data = sys_lozi.analysis_escape_phase_diagram(
    map_factory = lambda a: CoupledMapLattice.from_lozi(a=a, b=0.5),
    a_range     = (1.0, 2.5),
    eps_range   = (0.0, 0.6),
    sweep_label = 'a',
    cmap        = CMAP,
    output      = f"{OUT}/lozi/escape_map_a.png",
    show        = False,
    **GRID,
)
if rank == 0:
    sys_lozi.save_data(data, 'escape_map_a.npz')


# ══════════════════════════════════════════════════════════════════════════════
# LOZI MAP — sweep b  (a fixed at 1.7)
# ══════════════════════════════════════════════════════════════════════════════

data = sys_lozi.analysis_escape_phase_diagram(
    map_factory = lambda b: CoupledMapLattice.from_lozi(a=1.7, b=b),
    a_range     = (0.1, 0.9),
    eps_range   = (0.0, 0.6),
    sweep_label = 'b',
    cmap        = CMAP,
    output      = f"{OUT}/lozi/escape_map_b.png",
    show        = False,
    **GRID,
)
if rank == 0:
    sys_lozi.save_data(data, 'escape_map_b.npz')
