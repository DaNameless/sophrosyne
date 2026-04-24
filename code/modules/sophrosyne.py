"""
╔══════════════════════════════════════╗
║             SOPHROSYNE               ║
║              v1.0.0                  ║
║   Coupled Dynamical Systems Solver   ║
║            by R.S.S.G.               ║
╚══════════════════════════════════════╝

A modular Python framework for simulating mean-field coupled
dynamical systems using RK4 integration.

Supported systems:
    - Linz-Sprott 
    - Rössler 
    - Rössler globally coupled
    - Shimizu-Morioka (B=0 variant)
    - Easily extensible via DynamicalSystem base class

Usage:
    from sophrosyne import LinzSprott, RK4Integrator, SimulationRunner

    runner = SimulationRunner(
        system_cls=LinzSprott,
        system_params={"a": 0.553},
        N=100, epsilon=0.001, dt=0.005, steps=200000
    )
    result = runner.run()

Author : R.S.S.G.
Created: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Type, Dict, Any, Tuple, List
import os
import time
import warnings
import multiprocessing as mp
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp

# ──────────────────────────────────────────────
# Optional heavy imports (graceful degradation)
# ──────────────────────────────────────────────
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# ══════════════════════════════════════════════
#  1. DYNAMICAL SYSTEMS
# ══════════════════════════════════════════════

class DynamicalSystem(ABC):
    """
    Abstract base class for coupled dynamical systems.

    Subclasses must implement:
        - slope(t, s, means) -> np.ndarray of shape (N, dim)
        - dim (property): phase-space dimension
        - default_ic (property): default initial condition (array of shape (dim,))
        - name (property): human-readable name

    Convention:
        s      : state array of shape (N, dim)
        means  : dict {"x0": float, "x1": float, ...} with mean of each component
        epsilon: coupling strength (stored as self.epsilon)
    """

    def __init__(self, epsilon: float = 0.0, **params):
        self.epsilon = epsilon
        self._params = params
        for k, v in params.items():
            setattr(self, k, v)

    @abstractmethod
    def slope(self, t: float, s: np.ndarray, means: Dict[str, float]) -> np.ndarray:
        """
        Compute the right-hand side ds/dt.

        Parameters
        ----------
        t     : current time
        s     : state array, shape (N, dim)
        means : dict with keys "x0", "x1", ... holding the ensemble mean
                of each component

        Returns
        -------
        np.ndarray of shape (N, dim)
        """
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Phase-space dimension."""
        ...

    @property
    @abstractmethod
    def default_ic(self) -> np.ndarray:
        """Default initial condition, shape (dim,)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable system name."""
        ...

    @property
    def labels(self) -> List[str]:
        """Axis labels for each component. Override for custom names."""
        return [f"x{i}" for i in range(self.dim)]

    def _compute_means(self, s: np.ndarray) -> Dict[str, float]:
        """Compute ensemble means from state array."""
        return {f"x{i}": np.mean(s[:, i]) for i in range(self.dim)}

    def __repr__(self):
        param_str = ", ".join(f"{k}={v}" for k, v in self._params.items())
        return f"{self.name}(ε={self.epsilon}, {param_str})"


class ForcedSystem(DynamicalSystem):
    """
    Wraps any DynamicalSystem and adds a constant forcing h to one component:

        d(component)/dt  +=  h

    The wrapped system is otherwise unchanged, so ForcedSystem works anywhere
    a plain DynamicalSystem is accepted (SimulationRunner, BifurcationAnalyzer,
    EscapeMapAnalyzer, Plotter, …).

    Parameters
    ----------
    inner_cls : DynamicalSystem subclass to wrap (keyword arg for pickling)
    h         : forcing amplitude (constant)
    component : index of the component that receives the forcing (default 0)
    **kwargs  : passed directly to inner_cls (e.g. a=0.5, epsilon=0.01)

    Examples
    --------
    # Direct use (SimulationRunner, Plotter — no multiprocessing):
    sys = ForcedSystem(LinzSprott, h=0.3, component=0, a=0.5)

    # Parallel use (BifurcationAnalyzer, EscapeMapAnalyzer — mp.Pool):
    # Pass ForcedSystem as system_cls and inner_cls via base_params so
    # everything is picklable (no lambdas).
    bif = BifurcationAnalyzer(
        ForcedSystem,
        base_params={"inner_cls": LinzSprott, "a": 0.5, "component": 0},
    )
    result = bif.compute("h", np.linspace(-1.0, 1.0, 300))
    """

    def __init__(self, inner_cls: Type[DynamicalSystem] = None,
                 h: float = 0.0, component: int = 0, **kwargs):
        self._inner    = inner_cls(**kwargs)
        self._h        = h
        self._comp     = component
        super().__init__(epsilon=self._inner.epsilon)
        self._params   = {"inner_cls": inner_cls, "h": h,
                          "component": component, **self._inner._params}

    def slope(self, t, s, means):
        dydt = self._inner.slope(t, s, means).copy()
        dydt[:, self._comp] += self._h
        return dydt

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def default_ic(self) -> np.ndarray:
        return self._inner.default_ic

    @property
    def name(self) -> str:
        return f"Forced({self._inner.name}, h={self._h}, comp={self._comp})"

    @property
    def labels(self) -> List[str]:
        return self._inner.labels


class LinzSprott(DynamicalSystem):
    """
    Coupled Linz-Sprott jerk system.

        dx/dt = y
        dy/dt = z
        dz/dt = (-a*z - y + |x| - 1) + ε*(z_mean - z)

    Parameters
    ----------
    a       : damping parameter (typical range ~0.5–0.6)
    epsilon : coupling strength
    """

    def __init__(self, a: float = 0.553, epsilon: float = 0.0):
        super().__init__(epsilon=epsilon, a=a)

    def slope(self, t, s, means):
        x, y, z = s[:, 0], s[:, 1], s[:, 2]
        z_mean = means["x2"]
        dx = y
        dy = z
        dz = (-self.a * z - y + np.abs(x) - 1.0) + self.epsilon * (z_mean - z)
        return np.stack([dx, dy, dz], axis=1)

    @property
    def dim(self):
        return 3

    @property
    def default_ic(self):
        return np.array([0.3077, -0.8528, -0.1290])

    @property
    def name(self):
        return "LinzSprott"

    @property
    def labels(self):
        return ["x", "y", "z"]


class Rossler(DynamicalSystem):
    """
    Coupled Rössler system with diffusive coupling on all components.

        dx/dt = -y - z              + ε*(x_mean - x)
        dy/dt =  a*y + x            + ε*(y_mean - y)
        dz/dt =  z*(x - c) + b      + ε*(z_mean - z)

    Parameters
    ----------
    a, b, c : Rössler parameters
    epsilon : coupling strength
    """

    def __init__(self, a: float = 0.2, b: float = 0.2, c: float = 5.7,
                 epsilon: float = 0.0):
        super().__init__(epsilon=epsilon, a=a, b=b, c=c)

    def slope(self, t, s, means):
        x, y, z = s[:, 0], s[:, 1], s[:, 2]
        dx = -y - z + self.epsilon * (means["x0"] - x)
        dy = self.a * y + x #+ self.epsilon * (means["x1"] - y)
        dz = z * (x - self.c) + self.b + self.epsilon * (means["x2"] - z)
        return np.stack([dx, dy, dz], axis=1)

    @property
    def dim(self):
        return 3

    @property
    def default_ic(self):
        return np.array([1.0, 1.0, 0.0])

    @property
    def name(self):
        return "Rössler"

    @property
    def labels(self):
        return ["x", "y", "z"]


class RosslerGloballyCoupled(DynamicalSystem):
    """
    Rössler system with global coupling on the velocity field.

        dX/dt = (1 - ε)*f_local(X) + ε*<f_local(X)>

    Parameters
    ----------
    a, b, c : Rössler parameters
    epsilon : coupling strength
    """

    def __init__(self, a: float = 0.2, b: float = 0.2, c: float = 5.7,
                 epsilon: float = 0.0):
        super().__init__(epsilon=epsilon, a=a, b=b, c=c)

    def slope(self, t, s, means):
        x, y, z = s[:, 0], s[:, 1], s[:, 2]
        # Local dynamics
        dx_loc = -y - z
        dy_loc = self.a * y + x
        dz_loc = z * (x - self.c) + self.b
        # Mean of local dynamics
        dx_mean = np.mean(dx_loc)
        dy_mean = np.mean(dy_loc)
        dz_mean = np.mean(dz_loc)
        z_mean = np.mean(z)
        # Globally coupled
        eps = self.epsilon
        dx = dx_loc#(1 - eps) * dx_loc + eps * dx_mean
        dy = dy_loc #(1 - eps) * dy_loc + eps * dy_mean
        dz = (1 - eps) * dz_loc + eps * z_mean
        return np.stack([dx, dy, dz], axis=1)

    @property
    def dim(self):
        return 3

    @property
    def default_ic(self):
        return np.array([1.0, 1.0, 0.0])

    @property
    def name(self):
        return "RösslerGlobal"

    @property
    def labels(self):
        return ["x", "y", "z"]


class ShimizuMorioka(DynamicalSystem):
    """
    Coupled Shimizu-Morioka system (B=0 variant).

        dx/dt = y
        dy/dt = x*(1 - z) - b*y        + ε*(y_mean - y)
        dz/dt = -a*(z - x²)            + ε*(z_mean - z)

    This is the system from the general form:
        dx/dt = y
        dy/dt = x*(1-z) - B*x³ - b*y
        dz/dt = -a*(z - x²)
    with B set to 0.

    Parameters
    ----------
    a       : slow-manifold rate (typical chaotic range ~0.35–0.45)
    b       : damping (typical ~0.75–1.0)
    epsilon : coupling strength
    """

    def __init__(self, a: float = 0.375, b: float = 0.81,
                 epsilon: float = 0.0):
        super().__init__(epsilon=epsilon, a=a, b=b)

    def slope(self, t, s, means):
        x, y, z = s[:, 0], s[:, 1], s[:, 2]
        dx = y
        dy = x * (1.0 - z) - self.b * y + self.epsilon * (means["x1"] - y)
        dz = -self.a * (z - x**2)        + self.epsilon * (means["x2"] - z)
        return np.stack([dx, dy, dz], axis=1)

    @property
    def dim(self):
        return 3

    @property
    def default_ic(self):
        return np.array([0.5, 0.0, 0.5])

    @property
    def name(self):
        return "ShimizuMorioka"

    @property
    def labels(self):
        return ["x", "y", "z"]

class ShimizuMorioka2(DynamicalSystem):
    """
    Coupled Shimizu-Morioka system (B=0 variant).

        dx/dt = y
        dy/dt = x*(1 - z) - b*y        + ε*(y_mean - y)
        dz/dt = -a*(z - x²)            + ε*(z_mean - z)

    This is the system from the general form:
        dx/dt = y
        dy/dt = x*(1-z) - B*x³ - b*y
        dz/dt = -a*(z - x²)
    with B set to 0.

    Parameters
    ----------
    a       : slow-manifold rate (typical chaotic range ~0.35–0.45)
    b       : damping (typical ~0.75–1.0)
    epsilon : coupling strength
    """

    def __init__(self, a: float = 0.375, b: float = 0.81,
                 epsilon: float = 0.0):
        super().__init__(epsilon=epsilon, a=a, b=b)

    def slope(self, t, s, means):
        x, y, z = s[:, 0], s[:, 1], s[:, 2]
        dx = y
        dy = x * (1.0 - z) - self.b * y 
        dz = -self.a * (z - x**2)        + self.epsilon * (means["x2"] - z)
        return np.stack([dx, dy, dz], axis=1)

    @property
    def dim(self):
        return 3

    @property
    def default_ic(self):
        return np.array([0.5, 0.0, 0.5])

    @property
    def name(self):
        return "ShimizuMorioka2"

    @property
    def labels(self):
        return ["x", "y", "z"]
# ══════════════════════════════════════════════
#  2. INTEGRATOR
# ══════════════════════════════════════════════

class RK4Integrator:
    """
    4th-order Runge-Kutta integrator for coupled dynamical systems.

    Mean fields are recomputed at every RK4 stage for full-order
    accuracy on the coupled system (no operator splitting).

    Parameters
    ----------
    system : DynamicalSystem instance
    dt     : time step
    """

    def __init__(self, system: DynamicalSystem, dt: float = 0.005):
        self.system = system
        self.dt = dt

    def step(self, t: float, s: np.ndarray) -> np.ndarray:
        """
        Advance the state by one time step.

        Parameters
        ----------
        t : current time
        s : state array, shape (N, dim)

        Returns
        -------
        s_new : updated state, shape (N, dim)
        """
        dt = self.dt
        sys = self.system

        # k1
        m = sys._compute_means(s)
        k1 = sys.slope(t, s, m)

        # k2
        s2 = s + 0.5 * dt * k1
        k2 = sys.slope(t + 0.5 * dt, s2, m)

        # k3
        s3 = s + 0.5 * dt * k2
        k3 = sys.slope(t + 0.5 * dt, s3, m)

        # k4
        s4 = s + dt * k3
        k4 = sys.slope(t + dt, s4, m)

        return s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ══════════════════════════════════════════════
#  3. ESCAPE DETECTOR
# ══════════════════════════════════════════════

@dataclass
class EscapeInfo:
    """Information about a detected escape/divergence event."""
    escaped: bool = False
    step: int = -1
    kind: str = ""          # "element", "mean", "nan"
    indices: np.ndarray = field(default_factory=lambda: np.array([]))
    means: np.ndarray = field(default_factory=lambda: np.array([]))

    def __repr__(self):
        if not self.escaped:
            return "EscapeInfo(bounded)"
        return (f"EscapeInfo(escaped=True, step={self.step}, "
                f"kind='{self.kind}', indices={self.indices})")


class EscapeDetector:
    """
    Checks for divergence in the state at every timestep.

    Detects three types of escape:
        - "element" : any individual oscillator exceeds threshold
        - "mean"    : any ensemble mean exceeds threshold
        - "nan"     : NaN or Inf appears in the state

    Parameters
    ----------
    threshold : escape threshold for absolute values (default 100)
    """

    def __init__(self, threshold: float = 100.0):
        self.threshold = threshold

    def check(self, s: np.ndarray, means: Dict[str, float],
              step: int) -> Optional[EscapeInfo]:
        """
        Check state for divergence. Returns EscapeInfo if escaped, else None.
        """
        # NaN / Inf check
        if not np.all(np.isfinite(s)):
            bad = np.where(~np.isfinite(s).all(axis=1))[0]
            mean_arr = np.array(list(means.values()))
            return EscapeInfo(True, step, "nan", bad, mean_arr)

        # Individual oscillator check
        mask = np.any(np.abs(s) > self.threshold, axis=1)
        if np.any(mask):
            mean_arr = np.array(list(means.values()))
            return EscapeInfo(True, step, "element", np.where(mask)[0], mean_arr)

        # Mean field check
        mean_arr = np.array(list(means.values()))
        if np.any(np.abs(mean_arr) > self.threshold):
            return EscapeInfo(True, step, "mean", np.array([]), mean_arr)

        return None


# ══════════════════════════════════════════════
#  4. SIMULATION RESULT
# ══════════════════════════════════════════════

@dataclass
class SimulationResult:
    """
    Container for simulation output.

    Attributes
    ----------
    tail       : np.ndarray, shape (window, N, dim) — last `window` steps
    t_tail     : np.ndarray, shape (window,) — time values for tail
    means_tail : np.ndarray, shape (window, dim) — ensemble means for tail
    escape     : EscapeInfo
    total_steps: int — actual number of steps completed
    dt         : float
    system     : DynamicalSystem
    N          : int
    full_traj  : Optional[np.ndarray] — full trajectory if save_full=True
    full_time  : Optional[np.ndarray]
    full_means : Optional[np.ndarray]
    avg_sigma  : Optional[np.ndarray] — time-averaged std over tail
    """
    tail: np.ndarray
    t_tail: np.ndarray
    means_tail: np.ndarray
    escape: EscapeInfo
    total_steps: int
    dt: float
    system: DynamicalSystem
    N: int
    full_traj: Optional[np.ndarray] = None
    full_time: Optional[np.ndarray] = None
    full_means: Optional[np.ndarray] = None
    avg_sigma: Optional[np.ndarray] = None


# ══════════════════════════════════════════════
#  5. SIMULATION RUNNER
# ══════════════════════════════════════════════

class SimulationRunner:
    """
    Main simulation orchestrator.

    Uses rolling buffers by default (memory-efficient for long runs).
    Optionally saves the full trajectory in memory.

    Parameters
    ----------
    system_cls    : class inheriting from DynamicalSystem
    system_params : dict of parameters passed to the system constructor
                    (excluding epsilon, which is passed separately)
    N             : number of coupled oscillators
    epsilon       : coupling strength
    dt            : integration time step
    steps         : total number of integration steps
    threshold     : escape detection threshold
    window        : size of the rolling tail buffer
    ic_spread     : spread of random perturbation around default IC
    ic_dist       : "uniform" or "normal" — distribution for IC perturbation
    save_full     : if True, store the entire trajectory in memory
    seed          : random seed (None for non-reproducible)
    """

    def __init__(
        self,
        system_cls: Type[DynamicalSystem] = LinzSprott,
        system_params: Optional[Dict[str, Any]] = None,
        N: int = 100,
        epsilon: float = 0.001,
        dt: float = 0.005,
        steps: int = 200_000,
        threshold: float = 100.0,
        window: int = 5000,
        ic_spread: float = 0.6,
        ic_dist: str = "uniform",
        save_full: bool = False,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        self.N = N
        self.epsilon = epsilon
        self.dt = dt
        self.steps = steps
        self.threshold = threshold
        self.window = window
        self.ic_spread = ic_spread
        self.ic_dist = ic_dist
        self.save_full = save_full
        self.seed = seed
        self.verbose = verbose

        # Build system.
        # For N=1 the mean field equals the single oscillator's value at every
        # RK4 stage k1, so coupling is zero.  However, the frozen-field RK4
        # (mean computed once from s, reused for k2/k3/k4) introduces a
        # spurious ε-dependent force at the substages because the stale mean
        # no longer equals the intermediate state values.  Forcing epsilon=0
        # for N=1 eliminates this artefact while leaving N>1 unchanged.
        params = system_params or {}
        effective_epsilon = 0.0 if N == 1 else epsilon
        self.system = system_cls(epsilon=effective_epsilon, **params)
        self.integrator = RK4Integrator(self.system, dt)
        self.detector = EscapeDetector(threshold)

    def _init_state(self, rng: np.random.Generator) -> np.ndarray:
        """Generate initial conditions: default IC + random perturbation."""
        ic = self.system.default_ic
        dim = self.system.dim
        if self.ic_dist == "uniform":
            perturbation = rng.uniform(-self.ic_spread, self.ic_spread, (self.N, dim))
        elif self.ic_dist == "normal":
            perturbation = rng.normal(0, self.ic_spread, (self.N, dim))
        else:
            raise ValueError(f"Unknown ic_dist: {self.ic_dist}")
        return ic + perturbation

    def run(self) -> SimulationResult:
        """
        Execute the simulation.

        Returns
        -------
        SimulationResult with rolling tail buffer (and optionally full trajectory).
        """
        if self.verbose:
            _print_logo()
            print(f"System   : {self.system}")
            print(f"N={self.N}, ε={self.epsilon}, dt={self.dt}, "
                  f"steps={self.steps}, threshold={self.threshold}")
            print(f"Window={self.window}, save_full={self.save_full}")
            print("-" * 50)

        rng = np.random.default_rng(self.seed)
        s = self._init_state(rng)
        dim = self.system.dim

        # Rolling buffers
        buf_s = deque(maxlen=self.window)
        buf_m = deque(maxlen=self.window)

        # Optional full storage
        if self.save_full:
            full_s = np.empty((self.steps, self.N, dim), dtype=np.float64)
            full_m = np.empty((self.steps, dim), dtype=np.float64)
        else:
            full_s = None
            full_m = None

        escape = EscapeInfo()
        t0 = time.time()

        for step in range(self.steps):
            means = self.system._compute_means(s)
            mean_arr = np.array(list(means.values()))

            # Store current state
            buf_s.append(s.copy())
            buf_m.append(mean_arr.copy())

            if self.save_full:
                full_s[step] = s
                full_m[step] = mean_arr

            # Escape check
            esc = self.detector.check(s, means, step)
            if esc is not None:
                escape = esc
                if self.verbose:
                    print(f"\n  *** ESCAPE at step {step} "
                          f"(t={step*self.dt:.4f})")
                    print(f"      Kind: {esc.kind}, Means: {esc.means}")
                break

            # Integrate
            s = self.integrator.step(step * self.dt, s)

            # Progress
            if self.verbose and (step + 1) % (self.steps // 10) == 0:
                elapsed = time.time() - t0
                pct = 100 * (step + 1) / self.steps
                print(f"  [{pct:5.1f}%] step {step+1:>8d}/{self.steps}  "
                      f"({elapsed:.1f}s elapsed)")

        total_steps = step + 1 if escape.escaped else self.steps
        elapsed = time.time() - t0
        if self.verbose:
            print(f"\nCompleted {total_steps} steps in {elapsed:.2f}s")
            print(f"Escaped: {escape.escaped}")

        # Build tail arrays
        tail = np.array(buf_s)
        means_tail = np.array(buf_m)
        t_tail = np.arange(
            max(0, total_steps - len(tail)),
            total_steps
        ) * self.dt

        # Time-averaged std over the tail
        # std across oscillators (axis=1) at each timestep, then average
        sig_per_step = np.std(tail, axis=1)       # (window, dim)
        avg_sigma = np.mean(sig_per_step, axis=0)  # (dim,)

        # Trim full arrays if escaped early
        if self.save_full:
            full_s = full_s[:total_steps]
            full_m = full_m[:total_steps]
            full_t = np.arange(total_steps) * self.dt
        else:
            full_t = None

        return SimulationResult(
            tail=tail,
            t_tail=t_tail,
            means_tail=means_tail,
            escape=escape,
            total_steps=total_steps,
            dt=self.dt,
            system=self.system,
            N=self.N,
            full_traj=full_s,
            full_time=full_t,
            full_means=full_m,
            avg_sigma=avg_sigma,
        )


# ══════════════════════════════════════════════
#  6. I/O
# ══════════════════════════════════════════════

def save_npz(result: SimulationResult, filepath: str):
    """
    Save simulation result to a compressed .npz file.

    Saves the tail buffer, means, escape info, and parameters.
    If save_full was True, also saves the full trajectory.
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    data = {
        "tail": result.tail,
        "t_tail": result.t_tail,
        "means_tail": result.means_tail,
        "total_steps": result.total_steps,
        "dt": result.dt,
        "N": result.N,
        "escaped": result.escape.escaped,
        "escape_step": result.escape.step,
        "escape_kind": result.escape.kind,
        "avg_sigma": result.avg_sigma,
        "system_name": result.system.name,
        "epsilon": result.system.epsilon,
    }
    if result.full_traj is not None:
        data["full_traj"] = result.full_traj
        data["full_time"] = result.full_time
        data["full_means"] = result.full_means

    np.savez_compressed(filepath, **data)
    print(f"Saved NPZ → {filepath}")


def save_hdf5(result: SimulationResult, filepath: str):
    """
    Save simulation result to an HDF5 file with gzip compression.

    Requires h5py. Saves tail, means, parameters, and optionally
    the full trajectory.
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required for HDF5 output. "
                          "Install with: pip install h5py")

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    with h5py.File(filepath, "w") as f:
        compress = dict(compression="gzip", compression_opts=4, chunks=True)

        # Tail data
        f.create_dataset("tail", data=result.tail, **compress)
        f.create_dataset("t_tail", data=result.t_tail, **compress)
        f.create_dataset("means_tail", data=result.means_tail, **compress)

        # Full trajectory (if available)
        if result.full_traj is not None:
            f.create_dataset("full_traj", data=result.full_traj, **compress)
            f.create_dataset("full_time", data=result.full_time, **compress)
            f.create_dataset("full_means", data=result.full_means, **compress)

        # Scalars and metadata
        f.create_dataset("avg_sigma", data=result.avg_sigma)
        f.attrs["total_steps"] = result.total_steps
        f.attrs["dt"] = result.dt
        f.attrs["N"] = result.N
        f.attrs["epsilon"] = result.system.epsilon
        f.attrs["system_name"] = result.system.name
        f.attrs["escaped"] = result.escape.escaped
        f.attrs["escape_step"] = result.escape.step
        f.attrs["escape_kind"] = result.escape.kind

    print(f"Saved HDF5 → {filepath}")


# ══════════════════════════════════════════════
#  7. PLOTTER
# ══════════════════════════════════════════════

class Plotter:
    """
    Publication-quality plots for simulation results.

    All methods are static — no state is kept between calls.

    Parameters common to all methods
    ---------------------------------
    output_dir : directory to save the PNG; if None, nothing is saved
    show       : if True, display interactively via plt.show()
    """

    @staticmethod
    def _finish(fig, fname: Optional[str], show: bool) -> None:
        """Save and/or show a figure, then close it."""
        if fname is not None:
            fig.savefig(fname, dpi=200, bbox_inches="tight")
            print(f"Saved plot → {fname}")
        if show:
            plt.show()
        if not show:
            plt.close(fig)

    @staticmethod
    def plot_tail_timeseries(
        result: SimulationResult,
        output_dir: Optional[str] = None,
        show: bool = False,
        max_traces: int = 10,
    ):
        """
        Plot the rolling-buffer tail: individual traces (gray) + mean (color).
        """
        if output_dir is None and not show:
            return
        tail = result.tail
        means = result.means_tail
        t = result.t_tail
        labels = result.system.labels
        dim = result.system.dim
        colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        fig, axes = plt.subplots(dim, 1, figsize=(20, 4 * dim),
                                 sharex=True, constrained_layout=True)
        if dim == 1:
            axes = [axes]

        num_plot = min(max_traces, result.N)
        for ax, i in zip(axes, range(dim)):
            for j in range(num_plot):
                ax.plot(t, tail[:, j, i], alpha=0.12, color="gray", linewidth=0.5)
            ax.plot(t, means[:, i],
                    label=fr"$\langle {labels[i]}(t) \rangle$",
                    color=colors[i % len(colors)], linewidth=2.5)
            ax.set_ylabel(fr"${labels[i]}(t)$", fontsize=16, labelpad=8)
            ax.tick_params(labelsize=12)
            ax.legend(fontsize=14, loc="upper right")

        axes[-1].set_xlabel("Time", fontsize=16, labelpad=8)
        sys_info = repr(result.system)
        fig.suptitle(f"Tail Time Series — {sys_info}", fontsize=18)

        fname = _safe_filename(result, "tail_timeseries", output_dir) if output_dir else None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        Plotter._finish(fig, fname, show)

    @staticmethod
    def plot_attractor_3d(
        result: SimulationResult,
        output_dir: Optional[str] = None,
        show: bool = False,
        components: Tuple[int, int, int] = (0, 1, 2),
        fraction: float = 0.5,
    ):
        """
        Plot the 3D phase-space attractor from the mean trajectory tail,
        colored by time progression.

        Parameters
        ----------
        components : tuple of 3 component indices to plot
        fraction   : fraction of the tail to use (from the end)
        """
        if output_dir is None and not show:
            return
        if result.system.dim < 3:
            warnings.warn("3D attractor requires dim >= 3, skipping.")
            return

        means = result.means_tail
        labels = result.system.labels
        ci, cj, ck = components

        n = max(2, int(fraction * len(means)))
        x = means[-n:, ci]
        y = means[-n:, cj]
        z = means[-n:, ck]

        fig = plt.figure(figsize=(10, 10), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")

        norm = plt.Normalize(0, len(x))
        colors = cm.magma(norm(np.arange(len(x))))
        for i in range(len(x) - 1):
            ax.plot(x[i:i + 2], y[i:i + 2], z[i:i + 2],
                    color=colors[i], linewidth=1.5)

        ax.set_xlabel(fr"${labels[ci]}$", fontsize=14, labelpad=8)
        ax.set_ylabel(fr"${labels[cj]}$", fontsize=14, labelpad=8)
        ax.set_zlabel(fr"${labels[ck]}$", fontsize=14, labelpad=8)
        ax.tick_params(labelsize=10)
        ax.grid(False)
        ax.view_init(elev=20, azim=-45)

        sm = plt.cm.ScalarMappable(cmap=cm.magma, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.1)
        cbar.set_label("Time progression", fontsize=12)
        cbar.ax.set_yticklabels([])

        sys_info = repr(result.system)
        fig.suptitle(f"Mean Attractor — {sys_info}", fontsize=16)

        fname = _safe_filename(result, "attractor_3d", output_dir) if output_dir else None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        Plotter._finish(fig, fname, show)

    @staticmethod
    def plot_sigma_evolution(
        result: SimulationResult,
        output_dir: Optional[str] = None,
        show: bool = False,
    ):
        """
        Plot the time evolution of the standard deviation across oscillators
        for each component, using the tail buffer.
        """
        if output_dir is None and not show:
            return
        tail = result.tail
        t = result.t_tail
        labels = result.system.labels
        dim = result.system.dim
        colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        sigma = np.std(tail, axis=1)  # (window, dim)

        fig, ax = plt.subplots(figsize=(16, 5), constrained_layout=True)
        for i in range(dim):
            ax.plot(t, sigma[:, i], label=fr"$\sigma_{{{labels[i]}}}$",
                    color=colors[i % len(colors)], linewidth=1.5)

        ax.set_xlabel("Time", fontsize=14)
        ax.set_ylabel(r"$\sigma$ (std across oscillators)", fontsize=14)
        ax.legend(fontsize=13)
        ax.tick_params(labelsize=11)
        sys_info = repr(result.system)
        fig.suptitle(f"Dispersion — {sys_info}", fontsize=16)

        fname = _safe_filename(result, "sigma", output_dir) if output_dir else None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        Plotter._finish(fig, fname, show)

    @staticmethod
    def plot_all(
        result: SimulationResult,
        output_dir: Optional[str] = None,
        show: bool = False,
    ):
        """Generate all standard plots."""
        Plotter.plot_tail_timeseries(result, output_dir=output_dir, show=show)
        Plotter.plot_attractor_3d(result, output_dir=output_dir, show=show)
        Plotter.plot_sigma_evolution(result, output_dir=output_dir, show=show)


# ══════════════════════════════════════════════
#  8. UTILITIES
# ══════════════════════════════════════════════

def _safe_str(x) -> str:
    """Convert a number to a filename-safe string."""
    return str(x).replace(".", "p").replace("-", "m")


def _safe_filename(result: SimulationResult, tag: str, output_dir: str) -> str:
    """Build a descriptive, filesystem-safe filename."""
    sys_name = result.system.name.replace("ö", "o")
    eps_str = _safe_str(result.system.epsilon)
    n_str = result.N
    return os.path.join(
        output_dir,
        f"{sys_name}_N{n_str}_e{eps_str}_{tag}.png"
    )


def _print_logo():
    logo = r"""
    ╔══════════════════════════════════════╗
    ║             SOPHROSYNE               ║
    ║              v1.0.0                  ║
    ║   Coupled Dynamical Systems Solver   ║
    ║            by R.S.S.G.               ║
    ╚══════════════════════════════════════╝
    """
    print(logo)


# ══════════════════════════════════════════════
#  9. PARAMETER SWEEP
# ══════════════════════════════════════════════

@dataclass
class SweepPoint:
    """Result of a single point in the parameter sweep."""
    N: int
    a: float
    epsilon: float
    escaped: bool
    escape_step: int          # -1 if bounded
    escape_kind: str          # "" if bounded
    total_steps: int
    avg_sigma: np.ndarray     # shape (dim,)
    wall_time: float          # seconds for this run


@dataclass
class SweepResult:
    """
    Full results of a parameter sweep over (N, a, ε).

    Attributes
    ----------
    points      : list of SweepPoint
    N_values    : sorted unique N values in the sweep
    a_values    : sorted unique a values
    eps_values  : sorted unique epsilon values
    system_name : name of the system used
    dt          : timestep used
    steps       : integration steps per run
    wall_time   : total wall-clock time for the sweep
    """
    points: List[SweepPoint]
    N_values: np.ndarray
    a_values: np.ndarray
    eps_values: np.ndarray
    system_name: str
    dt: float
    steps: int
    wall_time: float

    def escape_fraction(self, N: Optional[int] = None) -> float:
        """Fraction of runs that escaped, optionally filtered by N."""
        pts = self.points if N is None else [p for p in self.points if p.N == N]
        if not pts:
            return 0.0
        return sum(1 for p in pts if p.escaped) / len(pts)

    def to_grid(self, N: int, value: str = "escaped") -> np.ndarray:
        """
        Build a 2D array indexed by (a, epsilon) for a fixed N.

        Parameters
        ----------
        N     : which N slice to extract
        value : "escaped" (bool→int), "escape_step", or "avg_sigma_norm"

        Returns
        -------
        grid : np.ndarray of shape (len(a_values), len(eps_values))
        """
        a_idx = {v: i for i, v in enumerate(self.a_values)}
        e_idx = {v: i for i, v in enumerate(self.eps_values)}
        grid = np.full((len(self.a_values), len(self.eps_values)), np.nan)

        for p in self.points:
            if p.N != N:
                continue
            i, j = a_idx[p.a], e_idx[p.epsilon]
            if value == "escaped":
                grid[i, j] = float(p.escaped)
            elif value == "escape_step":
                grid[i, j] = p.escape_step if p.escaped else p.total_steps
            elif value == "avg_sigma_norm":
                grid[i, j] = np.linalg.norm(p.avg_sigma)
            else:
                raise ValueError(f"Unknown value: {value}")
        return grid

    def save(self, filepath: str):
        """Save sweep results to a compressed .npz file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        data = {
            "N_values": self.N_values,
            "a_values": self.a_values,
            "eps_values": self.eps_values,
            "system_name": self.system_name,
            "dt": self.dt,
            "steps": self.steps,
            "wall_time": self.wall_time,
            # Per-point arrays
            "pt_N": np.array([p.N for p in self.points]),
            "pt_a": np.array([p.a for p in self.points]),
            "pt_epsilon": np.array([p.epsilon for p in self.points]),
            "pt_escaped": np.array([p.escaped for p in self.points]),
            "pt_escape_step": np.array([p.escape_step for p in self.points]),
            "pt_total_steps": np.array([p.total_steps for p in self.points]),
            "pt_avg_sigma": np.array([p.avg_sigma for p in self.points]),
            "pt_wall_time": np.array([p.wall_time for p in self.points]),
        }
        np.savez_compressed(filepath, **data)
        print(f"Saved sweep → {filepath}")

    @staticmethod
    def load(filepath: str) -> "SweepResult":
        """Load a SweepResult from a .npz file."""
        d = np.load(filepath, allow_pickle=True)
        points = []
        for i in range(len(d["pt_N"])):
            points.append(SweepPoint(
                N=int(d["pt_N"][i]),
                a=float(d["pt_a"][i]),
                epsilon=float(d["pt_epsilon"][i]),
                escaped=bool(d["pt_escaped"][i]),
                escape_step=int(d["pt_escape_step"][i]),
                escape_kind="",
                total_steps=int(d["pt_total_steps"][i]),
                avg_sigma=d["pt_avg_sigma"][i],
                wall_time=float(d["pt_wall_time"][i]),
            ))
        return SweepResult(
            points=points,
            N_values=d["N_values"],
            a_values=d["a_values"],
            eps_values=d["eps_values"],
            system_name=str(d["system_name"]),
            dt=float(d["dt"]),
            steps=int(d["steps"]),
            wall_time=float(d["wall_time"]),
        )


def _sweep_worker(args: Tuple) -> SweepPoint:
    """
    Worker function for a single sweep point.

    Must be a top-level function (not a method) so that
    multiprocessing can pickle it.

    Parameters
    ----------
    args : tuple of (system_cls, system_params_base, N, a, epsilon,
                      dt, steps, threshold, window, ic_spread, ic_dist, seed)

    Returns
    -------
    SweepPoint
    """
    (system_cls, sys_params_base, N, a, epsilon,
     dt, steps, threshold, window, ic_spread, ic_dist, seed) = args

    # Merge 'a' into system params (overrides base)
    sys_params = dict(sys_params_base)
    sys_params["a"] = a

    t0 = time.time()
    runner = SimulationRunner(
        system_cls=system_cls,
        system_params=sys_params,
        N=N,
        epsilon=epsilon,
        dt=dt,
        steps=steps,
        threshold=threshold,
        window=window,
        ic_spread=ic_spread,
        ic_dist=ic_dist,
        save_full=False,
        seed=seed,
        verbose=False,
    )
    result = runner.run()
    wt = time.time() - t0

    return SweepPoint(
        N=N,
        a=a,
        epsilon=epsilon,
        escaped=result.escape.escaped,
        escape_step=result.escape.step,
        escape_kind=result.escape.kind,
        total_steps=result.total_steps,
        avg_sigma=result.avg_sigma,
        wall_time=wt,
    )


class ParameterSweep:
    """
    Parallel parameter sweep over (N, a, ε).

    Distributes independent runs across CPU cores using
    multiprocessing (single machine) or mpi4py (cluster).

    Parameters
    ----------
    system_cls      : DynamicalSystem subclass
    system_params   : base system parameters (a will be overridden per point)
    N_values        : list/array of N values to scan
    a_values        : list/array of a values to scan
    eps_values      : list/array of epsilon values to scan
    dt              : timestep for each run
    steps           : integration steps per run
    threshold       : escape threshold
    window          : rolling buffer size
    ic_spread       : IC perturbation spread
    ic_dist         : "uniform" or "normal"
    seed            : base seed (each run gets seed + index for reproducibility;
                      None for non-reproducible)
    n_workers       : number of parallel workers (default: CPU count)

    Usage
    -----
    sweep = ParameterSweep(
        system_cls=LinzSprott,
        N_values=[20, 50, 100],
        a_values=np.linspace(0.5, 0.6, 20),
        eps_values=np.logspace(-4, -1, 15),
    )
    result = sweep.run()                    # multiprocessing
    result = sweep.run_mpi()                # mpi4py on cluster
    SweepPlotter.phase_diagram(result, N=50)
    """

    def __init__(
        self,
        system_cls: Type[DynamicalSystem] = LinzSprott,
        system_params: Optional[Dict[str, Any]] = None,
        N_values: List[int] = None,
        a_values: np.ndarray = None,
        eps_values: np.ndarray = None,
        dt: float = 0.005,
        steps: int = 200_000,
        threshold: float = 100.0,
        window: int = 2000,
        ic_spread: float = 0.6,
        ic_dist: str = "uniform",
        seed: Optional[int] = None,
        n_workers: Optional[int] = None,
    ):
        self.system_cls = system_cls
        self.system_params = system_params or {}
        self.N_values = np.atleast_1d(N_values if N_values is not None else [100])
        self.a_values = np.atleast_1d(a_values if a_values is not None
                                      else np.array([0.553]))
        self.eps_values = np.atleast_1d(eps_values if eps_values is not None
                                        else np.array([0.001]))
        self.dt = dt
        self.steps = steps
        self.threshold = threshold
        self.window = window
        self.ic_spread = ic_spread
        self.ic_dist = ic_dist
        self.seed = seed
        self.n_workers = n_workers

    def _build_task_list(self) -> List[Tuple]:
        """Generate all (N, a, ε) combinations as worker arguments."""
        tasks = []
        idx = 0
        for N in self.N_values:
            for a in self.a_values:
                for eps in self.eps_values:
                    s = (self.seed + idx) if self.seed is not None else None
                    tasks.append((
                        self.system_cls, self.system_params,
                        int(N), float(a), float(eps),
                        self.dt, self.steps, self.threshold,
                        self.window, self.ic_spread, self.ic_dist, s
                    ))
                    idx += 1
        return tasks

    def run(self) -> SweepResult:
        """
        Execute the sweep using multiprocessing.

        Returns
        -------
        SweepResult
        """
        import multiprocessing as mp

        tasks = self._build_task_list()
        n_total = len(tasks)
        workers = self.n_workers or mp.cpu_count()

        _print_logo()
        print(f"Parameter Sweep")
        print(f"  System : {self.system_cls.__name__}")
        print(f"  Grid   : {len(self.N_values)} N × "
              f"{len(self.a_values)} a × {len(self.eps_values)} ε "
              f"= {n_total} runs")
        print(f"  Workers: {workers}")
        print(f"  Steps/run: {self.steps}")
        print("-" * 50)

        t0 = time.time()
        completed = 0

        # Use imap_unordered for progress reporting
        points = []
        with mp.Pool(processes=workers) as pool:
            for pt in pool.imap_unordered(_sweep_worker, tasks):
                points.append(pt)
                completed += 1
                if completed % max(1, n_total // 20) == 0 or completed == n_total:
                    elapsed = time.time() - t0
                    esc_count = sum(1 for p in points if p.escaped)
                    pct = 100 * completed / n_total
                    print(f"  [{pct:5.1f}%] {completed}/{n_total} done "
                          f"({esc_count} escaped)  "
                          f"[{elapsed:.1f}s elapsed]")

        wall = time.time() - t0
        esc_total = sum(1 for p in points if p.escaped)
        print(f"\nSweep complete: {n_total} runs in {wall:.1f}s "
              f"({esc_total} escaped, "
              f"{n_total - esc_total} bounded)")

        # Build a temporary system to get its name
        tmp = self.system_cls(**self.system_params, epsilon=0.0)

        return SweepResult(
            points=points,
            N_values=np.sort(np.unique(self.N_values)),
            a_values=np.sort(np.unique(self.a_values)),
            eps_values=np.sort(np.unique(self.eps_values)),
            system_name=tmp.name,
            dt=self.dt,
            steps=self.steps,
            wall_time=wall,
        )

    def run_mpi(self) -> Optional[SweepResult]:
        """
        Execute the sweep using mpi4py (for cluster environments).

        Call this from all MPI ranks. Only rank 0 returns the
        SweepResult; other ranks return None.

        Usage
        -----
        From the command line:
            mpirun -n 32 python -c "
                from sophrosyne import ParameterSweep, LinzSprott
                sweep = ParameterSweep(...)
                result = sweep.run_mpi()
                if result is not None:
                    result.save('sweep.npz')
            "
        """
        try:
            from mpi4py import MPI
        except ImportError:
            raise ImportError(
                "mpi4py is required for MPI sweeps. "
                "Install with: pip install mpi4py"
            )

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        tasks = self._build_task_list() if rank == 0 else None

        # Scatter tasks across ranks
        if rank == 0:
            n_total = len(tasks)
            _print_logo()
            print(f"MPI Parameter Sweep — {size} ranks, {n_total} tasks")
            # Distribute as evenly as possible
            chunks = [[] for _ in range(size)]
            for i, task in enumerate(tasks):
                chunks[i % size].append(task)
        else:
            chunks = None
            n_total = 0

        local_tasks = comm.scatter(chunks, root=0)
        n_total = comm.bcast(n_total if rank == 0 else 0, root=0)

        # Each rank processes its share
        local_points = []
        for task in local_tasks:
            pt = _sweep_worker(task)
            local_points.append(pt)

        # Gather all results to rank 0
        all_points = comm.gather(local_points, root=0)

        if rank == 0:
            points = [p for chunk in all_points for p in chunk]
            wall = MPI.Wtime()

            esc_total = sum(1 for p in points if p.escaped)
            print(f"MPI sweep complete: {len(points)} runs "
                  f"({esc_total} escaped, "
                  f"{len(points) - esc_total} bounded)")

            tmp = self.system_cls(**self.system_params, epsilon=0.0)
            return SweepResult(
                points=points,
                N_values=np.sort(np.unique(self.N_values)),
                a_values=np.sort(np.unique(self.a_values)),
                eps_values=np.sort(np.unique(self.eps_values)),
                system_name=tmp.name,
                dt=self.dt,
                steps=self.steps,
                wall_time=wall,
            )
        return None


# ══════════════════════════════════════════════
#  10. SWEEP PLOTTER
# ══════════════════════════════════════════════

class SweepPlotter:
    """
    Phase diagrams and summary plots for parameter sweeps.
    """

    @staticmethod
    def phase_diagram(
        result: SweepResult,
        N: int,
        value: str = "escaped",
        output_dir: str = ".",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """
        Plot a 2D phase diagram in (a, ε) space for a fixed N.

        Parameters
        ----------
        result : SweepResult from a sweep
        N      : which N slice to plot
        value  : "escaped" (binary), "escape_step", or "avg_sigma_norm"
        output_dir : where to save the PNG
        ax     : optional existing axes (for multi-panel figures)

        Returns
        -------
        fig : matplotlib Figure
        """
        grid = result.to_grid(N, value=value)
        a_vals = result.a_values
        eps_vals = result.eps_values

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
        else:
            fig = ax.get_figure()

        if value == "escaped":
            cmap = plt.cm.RdYlGn_r  # red = escaped, green = bounded
            vmin, vmax = 0, 1
            label = "Escaped (1) / Bounded (0)"
        elif value == "escape_step":
            cmap = plt.cm.viridis
            vmin, vmax = 0, result.steps
            label = "Escape step (or max steps)"
        elif value == "avg_sigma_norm":
            cmap = plt.cm.plasma
            vmin, vmax = None, None
            label = r"$\|\bar{\sigma}\|$"
        else:
            raise ValueError(f"Unknown value: {value}")

        im = ax.pcolormesh(
            eps_vals, a_vals, grid,
            cmap=cmap, vmin=vmin, vmax=vmax,
            shading="nearest",
        )
        ax.set_xlabel(r"$\epsilon$", fontsize=16)
        ax.set_ylabel(r"$a$", fontsize=16)
        ax.set_title(
            f"{result.system_name} — N={N}, "
            f"{result.steps} steps",
            fontsize=16,
        )
        ax.tick_params(labelsize=12)

        # Log scale for epsilon if range spans > 1 decade
        if eps_vals.max() / max(eps_vals.min(), 1e-15) > 10:
            ax.set_xscale("log")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label, fontsize=13)
        cbar.ax.tick_params(labelsize=11)

        if own_fig:
            os.makedirs(output_dir, exist_ok=True)
            fname = os.path.join(
                output_dir,
                f"phase_{result.system_name}_N{N}_{value}.png"
            )
            fig.savefig(fname, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved phase diagram → {fname}")

        return fig

    @staticmethod
    def phase_diagram_all_N(
        result: SweepResult,
        value: str = "escaped",
        output_dir: str = ".",
    ) -> plt.Figure:
        """
        Plot phase diagrams for all N values side by side.
        """
        N_vals = result.N_values
        n_panels = len(N_vals)
        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(7 * n_panels, 6),
            constrained_layout=True,
            squeeze=False,
        )
        for ax, N in zip(axes[0], N_vals):
            SweepPlotter.phase_diagram(result, N=int(N), value=value, ax=ax)

        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(
            output_dir,
            f"phase_allN_{result.system_name}_{value}.png"
        )
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved multi-N phase diagram → {fname}")
        return fig

    @staticmethod
    def escape_fraction_vs_N(
        result: SweepResult,
        output_dir: str = ".",
    ) -> plt.Figure:
        """
        Bar chart of escape fraction as a function of N.
        """
        N_vals = result.N_values
        fracs = [result.escape_fraction(N=int(N)) for N in N_vals]

        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.bar([str(int(n)) for n in N_vals], fracs, color="#d62728", alpha=0.85)
        ax.set_xlabel("N (oscillators)", fontsize=14)
        ax.set_ylabel("Escape fraction", fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{result.system_name} — escape fraction by N", fontsize=15)
        ax.tick_params(labelsize=12)

        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(
            output_dir,
            f"escape_vs_N_{result.system_name}.png"
        )
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved escape vs N → {fname}")
        return fig

    @staticmethod
    def plot_all(result: SweepResult, output_dir: str = "."):
        """Generate all sweep plots."""
        for N in result.N_values:
            SweepPlotter.phase_diagram(result, N=int(N), value="escaped",
                                       output_dir=output_dir)
            SweepPlotter.phase_diagram(result, N=int(N), value="escape_step",
                                       output_dir=output_dir)
            SweepPlotter.phase_diagram(result, N=int(N), value="avg_sigma_norm",
                                       output_dir=output_dir)
        if len(result.N_values) > 1:
            SweepPlotter.phase_diagram_all_N(result, output_dir=output_dir)
            SweepPlotter.escape_fraction_vs_N(result, output_dir=output_dir)


# ══════════════════════════════════════════════
#  11. CLI ENTRY POINT
# ══════════════════════════════════════════════

SYSTEM_REGISTRY: Dict[str, Type[DynamicalSystem]] = {
    "linzsprott": LinzSprott,
    "rossler": Rossler,
    "rossler_global": RosslerGloballyCoupled,
    "shimizu_morioka": ShimizuMorioka,
}


def main():
    """Command-line interface for SOPHROSYNE."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SOPHROSYNE — Coupled Dynamical Systems Simulator"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ---- Single run subcommand ----
    run_p = subparsers.add_parser("run", help="Run a single simulation")
    run_p.add_argument("--system", type=str, default="linzsprott",
                       choices=list(SYSTEM_REGISTRY.keys()))
    run_p.add_argument("--N", type=int, default=100)
    run_p.add_argument("--epsilon", type=float, default=0.001)
    run_p.add_argument("--dt", type=float, default=0.005)
    run_p.add_argument("--steps", type=int, default=200_000)
    run_p.add_argument("--threshold", type=float, default=100.0)
    run_p.add_argument("--window", type=int, default=5000)
    run_p.add_argument("--ic-spread", type=float, default=0.6)
    run_p.add_argument("--ic-dist", type=str, default="uniform",
                       choices=["uniform", "normal"])
    run_p.add_argument("--save-full", action="store_true")
    run_p.add_argument("--seed", type=int, default=None)
    run_p.add_argument("--output-dir", type=str, default="./output")
    run_p.add_argument("--save-npz", type=str, default=None)
    run_p.add_argument("--save-hdf5", type=str, default=None)
    run_p.add_argument("--plot", action="store_true")
    run_p.add_argument("--a", type=float, default=None)
    run_p.add_argument("--b", type=float, default=None)
    run_p.add_argument("--c", type=float, default=None)

    # ---- Sweep subcommand ----
    sw_p = subparsers.add_parser("sweep", help="Run a parameter sweep")
    sw_p.add_argument("--system", type=str, default="linzsprott",
                      choices=list(SYSTEM_REGISTRY.keys()))
    sw_p.add_argument("--N", type=str, default="100",
                      help="Comma-separated N values, e.g. '20,50,100'")
    sw_p.add_argument("--a-min", type=float, default=0.50)
    sw_p.add_argument("--a-max", type=float, default=0.60)
    sw_p.add_argument("--a-num", type=int, default=20)
    sw_p.add_argument("--eps-min", type=float, default=1e-4)
    sw_p.add_argument("--eps-max", type=float, default=1e-1)
    sw_p.add_argument("--eps-num", type=int, default=15)
    sw_p.add_argument("--eps-log", action="store_true", default=True,
                      help="Use log spacing for epsilon (default)")
    sw_p.add_argument("--eps-lin", action="store_true",
                      help="Use linear spacing for epsilon")
    sw_p.add_argument("--dt", type=float, default=0.005)
    sw_p.add_argument("--steps", type=int, default=200_000)
    sw_p.add_argument("--threshold", type=float, default=100.0)
    sw_p.add_argument("--window", type=int, default=2000)
    sw_p.add_argument("--ic-spread", type=float, default=0.6)
    sw_p.add_argument("--ic-dist", type=str, default="uniform",
                      choices=["uniform", "normal"])
    sw_p.add_argument("--seed", type=int, default=None)
    sw_p.add_argument("--workers", type=int, default=None)
    sw_p.add_argument("--output-dir", type=str, default="./output")
    sw_p.add_argument("--save", type=str, default=None,
                      help="Save sweep results to .npz")
    sw_p.add_argument("--plot", action="store_true")
    sw_p.add_argument("--b", type=float, default=None)
    sw_p.add_argument("--c", type=float, default=None)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "run":
        sys_params = {}
        for key in ["a", "b", "c"]:
            val = getattr(args, key)
            if val is not None:
                sys_params[key] = val

        system_cls = SYSTEM_REGISTRY[args.system]
        runner = SimulationRunner(
            system_cls=system_cls,
            system_params=sys_params,
            N=args.N,
            epsilon=args.epsilon,
            dt=args.dt,
            steps=args.steps,
            threshold=args.threshold,
            window=args.window,
            ic_spread=args.ic_spread,
            ic_dist=args.ic_dist,
            save_full=args.save_full,
            seed=args.seed,
        )
        result = runner.run()

        if args.save_npz:
            save_npz(result, os.path.join(args.output_dir, args.save_npz))
        if args.save_hdf5:
            save_hdf5(result, os.path.join(args.output_dir, args.save_hdf5))
        if args.plot:
            Plotter.plot_all(result, args.output_dir)

    elif args.command == "sweep":
        N_values = [int(x) for x in args.N.split(",")]
        a_values = np.linspace(args.a_min, args.a_max, args.a_num)
        if args.eps_lin:
            eps_values = np.linspace(args.eps_min, args.eps_max, args.eps_num)
        else:
            eps_values = np.logspace(
                np.log10(args.eps_min), np.log10(args.eps_max), args.eps_num
            )

        sys_params = {}
        for key in ["b", "c"]:
            val = getattr(args, key)
            if val is not None:
                sys_params[key] = val

        system_cls = SYSTEM_REGISTRY[args.system]
        sweep = ParameterSweep(
            system_cls=system_cls,
            system_params=sys_params,
            N_values=N_values,
            a_values=a_values,
            eps_values=eps_values,
            dt=args.dt,
            steps=args.steps,
            threshold=args.threshold,
            window=args.window,
            ic_spread=args.ic_spread,
            ic_dist=args.ic_dist,
            seed=args.seed,
            n_workers=args.workers,
        )
        result = sweep.run()

        if args.save:
            result.save(os.path.join(args.output_dir, args.save))
        if args.plot:
            SweepPlotter.plot_all(result, args.output_dir)


# ════════════════════════════════════════
# 12. Bifurcation Diagram
# ════════════════════════════════════════

# ── Module-level workers (must be top-level for multiprocessing pickling) ──

def _bif_worker(args: tuple) -> dict:
    """
    N=1 worker: integrates with DOP853 (adaptive, high-accuracy).
    All components extracted from the same trajectory.
    Integration is terminated early if the trajectory escapes (|y| > threshold).

    Memory note: t_eval only covers [t_transient, t_end], so solve_ivp
    stores solution values only for the steady-state window.  The solver
    still integrates through the transient using adaptive internal steps,
    but those intermediate states are never written to memory.
    """
    (system_cls, params, ic, t_transient, t_steady, dt,
     peak_keep, peak_distance, threshold) = args

    system = system_cls(**params)
    dim    = system.dim
    labels = system.labels

    t_end  = t_transient + t_steady
    # Only request output from t_transient onward — transient steps are
    # integrated internally by the solver without being stored.
    t_eval = np.arange(t_transient, t_end + dt / 2.0, dt)

    def rhs(t, y):
        s     = y.reshape(1, dim)
        means = {f"x{i}": float(s[0, i]) for i in range(dim)}
        return system.slope(t, s, means).ravel()

    def escape_event(t, y):
        return threshold - np.max(np.abs(y))
    escape_event.terminal  = True
    escape_event.direction = -1

    out = {
        "peaks": {lab: np.empty(0) for lab in labels},
        "fixed": {lab: np.nan     for lab in labels},
    }

    try:
        sol = solve_ivp(
            rhs,
            (0.0, t_end),           # integrate from t=0, but only store from t_transient
            np.asarray(ic, dtype=float),
            t_eval=t_eval,
            method="DOP853",
            rtol=1e-8,
            atol=1e-8,
            dense_output=False,
            events=escape_event,
        )

        # Escaped or solver failed → return empty (no data for this param value)
        if sol.status == 1 or not sol.success:
            return out

        steady = sol.y   # (dim, n_steady) — already only steady-state points

        for idx, lab in enumerate(labels):
            _extract_peaks(steady[idx], lab, peak_idx_fn=find_peaks,
                           peak_distance=peak_distance, peak_keep=peak_keep, out=out)

    except Exception:
        pass

    return out


def _bif_worker_coupled(args: tuple) -> dict:
    """
    N>1 worker: uses SimulationRunner (RK4 + rolling buffer).
    Memory cost is O(window × N × dim) regardless of total run length.
    Extracts signal from the mean field or a single oscillator.
    """
    (system_cls, params, N, epsilon, steps, window, dt,
     peak_keep, peak_distance, variable, oscillator_index) = args

    runner = SimulationRunner(
        system_cls=system_cls,
        system_params=params,
        N=N,
        epsilon=epsilon,
        dt=dt,
        steps=steps,
        window=window,
        verbose=False,
    )
    result = runner.run()

    labels = result.system.labels

    out = {
        "peaks": {lab: np.empty(0) for lab in labels},
        "fixed": {lab: np.nan     for lab in labels},
    }

    # Discard escaped trajectories — tail data is unreliable after divergence
    if result.escape.escaped:
        return out

    for idx, lab in enumerate(labels):
        if variable == "mean":
            sig = result.means_tail[:, idx]
        else:
            sig = result.tail[:, oscillator_index, idx]

        _extract_peaks(sig, lab, peak_idx_fn=find_peaks,
                       peak_distance=peak_distance, peak_keep=peak_keep, out=out)

    return out


def _extract_peaks(sig, lab, peak_idx_fn, peak_distance, peak_keep, out):
    """Shared peak-extraction logic used by both workers."""
    peak_idx, _ = peak_idx_fn(sig, distance=peak_distance)
    if peak_idx.size:
        sel = peak_idx[-min(peak_keep, peak_idx.size):]
        out["peaks"][lab] = sig[sel]
    else:
        out["fixed"][lab] = float(np.mean(sig))


@dataclass
class BifurcationResult:
    param_values: np.ndarray
    peaks: Dict[str, List[np.ndarray]]  # label -> one array of maxima per param value
    fixed: Dict[str, np.ndarray]        # label -> fixed-point value per param (nan if oscillatory)
    param_name: str
    labels: List[str]


class BifurcationAnalyzer:
    """
    Compute bifurcation diagrams for any DynamicalSystem in SOPHROSYNE.

    Two backends are selected automatically:
      - N=1, epsilon=0 → DOP853 via solve_ivp (adaptive, high-accuracy)
      - N>1  or  epsilon>0 → SimulationRunner (RK4 + rolling buffer, memory-efficient)

    Parameters
    ----------
    system_cls      : DynamicalSystem subclass
    base_params     : fixed system parameters (do not include the swept parameter)
    N               : number of coupled oscillators
    epsilon         : coupling strength (only used when N>1)
    ic              : initial condition for N=1 mode (defaults to system.default_ic)
    t_transient     : time discarded as transient
    t_steady        : time analysed after the transient
    dt              : time step (t_eval step for DOP853; fixed step for RK4)
    peak_keep       : number of last peaks to retain per component per param value
    min_peak_period : minimum time between consecutive peaks (time units)
    threshold       : escape threshold — integration stopped when |y| > threshold (N=1 only)
    variable        : "mean" (ensemble mean field) or "single" (one oscillator); N>1 only
    oscillator_index: which oscillator to track when variable="single"
    n_jobs          : parallel workers; None → os.cpu_count()
    """

    def __init__(
        self,
        system_cls: Type[DynamicalSystem],
        base_params: Optional[Dict[str, Any]] = None,
        N: int = 1,
        epsilon: float = 0.0,
        ic: Optional[np.ndarray] = None,
        t_transient: float = 2000.0,
        t_steady: float = 2000.0,
        dt: float = 0.01,
        peak_keep: int = 20,
        min_peak_period: float = 0.5,
        threshold: float = 100.0,
        variable: str = "mean",
        oscillator_index: int = 0,
        n_jobs: Optional[int] = None,
    ):
        self.system_cls        = system_cls
        self.base_params       = base_params or {}
        self.N                 = N
        self.epsilon           = epsilon
        self.t_transient       = t_transient
        self.t_steady          = t_steady
        self.dt                = dt
        self.peak_keep         = peak_keep
        self.peak_distance     = max(1, int(min_peak_period / dt))
        self.threshold         = threshold
        self.variable          = variable
        self.oscillator_index  = oscillator_index
        self.n_jobs            = n_jobs or os.cpu_count() or 1

        # Coupled mode: convert real time to step counts for SimulationRunner
        self._steps  = int((t_transient + t_steady) / dt)
        self._window = int(t_steady / dt)

        # N=1 mode: resolve IC
        if ic is not None:
            self.ic = np.asarray(ic, dtype=float)
        else:
            tmp = system_cls(**self.base_params)
            self.ic = tmp.default_ic.astype(float)

    @property
    def _coupled(self) -> bool:
        return self.N > 1 or self.epsilon != 0.0

    def compute(
        self,
        param_name: str,
        param_values: np.ndarray,
        verbose: bool = True,
    ) -> BifurcationResult:
        """
        Sweep `param_name` over `param_values` in parallel.

        All state-space components are computed per integration. Fixed points
        (no peaks detected) are stored separately so they remain visible in the plot.
        """
        param_values = np.asarray(param_values)

        if self._coupled:
            args_list = [
                (
                    self.system_cls,
                    {**self.base_params, param_name: float(v)},
                    self.N,
                    self.epsilon,
                    self._steps,
                    self._window,
                    self.dt,
                    self.peak_keep,
                    self.peak_distance,
                    self.variable,
                    self.oscillator_index,
                )
                for v in param_values
            ]
            worker = _bif_worker_coupled
            mode   = f"N={self.N}, ε={self.epsilon}, variable='{self.variable}' [RK4]"
        else:
            args_list = [
                (
                    self.system_cls,
                    {**self.base_params, param_name: float(v)},
                    self.ic,
                    self.t_transient,
                    self.t_steady,
                    self.dt,
                    self.peak_keep,
                    self.peak_distance,
                    self.threshold,
                )
                for v in param_values
            ]
            worker = _bif_worker
            mode   = "N=1 [DOP853]"

        if verbose:
            print(f"Bifurcation sweep: {len(param_values)} values of '{param_name}' "
                  f"on {self.n_jobs} workers — {mode} …")

        t0 = time.time()
        with mp.Pool(processes=self.n_jobs) as pool:
            raw = pool.map(worker, args_list)
        if verbose:
            print(f"Done in {time.time() - t0:.1f} s")

        tmp    = self.system_cls(**self.base_params)
        labels = tmp.labels

        peaks: Dict[str, List[np.ndarray]] = {lab: [] for lab in labels}
        fixed: Dict[str, List[float]]      = {lab: [] for lab in labels}

        for res in raw:
            for lab in labels:
                peaks[lab].append(res["peaks"][lab])
                fixed[lab].append(res["fixed"][lab])

        return BifurcationResult(
            param_values=param_values,
            peaks=peaks,
            fixed={lab: np.asarray(fixed[lab]) for lab in labels},
            param_name=param_name,
            labels=labels,
        )


# ════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════

class BifurcationPlotter:

    @staticmethod
    def plot(
        result: BifurcationResult,
        output: Optional[str] = None,
        show: bool = False,
        max_points: int = 200_000,
        title: Optional[str] = None,
        markersize: float = 0.3,
    ) -> None:
        """
        Three-panel bifurcation diagram, one subplot per state-space component.

        Oscillatory regimes → black scatter (peaks).
        Fixed-point regimes → red scatter (mean value).

        Parameters
        ----------
        output     : file path to save (PNG/EPS/…); skipped if None
        show       : display interactively
        max_points : max scatter points per panel (uniform downsampling)
        markersize : dot size for oscillatory scatter
        """
        if not output and not show:
            return
        labels = result.labels
        n      = len(labels)

        fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n),
                                 sharex=True, constrained_layout=True)
        if n == 1:
            axes = [axes]

        for ax, lab in zip(axes, labels):
            x_osc, y_osc = [], []
            x_fp,  y_fp  = [], []

            for param, pk_arr, fval in zip(
                result.param_values,
                result.peaks[lab],
                result.fixed[lab],
            ):
                if pk_arr.size > 0:
                    stride  = max(1, pk_arr.size // max_points)
                    sampled = pk_arr[::stride]
                    x_osc.extend([param] * len(sampled))
                    y_osc.extend(sampled)
                elif not np.isnan(fval):
                    x_fp.append(param)
                    y_fp.append(fval)

            if x_osc:
                ax.scatter(x_osc, y_osc, s=markersize, color="black", linewidths=0)
            if x_fp:
                ax.scatter(x_fp, y_fp, s=markersize * 3, color="tomato", linewidths=0)

            ax.set_ylabel(f"Maxima of ${lab}(t)$", fontsize=12)
            ax.grid(True, linewidth=0.2)

        axes[-1].set_xlabel(result.param_name, fontsize=13)
        fig.suptitle(title or "Bifurcation Diagram", fontsize=14)

        if output:
            fig.savefig(output, dpi=200, bbox_inches="tight")
            print(f"Saved → {output}")

        if show:
            plt.show()

        if not show:
            plt.close(fig)

# ════════════════════════════════════════
# 13. Escape Map
# ════════════════════════════════════════

def _escape_map_worker(args: tuple) -> tuple:
    """
    Fixed-N worker: run one (params, epsilon, N) point.
    Returns (escaped: bool, escape_time in time units — nan if bounded).
    """
    system_cls, params, N, epsilon, dt, steps, window, threshold, seed = args
    try:
        runner = SimulationRunner(
            system_cls=system_cls,
            system_params=params,
            N=N, epsilon=epsilon,
            dt=dt, steps=steps, window=window,
            threshold=threshold, seed=seed,
            verbose=False,
        )
        result   = runner.run()
        escaped  = result.escape.escaped
        esc_time = float(result.escape.step * dt) if escaped else np.nan
    except Exception:
        escaped, esc_time = False, np.nan
    return escaped, esc_time


def _escape_min_N_worker(args: tuple) -> float:
    """
    Min-N worker: binary-search the smallest N for which the system stays
    bounded.  Returns N_c as float, or np.nan if even N_max escapes.

    A point is considered "escaping at N" if ANY of the n_trials runs
    with that N escapes (same convention as the discrete case).
    """
    (system_cls, params, epsilon,
     dt, steps, window, threshold,
     N_min, N_max, n_trials, seed) = args

    def _escapes(N: int) -> bool:
        for trial in range(n_trials):
            try:
                runner = SimulationRunner(
                    system_cls=system_cls,
                    system_params=params,
                    N=N, epsilon=epsilon,
                    dt=dt, steps=steps, window=window,
                    threshold=threshold, seed=seed + trial,
                    verbose=False,
                )
                if runner.run().escape.escaped:
                    return True
            except Exception:
                return True
        return False

    if not _escapes(N_min):
        return float(N_min)
    if _escapes(N_max):
        return np.nan          # always escapes

    lo, hi = N_min, N_max
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _escapes(mid):
            lo = mid
        else:
            hi = mid
    return float(hi)


@dataclass
class EscapeMapResult:
    """Result from a fixed-N escape sweep."""
    param1_values: np.ndarray   # x-axis (n_p1,)
    param2_values: np.ndarray   # y-axis (n_p2,)
    escaped:       np.ndarray   # bool   (n_p2, n_p1)
    escape_time:   np.ndarray   # float  (n_p2, n_p1), nan where bounded
    param1_name:   str
    param2_name:   str


@dataclass
class EscapeMinNResult:
    """Result from a min-N escape sweep."""
    param1_values: np.ndarray   # x-axis (n_p1,)
    param2_values: np.ndarray   # y-axis (n_p2,)
    min_N:         np.ndarray   # float  (n_p2, n_p1), nan where always-escaping
    param1_name:   str
    param2_name:   str
    N_max:         int

    def trim(self, param1_range=None, param2_range=None) -> "EscapeMinNResult":
        """
        Return a new EscapeMinNResult restricted to the given ranges.

        Parameters
        ----------
        param1_range : (lo, hi) or None
        param2_range : (lo, hi) or None
        """
        p1, p2, Z = self.param1_values, self.param2_values, self.min_N
        if param1_range is not None:
            mask = (p1 >= param1_range[0]) & (p1 <= param1_range[1])
            p1, Z = p1[mask], Z[:, mask]
        if param2_range is not None:
            mask = (p2 >= param2_range[0]) & (p2 <= param2_range[1])
            p2, Z = p2[mask], Z[mask, :]
        return EscapeMinNResult(
            param1_values=p1, param2_values=p2, min_N=Z,
            param1_name=self.param1_name, param2_name=self.param2_name,
            N_max=self.N_max,
        )


class EscapeMapAnalyzer:
    """
    Sweep two parameters on a 2-D grid and either:
      - record escape/no-escape for a fixed N  (compute)
      - find the minimum N that prevents escape (compute_min_N)

    One swept parameter can be "epsilon"; it routes to SimulationRunner
    directly. All others go to system_params.

    Parameters
    ----------
    system_cls  : DynamicalSystem subclass
    base_params : fixed system parameters (do not include swept params)
    N           : number of oscillators used in compute() (ignored in compute_min_N)
    epsilon     : coupling strength (fixed unless swept)
    dt          : integration time step
    steps       : total integration steps per trial
    window      : rolling tail window (steps)
    threshold   : escape detection threshold
    n_jobs      : parallel workers; None → os.cpu_count()
    """

    def __init__(
        self,
        system_cls:  Type[DynamicalSystem],
        base_params: Optional[Dict[str, Any]] = None,
        N:           int   = 50,
        epsilon:     float = 0.0,
        dt:          float = 0.005,
        steps:       int   = 100_000,
        window:      int   = 5_000,
        threshold:   float = 100.0,
        n_jobs:      Optional[int] = None,
    ):
        self.system_cls  = system_cls
        self.base_params = base_params or {}
        self.N           = N
        self.epsilon     = epsilon
        self.dt          = dt
        self.steps       = steps
        self.window      = window
        self.threshold   = threshold
        self.n_jobs      = os.cpu_count() or 1 if (n_jobs is None or n_jobs < 1) else n_jobs

    def _build_params(self, p1, p1_name, p2, p2_name):
        """Resolve (p1, p2) into (system_params dict, epsilon float)."""
        params  = dict(self.base_params)
        epsilon = self.epsilon
        if p1_name == "epsilon":
            epsilon = float(p1)
        else:
            params[p1_name] = float(p1)
        if p2_name == "epsilon":
            epsilon = float(p2)
        else:
            params[p2_name] = float(p2)
        return params, epsilon

    @staticmethod
    def _mpi():
        """Return (comm, rank, size). Falls back to (None, 0, 1) without mpi4py."""
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            return comm, comm.Get_rank(), comm.Get_size()
        except ImportError:
            return None, 0, 1

    def _run_tasks(self, tasks, worker, verbose, label):
        """Distribute tasks via MPI (if available) or mp.Pool."""
        comm, rank, size = self._mpi()
        t0 = time.time()

        if size > 1:    # ── MPI ──────────────────────────────────────────────
            if rank == 0 and verbose:
                print(f"{label}: {len(tasks)} points on {size} MPI ranks …")
            my_indices = range(rank, len(tasks), size)
            my_results = [(k, worker(tasks[k])) for k in my_indices]
            gathered   = comm.gather(my_results, root=0)
            if rank != 0:
                return None
            flat = sorted([r for sub in gathered for r in sub], key=lambda x: x[0])
            raw  = [r for _, r in flat]
        else:           # ── mp.Pool ──────────────────────────────────────────
            if verbose:
                print(f"{label}: {len(tasks)} points on {self.n_jobs} workers …")
            with mp.Pool(processes=self.n_jobs) as pool:
                raw = pool.map(worker, tasks)

        if verbose and (size == 1 or (size > 1 and comm.Get_rank() == 0)):
            print(f"Done in {time.time() - t0:.1f} s")
        return raw

    def compute(
        self,
        param1_name:   str,
        param1_values: np.ndarray,
        param2_name:   str = "epsilon",
        param2_values: Optional[np.ndarray] = None,
        seed:          int = 42,
        verbose:       bool = True,
    ):
        """
        Fixed-N sweep: detect whether the system escapes at each (p1, p2) point.
        Coloured by escape time; grey where bounded.
        Returns None on non-root MPI ranks.
        """
        param1_values = np.asarray(param1_values)
        param2_values = np.asarray(
            param2_values if param2_values is not None
            else np.linspace(0.0, 0.5, 40)
        )

        tasks = []
        for p2 in param2_values:
            for p1 in param1_values:
                params, epsilon = self._build_params(p1, param1_name, p2, param2_name)
                tasks.append((self.system_cls, params, self.N, epsilon,
                               self.dt, self.steps, self.window, self.threshold, seed))

        raw = self._run_tasks(tasks, _escape_map_worker, verbose,
                              f"Escape map (fixed N={self.N})")
        if raw is None:
            return None

        n1, n2 = len(param1_values), len(param2_values)
        return EscapeMapResult(
            param1_values=param1_values,
            param2_values=param2_values,
            escaped    =np.array([r[0] for r in raw], dtype=bool).reshape(n2, n1),
            escape_time=np.array([r[1] for r in raw], dtype=float).reshape(n2, n1),
            param1_name=param1_name,
            param2_name=param2_name,
        )

    def compute_min_N(
        self,
        param1_name:   str,
        param1_values: np.ndarray,
        param2_name:   str = "epsilon",
        param2_values: Optional[np.ndarray] = None,
        N_min:         int = 1,
        N_max:         int = 10_000,
        n_trials:      int = 3,
        seed:          int = 42,
        verbose:       bool = True,
    ):
        """
        Min-N sweep: for each (p1, p2) find the smallest N that keeps the
        system bounded.  Coloured by log₁₀(N_c); grey where always-escaping.
        Returns None on non-root MPI ranks.
        """
        param1_values = np.asarray(param1_values)
        param2_values = np.asarray(
            param2_values if param2_values is not None
            else np.linspace(0.0, 0.5, 40)
        )

        tasks = []
        for p2 in param2_values:
            for p1 in param1_values:
                params, epsilon = self._build_params(p1, param1_name, p2, param2_name)
                tasks.append((
                    self.system_cls, params, epsilon,
                    self.dt, self.steps, self.window, self.threshold,
                    N_min, N_max, n_trials, seed,
                ))

        raw = self._run_tasks(tasks, _escape_min_N_worker, verbose,
                              f"Escape map (min-N, N_max={N_max})")
        if raw is None:
            return None

        n1, n2 = len(param1_values), len(param2_values)
        return EscapeMinNResult(
            param1_values=param1_values,
            param2_values=param2_values,
            min_N      =np.array(raw, dtype=float).reshape(n2, n1),
            param1_name=param1_name,
            param2_name=param2_name,
            N_max      =N_max,
        )

    def compute_min_N_adaptive(
        self,
        param1_name:   str,
        param1_range:  tuple,
        param2_name:   str   = "epsilon",
        param2_range:  tuple = (0.0, 1.0),
        coarse_res:    int   = 20,
        fine_res:      int   = 100,
        tol:           float = 0.5,
        margin_frac:   float = 0.05,
        N_min:         int   = 1,
        N_max:         int   = 10_000,
        n_trials:      int   = 3,
        seed:          int   = 42,
        verbose:       bool  = True,
    ) -> "Optional[EscapeMinNResult]":
        """
        Two-pass adaptive min-N sweep.

        Pass 1 runs a coarse grid over the full (param1, param2) domain.
        Pass 2 concentrates a fine grid in the region where N_c transitions:
        cells adjacent to a nan↔finite boundary or where neighbouring log₁₀(N_c)
        values differ by more than *tol*.

        Both passes call compute_min_N internally, so MPI parallelism is fully
        preserved.  The two results are merged with nearest-neighbour
        interpolation onto a uniform output grid at *fine_res* resolution.

        Returns None on non-root MPI ranks.
        """
        from scipy.interpolate import griddata

        comm, rank, size = self._mpi()

        p1_min, p1_max = param1_range
        p2_min, p2_max = param2_range

        # ── Pass 1: coarse grid over full domain ──────────────────────────
        p1_coarse = np.linspace(p1_min, p1_max, coarse_res)
        p2_coarse = np.linspace(p2_min, p2_max, coarse_res)

        coarse_result = self.compute_min_N(
            param1_name, p1_coarse,
            param2_name, p2_coarse,
            N_min=N_min, N_max=N_max,
            n_trials=n_trials, seed=seed, verbose=verbose,
        )

        # ── Detect active region on rank 0, then broadcast ────────────────
        if rank == 0:
            Z     = coarse_result.min_N   # shape (coarse_res, coarse_res)
            log_Z = np.where(np.isfinite(Z), np.log10(np.maximum(Z, 1.0)), np.nan)
            active = np.zeros_like(Z, dtype=bool)

            # Horizontal neighbors (along param1)
            a, b = log_Z[:, :-1], log_Z[:, 1:]
            flag = ((np.isnan(a) != np.isnan(b)) |
                    (np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) > tol)))
            active[:, :-1] |= flag
            active[:, 1:]  |= flag

            # Vertical neighbors (along param2)
            a, b = log_Z[:-1, :], log_Z[1:, :]
            flag = ((np.isnan(a) != np.isnan(b)) |
                    (np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) > tol)))
            active[:-1, :] |= flag
            active[1:,  :] |= flag

            rows, cols = np.where(active)
            if len(rows) == 0:
                region = (p1_min, p1_max, p2_min, p2_max, False)
            else:
                c_lo = max(0,            cols.min() - 1)
                c_hi = min(coarse_res-1, cols.max() + 1)
                r_lo = max(0,            rows.min() - 1)
                r_hi = min(coarse_res-1, rows.max() + 1)
                p1_lo, p1_hi = p1_coarse[c_lo], p1_coarse[c_hi]
                p2_lo, p2_hi = p2_coarse[r_lo], p2_coarse[r_hi]

                p1_span = p1_hi - p1_lo or (p1_max - p1_min)
                p2_span = p2_hi - p2_lo or (p2_max - p2_min)
                p1_lo = max(p1_min, p1_lo - margin_frac * p1_span)
                p1_hi = min(p1_max, p1_hi + margin_frac * p1_span)
                p2_lo = max(p2_min, p2_lo - margin_frac * p2_span)
                p2_hi = min(p2_max, p2_hi + margin_frac * p2_span)

                region = (p1_lo, p1_hi, p2_lo, p2_hi, True)
        else:
            region = None

        if size > 1:
            region = comm.bcast(region, root=0)

        p1_lo, p1_hi, p2_lo, p2_hi, has_active = region

        if not has_active:
            if verbose and rank == 0:
                print("No active region — returning coarse result.")
            return coarse_result   # None on non-root ranks

        if verbose and rank == 0:
            print(f"Active region: {param1_name}=[{p1_lo:.4f}, {p1_hi:.4f}], "
                  f"{param2_name}=[{p2_lo:.4f}, {p2_hi:.4f}]")

        # ── Pass 2: fine grid over active region ──────────────────────────
        p1_fine = np.linspace(p1_lo, p1_hi, fine_res)
        p2_fine = np.linspace(p2_lo, p2_hi, fine_res)

        fine_result = self.compute_min_N(
            param1_name, p1_fine,
            param2_name, p2_fine,
            N_min=N_min, N_max=N_max,
            n_trials=n_trials, seed=seed, verbose=verbose,
        )

        if rank != 0:
            return None

        # ── Merge onto full-domain output grid ────────────────────────────
        p1_out = np.linspace(p1_min, p1_max, fine_res)
        p2_out = np.linspace(p2_min, p2_max, fine_res)
        P1_out, P2_out = np.meshgrid(p1_out, p2_out)

        P1_c, P2_c = np.meshgrid(p1_coarse, p2_coarse)
        P1_f, P2_f = np.meshgrid(p1_fine,   p2_fine)

        # Encode nan as -1 (all valid N_c ≥ 1) so griddata can handle it
        def _enc(arr):
            return np.where(np.isfinite(arr), arr, -1.0)

        pts = np.column_stack([
            np.concatenate([P1_c.ravel(), P1_f.ravel()]),
            np.concatenate([P2_c.ravel(), P2_f.ravel()]),
        ])
        vals = np.concatenate([
            _enc(coarse_result.min_N.ravel()),
            _enc(fine_result.min_N.ravel()),
        ])

        merged = griddata(pts, vals, (P1_out, P2_out), method='nearest')
        merged = np.where(merged < 0, np.nan, merged)

        return EscapeMinNResult(
            param1_values=p1_out,
            param2_values=p2_out,
            min_N        =merged,
            param1_name  =param1_name,
            param2_name  =param2_name,
            N_max        =N_max,
        )


class EscapeMapPlotter:

    @staticmethod
    def _finish(fig, output, show):
        if output:
            fig.savefig(output, dpi=200, bbox_inches="tight")
            print(f"Saved → {output}")
        if show:
            plt.show()
        if not show:
            plt.close(fig)

    @staticmethod
    def plot(
        result: EscapeMapResult,
        output: Optional[str] = None,
        show:   bool = False,
        cmap:   str  = "viridis",
        title:  Optional[str] = None,
    ) -> None:
        """
        Fixed-N escape map.
        Bounded → grey.  Escaped → coloured by escape time.
        """
        if not output and not show:
            return
        from matplotlib.patches import Patch

        esc_time = result.escape_time.copy()
        cm_ = plt.cm.get_cmap(cmap).copy()
        cm_.set_bad(color="#CCCCCC")

        # ε on x-axis, a (param1) on y-axis → transpose
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        im = ax.pcolormesh(
            result.param2_values, result.param1_values, esc_time.T,
            cmap=cm_, vmin=np.nanmin(esc_time), vmax=np.nanmax(esc_time),
            shading="nearest",
        )
        if result.escaped.any() and not result.escaped.all():
            ax.contour(result.param2_values, result.param1_values,
                       result.escaped.T.astype(float),
                       levels=[0.5], colors="black", linewidths=0.8)

        fig.colorbar(im, ax=ax, pad=0.02, label="Escape time")
        ax.legend(handles=[Patch(facecolor="#CCCCCC", edgecolor="black",
                                 linewidth=0.5, label="Bounded")],
                  loc="best", fontsize=8)
        ax.set_xlabel(result.param2_name, fontsize=12)
        ax.set_ylabel(result.param1_name, fontsize=12)
        ax.set_title(title or "Escape Map", fontsize=13)
        EscapeMapPlotter._finish(fig, output, show)

    @staticmethod
    def plot_min_N(
        result: EscapeMinNResult,
        output: Optional[str] = None,
        show:   bool = False,
        cmap:   str  = "plasma",
        title:  Optional[str] = None,
    ) -> None:
        """
        Min-N escape map.
        Colour = log₁₀(N_c).  Grey = always escapes (N_c > N_max).
        Black contour marks the escape boundary.
        """
        if not output and not show:
            return
        from matplotlib.patches import Patch

        log_N = np.where(np.isnan(result.min_N), np.nan, np.log10(result.min_N))
        cm_ = plt.cm.get_cmap(cmap).copy()
        cm_.set_bad(color="#CCCCCC")

        vmin, vmax = np.nanmin(log_N), np.nanmax(log_N)

        # ε on x-axis, a (param1) on y-axis → transpose
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        im = ax.pcolormesh(
            result.param2_values, result.param1_values, log_N.T,
            cmap=cm_, vmin=vmin, vmax=vmax, shading="nearest",
        )

        escape_mask = np.isnan(result.min_N)
        if escape_mask.any() and not escape_mask.all():
            ax.contour(result.param2_values, result.param1_values,
                       escape_mask.T.astype(float),
                       levels=[0.5], colors="black", linewidths=0.8)

        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(r"$\log_{10}(N_c)$", fontsize=10)
        tick_vals = np.linspace(vmin, vmax, 5)
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels([rf"$10^{{{v:.1f}}}$" for v in tick_vals])

        ax.legend(
            handles=[Patch(facecolor="#CCCCCC", edgecolor="black", linewidth=0.5,
                           label=rf"Always escapes ($N>{result.N_max}$)")],
            loc="best", fontsize=8,
        )
        ax.set_xlabel(result.param2_name, fontsize=12)
        ax.set_ylabel(result.param1_name, fontsize=12)
        ax.set_title(title or "Escape Map — min $N$", fontsize=13)
        EscapeMapPlotter._finish(fig, output, show)


# ════════════════════════════════════════
# 14. Histogram
# ════════════════════════════════════════

class HistogramPlotter:
    """
    Plot histograms of state-component distributions from a SimulationResult.

    Each subplot shows the distribution of one component (x, y, z, …)
    pooled across all oscillators and all time steps in the tail window.
    Optionally overlays a Gaussian fit.
    """

    @staticmethod
    def _gauss_overlay(ax, data, color="black", ls="--", lw=1.2):
        data = data[np.isfinite(data)]
        if data.size == 0:
            return
        mu, sigma = data.mean(), data.std()
        x = np.linspace(data.min(), data.max(), 300)
        ax.plot(x, np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                / (sigma * np.sqrt(2 * np.pi)),
                color=color, linewidth=lw, ls=ls,
                label=rf"$\mathcal{{N}}({mu:.2f},\,{sigma:.2f}^2)$")

    @staticmethod
    def plot(
        result:          "SimulationResult",
        output:          Optional[str] = None,
        show:            bool = False,
        bins:            int  = 60,
        gaussian_fit:    Optional[str] = None,
        plot_elements:   bool = True,
        plot_mean_field: bool = False,
        title:           Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        result          : SimulationResult from SimulationRunner.run()
        bins            : number of histogram bins
        gaussian_fit    : fit Gaussian to 'elements', 'mean', 'both', or None
        plot_elements   : plot individual-oscillator distribution (pooled over N and time)
        plot_mean_field : overlay mean-field h_t = mean_i(x_i) distribution
        output          : save path; skipped if None
        show            : display interactively
        """
        if not output and not show:
            return
        tail   = result.tail          # (window, N, dim)
        labels = result.system.labels
        dim    = result.system.dim
        colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        fig, axes = plt.subplots(1, dim, figsize=(5 * dim, 4),
                                 constrained_layout=True)
        if dim == 1:
            axes = [axes]

        fit_elements = gaussian_fit in ("elements", "both")
        fit_mean     = gaussian_fit in ("mean", "both")

        for idx, (ax, lab) in enumerate(zip(axes, labels)):
            data = tail[:, :, idx].ravel()        # (window*N,) — all elements
            mf   = tail[:, :, idx].mean(axis=1)  # (window,)   — mean field

            if plot_elements:
                ax.hist(data, bins=bins, density=True,
                        color=colors[idx % len(colors)], alpha=0.7,
                        edgecolor="white", linewidth=0.3,
                        label="Elements")
            if fit_elements and plot_elements:
                HistogramPlotter._gauss_overlay(ax, data)

            if plot_mean_field:
                ax.hist(mf, bins=bins, density=True,
                        color="black", alpha=0.9,
                        histtype="step", linewidth=1.0,
                        label=r"$h_t$ (mean field)")
            if fit_mean and plot_mean_field:
                HistogramPlotter._gauss_overlay(ax, mf, color="gray", ls=":")

            if plot_elements or plot_mean_field or gaussian_fit:
                ax.legend(fontsize=8)

            ax.set_xlabel(rf"${lab}$", fontsize=12)
            ax.set_ylabel("Density", fontsize=11)
            ax.set_title(rf"Distribution of ${lab}$", fontsize=11)

        sys_info = repr(result.system)
        fig.suptitle(title or f"State Distributions — {sys_info}", fontsize=12)

        if output:
            fig.savefig(output, dpi=200, bbox_inches="tight")
            print(f"Saved → {output}")

        if show:
            plt.show()

        if not show:
            plt.close(fig)


if __name__ == "__main__":
    main()