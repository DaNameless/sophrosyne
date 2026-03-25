"""
╔══════════════════════════════════════╗
║             SOPHROSYNE               ║
║              v1.0.0                  ║
║   Coupled Dynamical Systems Solver   ║
║            by R.S.S.G.               ║
╚══════════════════════════════════════╝ - Discrete Map Version

Numerical Analysis of Mean-Field Convergence in Globally Coupled Discrete Maps
===============================================================================

System:
  x_{t+1}^{(i)} = (1-eps)*f(state_t^{(i)}) + eps*h_t
  h_t = (1/N) * sum_j f(state_t^{(j)})

  Supports 1D and multi-dimensional maps (e.g. Lozi).

Usage — 1D map (tent, logistic):
  sys = CoupledMapLattice(lambda x: tent(x, a=1.99), name="tent_a1.99")
  sys.run_all(eps=0.1)

Usage — 2D map (Lozi):
  def lozi_obs(state):           # f(x,y) = 1 - a|x| + y
      x, y = state
      return 1 - 1.7 * np.abs(x) + y

  def lozi_step(state, fxy, eps, h):
      x, y = state
      return (1 - eps) * fxy + eps * h,  0.5 * x

  def lozi_init(N, rng):
      x = rng.uniform(-1, 1, N)
      return x, 0.5 * x + rng.uniform(-0.01, 0.01, N)

  sys = CoupledMapLattice(lozi_obs, step_fn=lozi_step, init_fn=lozi_init, name="lozi")
  sys.run_all(eps=0.1)

Sweep a map parameter (e.g. 'a' in tent or Lozi):
  factory = lambda a: CoupledMapLattice(lambda x: tent(x, a=a), name=f"tent_a{a:.2f}")
  sys.analysis_meanfield_bifurcation(sweep='map_param', map_factory=factory,
                                     param_range=(0.5, 2.0), eps_fixed=0.1)

Analyses:
  1. Time series of h_t for different N
  2. Fluctuation scaling: sigma_h vs N  (expect slope -1/2)
  3. Distribution of h_t, rescaled collapse, Q-Q plot
  4. Self-consistency check
  5. Phase diagram: scaling exponent alpha(eps)
  6. Mean-field bifurcation diagram
  7. Minimum N to prevent escape

Author : R.S.S.G.
Created: 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

try:
    from mpi4py import MPI
    _MPI_AVAILABLE = True
except ImportError:
    _MPI_AVAILABLE = False


# ── Built-in maps ──────────────────────────────────────────────────────────
def tent(x, a=1.99):
    """Tent map: f(x) = (a/2)(1 - |1 - 2x|)"""
    return (a / 2.0) * (1.0 - np.abs(1.0 - 2.0 * x))


def logistic(x, r=3.9):
    """Logistic map: f(x) = r*x*(1-x)"""
    return r * x * (1.0 - x)


def lozi_f(x, y, a=1.7):
    """Lozi map observable: f(x,y) = 1 - a|x| + y"""
    return 1.0 - a * np.abs(x) + y


# ══════════════════════════════════════════════════════════════════════════════
# COUPLED MAP LATTICE ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
class CoupledMapLattice:
    """
    Globally coupled map system analyzer.

        h_t   = (1/N) * sum_j obs_fn(state_t^{(j)})
        state_{t+1}^{(i)} = step_fn(state_t^{(i)}, obs_fn(state_t^{(i)}), eps, h_t)

    Parameters
    ----------
    obs_fn : callable
        Observable f(state) -> ndarray of length N.
        For 1D maps: state is a 1D array; obs_fn is just the map f(x).
        For 2D maps (e.g. Lozi): state is a tuple of arrays; obs_fn unpacks it.

    step_fn : callable, optional
        Full state update: step_fn(state, f_val, eps, h) -> new_state.
        Defaults to the standard 1D mean-field step:
            new_state = (1 - eps) * f_val + eps * h

    init_fn : callable, optional
        Initial state generator: init_fn(N, rng) -> state.
        Defaults to rng.random(N) for 1D maps.

    name : str
        Label used in plot titles and saved filenames.
    """

    def __init__(self, obs_fn, step_fn=None, init_fn=None, name="Map", output_dir="."):
        self.obs_fn     = obs_fn
        self.step_fn    = step_fn or (lambda state, fv, eps, h: (1.0 - eps) * fv + eps * h)
        self.init_fn    = init_fn or (lambda N, rng: rng.random(N))
        self.name       = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _savepath(self, fname):
        """Return full path for a figure filename inside output_dir."""
        return self.output_dir / fname

    # ── Built-in map constructors ──────────────────────────────────────────
    @classmethod
    def from_tent(cls, a=1.99, output_dir="."):
        return cls(
            obs_fn     = lambda x: tent(x, a=a),
            name       = f"tent_a{a}",
            output_dir = output_dir,
        )

    @classmethod
    def from_logistic(cls, r=3.9, output_dir="."):
        return cls(
            obs_fn     = lambda x: logistic(x, r=r),
            name       = f"logistic_r{r}",
            output_dir = output_dir,
        )

    @classmethod
    def from_lozi(cls, a=1.7, b=0.5, output_dir="."):
        def obs(state):
            x, y = state
            return lozi_f(x, y, a=a)

        def step(state, fxy, eps, h):
            x, _ = state
            return (1.0 - eps) * fxy + eps * h,  b * x

        def init(N, rng):
            x = rng.uniform(-1, 1, N)
            return x, b * x + rng.uniform(-0.01, 0.01, N)

        return cls(obs_fn=obs, step_fn=step, init_fn=init,
                   name=f"lozi_a{a}_b{b}", output_dir=output_dir)

    # ── MPI helper ────────────────────────────────────────────────────────
    @staticmethod
    def _mpi():
        """Return (comm, rank, size). Falls back to (None, 0, 1) without mpi4py."""
        if _MPI_AVAILABLE:
            comm = MPI.COMM_WORLD
            return comm, comm.Get_rank(), comm.Get_size()
        return None, 0, 1

    # ── Core simulation ────────────────────────────────────────────────────
    def simulate(self, N, eps, T_total=5000, T_transient=2000, seed=None,
                 obs_fn=None, step_fn=None, init_fn=None):
        """
        Run the coupled system and return h_t time series after transient.

        The obs_fn/step_fn/init_fn overrides let sweep methods inject a
        different map without mutating self (used internally).
        """
        f    = obs_fn  or self.obs_fn
        step = step_fn or self.step_fn
        init = init_fn or self.init_fn

        rng   = np.random.default_rng(seed)
        state = init(N, rng)
        h_series = np.empty(T_total - T_transient)

        for t in range(T_total):
            f_val = f(state)
            h     = np.mean(f_val)
            if t >= T_transient:
                h_series[t - T_transient] = h
            state = step(state, f_val, eps, h)

        return h_series

    # ── Trajectory plot ────────────────────────────────────────────────────
    def plot_trajectories(self, N, eps, n_show=5,
                          T_total=3000, T_transient=2000,
                          seed=None, save=True):
        """
        Evolve the system and plot the last (T_total - T_transient) steps.

        Parameters
        ----------
        N       : total number of particles in the simulation
        eps     : coupling strength
        n_show  : how many individual particle trajectories to overlay
        save    : whether to save the figure to output_dir
        """
        f    = self.obs_fn
        step = self.step_fn
        init = self.init_fn

        rng   = np.random.default_rng(seed)
        state = init(N, rng)

        T_rec = T_total - T_transient
        # record the x-component of each tracked particle and h_t
        x_tracks = np.empty((T_rec, n_show))
        h_series = np.empty(T_rec)

        # pick n_show particle indices to track
        rng2    = np.random.default_rng((seed or 0) + 1)
        indices = rng2.choice(N, size=n_show, replace=False)

        for t in range(T_total):
            f_val = f(state)
            h     = np.mean(f_val)
            state = step(state, f_val, eps, h)

            if t >= T_transient:
                i = t - T_transient
                h_series[i] = h
                # extract x-component (works for 1D array or tuple)
                x = state[0] if isinstance(state, tuple) else state
                x_tracks[i] = x[indices]

        time = np.arange(T_rec)

        fig, ax = plt.subplots(figsize=(14, 5))
        for j in range(n_show):
            ax.plot(time, x_tracks[:, j], lw=0.7, alpha=0.6,
                    label=f'particle {indices[j]}')
        ax.plot(time, h_series, 'k-', lw=1.5, label='$h_t$ (mean field)')

        ax.set_xlabel('Time step', fontsize=12)
        ax.set_ylabel('$x$', fontsize=12)
        ax.set_title(f'Trajectories  ({self.name},  N={N},  eps={eps})', fontsize=13)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            fname = f'{self.name}_eps{eps}_N{N}_trajectories.png'
            plt.savefig(self._savepath(fname), dpi=150, bbox_inches='tight')
            print(f'  -> Saved {fname}')

        plt.show()
        plt.close()

    def _system_escapes(self, N, eps, T_total=4000, x_bound=100,
                        n_trials=3, seed=42,
                        obs_fn=None, step_fn=None, init_fn=None):
        f    = obs_fn  or self.obs_fn
        step = step_fn or self.step_fn
        init = init_fn or self.init_fn

        for trial in range(n_trials):
            rng   = np.random.default_rng(seed + trial)
            state = init(N, rng)
            for _ in range(T_total):
                f_val = f(state)
                h     = np.mean(f_val)
                state = step(state, f_val, eps, h)
                # Check first component for escapes (works for 1D and tuples)
                x = state[0] if isinstance(state, tuple) else state
                if not np.all(np.isfinite(x)) or np.any(np.abs(x) > x_bound):
                    return True
        return False

    def _resolve_factory(self, p, map_factory):
        """
        Call map_factory(p) and return (obs_fn, step_fn, init_fn).
        Accepts factories that return either a CoupledMapLattice instance
        or a plain callable (backwards-compatible 1D shorthand).
        """
        result = map_factory(p)
        if isinstance(result, CoupledMapLattice):
            return result.obs_fn, result.step_fn, result.init_fn
        # Plain callable → treat as 1D obs_fn with default step/init
        return result, self.step_fn, self.init_fn

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS 1: Time series visualization
    # ══════════════════════════════════════════════════════════════════════
    def analysis_timeseries(self, eps, N_values=(100, 1000, 10000, 100000)):
        print("=" * 60)
        print(f"ANALYSIS 1: Time series of h_t  ({self.name}, eps={eps})")
        print("=" * 60)

        fig, axes = plt.subplots(len(N_values), 1, figsize=(14, 10), sharex=True)
        T_show = 200

        for ax, N in zip(axes, N_values):
            h = self.simulate(N, eps, T_total=T_show + 10000, T_transient=10000, seed=42)
            h = h[:T_show]
            ax.plot(h, 'b-', linewidth=0.5, alpha=0.8)
            ax.axhline(y=np.mean(h), color='r', linestyle='--', alpha=0.5)
            ax.set_ylabel('$h_t$', fontsize=11)
            ax.set_title(f'N = {N}   (sigma = {np.std(h):.5f})', fontsize=12)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time step', fontsize=12)
        plt.suptitle(f'Mean Field Time Series ({self.name}, eps={eps})', fontsize=14, y=1.01)
        plt.tight_layout()
        fname = f'./{self.name}_eps{eps}_timeseries.png'
        plt.savefig(self._savepath(fname), dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"  -> Saved {fname}\n")

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS 2: Fluctuation scaling  sigma_h  vs  N
    # ══════════════════════════════════════════════════════════════════════
    def analysis_fluctuation_scaling(self, eps, N_values=(100, 1000, 10000, 100000)):
        print("=" * 60)
        print(f"ANALYSIS 2: Fluctuation scaling  ({self.name}, eps={eps})")
        print("=" * 60)

        sigmas, means = [], []
        n_trials = 5

        for N in N_values:
            trial_sigmas, trial_means = [], []
            for trial in range(n_trials):
                h = self.simulate(N, eps, T_total=8000, T_transient=3000, seed=42 + trial)
                trial_sigmas.append(np.std(h))
                trial_means.append(np.mean(h))
            sigmas.append(np.mean(trial_sigmas))
            means.append(np.mean(trial_means))
            print(f"  N={N:>6d}:  <h> = {means[-1]:.6f},  sigma_h = {sigmas[-1]:.6f}")

        sigmas = np.array(sigmas)
        N_arr  = np.array(N_values, dtype=float)

        slope, intercept, r, p, se = stats.linregress(np.log(N_arr), np.log(sigmas))
        print(f"\n  Log-log fit:  sigma_h ~ N^{slope:.4f}  (R^2 = {r**2:.6f})")
        print(f"  Expected: -0.5000,  Deviation: {abs(slope + 0.5):.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.loglog(N_arr, sigmas, 'ko-', markersize=7, label='Measured sigma_h')
        N_fit = np.linspace(N_arr[0], N_arr[-1], 100)
        ax.loglog(N_fit, np.exp(intercept) * N_fit**slope, 'r--', lw=2,
                  label=f'Fit: N^{{{slope:.3f}}}')
        ax.loglog(N_fit, sigmas[0] * (N_fit / N_arr[0])**(-0.5),
                  'b:', lw=1.5, alpha=0.7, label='N^{-1/2} reference')
        ax.set_xlabel('N', fontsize=12); ax.set_ylabel('sigma_h', fontsize=12)
        ax.set_title(f'Fluctuation Scaling ({self.name}, eps={eps})', fontsize=13)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.semilogx(N_arr, means, 'ko-', markersize=7)
        h_inf = means[-1]
        ax.axhline(y=h_inf, color='r', linestyle='--', label=f'h_bar ~ {h_inf:.4f}')
        ax.set_xlabel('N', fontsize=12); ax.set_ylabel('<h_t>', fontsize=12)
        ax.set_title('Mean Field Convergence', fontsize=13)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f'./{self.name}_eps{eps}_scaling.png'
        plt.savefig(self._savepath(fname), dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"  -> Saved {fname}\n")

        return slope

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS 3: Distribution of h_t and rescaled collapse
    # ══════════════════════════════════════════════════════════════════════
    def analysis_distribution(self, eps, N_values=(100, 1000, 10000, 100000)):
        print("=" * 60)
        print(f"ANALYSIS 3: Distribution & rescaled collapse  ({self.name}, eps={eps})")
        print("=" * 60)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        all_rescaled = {}

        for i, N in enumerate(N_values):
            h = self.simulate(N, eps, T_total=12000, T_transient=2000, seed=42)
            h = h[np.isfinite(h)]
            if len(h) == 0:
                print(f"  N={N:>6d}: simulation diverged, skipping")
                continue
            h_mean, h_std = np.mean(h), np.std(h)
            rescaled = (h - h_mean) * np.sqrt(N)
            all_rescaled[N] = rescaled

            _, shapiro_p = stats.shapiro(h[:5000])
            print(f"  N={N:>6d}: <h>={h_mean:.6f}, sigma={h_std:.6f}, Shapiro p={shapiro_p:.4f}")

            axes[0].hist(h, bins=60, density=True, alpha=0.5, color=colors[i], label=f'N={N}')
            axes[1].hist(rescaled, bins=60, density=True, alpha=0.5, color=colors[i], label=f'N={N}')

        last_key = next((N for N in reversed(N_values) if N in all_rescaled), None)
        if last_key is None:
            print("  All simulations diverged. Skipping plots.")
            plt.close(); return
        ref_std = np.std(all_rescaled[last_key])
        xg = np.linspace(-4 * ref_std, 4 * ref_std, 200)
        axes[1].plot(xg, stats.norm.pdf(xg, 0, ref_std), 'k-', lw=2, label='Gaussian')

        axes[0].set_xlabel('h_t'); axes[0].set_ylabel('Density')
        axes[0].set_title('Distribution of h_t'); axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('(h_t - h_bar)*sqrt(N)'); axes[1].set_ylabel('Density')
        axes[1].set_title('Rescaled (should collapse)'); axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        h_largest = self.simulate(last_key, eps, T_total=12000, T_transient=2000, seed=42)
        h_largest = h_largest[np.isfinite(h_largest)]
        if len(h_largest) > 0:
            z = (h_largest - np.mean(h_largest)) / np.std(h_largest)
            stats.probplot(z, dist="norm", plot=axes[2])
        axes[2].set_title(f'Q-Q Plot (N={last_key})'); axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f'./{self.name}_eps{eps}_distribution.png'
        plt.savefig(self._savepath(fname), dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"  -> Saved {fname}\n")

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS 4: Self-consistency check
    # ══════════════════════════════════════════════════════════════════════
    def analysis_self_consistency(self, eps, h_range=(0.01, 0.99), N_sc=10000,
                                  T_sc=200000, T_trans_sc=50000):
        """
        Parameters
        ----------
        h_range : (float, float)
            Range of candidate h_bar values. Set to cover the expected
            mean field of your map (e.g. (-0.5, 1.5) for Lozi).
        """
        print("=" * 60)
        print(f"ANALYSIS 4: Self-consistency  ({self.name}, eps={eps})")
        print("=" * 60)

        def single_site_average(h_bar):
            # Use N=1 with init_fn so dimensionality is handled correctly
            state = self.init_fn(1, np.random.default_rng(0))
            total = 0.0
            for t in range(T_sc):
                f_val = self.obs_fn(state)
                if t >= T_trans_sc:
                    total += float(f_val[0])
                state = self.step_fn(state, f_val, eps, h_bar)
            return total / (T_sc - T_trans_sc)

        h_candidates = np.linspace(h_range[0], h_range[1], 50)
        f_averages   = np.array([single_site_average(h) for h in h_candidates])
        residuals    = f_averages - h_candidates

        sign_changes = np.where(np.diff(np.sign(residuals)))[0]
        h_star_list  = []
        for idx in sign_changes:
            h1, h2 = h_candidates[idx], h_candidates[idx + 1]
            r1, r2 = residuals[idx], residuals[idx + 1]
            h_star_list.append(h1 - r1 * (h2 - h1) / (r2 - r1))

        h_sim      = self.simulate(N_sc, eps, T_total=10000, T_transient=3000, seed=42)
        h_sim_mean = np.mean(h_sim)

        print(f"  Self-consistent h_bar: {h_star_list}")
        print(f"  Simulation <h> (N={N_sc}): {h_sim_mean:.6f}")
        if h_star_list:
            print(f"  |diff| = {abs(h_star_list[0] - h_sim_mean):.6f}")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(h_candidates, f_averages, 'b-', lw=2, label='<f(state)> under rho*')
        ax.plot(h_candidates, h_candidates, 'r--', lw=1.5, label='y = h_bar')
        for hs in h_star_list:
            ax.plot(hs, hs, 'go', ms=12, zorder=5, label=f'Fixed point: {hs:.4f}')
        ax.axvline(h_sim_mean, color='purple', ls='--', alpha=0.5,
                   label=f'Sim mean: {h_sim_mean:.4f}')
        ax.set_xlabel('h_bar (candidate)'); ax.set_ylabel('<f>')
        ax.set_title(f'Self-Consistency ({self.name}, eps={eps})')
        ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()

        fname = f'./{self.name}_eps{eps}_self_consistency.png'
        plt.savefig(self._savepath(fname), dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"  -> Saved {fname}\n")

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS 5: Phase diagram — alpha(eps)
    # ══════════════════════════════════════════════════════════════════════
    def analysis_phase_diagram(self, eps_values=None, N_values=(100, 1000, 10000, 100000),
                               n_jobs=1):
        if eps_values is None:
            eps_values = np.arange(0.02, 0.52, 0.03)
        comm, rank, size = self._mpi()
        if rank == 0:
            print("=" * 60)
            print(f"ANALYSIS 5: Phase diagram alpha(eps)  ({self.name})")
            print("=" * 60)

        N_arr = np.array(N_values, dtype=float)

        def _compute_eps(i, eps):
            sigmas = [np.std(self.simulate(N, eps, T_total=6000, T_transient=2000, seed=42))
                      for N in N_values]
            slope, _, r, _, se = stats.linregress(np.log(N_arr),
                                                   np.log(np.array(sigmas) + 1e-15))
            return i, -slope, se

        if size > 1:   # MPI
            my_indices = list(range(rank, len(eps_values), size))
            my_results = [_compute_eps(i, eps_values[i]) for i in my_indices]
            all_results = comm.gather(my_results, root=0)
            if rank != 0:
                return
            all_results = sorted([r for sub in all_results for r in sub], key=lambda x: x[0])
        else:           # joblib
            all_results = Parallel(n_jobs=n_jobs)(
                delayed(_compute_eps)(i, eps) for i, eps in enumerate(eps_values)
            )
            all_results = sorted(all_results, key=lambda x: x[0])

        for _, alpha, se in all_results:
            print(f"  eps={eps_values[all_results.index((_, alpha, se))%len(eps_values)]:.2f}:"
                  f"  alpha = {alpha:.4f} +/- {se:.4f}")

        alphas       = np.array([r[1] for r in all_results])
        alpha_errors = np.array([r[2] for r in all_results])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.errorbar(eps_values, alphas, yerr=alpha_errors, fmt='ko-', ms=6, capsize=3)
        ax.axhline(0.5, color='r', ls='--', lw=1.5, label='CLT (alpha=1/2)')
        ax.fill_between(eps_values, 0.45, 0.55, alpha=0.1, color='red')
        ax.set_xlabel('eps'); ax.set_ylabel('alpha')
        ax.set_title(f'Scaling Exponent ({self.name})'); ax.legend()
        ax.grid(True, alpha=0.3); ax.set_ylim(-0.1, 1.1)

        ax = axes[1]
        for eps_s, col in zip([0.05, 0.15, 0.3, 0.45],
                               ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']):
            sigs = [np.std(self.simulate(N, eps_s, T_total=6000, T_transient=2000, seed=42))
                    for N in N_values]
            ax.loglog(N_arr, sigs, 'o-', color=col, ms=6, label=f'eps={eps_s}')
        ax.loglog(N_arr, 0.3 * N_arr**(-0.5), 'k--', lw=1, alpha=0.5, label='N^{-1/2}')
        ax.set_xlabel('N'); ax.set_ylabel('sigma_h')
        ax.set_title('Scaling at Different Couplings'); ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f'./{self.name}_phase_diagram.png'
        plt.savefig(self._savepath(fname), dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"  -> Saved {fname}\n")

    # ══════════════════════════════════════════════════════════════════════
    # BIFURCATION DIAGRAM (single-site map g = (1-eps)*f(state) + eps*h_bar)
    # ══════════════════════════════════════════════════════════════════════
    def bifurcation_diagram(self, bifurcation_param='eps',
                            eps_range=(0.0, 1.0), eps_fixed=0.1,
                            map_factory=None, param_range=(0.5, 2.0),
                            h_bar=0.65, n_param=500,
                            T_iter=800, T_trans=600):
        """
        Bifurcation diagram of the single-site map driven by a fixed h_bar.

        bifurcation_param : 'eps'
            Sweep coupling strength. Uses self.obs_fn / self.step_fn.
        bifurcation_param : 'map_param'
            Sweep a map parameter. Requires map_factory(p) -> CoupledMapLattice
            (or a plain 1D callable for backwards compat).
        """
        if bifurcation_param == 'eps':
            param_values = np.linspace(eps_range[0], eps_range[1], n_param)
            xlabel = 'eps'
        elif bifurcation_param == 'map_param':
            if map_factory is None:
                raise ValueError("map_factory required when bifurcation_param='map_param'")
            param_values = np.linspace(param_range[0], param_range[1], n_param)
            xlabel = 'map param'
        else:
            raise ValueError("bifurcation_param must be 'eps' or 'map_param'")

        all_param, all_x = [], []

        for p in param_values:
            if bifurcation_param == 'eps':
                f_obs, f_step, f_init, eps = self.obs_fn, self.step_fn, self.init_fn, p
            else:
                f_obs, f_step, f_init = self._resolve_factory(p, map_factory)
                eps = eps_fixed

            state   = f_init(1, np.random.default_rng(0))
            escaped = False

            for _ in range(T_trans):
                f_val = f_obs(state)
                state = f_step(state, f_val, eps, h_bar)
                x = state[0] if isinstance(state, tuple) else state
                if not np.isfinite(x[0]) or abs(x[0]) > 100:
                    escaped = True
                    break

            if escaped:
                continue

            for _ in range(T_iter - T_trans):
                f_val = f_obs(state)
                state = f_step(state, f_val, eps, h_bar)
                x = state[0] if isinstance(state, tuple) else state
                if not np.isfinite(x[0]) or abs(x[0]) > 100:
                    break
                all_param.append(p)
                all_x.append(float(x[0]))

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.scatter(all_param, all_x, s=0.01, c='black', alpha=0.3)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('x (attractor)', fontsize=12)
        if bifurcation_param == 'eps':
            title = f'(1-eps)*f(state) + eps*h_bar  ({self.name}, h_bar={h_bar})'
        else:
            title = f'(1-eps)*f(state) + eps*h_bar  ({self.name}, eps={eps_fixed}, h_bar={h_bar})'
        ax.set_title(title, fontsize=13)
        ax.set_ylim(-1,2)
        ax.grid(True, alpha=0.3)
        fname = f'./{self.name}_bifurcation_{bifurcation_param}.png'
        plt.savefig(self._savepath(fname), dpi=150, bbox_inches='tight')
        plt.show()
        plt.tight_layout()
        plt.close()
        print(f"  -> Saved {fname}\n")

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS 6: Mean-Field Bifurcation Diagram
    # ══════════════════════════════════════════════════════════════════════
    def analysis_meanfield_bifurcation(self, sweep='eps',
                                       eps_range=(0.0, 1.0), eps_fixed=0.4,
                                       map_factory=None, param_range=(0.5, 3.0),
                                       N=10000, n_param=500,
                                       T_total=6000, T_transient=4000,
                                       seed=42):
        """
        Bifurcation diagram of the mean field h_t from the full N-body simulation.

        sweep : 'eps'
            Sweep coupling strength (uses self's map functions).
        sweep : 'map_param'
            Sweep a map parameter. Requires map_factory(p) -> CoupledMapLattice
            (or a plain 1D callable).
        """
        comm, rank, size = self._mpi()
        if sweep == 'eps':
            param_values = np.linspace(eps_range[0], eps_range[1], n_param)
            xlabel, fixed_label = 'eps', self.name
        elif sweep == 'map_param':
            if map_factory is None:
                raise ValueError("map_factory required when sweep='map_param'")
            param_values = np.linspace(param_range[0], param_range[1], n_param)
            xlabel, fixed_label = 'map param', f'eps={eps_fixed}'
        else:
            raise ValueError("sweep must be 'eps' or 'map_param'")

        if rank == 0:
            print("=" * 60)
            print(f"ANALYSIS 6: Mean-field bifurcation  ({self.name}, sweep={sweep}, N={N})")
            print("=" * 60)

        n_keep = 300
        my_pairs = []   # list of (param_value, h_value)

        my_indices = list(range(rank, len(param_values), size))
        for count, i in enumerate(my_indices):
            p = param_values[i]
            if sweep == 'eps':
                f_obs, f_step, f_init, eps = self.obs_fn, self.step_fn, self.init_fn, p
            else:
                f_obs, f_step, f_init = self._resolve_factory(p, map_factory)
                eps = eps_fixed

            h = self.simulate(N, eps, T_total=T_total, T_transient=T_transient,
                              seed=seed, obs_fn=f_obs, step_fn=f_step, init_fn=f_init)
            h_tail = h[-n_keep:]
            mask   = np.isfinite(h_tail) & (np.abs(h_tail) < 20)
            my_pairs.extend((p, hv) for hv in h_tail[mask])

            if (count + 1) % 50 == 0:
                print(f"  [rank {rank}] {count+1}/{len(my_indices)} done")

        if comm is not None:
            all_pairs = comm.gather(my_pairs, root=0)
            if rank != 0:
                return
            all_pairs = [item for sub in all_pairs for item in sub]
        else:
            all_pairs = my_pairs

        all_param = np.array([p for p, _ in all_pairs])
        all_h     = np.array([h for _, h in all_pairs])
        print(f"  Total points: {len(all_h)}")

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.scatter(all_param, all_h, s=0.05, c='black', alpha=0.4, rasterized=True)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel('$h_t$ (mean field)', fontsize=13)
        ax.set_title(f'Mean-Field Bifurcation ({self.name}, sweep={sweep})', fontsize=14)
        ax.grid(True, alpha=0.3)

        if len(all_h) > 0:
            y_lo, y_hi = np.percentile(all_h, [0.1, 99.9])
            margin = 0.05 * max(y_hi - y_lo, 0.01)
            ax.set_ylim(y_lo - margin, y_hi + margin)

        plt.tight_layout()
        fname = (f'./{self.name}_meanfield_bif_{sweep}_'
                 f'{fixed_label.replace("=", "")}_N{N}.png')
        plt.savefig(self._savepath(fname), dpi=200, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"  -> Saved {fname}\n")

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS 7: Minimum N to prevent escape
    # ══════════════════════════════════════════════════════════════════════
    def find_min_N(self, eps, N_min=2, N_max=100000, T_total=4000,
                   x_bound=100, n_trials=3, seed=42,
                   obs_fn=None, step_fn=None, init_fn=None):
        kw = dict(T_total=T_total, x_bound=x_bound, n_trials=n_trials,
                  seed=seed, obs_fn=obs_fn, step_fn=step_fn, init_fn=init_fn)
        if not self._system_escapes(N_min, eps, **kw):
            return N_min
        if self._system_escapes(N_max, eps, **kw):
            return None
        lo, hi = N_min, N_max
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self._system_escapes(mid, eps, **kw):
                lo = mid
            else:
                hi = mid
        return hi

    def analysis_min_N_escape(self, sweep='eps',
                              eps_range=(0.0, 0.5), eps_fixed=0.1,
                              map_factory=None, param_range=(1.0, 3.0),
                              n_param=60, N_min=2, N_max=100000,
                              T_total=4000, x_bound=100, n_trials=3, seed=42):
        """
        sweep : 'eps' or 'map_param' (requires map_factory).
        """
        comm, rank, size = self._mpi()
        if sweep == 'eps':
            param_values = np.linspace(eps_range[0], eps_range[1], n_param)
            fixed_label, xlabel = self.name, 'eps'
        elif sweep == 'map_param':
            if map_factory is None:
                raise ValueError("map_factory required when sweep='map_param'")
            param_values = np.linspace(param_range[0], param_range[1], n_param)
            fixed_label, xlabel = f'eps={eps_fixed}', 'map param'
        else:
            raise ValueError("sweep must be 'eps' or 'map_param'")

        if rank == 0:
            print("=" * 60)
            print(f"ANALYSIS 7: Minimum N to prevent escape  ({self.name}, sweep={sweep})")
            print("=" * 60)

        my_indices = list(range(rank, len(param_values), size))
        my_results = []
        for i in my_indices:
            p = param_values[i]
            if sweep == 'eps':
                f_obs, f_step, f_init, eps = self.obs_fn, self.step_fn, self.init_fn, p
            else:
                f_obs, f_step, f_init = self._resolve_factory(p, map_factory)
                eps = eps_fixed

            Nc = self.find_min_N(eps, N_min=N_min, N_max=N_max, T_total=T_total,
                                  x_bound=x_bound, n_trials=n_trials, seed=seed,
                                  obs_fn=f_obs, step_fn=f_step, init_fn=f_init)
            my_results.append((i, p, Nc))
            Nc_str = f"{Nc}" if Nc is not None else f"> {N_max}"
            print(f"  [rank {rank}] {xlabel}={p:.4f}:  N_min = {Nc_str}")

        if comm is not None:
            all_results = comm.gather(my_results, root=0)
            if rank != 0:
                return None, None
            all_results = sorted([r for sub in all_results for r in sub], key=lambda x: x[0])
        else:
            all_results = my_results

        results_param = [r[1] for r in all_results]
        results_N     = [r[2] for r in all_results]

        p_finite = [p for p, n in zip(results_param, results_N) if n is not None]
        N_finite = [n for n in results_N if n is not None]
        p_inf    = [p for p, n in zip(results_param, results_N) if n is None]

        fig, ax = plt.subplots(figsize=(12, 6))
        if p_finite:
            ax.semilogy(p_finite, N_finite, 'ko-', ms=5, lw=1.5,
                        label='$N_{\\mathrm{min}}$ (bounded)')
        if p_inf:
            ax.axvspan(min(p_inf) - 0.01, max(p_inf) + 0.01, color='red', alpha=0.12,
                       label=f'Escapes for all N <= {N_max}')
            for pi in p_inf:
                ax.semilogy(pi, N_max, 'rx', ms=9, mew=2)

        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel('$N_{\\mathrm{min}}$', fontsize=13)
        ax.set_title(f'Minimum N to Prevent Escape  ({self.name}, {fixed_label})', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        fname = f'./{self.name}_min_N_escape_{sweep}.png'
        plt.savefig(self._savepath(fname), dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"  -> Saved {fname}\n")

        return results_param, results_N

    # ── Convenience: run all standard analyses ─────────────────────────────
    def run_all(self, eps, N_values=(100, 1000, 10000, 100000)):
        """Run analyses 1-5 with the given coupling strength."""
        self.analysis_timeseries(eps, N_values)
        self.analysis_fluctuation_scaling(eps, N_values)
        self.analysis_distribution(eps, N_values)
        self.analysis_self_consistency(eps)
        self.analysis_phase_diagram(N_values=N_values)


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Built-in maps ──────────────────────────────────────────────────────
    CoupledMapLattice.from_tent(a=1.99).run_all(eps=0.1)
    CoupledMapLattice.from_logistic(r=3.9).run_all(eps=0.1)
    CoupledMapLattice.from_lozi(a=1.7, b=0.5).run_all(eps=0.1)

    # ── Sweep a map parameter (example) ───────────────────────────────────
    # sys = CoupledMapLattice.from_lozi(a=1.7, b=0.5)
    # sys.analysis_meanfield_bifurcation(
    #     sweep='map_param',
    #     map_factory=lambda a: CoupledMapLattice.from_lozi(a=a, b=0.5),
    #     param_range=(1.0, 2.0), eps_fixed=0.1)
