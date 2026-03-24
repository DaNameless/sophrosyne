"""
Numerical Analysis of Mean-Field Convergence in Globally Coupled Tent Maps
===========================================================================

System:
  x_{t+1}^{(i)} = (1-eps)*f(x_t^{(i)}) + eps*h_t
  h_t = (1/N) * sum_j f(x_t^{(j)))
  f(x) = (a/2)(1 - |1 - 2x|)       (tent map with parameter a)

Analyses:
  1. Time series of h_t for different N
  2. Fluctuation scaling: sigma_h vs N  (expect slope -1/2)
  3. Distribution of h_t, rescaled collapse, Q-Q plot
  4. Self-consistency check
  5. Phase diagram: scaling exponent alpha(eps)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ── Tent map: f(x) = (a/2)(1 - |1 - 2x|) ──
def tent(x, a):
    return (a / 2.0) * (1.0 - np.abs(1.0 - 2.0 * x))

# ── Simulate coupled system, return h_t time series ──
def simulate(N, eps, a, T_total=5000, T_transient=2000, seed=None):
    rng = np.random.default_rng(seed)
    x = rng.random(N)
    h_series = np.empty(T_total - T_transient)

    for t in range(T_total):
        fx = tent(x, a)
        h = np.mean(fx)
        if t >= T_transient:
            h_series[t - T_transient] = h
        x = (1.0 - eps) * fx + eps * h

    return h_series


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Time series visualization
# ══════════════════════════════════════════════════════════════════════
def analysis_timeseries(eps, a, N_values=[100, 1000, 10000, 100000]):
    print("=" * 60)
    print(f"ANALYSIS 1: Time series of h_t  (eps={eps}, a={a})")
    print("=" * 60)

    fig, axes = plt.subplots(len(N_values), 1, figsize=(14, 10), sharex=True)
    T_show = 200

    for ax, N in zip(axes, N_values):
        h = simulate(N, eps, a, T_total=T_show + 10000, T_transient=10000, seed=42)
        h = h[:T_show]
        ax.plot(h, 'b-', linewidth=0.5, alpha=0.8)
        ax.axhline(y=np.mean(h), color='r', linestyle='--', alpha=0.5)
        ax.set_ylabel('$h_t$', fontsize=11)
        ax.set_title(f'N = {N}   (sigma = {np.std(h):.5f})', fontsize=12)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time step', fontsize=12)
    plt.suptitle(f'Mean Field Time Series (eps={eps}, a={a})', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'./a{a}_eps{eps}_timeseries.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  -> Saved fig1_timeseries.png\n")


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Fluctuation scaling  sigma_h  vs  N
# ══════════════════════════════════════════════════════════════════════
def analysis_fluctuation_scaling(eps, a, N_values=[100, 1000, 10000, 100000]):
    print("=" * 60)
    print(f"ANALYSIS 2: Fluctuation scaling  (eps={eps}, a={a})")
    print("=" * 60)

    sigmas = []
    means = []
    n_trials = 5

    for N in N_values:
        trial_sigmas = []
        trial_means = []
        for trial in range(n_trials):
            h = simulate(N, eps, a, T_total=8000, T_transient=3000, seed=42 + trial)
            trial_sigmas.append(np.std(h))
            trial_means.append(np.mean(h))
        sigmas.append(np.mean(trial_sigmas))
        means.append(np.mean(trial_means))
        print(f"  N={N:>6d}:  <h> = {means[-1]:.6f},  sigma_h = {sigmas[-1]:.6f}")

    sigmas = np.array(sigmas)
    N_arr = np.array(N_values, dtype=float)

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
    ax.set_xlabel('N', fontsize=12)
    ax.set_ylabel('sigma_h', fontsize=12)
    ax.set_title(f'Fluctuation Scaling (eps={eps}, a={a})', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogx(N_arr, means, 'ko-', markersize=7)
    h_inf = means[-1]
    ax.axhline(y=h_inf, color='r', linestyle='--',
               label=f'h_bar ~ {h_inf:.4f}')
    ax.set_xlabel('N', fontsize=12)
    ax.set_ylabel('<h_t>', fontsize=12)
    ax.set_title('Mean Field Convergence', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.savefig(f'./a{a}_eps{eps}_scaling.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Saved fig2_scaling.png\n")

    return slope


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Distribution of h_t and rescaled collapse
# ══════════════════════════════════════════════════════════════════════
def analysis_distribution(eps, a, N_values = [100, 1000, 10000, 100000]):
    print("=" * 60)
    print(f"ANALYSIS 3: Distribution & rescaled collapse  (eps={eps}, a={a})")
    print("=" * 60)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    all_rescaled = {}

    for i, N in enumerate(N_values):
        h = simulate(N, eps, a, T_total=12000, T_transient=2000, seed=42)
        h_mean, h_std = np.mean(h), np.std(h)
        rescaled = (h - h_mean) * np.sqrt(N)
        all_rescaled[N] = rescaled

        sample = h[:5000]
        _, shapiro_p = stats.shapiro(sample)
        print(f"  N={N:>6d}: <h>={h_mean:.6f}, sigma={h_std:.6f}, Shapiro p={shapiro_p:.4f}")

        axes[0].hist(h, bins=60, density=True, alpha=0.5, color=colors[i], label=f'N={N}')
        axes[1].hist(rescaled, bins=60, density=True, alpha=0.5, color=colors[i], label=f'N={N}')

    ref_std = np.std(all_rescaled[N_values[-1]])
    xg = np.linspace(-4 * ref_std, 4 * ref_std, 200)
    axes[1].plot(xg, stats.norm.pdf(xg, 0, ref_std), 'k-', lw=2, label='Gaussian')

    axes[0].set_xlabel('h_t'); axes[0].set_ylabel('Density')
    axes[0].set_title('Distribution of h_t'); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('(h_t - h_bar)*sqrt(N)'); axes[1].set_ylabel('Density')
    axes[1].set_title('Rescaled (should collapse)'); axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    h_largest = simulate(N_values[-1], eps, a, T_total=12000, T_transient=2000, seed=42)
    z = (h_largest - np.mean(h_largest)) / np.std(h_largest)
    stats.probplot(z, dist="norm", plot=axes[2])
    axes[2].set_title(f'Q-Q Plot (N={N_values[-1]})'); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'./a{a}_eps{eps}_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Saved fig3_distribution.png\n")


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 4: Self-consistency check
# ══════════════════════════════════════════════════════════════════════
def analysis_self_consistency(eps, a):
    print("=" * 60)
    print(f"ANALYSIS 4: Self-consistency  (eps={eps}, a={a})")
    print("=" * 60)

    def single_site_average(h_bar, eps, a, T=200000, T_trans=50000):
        x = 0.37
        total = 0.0
        for t in range(T):
            fx = (a / 2.0) * (1.0 - abs(1.0 - 2.0 * x))
            if t >= T_trans:
                total += fx
            x = (1.0 - eps) * fx + eps * h_bar
        return total / (T - T_trans)

    h_candidates = np.linspace(0.01, a / 2.0 - 0.01, 50)
    f_averages = np.array([single_site_average(h, eps, a) for h in h_candidates])
    residuals = f_averages - h_candidates

    sign_changes = np.where(np.diff(np.sign(residuals)))[0]
    h_star_list = []
    for idx in sign_changes:
        h1, h2 = h_candidates[idx], h_candidates[idx + 1]
        r1, r2 = residuals[idx], residuals[idx + 1]
        h_star_list.append(h1 - r1 * (h2 - h1) / (r2 - r1))

    h_sim = simulate(10000, eps, a, T_total=10000, T_transient=3000, seed=42)
    h_sim_mean = np.mean(h_sim)

    print(f"  Self-consistent h_bar: {h_star_list}")
    print(f"  Simulation <h> (N=10000): {h_sim_mean:.6f}")
    if h_star_list:
        print(f"  |diff| = {abs(h_star_list[0] - h_sim_mean):.6f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(h_candidates, f_averages, 'b-', lw=2, label='<f(x)> under rho*')
    ax.plot(h_candidates, h_candidates, 'r--', lw=1.5, label='y = h_bar')
    for hs in h_star_list:
        ax.plot(hs, hs, 'go', ms=12, zorder=5, label=f'Fixed point: {hs:.4f}')
    ax.axvline(h_sim_mean, color='purple', ls='--', alpha=0.5, label=f'Sim mean: {h_sim_mean:.4f}')
    ax.set_xlabel('h_bar (candidate)'); ax.set_ylabel('<f>')
    ax.set_title(f'Self-Consistency (eps={eps}, a={a})')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.savefig(f'./a{a}_eps{eps}_self_consistence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Saved fig4_self_consistency.png\n")


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 5: Phase diagram — alpha(eps)
# ══════════════════════════════════════════════════════════════════════
def analysis_phase_diagram(a, N_values = [100, 1000, 10000, 100000]):
    print("=" * 60)
    print(f"ANALYSIS 5: Phase diagram alpha(eps)  (a={a})")
    print("=" * 60)

    eps_values = np.arange(0.02, 0.52, 0.03)

    N_arr = np.array(N_values, dtype=float)

    alphas, alpha_errors = [], []

    for eps in eps_values:
        sigmas = []
        for N in N_values:
            h = simulate(N, eps, a, T_total=6000, T_transient=2000, seed=42)
            sigmas.append(np.std(h))
        sigmas = np.array(sigmas)
        slope, intercept, r, p, se = stats.linregress(np.log(N_arr), np.log(sigmas + 1e-15))
        alphas.append(-slope)
        alpha_errors.append(se)
        print(f"  eps={eps:.2f}:  alpha = {-slope:.4f} +/- {se:.4f}  (R^2={r**2:.4f})")

    alphas = np.array(alphas)
    alpha_errors = np.array(alpha_errors)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.errorbar(eps_values, alphas, yerr=alpha_errors, fmt='ko-', ms=6, capsize=3)
    ax.axhline(0.5, color='r', ls='--', lw=1.5, label='CLT (alpha=1/2)')
    ax.fill_between(eps_values, 0.45, 0.55, alpha=0.1, color='red')
    ax.set_xlabel('eps'); ax.set_ylabel('alpha')
    ax.set_title(f'Scaling Exponent (a={a})'); ax.legend()
    ax.grid(True, alpha=0.3); ax.set_ylim(-0.1, 1.1)

    ax = axes[1]
    for eps_s, col in zip([0.05, 0.15, 0.3, 0.45],
                          ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']):
        sigs = [np.std(simulate(N, eps_s, a, T_total=6000, T_transient=2000, seed=42))
                for N in N_values]
        ax.loglog(N_arr, sigs, 'o-', color=col, ms=6, label=f'eps={eps_s}')
    ax.loglog(N_arr, 0.3 * N_arr**(-0.5), 'k--', lw=1, alpha=0.5, label='N^{-1/2}')
    ax.set_xlabel('N'); ax.set_ylabel('sigma_h')
    ax.set_title('Scaling at Different Couplings'); ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.savefig(f'./a{a}_eps{eps}_phase_diagram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Saved fig5_phase_diagram.png\n")


# ══════════════════════════════════════════════════════════════════════
# BIFURCATION DIAGRAM
# ══════════════════════════════════════════════════════════════════════
def bifurcation_diagram(bifurcation_param='a',
                        a_range=(0.5, 2.0), eps_range=(0.0, 1.0),
                        a_fixed=1.99, eps_fixed=0.1, h_bar=0.65,
                        n_param=500, T_iter=800, T_trans=600):
    """
    Bifurcation diagram of g(x) = (1-eps)*f(x,a) + eps*h_bar
    with h_bar a fixed constant passed as argument.
    """
    if bifurcation_param == 'a':
        param_values = np.linspace(a_range[0], a_range[1], n_param)
        xlabel = 'a'
    elif bifurcation_param == 'eps':
        param_values = np.linspace(eps_range[0], eps_range[1], n_param)
        xlabel = 'eps'
    else:
        raise ValueError("bifurcation_param must be 'a' or 'eps'")

    all_param = []
    all_x = []

    for i, p in enumerate(param_values):
        if bifurcation_param == 'a':
            a, eps = p, eps_fixed
        else:
            a, eps = a_fixed, p

        x = 0.37
        escaped = False

        for t in range(T_trans):
            fx = (a / 2.0) * (1.0 - abs(1.0 - 2.0 * x))
            x = (1.0 - eps) * fx + eps * h_bar
            if not np.isfinite(x) or abs(x) > 100:
                escaped = True
                break

        if escaped:
            continue

        for t in range(T_iter - T_trans):
            fx = (a / 2.0) * (1.0 - abs(1.0 - 2.0 * x))
            x = (1.0 - eps) * fx + eps * h_bar
            if not np.isfinite(x) or abs(x) > 100:
                break
            all_param.append(p)
            all_x.append(x)

    all_param = np.array(all_param)
    all_x = np.array(all_x)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.scatter(all_param, all_x, s=0.01, c='black', alpha=0.3)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('x (attractor)', fontsize=12)
    if bifurcation_param == 'a':
        ax.set_title(f'g(x) = (1-eps)*f(x,a) + eps*h_bar   '
                     f'(eps={eps_fixed}, h_bar={h_bar})', fontsize=13)
    else:
        ax.set_title(f'g(x) = (1-eps)*f(x,a) + eps*h_bar   '
                     f'(a={a_fixed}, h_bar={h_bar})', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5,1.5)
    plt.show()
    plt.tight_layout()
    plt.savefig(f'bifurcation_{bifurcation_param}_{a_fixed if xlabel=="eps" else eps_fixed}.png', dpi=150, bbox_inches='tight')
    plt.close()

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 6: Mean-Field Bifurcation Diagram
# ══════════════════════════════════════════════════════════════════════
def analysis_meanfield_bifurcation(sweep='a',
                                   a_range=(0.5, 3.0), eps_range=(0.0, 1.0),
                                   a_fixed=3, eps_fixed=0.4,
                                   N=10000, n_param=500,
                                   T_total=6000, T_transient=4000,
                                   seed=42):
    """
    Bifurcation diagram of the mean field h_t from the full N-body
    coupled simulation, sweeping either 'a' or 'eps'.
    """
    print("=" * 60)
    if sweep == 'a':
        param_values = np.linspace(a_range[0], a_range[1], n_param)
        xlabel, fixed_label = 'a', f'eps={eps_fixed}'
    elif sweep == 'eps':
        param_values = np.linspace(eps_range[0], eps_range[1], n_param)
        xlabel, fixed_label = 'eps', f'a={a_fixed}'
    else:
        raise ValueError("sweep must be 'a' or 'eps'")
    print(f"ANALYSIS 6: Mean-field bifurcation  (sweep {sweep}, {fixed_label}, N={N})")
    print("=" * 60)

    all_param = []
    all_h = []
    n_keep = 300  # attractor points per parameter value

    for i, p in enumerate(param_values):
        if sweep == 'a':
            a, eps = p, eps_fixed
        else:
            a, eps = a_fixed, p

        h = simulate(N, eps, a, T_total=T_total, T_transient=T_transient, seed=seed)
        h_tail = h[-n_keep:]
        mask = np.isfinite(h_tail) & (np.abs(h_tail) < 100)
        h_clean = h_tail[mask]

        for hv in h_clean:
            all_param.append(p)
            all_h.append(hv)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_param} done")

    all_param = np.array(all_param)
    all_h = np.array(all_h)
    print(f"  Total points: {len(all_h)}")

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.scatter(all_param, all_h, s=0.05, c='black', alpha=0.4, rasterized=True)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel('$h_t$ (mean field)', fontsize=13)
    ax.set_title(f'Mean-Field Bifurcation Diagram  ({fixed_label}, N={N})', fontsize=14)
    ax.grid(True, alpha=0.3)

    if len(all_h) > 0:
        y_lo, y_hi = np.percentile(all_h, [0.1, 99.9])
        margin = 0.05 * max(y_hi - y_lo, 0.01)
        ax.set_ylim(y_lo - margin, y_hi + margin)

    plt.tight_layout()
    plt.savefig(f'./meanfield_bif_{sweep}_{fixed_label.replace("=","")}_N{N}.png',
                dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  -> Saved meanfield_bif_{sweep}_{fixed_label.replace('=','')}_N{N}.png\n")



# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 7: Minimum N to prevent escape
# ══════════════════════════════════════════════════════════════════════
def _system_escapes(N, eps, a, T_total=4000, x_bound=100,
                    n_trials=3, seed=42):
    """
    Run the coupled simulation and return True if the system escapes
    (any state becomes non-finite or |x_i| > x_bound) in ANY trial.
 
    Multiple trials with different seeds are used because escape can be
    stochastic — a single lucky initial condition might survive while
    most don't.
    """
    for trial in range(n_trials):
        rng = np.random.default_rng(seed + trial)
        x = rng.random(N)
        for t in range(T_total):
            fx = tent(x, a)
            h = np.mean(fx)
            x = (1.0 - eps) * fx + eps * h
            if not np.all(np.isfinite(x)) or np.any(np.abs(x) > x_bound):
                return True
    return False
 
 
def find_min_N(eps, a, N_min=2, N_max=100000, T_total=4000,
               x_bound=100, n_trials=3, seed=42):
    """
    Binary search for the minimum N that keeps the coupled tent-map
    system bounded for the given (eps, a).
 
    Returns
    -------
    N_crit : int or None
        Smallest N in [N_min, N_max] for which the system does NOT
        escape.  None if even N_max escapes.
    """
    # Quick check: if the system is already safe at N_min, return that
    if not _system_escapes(N_min, eps, a, T_total, x_bound, n_trials, seed):
        return N_min
 
    # If even N_max escapes, signal that no safe N was found
    if _system_escapes(N_max, eps, a, T_total, x_bound, n_trials, seed):
        return None
 
    # Binary search: invariant  lo escapes, hi survives
    lo, hi = N_min, N_max
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _system_escapes(mid, eps, a, T_total, x_bound, n_trials, seed):
            lo = mid
        else:
            hi = mid
    return hi
 
 
def analysis_min_N_escape(sweep='a',
                          a_range=(1.0, 3.0), eps_range=(0.0, 0.5),
                          a_fixed=2.5, eps_fixed=0.1,
                          n_param=60,
                          N_min=2, N_max=100000,
                          T_total=4000, x_bound=100,
                          n_trials=3, seed=42):
    """
    Sweep one parameter (a or eps) while keeping the other fixed and,
    for each value, find the minimum N that prevents the system from
    escaping.
 
    Parameters
    ----------
    sweep : str
        'a'   — sweep a over a_range with eps = eps_fixed
        'eps' — sweep eps over eps_range with a = a_fixed
    a_range, eps_range : tuple
        (min, max) of the swept parameter.
    a_fixed, eps_fixed : float
        Value of the parameter that is NOT swept.
    n_param : int
        Number of parameter values to sample.
    N_min, N_max : int
        Bounds for the binary search over N.
    T_total : int
        Simulation length used for each escape test.
    x_bound : float
        Threshold above which |x_i| is considered escaped.
    n_trials : int
        Independent initial-condition trials per (N, param) test.
    seed : int
        Base RNG seed for reproducibility.
 
    Produces
    --------
    - Console table of (param, N_crit)
    - Figure with N_crit vs swept parameter (log-scale y-axis)
    """
    print("=" * 60)
    if sweep == 'a':
        param_values = np.linspace(a_range[0], a_range[1], n_param)
        fixed_label = f'eps={eps_fixed}'
        xlabel = 'a'
    elif sweep == 'eps':
        param_values = np.linspace(eps_range[0], eps_range[1], n_param)
        fixed_label = f'a={a_fixed}'
        xlabel = 'eps'
    else:
        raise ValueError("sweep must be 'a' or 'eps'")
 
    print(f"ANALYSIS 7: Minimum N to prevent escape  "
          f"(sweep {sweep}, {fixed_label})")
    print("=" * 60)
 
    results_param = []
    results_N = []
 
    for i, p in enumerate(param_values):
        if sweep == 'a':
            a, eps = p, eps_fixed
        else:
            a, eps = a_fixed, p
 
        Nc = find_min_N(eps, a, N_min=N_min, N_max=N_max,
                        T_total=T_total, x_bound=x_bound,
                        n_trials=n_trials, seed=seed)
 
        results_param.append(p)
        results_N.append(Nc)
 
        Nc_str = f"{Nc}" if Nc is not None else f"> {N_max}"
        print(f"  {xlabel}={p:.4f}:  N_min = {Nc_str}")
 
    # ── Separate finite / infinite results for plotting ──
    p_finite = [p for p, n in zip(results_param, results_N) if n is not None]
    N_finite = [n for n in results_N if n is not None]
    p_inf    = [p for p, n in zip(results_param, results_N) if n is None]
 
    # ── Plot ──
    fig, ax = plt.subplots(figsize=(12, 6))
 
    if p_finite:
        ax.semilogy(p_finite, N_finite, 'ko-', ms=5, lw=1.5,
                    label='$N_{\\mathrm{min}}$ (bounded)')
 
    if p_inf:
        ax.axvspan(min(p_inf) - 0.01, max(p_inf) + 0.01,
                   color='red', alpha=0.12, label=f'Escapes for all N ≤ {N_max}')
        for pi in p_inf:
            ax.semilogy(pi, N_max, 'rx', ms=9, mew=2)
 
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel('$N_{\\mathrm{min}}$', fontsize=13)
    ax.set_title(f'Minimum N to Prevent Escape  ({fixed_label})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
 
    plt.tight_layout()
    fname = f'./min_N_escape_{sweep}_{fixed_label.replace("=","")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  -> Saved {fname}\n")
 
    return results_param, results_N
 
 
# ══════════════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ═══════════════════════════════════════════
    # PARAMETERS — change these
    # ═══════════════════════════════════════════
    A   = 1.99    # tent map parameter
    EPS = 0.1     # coupling strength
 
    print(f"\n{'=' * 60}")
    print(f"  GLOBALLY COUPLED TENT MAPS: a={A}, eps={EPS}")
    print(f"{'=' * 60}\n")
 
    analysis_timeseries(EPS, A)
    analysis_fluctuation_scaling(EPS, A)
    analysis_distribution(EPS, A)
    analysis_self_consistency(EPS, A)
    analysis_phase_diagram(A)
 
    # ── Analysis 7 examples ──
    # Sweep a with fixed eps: where does the system start needing
    # larger populations to stay bounded?
    # analysis_min_N_escape(sweep='a', eps_fixed=0.1, a_range=(1.5, 3.0))
 
    # Sweep eps with fixed a: how does coupling rescue a chaotic map?
    # analysis_min_N_escape(sweep='eps', a_fixed=2.5, eps_range=(0.0, 0.5))
 