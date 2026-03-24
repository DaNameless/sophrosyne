"""
Numerical Analysis of Mean-Field Convergence in Globally Coupled Lozi Maps
===========================================================================

System:
  x_{t+1}^{(i)} = (1-eps)*f(x_t^{(i)}, y_t^{(i)}) + eps*h_t
  y_{t+1}^{(i)} = beta*x_t^{(i)}
  h_t = (1/N) * sum_j f(x_t^{(j)}, y_t^{(j)})
  f(x, y) = 1 - alpha*|x| + y

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


def f(x, y, a):
    return 1 - a * np.abs(x) + y


# ── Simulate coupled system, return h_t time series ──
def simulate(N, eps, a, b, T_total=5000, T_transient=2000, seed=None):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, N)
    y = b * x + rng.uniform(-0.01, 0.01, N)  # y ≈ b*x with small noise
    h_series = np.empty(T_total - T_transient)

    for t in range(T_total):
        fxy = f(x, y, a)
        h = np.mean(fxy)
        if t >= T_transient:
            h_series[t - T_transient] = h
        y = b * x
        x = (1.0 - eps) * fxy + eps * h
    return h_series


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Time series visualization
# ══════════════════════════════════════════════════════════════════════
def analysis_timeseries(eps, a, b, N_values=[100, 1000, 10000, 100000]):
    print("=" * 60)
    print(f"ANALYSIS 1: Time series of h_t  (eps={eps}, a={a}, b={b})")
    print("=" * 60)

    fig, axes = plt.subplots(len(N_values), 1, figsize=(14, 10), sharex=True)
    T_show = 200

    for ax, N in zip(axes, N_values):
        h = simulate(N, eps, a, b, T_total=T_show + 10000, T_transient=10000, seed=42)
        h = h[:T_show]
        ax.plot(h, 'b-', linewidth=0.5, alpha=0.8)
        ax.axhline(y=np.mean(h), color='r', linestyle='--', alpha=0.5)
        ax.set_ylabel('$h_t$', fontsize=11)
        ax.set_title(f'N = {N}   (sigma = {np.std(h):.5f})', fontsize=12)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time step', fontsize=12)
    plt.suptitle(f'Mean Field Time Series (eps={eps}, a={a}, b={b})', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'./a{a}_b{b}_eps{eps}_timeseries.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  -> Saved timeseries.png\n")


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Fluctuation scaling  sigma_h  vs  N
# ══════════════════════════════════════════════════════════════════════
def analysis_fluctuation_scaling(eps, a, b, N_values=[100, 1000, 10000, 100000]):
    print("=" * 60)
    print(f"ANALYSIS 2: Fluctuation scaling  (eps={eps}, a={a}, b={b})")
    print("=" * 60)

    sigmas = []
    means = []
    n_trials = 5

    for N in N_values:
        trial_sigmas = []
        trial_means = []
        for trial in range(n_trials):
            h = simulate(N, eps, a, b, T_total=8000, T_transient=3000, seed=42 + trial)
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
    ax.set_title(f'Fluctuation Scaling (eps={eps}, a={a}, b={b})', fontsize=13)
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
    plt.savefig(f'./a{a}_b{b}_eps{eps}_scaling.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Saved scaling.png\n")

    return slope


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Distribution of h_t and rescaled collapse
# ══════════════════════════════════════════════════════════════════════
def analysis_distribution(eps, a, b, N_values=[100, 1000, 10000, 100000]):
    print("=" * 60)
    print(f"ANALYSIS 3: Distribution & rescaled collapse  (eps={eps}, a={a}, b={b})")
    print("=" * 60)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    all_rescaled = {}

    for i, N in enumerate(N_values):
        h = simulate(N, eps, a, b, T_total=12000, T_transient=2000, seed=42)
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

    h_largest = simulate(N_values[-1], eps, a, b, T_total=12000, T_transient=2000, seed=42)
    z = (h_largest - np.mean(h_largest)) / np.std(h_largest)
    stats.probplot(z, dist="norm", plot=axes[2])
    axes[2].set_title(f'Q-Q Plot (N={N_values[-1]})'); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'./a{a}_b{b}_eps{eps}_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Saved distribution.png\n")


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 4: Self-consistency check
# ══════════════════════════════════════════════════════════════════════
def analysis_self_consistency(eps, a, b):
    print("=" * 60)
    print(f"ANALYSIS 4: Self-consistency  (eps={eps}, a={a}, b={b})")
    print("=" * 60)

    def single_site_average(h_bar, eps, a, b, T=200000, T_trans=50000):
        x, y = 0.37, 0.4
        total = 0.0
        for t in range(T):
            # Compute f at current (x, y)
            fx = 1.0 - a * np.abs(x) + y
            if t >= T_trans:
                total += fx
            # Update state: y_{t+1} = beta * x_t  (old x), then x
            y = b * x
            x = (1.0 - eps) * fx + eps * h_bar
        return total / (T - T_trans)

    h_candidates = np.linspace(0.01, a / 2.0 - 0.01, 50)
    f_averages = np.array([single_site_average(h, eps, a, b) for h in h_candidates])
    residuals = f_averages - h_candidates

    sign_changes = np.where(np.diff(np.sign(residuals)))[0]
    h_star_list = []
    for idx in sign_changes:
        h1, h2 = h_candidates[idx], h_candidates[idx + 1]
        r1, r2 = residuals[idx], residuals[idx + 1]
        h_star_list.append(h1 - r1 * (h2 - h1) / (r2 - r1))

    h_sim = simulate(10000, eps, a, b, T_total=10000, T_transient=3000, seed=42)
    h_sim_mean = np.mean(h_sim)

    print(f"  Self-consistent h_bar: {h_star_list}")
    print(f"  Simulation <h> (N=10000): {h_sim_mean:.6f}")
    if h_star_list:
        print(f"  |diff| = {abs(h_star_list[0] - h_sim_mean):.6f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(h_candidates, f_averages, 'b-', lw=2, label='<f(x,y)> under rho*')
    ax.plot(h_candidates, h_candidates, 'r--', lw=1.5, label='y = h_bar')
    for hs in h_star_list:
        ax.plot(hs, hs, 'go', ms=12, zorder=5, label=f'Fixed point: {hs:.4f}')
    ax.axvline(h_sim_mean, color='purple', ls='--', alpha=0.5, label=f'Sim mean: {h_sim_mean:.4f}')
    ax.set_xlabel('h_bar (candidate)'); ax.set_ylabel('<f>')
    ax.set_title(f'Self-Consistency (eps={eps}, a={a}, b={b})')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.savefig(f'./a{a}_b{b}_eps{eps}_self_consistency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Saved self_consistency.png\n")


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 5: Phase diagram — alpha(eps)
# ══════════════════════════════════════════════════════════════════════
def analysis_phase_diagram(a, b, N_values=[100, 1000, 10000, 100000]):
    print("=" * 60)
    print(f"ANALYSIS 5: Phase diagram alpha(eps)  (a={a}, b={b})")
    print("=" * 60)

    eps_values = np.arange(0.02, 0.52, 0.03)
    N_arr = np.array(N_values, dtype=float)
    alphas, alpha_errors = [], []

    for eps in eps_values:
        sigmas = []
        for N in N_values:
            h = simulate(N, eps, a, b, T_total=6000, T_transient=2000, seed=42)
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
    ax.set_title(f'Scaling Exponent (a={a}, b={b})'); ax.legend()
    ax.grid(True, alpha=0.3); ax.set_ylim(-0.1, 1.1)

    ax = axes[1]
    for eps_s, col in zip([0.05, 0.15, 0.3, 0.45],
                          ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']):
        sigs = [np.std(simulate(N, eps_s, a, b, T_total=6000, T_transient=2000, seed=42))
                for N in N_values]
        ax.loglog(N_arr, sigs, 'o-', color=col, ms=6, label=f'eps={eps_s}')
    ax.loglog(N_arr, 0.3 * N_arr**(-0.5), 'k--', lw=1, alpha=0.5, label='N^{-1/2}')
    ax.set_xlabel('N'); ax.set_ylabel('sigma_h')
    ax.set_title('Scaling at Different Couplings'); ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.savefig(f'./a{a}_b{b}_phase_diagram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Saved phase_diagram.png\n")


# ══════════════════════════════════════════════════════════════════════
# BIFURCATION DIAGRAM (single-site under fixed h_bar)
# ══════════════════════════════════════════════════════════════════════
def bifurcation_diagram(bifurcation_param='a',
                        a_range=(1, 2.0), eps_range=(0.0, 1.0), b_range=(-1, 1.0),
                        a_fixed=1.99, eps_fixed=0.1, b_fixed=0.5, h_bar=0.65,
                        n_param=500, T_iter=800, T_trans=600):
    """
    Bifurcation diagram of the 2D Lozi map driven by a fixed mean field h_bar:
      x_{t+1} = (1-eps)*f(x_t, y_t) + eps*h_bar
      y_{t+1} = b * x_t
    bifurcation_param: 'a', 'eps', or 'b'
    """
    if bifurcation_param == 'a':
        param_values = np.linspace(a_range[0], a_range[1], n_param)
        xlabel = 'a'
    elif bifurcation_param == 'eps':
        param_values = np.linspace(eps_range[0], eps_range[1], n_param)
        xlabel = 'eps'
    elif bifurcation_param == 'b':
        param_values = np.linspace(b_range[0], b_range[1], n_param)
        xlabel = 'b'
    else:
        raise ValueError("bifurcation_param must be 'a', 'eps', or 'b'")

    all_param, all_x, all_y = [], [], []

    for p in param_values:
        a   = p if bifurcation_param == 'a'   else a_fixed
        eps = p if bifurcation_param == 'eps' else eps_fixed
        b   = p if bifurcation_param == 'b'   else b_fixed

        x, y = 0.37, 0.4
        escaped = False

        for _ in range(T_trans):
            fx = 1.0 - a * np.abs(x) + y
            y  = b * x
            x  = (1.0 - eps) * fx + eps * h_bar
            if not np.isfinite(x) or abs(x) > 1e6:
                escaped = True
                break

        if escaped:
            continue

        for _ in range(T_iter - T_trans):
            fx = 1.0 - a * np.abs(x) + y
            y  = b * x
            x  = (1.0 - eps) * fx + eps * h_bar
            if not np.isfinite(x) or abs(x) > 1e6:
                break
            all_param.append(p)
            all_x.append(x)
            all_y.append(y)

    all_param = np.array(all_param)
    all_x     = np.array(all_x)
    all_y     = np.array(all_y)

    if bifurcation_param == 'a':
        title_fixed = f'eps={eps_fixed}, b={b_fixed}'
    elif bifurcation_param == 'eps':
        title_fixed = f'a={a_fixed}, b={b_fixed}'
    else:
        title_fixed = f'a={a_fixed}, eps={eps_fixed}'

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    axes[0].scatter(all_param, all_x, s=0.01, c='black', alpha=0.3)
    axes[0].set_ylabel('x (attractor)', fontsize=12)
    axes[0].set_ylim(-1.5, 1.5)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Lozi g(x,y) = (1-eps)*f(x,a)+eps*h_bar  ({title_fixed}, h_bar={h_bar})',
                      fontsize=13)

    axes[1].scatter(all_param, all_y, s=0.01, c='steelblue', alpha=0.3)
    axes[1].set_ylabel('y (attractor)', fontsize=12)
    axes[1].set_ylim(-1.5, 1.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel(xlabel, fontsize=12)

    plt.tight_layout()
    plt.show()
    plt.savefig(f'bifurcation_{bifurcation_param}_fixed_{title_fixed.replace(", ","_").replace("=","")}.png',
                dpi=150, bbox_inches='tight')
    plt.close()

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 6: Mean-Field Bifurcation Diagram
# ══════════════════════════════════════════════════════════════════════

def analysis_meanfield_bifurcation(sweep='a',
                                   a_range=(0.5, 3.0), eps_range=(0.0, 1.0), b_range=(0.0, 1.0),
                                   a_fixed=3, eps_fixed=0.4, b_fixed=0.5,
                                   N=10000, n_param=500,
                                   T_total=6000, T_transient=4000,
                                   seed=42):
    """
    Bifurcation diagram of the mean field h_t from the full N-body
    coupled simulation, sweeping 'a', 'eps', or 'b'.
    """
    print("=" * 60)
    if sweep == 'a':
        param_values = np.linspace(a_range[0], a_range[1], n_param)
        xlabel       = 'a'
        fixed_label  = f'eps={eps_fixed}, b={b_fixed}'
    elif sweep == 'eps':
        param_values = np.linspace(eps_range[0], eps_range[1], n_param)
        xlabel       = 'eps'
        fixed_label  = f'a={a_fixed}, b={b_fixed}'
    elif sweep == 'b':
        param_values = np.linspace(b_range[0], b_range[1], n_param)
        xlabel       = 'b'
        fixed_label  = f'a={a_fixed}, eps={eps_fixed}'
    else:
        raise ValueError("sweep must be 'a', 'eps', or 'b'")
    print(f"ANALYSIS 6: Mean-field bifurcation  (sweep {sweep}, {fixed_label}, N={N})")
    print("=" * 60)

    all_param, all_h = [], []
    n_keep = 300

    for i, p in enumerate(param_values):
        a_use   = p       if sweep == 'a'   else a_fixed
        eps_use = p       if sweep == 'eps' else eps_fixed
        b_use   = p       if sweep == 'b'   else b_fixed

        h = simulate(N, eps_use, a_use, b_use,
                     T_total=T_total, T_transient=T_transient, seed=seed)
        h_tail  = h[-n_keep:]
        mask    = np.isfinite(h_tail) & (np.abs(h_tail) < 1e6)
        h_clean = h_tail[mask]

        for hv in h_clean:
            all_param.append(p)
            all_h.append(hv)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_param} done")

    all_param = np.array(all_param)
    all_h     = np.array(all_h)
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
    plt.savefig(f'./meanfield_bif_{sweep}_{fixed_label.replace(", ","_").replace("=","")}_N{N}.png',
                dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  -> Saved meanfield_bif png\n")


# ══════════════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ═══════════════════════════════════════════
    # PARAMETERS — change these
    # ═══════════════════════════════════════════
    A   = 1.99   # Lozi alpha parameter
    B   = 0.5    # Lozi beta parameter   ← BUG FIX 14: was never defined in main
    EPS = 0.1    # coupling strength

    print(f"\n{'=' * 60}")
    print(f"  GLOBALLY COUPLED LOZI MAPS: a={A}, b={B}, eps={EPS}")
    print(f"{'=' * 60}\n")

    analysis_timeseries(EPS, A, B)
    analysis_fluctuation_scaling(EPS, A, B)
    analysis_distribution(EPS, A, B)
    analysis_self_consistency(EPS, A, B)
    analysis_phase_diagram(A, B)