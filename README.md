# Sophrosyne

**Coupled Dynamical Systems Simulator** — A Python toolkit for studying emergent phenomena, escape dynamics, and divergence control in globally coupled continuous and discrete systems.

---

## Overview

Sophrosyne numerically integrates globally coupled dynamical systems using a **4th-order Runge–Kutta (RK4)** scheme. It is designed to explore how collections of coupled oscillators can exhibit emergent behavior — synchronization, partial escape, or full divergence — depending on coupling strength, system parameters, and initial conditions.

Given $N$ coupled units, each governed by a vector field $\mathbf{F}$, the general form of the coupled system is:

$$
\dot{\mathbf{x}}_i = \mathbf{F}(\mathbf{x}_i; \, a, b, c) \;+\; \frac{\varepsilon}{N} \sum_{j=1}^{N} \mathbf{H}(\mathbf{x}_j - \mathbf{x}_i), \quad i = 1, \dots, N
$$

where $\varepsilon$ is the global coupling strength and $\mathbf{H}$ is the coupling function.

---

## Supported Systems

| System | Key | Description |
|--------|-----|-------------|
| **Linz–Sprott** | `linzsprott` | 3D flow with tunable divergence via parameter $a$ |
| **Rössler** | `rossler` | Classic chaotic attractor $(a, b, c)$ |
| **Rössler (globally coupled)** | `rossler_global` | $N$ Rössler oscillators with global mean-field coupling |
| **Shimizu–Morioka ($B = 0$)** | `shimizu_morioka` | Reduced Shimizu–Morioka system |

> More systems can be added to the `SYSTEM_REGISTRY`, or *the user can supply custom differential equations (in progress)*.

---

## Project Structure

```
sophrosyne/
├── code/
│   ├── modules/          # Python modules (.py)
│   │   └── sophrosyne.py # Main module: systems, solver, CLI
│   └── notebooks/        # Jupyter notebooks for analysis
├── outputs/              # Saved figures & data (git-ignored)
└── README.md
```

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/<your-username>/sophrosyne.git
cd sophrosyne
pip install -e .
```

Required dependencies:

- `numpy`
- `scipy`
- `matplotlib`

---

## Usage as Script

### Single run

Run a simulation with default or custom parameters:

```bash
python sophrosyne.py run --system linzsprott --N 100 --epsilon 0.001 \
    --dt 0.005 --steps 200000 --threshold 100.0 --window 5000 \
    --ic-spread 0.6 --ic-dist uniform --plot --output-dir ./output
```

### Parameter sweep

Scan over coupling strength $\varepsilon$ and a system parameter $a$ for different ensemble sizes $N$:

```bash
python sophrosyne.py sweep --system linzsprott \
    --N 20,50,100 --a-min 0.50 --a-max 0.60 --a-num 20 \
    --eps-min 1e-4 --eps-max 1e-1 --eps-num 15 --eps-log \
    --steps 200000 --plot --save sweep_results.npz --output-dir ./output
```

---

## Command-Line Reference

### `run` — Single simulation

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--system` | str | `linzsprott` | Dynamical system to simulate |
| `--N` | int | `100` | Number of coupled units |
| `--epsilon` | float | `0.001` | Global coupling strength $\varepsilon$ |
| `--dt` | float | `0.005` | Integration time step |
| `--steps` | int | `200000` | Total number of time steps |
| `--threshold` | float | `100.0` | Divergence detection threshold |
| `--window` | int | `5000` | Observation window (steps) |
| `--ic-spread` | float | `0.6` | Spread of initial conditions |
| `--ic-dist` | str | `uniform` | Distribution of ICs: `uniform` or `normal` |
| `--seed` | int | `None` | Random seed for reproducibility |
| `--a`, `--b`, `--c` | float | `None` | System-specific parameters |
| `--save-full` | flag | — | Save full trajectory data |
| `--save-npz` | str | `None` | Save results to `.npz` file |
| `--save-hdf5` | str | `None` | Save results to `.hdf5` file |
| `--plot` | flag | — | Generate plots after simulation |
| `--output-dir` | str | `./output` | Output directory |

### `sweep` — Parameter sweep

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--system` | str | `linzsprott` | Dynamical system to simulate |
| `--N` | str | `100` | Comma-separated ensemble sizes, e.g. `20,50,100` |
| `--a-min`, `--a-max` | float | `0.50`, `0.60` | Range of parameter $a$ |
| `--a-num` | int | `20` | Number of $a$ values |
| `--eps-min`, `--eps-max` | float | `1e-4`, `1e-1` | Range of coupling $\varepsilon$ |
| `--eps-num` | int | `15` | Number of $\varepsilon$ values |
| `--eps-log` / `--eps-lin` | flag | `--eps-log` | Logarithmic or linear spacing for $\varepsilon$ |
| `--workers` | int | `None` | Number of parallel workers |
| `--save` | str | `None` | Save sweep results to `.npz` |
| `--plot` | flag | — | Generate sweep plots |
| `--output-dir` | str | `./output` | Output directory |

> Flags `--dt`, `--steps`, `--threshold`, `--window`, `--ic-spread`, `--ic-dist`, `--seed`, `--b`, `--c` are shared with `run`.

---

## Importing the Module

You can import the classes directly in your own scripts or notebooks:

```python
%load_ext autoreload
%autoreload 2

from modules.sophrosyne import LinzSprott, SimulationRunner, Plotter
```

### Example: single simulation from a notebook

```python
runner = SimulationRunner(
    system_cls=LinzSprott,
    system_params={"a": 0.55},
    N=100,
    epsilon=0.01,
    dt=0.005,
    steps=200_000,
    threshold=100.0,
    window=5000,
    ic_spread=0.6,
    ic_dist="uniform",
    seed=42,
)

result = runner.run()
Plotter.plot_all(result, "./outputs")
```

---

## Output Files

Depending on the provided arguments, the following may be generated:

- `*.npz` / `*.hdf5` — Simulation data (trajectories, parameters, metadata)
- `*.png` — Plots of dynamics, divergence fraction, sweep phase diagrams
- Sweep results — $(\varepsilon, a)$ phase diagrams showing survival/escape fractions for each $N$

---

## Author

- Rolando Sebastián Sánchez García
