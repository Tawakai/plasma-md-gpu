"""Finite-size scaling of the electron self-diffusion coefficient from real GPU MD runs.

For each system size the GPU-resident engine is run at Gamma = 0.5, r_s = 1 with the
Kelbg potential; the measured ``(N, D_e*, sigma)`` go to ``results/diffusion_fss_real.csv``
and are fitted with ``D(N) = D_inf + c N^(-1/3)`` (Yeh-Hummer) and ``D(N) = D_inf + c / N``.
A second pair of runs compares ``g_ab(r)`` for the Deutsch and Kelbg potentials.

Every number in the CSV and on the figures comes from a simulation executed by this
script; nothing is hard-coded.

Usage
-----
    python scripts/run_real_fss_gpu.py                 # production: 125, 250, 500, 1000 pairs
    python scripts/run_real_fss_gpu.py --quick         # minutes-long smoke run
    python scripts/run_real_fss_gpu.py --n-pairs 64 125 250 --n-prod 20000 --n-blocks 10
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from plasmamd.gpu.kelbg_forces import POTENTIAL_DEUTSCH, POTENTIAL_KELBG
from plasmamd.gpu.md_engine_gpu import MDConfig, MDResult, run_md
from plasmamd.telemetry import SystemMonitor

plt.switch_backend("Agg")  # files only; no display is needed on a compute node

FloatArray = NDArray[np.float64]

GAMMA = 0.5
R_S = 1.0
DEFAULT_N_PAIRS = (125, 250, 500, 1000)
QUICK_N_PAIRS = (64, 125, 250)
CSV_NAME = "diffusion_fss_real.csv"
FSS_FIGURE = "real_fss_kelbg"
RDF_FIGURE = "rdf_deutsch_vs_kelbg"
FIG_WIDTH_IN = 160.0 / 25.4
FIG_DPI = 300
Z_95 = 1.959963984540054  # two-sided 95 % quantile of the normal distribution
MODEL_EXPONENTS = {"yeh_hummer": -1.0 / 3.0, "inverse_n": -1.0}


@dataclass(frozen=True)
class RunSettings:
    """MD settings shared by all system sizes."""

    n_equil: int = 4000
    n_prod: int = 40000
    n_blocks: int = 8
    sample_stride: int = 10
    dt: float = 0.03
    seed: int = 1
    backend: str = "gpu"
    rdf_n_pairs: int = 250
    rdf_n_equil: int = 4000
    rdf_n_prod: int = 20000
    rdf_stride: int = 20
    rdf_bins: int = 100


QUICK_SETTINGS = RunSettings(
    n_equil=1000, n_prod=7500, n_blocks=5, rdf_n_pairs=64, rdf_n_equil=1000, rdf_n_prod=4000
)


@dataclass(frozen=True)
class FssPoint:
    """One measured point of the size dependence."""

    n_pairs: int
    n_particles: int
    d_e: float
    d_e_err: float
    kt_electron: float
    energy_drift: float
    steps_per_second: float


@dataclass(frozen=True)
class FitResult:
    """Weighted least-squares fit of ``D(N) = D_inf + c N^exponent``."""

    model: str
    exponent: float
    d_inf: float
    d_inf_err: float
    c: float
    c_err: float
    chi2: float
    dof: int

    @property
    def chi2_per_dof(self) -> float:
        """Reduced chi-square (``nan`` when there are no degrees of freedom)."""
        return self.chi2 / self.dof if self.dof > 0 else float("nan")

    @property
    def d_inf_ci95(self) -> tuple[float, float]:
        """95 % confidence interval of ``D_inf`` (normal quantile: the sigmas are absolute)."""
        half = Z_95 * self.d_inf_err
        return self.d_inf - half, self.d_inf + half

    def predict(self, n: FloatArray) -> FloatArray:
        """Model value at particle numbers ``n``."""
        return self.d_inf + self.c * np.asarray(n, dtype=np.float64) ** self.exponent


@dataclass(frozen=True)
class PipelineResult:
    """Everything the pipeline produced."""

    points: list[FssPoint]
    fits: dict[str, FitResult]
    csv_path: Path
    figure_paths: list[Path]


# --------------------------------------------------------------------------------------
# Simulations
# --------------------------------------------------------------------------------------


def _relative_energy_drift(result: MDResult) -> float:
    """``(E_end - E_start)`` of the NVE stage in units of the mean kinetic energy."""
    n_pairs = result.config.n_pairs
    kinetic = 1.5 * n_pairs * float(np.mean(result.kt_ion + result.kt_electron))
    return float((result.e_total[-1] - result.e_total[0]) / kinetic)


def run_fss(n_pairs_list: Sequence[int], settings: RunSettings) -> list[FssPoint]:
    """Run the MD engine once per system size and collect ``D_e*`` with its standard error."""
    if settings.n_blocks < 2:
        raise ValueError("n_blocks >= 2 is required to estimate the error of D_e")
    points = []
    for index, n_pairs in enumerate(n_pairs_list):
        config = MDConfig(
            n_pairs=n_pairs, gamma=GAMMA, r_s=R_S, potential_type=POTENTIAL_KELBG,
            dt=settings.dt, n_equil=settings.n_equil, n_prod=settings.n_prod,
            n_blocks=settings.n_blocks, sample_stride=settings.sample_stride,
            seed=settings.seed + index, backend=settings.backend,
        )
        result = run_md(config)
        point = FssPoint(
            n_pairs=n_pairs,
            n_particles=2 * n_pairs,
            d_e=result.d_e,
            d_e_err=result.d_e_err,
            kt_electron=float(result.kt_electron.mean()),
            energy_drift=_relative_energy_drift(result),
            steps_per_second=result.steps_per_second,
        )
        print(
            f"N_pairs={n_pairs:5d}  D_e*={point.d_e:.3f} +- {point.d_e_err:.3f}  "
            f"<kT_e>={point.kt_electron:.4f}  dE/E_kin={point.energy_drift:+.1e}  "
            f"{point.steps_per_second:.0f} steps/s"
        )
        points.append(point)
    return points


def write_csv(points: Sequence[FssPoint], path: Path) -> None:
    """Write the measured points; columns are the fields of :class:`FssPoint`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(FssPoint.__dataclass_fields__)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for point in points:
            writer.writerow([repr(getattr(point, name)) for name in fields])


def read_csv(path: Path) -> list[FssPoint]:
    """Read points written by :func:`write_csv`."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        FssPoint(
            n_pairs=int(row["n_pairs"]),
            n_particles=int(row["n_particles"]),
            d_e=float(row["d_e"]),
            d_e_err=float(row["d_e_err"]),
            kt_electron=float(row["kt_electron"]),
            energy_drift=float(row["energy_drift"]),
            steps_per_second=float(row["steps_per_second"]),
        )
        for row in rows
    ]


# --------------------------------------------------------------------------------------
# Fits
# --------------------------------------------------------------------------------------


def fit_power_model(n: FloatArray, d: FloatArray, sigma: FloatArray, model: str) -> FitResult:
    """Weighted fit of ``D(N) = D_inf + c N^exponent`` with an analytic Jacobian."""
    exponent = MODEL_EXPONENTS[model]
    n = np.asarray(n, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if n.size < 2:
        raise ValueError("at least two system sizes are needed for a two-parameter fit")
    if not (np.all(np.isfinite(sigma)) and np.all(sigma > 0.0)):
        raise ValueError("all standard errors must be finite and positive")

    def model_fn(n_values: FloatArray, d_inf: float, c: float) -> FloatArray:
        return d_inf + c * n_values**exponent

    def jacobian(n_values: FloatArray, d_inf: float, c: float) -> FloatArray:
        return np.column_stack([np.ones_like(n_values), n_values**exponent])

    popt, pcov = curve_fit(
        model_fn, n, d, p0=(float(d.mean()), 0.0), sigma=sigma, absolute_sigma=True, jac=jacobian
    )
    residual = (d - model_fn(n, *popt)) / sigma
    perr = np.sqrt(np.diag(pcov))
    return FitResult(
        model=model, exponent=exponent, d_inf=float(popt[0]), d_inf_err=float(perr[0]),
        c=float(popt[1]), c_err=float(perr[1]), chi2=float(residual @ residual), dof=n.size - 2,
    )


def fit_fss(points: Sequence[FssPoint]) -> dict[str, FitResult]:
    """Fit both size-dependence models to the measured points."""
    n = np.array([p.n_particles for p in points], dtype=np.float64)
    d = np.array([p.d_e for p in points])
    sigma = np.array([p.d_e_err for p in points])
    return {model: fit_power_model(n, d, sigma, model) for model in MODEL_EXPONENTS}


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10, "axes.labelsize": 10, "legend.fontsize": 10,
            "xtick.labelsize": 10, "ytick.labelsize": 10, "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans", "axes.linewidth": 0.8, "lines.linewidth": 1.2,
            "xtick.direction": "in", "ytick.direction": "in", "savefig.dpi": FIG_DPI,
        }
    )


def _save(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=FIG_DPI)
    plt.close(fig)
    return paths


def plot_fss(points: Sequence[FssPoint], fits: dict[str, FitResult], stem: Path) -> list[Path]:
    """``D_e*`` against ``N^(-1/3)`` with both fits and the extrapolated ``D_inf``."""
    _apply_style()
    n = np.array([p.n_particles for p in points], dtype=np.float64)
    d = np.array([p.d_e for p in points])
    sigma = np.array([p.d_e_err for p in points])
    x_max = 1.08 * float(n.min()) ** (-1.0 / 3.0)
    x_line = np.linspace(1e-4, x_max, 300)
    n_line = x_line**-3.0

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 0.62 * FIG_WIDTH_IN))
    ax.errorbar(
        n ** (-1.0 / 3.0), d, yerr=sigma, fmt="o", color="black", markersize=5, capsize=3,
        label="МД, потенциал Кельбга",
    )
    yeh, inv = fits["yeh_hummer"], fits["inverse_n"]
    ax.plot(
        x_line, yeh.predict(n_line), "-", color="black",
        label=rf"$D_\infty + cN^{{-1/3}}$, $\chi^2/\nu = {yeh.chi2_per_dof:.2f}$",
    )
    ax.plot(
        x_line, inv.predict(n_line), "--", color="0.35",
        label=rf"$D_\infty + c/N$, $\chi^2/\nu = {inv.chi2_per_dof:.2f}$",
    )
    ax.errorbar(
        [0.0], [yeh.d_inf], yerr=[Z_95 * yeh.d_inf_err], fmt="s", color="black",
        markerfacecolor="white", markersize=6, capsize=3, clip_on=False,
        label=rf"$D_\infty^* = {yeh.d_inf:.2f} \pm {Z_95 * yeh.d_inf_err:.2f}$ (95 %)",
    )
    ax.set_xlim(0.0, x_max)
    ax.set_xlabel(r"$N^{-1/3}$")
    ax.set_ylabel(r"$D_e^* = D_e / (a^2 \omega_{pe})$")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, stem)


def run_rdf_comparison(settings: RunSettings) -> dict[int, MDResult]:
    """Run Deutsch and Kelbg systems and accumulate ``g_ab(r)``.

    Static properties of a classical system do not depend on the masses, so the ions are
    given the electron mass in the equations of motion (the thermal lengths keep the real
    mass ratio).  Hansen & McDonald (1981, Sec. III) used the same device.
    """
    results = {}
    for potential_type in (POTENTIAL_DEUTSCH, POTENTIAL_KELBG):
        config = MDConfig(
            n_pairs=settings.rdf_n_pairs, gamma=GAMMA, r_s=R_S, potential_type=potential_type,
            ion_dynamic_mass=1.0, dt=settings.dt, n_equil=settings.rdf_n_equil,
            n_prod=settings.rdf_n_prod, n_blocks=1, sample_stride=settings.sample_stride,
            rdf_stride=settings.rdf_stride, rdf_bins=settings.rdf_bins, seed=settings.seed,
            backend=settings.backend,
        )
        results[potential_type] = run_md(config)
    return results


def plot_rdf_comparison(results: dict[int, MDResult], stem: Path) -> list[Path]:
    """``g_ii``, ``g_ie``, ``g_ee`` for the Deutsch (grey) and Kelbg (black) potentials."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 0.62 * FIG_WIDTH_IN))
    pair_styles = (("g_ii", "-", "ii"), ("g_ie", "--", "ie"), ("g_ee", ":", "ee"))
    potentials = ((POTENTIAL_DEUTSCH, "0.55", "Дойч"), (POTENTIAL_KELBG, "black", "Кельбг"))
    for potential_type, color, name in potentials:
        result = results[potential_type]
        for attribute, linestyle, pair in pair_styles:
            ax.plot(
                result.rdf_r, getattr(result, attribute), linestyle=linestyle, color=color,
                label=rf"$g_{{{pair}}}$, {name}",
            )
    ax.axhline(1.0, color="0.8", linewidth=0.6, zorder=0)
    ax.set_xlim(0.0, float(results[POTENTIAL_KELBG].rdf_r[-1]))
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"$r / a$")
    ax.set_ylabel(r"$g_{ab}(r)$")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return _save(fig, stem)


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------


def run_pipeline(
    n_pairs_list: Sequence[int] = DEFAULT_N_PAIRS,
    settings: RunSettings | None = None,
    output_root: Path | str = ".",
) -> PipelineResult:
    """Simulate, write the CSV, fit and draw both figures under ``output_root``."""
    settings = settings or RunSettings()
    root = Path(output_root)
    csv_path = root / "results" / CSV_NAME

    with SystemMonitor(interval=0.25) as mon:
        mon.set_stage("FSS_MD")
        points = run_fss(n_pairs_list, settings)
        mon.set_stage("RDF_Comparison")
        rdf_results = run_rdf_comparison(settings)
    mon.print_summary()
    mon.save_csv(root / "results" / "telemetry_timeline.csv")

    write_csv(points, csv_path)
    fits = fit_fss(read_csv(csv_path))
    for fit in fits.values():
        low, high = fit.d_inf_ci95
        print(
            f"{fit.model:>10s}: D_inf*={fit.d_inf:.3f} +- {fit.d_inf_err:.3f}  "
            f"95% CI [{low:.3f}, {high:.3f}]  c={fit.c:.2f}  chi2/dof={fit.chi2_per_dof:.2f}"
        )
    figures = plot_fss(points, fits, root / "figures" / FSS_FIGURE)
    figures += plot_rdf_comparison(rdf_results, root / "figures" / RDF_FIGURE)
    figures += mon.plot(root / "figures" / "system_telemetry")
    return PipelineResult(points=points, fits=fits, csv_path=csv_path, figure_paths=figures)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="short smoke run (small N, few steps)")
    parser.add_argument("--n-pairs", type=int, nargs="+", help="list of N_pairs to simulate")
    parser.add_argument("--n-equil", type=int, help="thermalisation steps per size")
    parser.add_argument("--n-prod", type=int, help="NVE steps per size")
    parser.add_argument("--n-blocks", type=int, help="independent MSD segments per run")
    parser.add_argument("--dt", type=float, help="time step in 1/omega_pe")
    parser.add_argument("--seed", type=int, help="base random seed")
    parser.add_argument("--backend", choices=("gpu", "cpu"), help="'cpu' = reference backend")
    parser.add_argument("--output-root", type=Path, default=Path("."), help="where to write files")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = _parse_args()
    base = QUICK_SETTINGS if args.quick else RunSettings()
    overrides = {
        name: getattr(args, name)
        for name in ("n_equil", "n_prod", "n_blocks", "dt", "seed", "backend")
        if getattr(args, name) is not None
    }
    settings = RunSettings(**{**base.__dict__, **overrides})
    n_pairs_list = args.n_pairs or (QUICK_N_PAIRS if args.quick else DEFAULT_N_PAIRS)
    if not math.isfinite(settings.dt) or settings.dt <= 0.0:
        raise SystemExit("dt must be positive")
    result = run_pipeline(n_pairs_list, settings, args.output_root)
    print(f"CSV: {result.csv_path}")
    for path in result.figure_paths:
        print(f"Figure: {path}")


if __name__ == "__main__":
    main()
