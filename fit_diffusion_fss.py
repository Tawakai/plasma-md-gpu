"""Weighted finite-size scaling fit of electron diffusion D_e*(N) via Yeh-Hummer vs 1/N models."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_dist
from scipy.stats import t as student_t


@dataclass(frozen=True)
class FSSModelFit:
    """Container for weighted least-squares fit results of a finite-size scaling model."""

    model_name: str
    d_inf: float
    d_inf_se: float
    d_inf_ci95_halfwidth: float
    d_inf_ci95_low: float
    d_inf_ci95_high: float
    d_inf_se_rescaled: float
    c_slope: float
    c_slope_se: float
    chi2: float
    dof: int
    chi2_per_dof: float
    p_value: float


def model_yeh_hummer(
    n_val: NDArray[np.float64],
    d_inf: float,
    c_val: float,
) -> NDArray[np.float64]:
    """Yeh-Hummer hydrodynamic finite-size scaling law: D(N) = D_inf + c * N^(-1/3).

    Parameters
    ----------
    n_val : np.ndarray, shape (M,), dtype float64
        Number of ion-electron pairs N [dimensionless].
    d_inf : float
        Thermodynamic-limit diffusion coefficient D_inf^* [a^2 omega_pe].
    c_val : float
        Hydrodynamic finite-size scaling amplitude c [a^2 omega_pe].

    Returns
    -------
    d_pred : np.ndarray, shape (M,), dtype float64
        Predicted reduced diffusion coefficient D^*(N) [a^2 omega_pe].
    """
    return d_inf + c_val * np.power(n_val, -1.0 / 3.0)


def model_inv_n(
    n_val: NDArray[np.float64],
    d_inf: float,
    c_val: float,
) -> NDArray[np.float64]:
    """Alternative ensemble 1/N finite-size scaling law: D(N) = D_inf + c / N.

    Parameters
    ----------
    n_val : np.ndarray, shape (M,), dtype float64
        Number of ion-electron pairs N [dimensionless].
    d_inf : float
        Thermodynamic-limit diffusion coefficient D_inf^* [a^2 omega_pe].
    c_val : float
        1/N finite-size scaling coefficient c [a^2 omega_pe].

    Returns
    -------
    d_pred : np.ndarray, shape (M,), dtype float64
        Predicted reduced diffusion coefficient D^*(N) [a^2 omega_pe].
    """
    return d_inf + c_val / n_val


def fit_scaling_model(
    n_arr: NDArray[np.float64],
    d_arr: NDArray[np.float64],
    sigma_arr: NDArray[np.float64],
    mode: str = "yeh_hummer",
    conf_level: float = 0.95,
) -> FSSModelFit:
    """Perform weighted curve_fit with absolute_sigma=True and compute chi^2/dof and CI.

    Parameters
    ----------
    n_arr : np.ndarray, shape (M,), dtype float64
        System sizes N [dimensionless].
    d_arr : np.ndarray, shape (M,), dtype float64
        Measured reduced diffusion coefficients D_e^* [a^2 omega_pe].
    sigma_arr : np.ndarray, shape (M,), dtype float64
        1-sigma statistical uncertainties of D_e^* [a^2 omega_pe].
    mode : str, optional
        Model identifier: "yeh_hummer" for D_inf + c*N^(-1/3) or "inv_n" for D_inf + c/N.
    conf_level : float, optional
        Confidence level for Student-t interval (default 0.95) [dimensionless].

    Returns
    -------
    result : FSSModelFit
        Fitted parameters, standard errors, 95% CI, chi^2/dof, and goodness-of-fit p-value.
    """
    n_c = np.ascontiguousarray(n_arr, dtype=np.float64)
    d_c = np.ascontiguousarray(d_arr, dtype=np.float64)
    s_c = np.ascontiguousarray(sigma_arr, dtype=np.float64)

    if n_c.ndim != 1 or d_c.shape != n_c.shape or s_c.shape != n_c.shape:
        raise ValueError("Input arrays n_arr, d_arr, sigma_arr must be 1D with identical shape.")
    if n_c.size < 3:
        raise ValueError(f"At least 3 data points are required, got {n_c.size}.")
    if np.any(s_c <= 0.0):
        raise ValueError("All uncertainties sigma_arr must be strictly positive.")

    if mode == "yeh_hummer":
        func = model_yeh_hummer
        label = "Yeh-Hummer: D_inf + c * N^(-1/3)"

        def jac(n_val, _d, _c):
            return np.column_stack([np.ones_like(n_val), np.power(n_val, -1.0 / 3.0)])
    elif mode == "inv_n":
        func = model_inv_n
        label = "Inverse-N:  D_inf + c / N"

        def jac(n_val, _d, _c):
            return np.column_stack([np.ones_like(n_val), 1.0 / n_val])
    else:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'yeh_hummer' or 'inv_n'.")

    p0 = (float(d_c[-1]), 0.0)
    popt, pcov = curve_fit(
        func,
        n_c,
        d_c,
        p0=p0,
        sigma=s_c,
        absolute_sigma=True,
        jac=jac,
        ftol=1e-15,
        xtol=1e-15,
        gtol=1e-15,
    )

    d_inf = float(popt[0])
    c_val = float(popt[1])
    d_inf_se = float(np.sqrt(pcov[0, 0]))
    c_se = float(np.sqrt(pcov[1, 1]))

    residuals = (d_c - func(n_c, d_inf, c_val)) / s_c
    chi2_val = float(np.sum(residuals * residuals))
    dof = int(n_c.size - 2)
    chi2_red = chi2_val / float(dof)
    p_val = float(chi2_dist.sf(chi2_val, dof))

    alpha_tail = 0.5 * (1.0 + conf_level)
    t_crit = float(student_t.ppf(alpha_tail, df=dof))
    ci_half = t_crit * d_inf_se
    d_inf_se_rescaled = d_inf_se * float(np.sqrt(max(0.0, chi2_red)))

    return FSSModelFit(
        model_name=label,
        d_inf=d_inf,
        d_inf_se=d_inf_se,
        d_inf_ci95_halfwidth=ci_half,
        d_inf_ci95_low=d_inf - ci_half,
        d_inf_ci95_high=d_inf + ci_half,
        d_inf_se_rescaled=d_inf_se_rescaled,
        c_slope=c_val,
        c_slope_se=c_se,
        chi2=chi2_val,
        dof=dof,
        chi2_per_dof=chi2_red,
        p_value=p_val,
    )


def load_diffusion_csv(
    csv_path: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Load (N, D_e_star, sigma) columns from a CSV file (with or without header).

    Parameters
    ----------
    csv_path : str or Path
        Path to CSV file containing 3 numeric columns: N, D_e*, and uncertainty sigma.

    Returns
    -------
    n_arr, d_arr, sigma_arr : tuple of np.ndarray, shape (M,), dtype float64
        Parsed arrays of system sizes, diffusion coefficients, and standard errors.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    n_list: list[float] = []
    d_list: list[float] = []
    s_list: list[float] = []

    raw_lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for line in raw_lines:
        cleaned = line.replace(";", ",")
        raw_parts = cleaned.split(",") if "," in cleaned else cleaned.split()
        tokens = [tok.strip() for tok in raw_parts if tok.strip()]
        if len(tokens) < 3:
            continue
        try:
            n_val = float(tokens[0])
            d_val = float(tokens[1])
            s_val = float(tokens[2])
        except ValueError:
            continue
        n_list.append(n_val)
        d_list.append(d_val)
        s_list.append(s_val)

    if len(n_list) < 3:
        raise ValueError(f"Failed to parse at least 3 numeric rows from {path}.")

    return (
        np.array(n_list, dtype=np.float64),
        np.array(d_list, dtype=np.float64),
        np.array(s_list, dtype=np.float64),
    )


def run_comparison(csv_path: str | Path, out_json: str | Path | None = None) -> dict[str, object]:
    """Fit both Yeh-Hummer and 1/N models, print comparison table, and optionally save JSON."""
    n_arr, d_arr, sigma_arr = load_diffusion_csv(csv_path)

    fit_yh = fit_scaling_model(n_arr, d_arr, sigma_arr, mode="yeh_hummer")
    fit_1n = fit_scaling_model(n_arr, d_arr, sigma_arr, mode="inv_n")

    dist_yh = abs(fit_yh.chi2_per_dof - 1.0)
    dist_1n = abs(fit_1n.chi2_per_dof - 1.0)
    preferred = fit_yh if dist_yh <= dist_1n else fit_1n
    best_chi2_model = fit_yh if fit_yh.chi2 < fit_1n.chi2 else fit_1n

    n_int_list = n_arr.astype(int).tolist()
    print("=" * 78)
    print(f"Finite-Size Scaling of D_e* (M = {len(n_arr)} points: N = {n_int_list})")
    print("=" * 78)
    for fit in (fit_yh, fit_1n):
        print(f"\nModel: {fit.model_name}")
        print(
            f"  D_inf* (weighted LS) = {fit.d_inf:.6f} +/- {fit.d_inf_se:.6f} "
            f"(1 sigma, absolute_sigma=True)"
        )
        print(
            f"  95% CI (t_3 = 3.182) = [{fit.d_inf_ci95_low:.6f}, {fit.d_inf_ci95_high:.6f}] "
            f"(+/- {fit.d_inf_ci95_halfwidth:.6f})"
        )
        print(
            f"  D_inf* SE (rescaled) = +/- {fit.d_inf_se_rescaled:.6f} (scaled by sqrt(chi^2/dof))"
        )
        print(f"  Slope c              = {fit.c_slope:.6f} +/- {fit.c_slope_se:.6f}")
        print(
            f"  chi^2 / dof          = {fit.chi2:.4f} / {fit.dof} = "
            f"{fit.chi2_per_dof:.4f} (p = {fit.p_value:.4f})"
        )

    print("\n" + "-" * 78)
    print(
        f"Best model by lowest chi^2/dof: {best_chi2_model.model_name}\n"
        f"  => D_inf* = {best_chi2_model.d_inf:.5f} +/- {best_chi2_model.d_inf_se:.5f} (1 sigma)\n"
        f"  => 95% CI = {best_chi2_model.d_inf:.5f} +/- "
        f"{best_chi2_model.d_inf_ci95_halfwidth:.5f} "
        f"[{best_chi2_model.d_inf_ci95_low:.5f}, {best_chi2_model.d_inf_ci95_high:.5f}]"
    )
    print("=" * 78)

    summary: dict[str, object] = {
        "n": n_arr.tolist(),
        "d_e_star": d_arr.tolist(),
        "sigma": sigma_arr.tolist(),
        "yeh_hummer": asdict(fit_yh),
        "inv_n": asdict(fit_1n),
        "preferred_by_chi2": best_chi2_model.model_name,
        "preferred_closer_to_unit_chi2_red": preferred.model_name,
    }
    if out_json is not None:
        Path(out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Fit Yeh-Hummer and 1/N scaling laws to D_e*(N) CSV."
    )
    parser.add_argument("csv_file", type=Path, help="Path to CSV with columns: N, D_e*, sigma.")
    parser.add_argument("--out", type=Path, default=None, help="Optional output JSON path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    run_comparison(args.csv_file, args.out)


if __name__ == "__main__":
    main()
