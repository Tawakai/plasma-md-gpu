"""Tests for weighted finite-size scaling curve_fit (Yeh-Hummer N^(-1/3) vs 1/N)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fit_diffusion_fss import fit_scaling_model, load_diffusion_csv, run_comparison


def test_weighted_curve_fit_matches_exact_analytical_gls() -> None:
    """Verify curve_fit D_inf, c, SE, and chi^2/dof against exact analytical GLS equations."""
    n_arr = np.array([125.0, 500.0, 2000.0, 8000.0, 32000.0], dtype=np.float64)
    sigma_arr = np.array([0.18, 0.12, 0.09, 0.06, 0.04], dtype=np.float64)

    d_true = 7.85
    c_true = -3.20
    shifts = np.array([0.4, -0.7, 0.5, -0.6, 0.3], dtype=np.float64)
    d_arr = d_true + c_true * np.power(n_arr, -1.0 / 3.0) + shifts * sigma_arr

    fit_yh = fit_scaling_model(n_arr, d_arr, sigma_arr, mode="yeh_hummer")
    fit_1n = fit_scaling_model(n_arr, d_arr, sigma_arr, mode="inv_n")

    x = np.power(n_arr, -1.0 / 3.0)
    w = 1.0 / (sigma_arr * sigma_arr)
    s_w = float(np.sum(w))
    s_x = float(np.sum(w * x))
    s_y = float(np.sum(w * d_arr))
    s_xx = float(np.sum(w * x * x))
    s_xy = float(np.sum(w * x * d_arr))
    det = s_w * s_xx - s_x * s_x

    d_inf_exact = (s_xx * s_y - s_x * s_xy) / det
    c_exact = (s_w * s_xy - s_x * s_y) / det
    se_d_inf_exact = float(np.sqrt(s_xx / det))
    se_c_exact = float(np.sqrt(s_w / det))
    chi2_exact = float(np.sum(w * (d_arr - (d_inf_exact + c_exact * x)) ** 2))

    np.testing.assert_allclose(fit_yh.d_inf, d_inf_exact, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(fit_yh.c_slope, c_exact, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(fit_yh.d_inf_se, se_d_inf_exact, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(fit_yh.c_slope_se, se_c_exact, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(fit_yh.chi2, chi2_exact, rtol=1e-8, atol=1e-10)
    assert fit_yh.dof == 3
    assert fit_yh.chi2_per_dof < fit_1n.chi2_per_dof


def test_csv_loader_and_cli_runner(tmp_path: Path) -> None:
    """Verify CSV parsing and JSON output on a 5-point table."""
    csv_file = tmp_path / "de_scaling.csv"
    out_json = tmp_path / "fss_out.json"
    csv_file.write_text(
        "N,D_e_star,sigma\n"
        "125,7.22,0.18\n"
        "500,7.45,0.11\n"
        "2000,7.60,0.08\n"
        "8000,7.70,0.05\n"
        "32000,7.76,0.04\n",
        encoding="utf-8",
    )
    n_arr, _, _ = load_diffusion_csv(csv_file)
    assert len(n_arr) == 5
    res = run_comparison(csv_file, out_json)
    assert out_json.exists()
    assert "yeh_hummer" in res and "inv_n" in res
