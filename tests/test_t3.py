"""Tests for static/dynamic observables: RDF, VACF, MSD, and S_ZZ(q, omega) (T3)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from analyze import analyze_h5
from plasmamd.md import run_simulation
from plasmamd.observables import (
    charge_dsf,
    rdf,
)
from plasmamd.state import PlasmaState
from run_md import save_trajectory_h5


def test_rdf_neighbor_count_normalization() -> None:
    """Criterion 1: sum 4*pi*r^2*n*g(r)*dr up to L/2 gives exact average neighbor count."""
    rng = np.random.default_rng(42)
    n_pairs = 125
    state = PlasmaState(gamma=0.5, r_s=1.0, n_pairs=n_pairs)
    box = state.box
    dr = 0.02
    r_max = 0.5 * box
    n_bins = int(math.floor(r_max / dr))
    r_cutoff = n_bins * dr

    n_frames = 20
    positions = rng.uniform(0.0, box, size=(n_frames, state.n_total, 3)).astype(np.float64)
    species = np.array([0] * n_pairs + [1] * n_pairs, dtype=np.int8)

    r, g_ii, g_ie, g_ee = rdf(positions, species, box, dr=dr, r_max=r_max)
    n_density = float(n_pairs) / (box**3)

    integral_ie = float(np.sum(4.0 * math.pi * (r**2) * n_density * g_ie * dr))
    integral_ii = float(np.sum(4.0 * math.pi * (r**2) * n_density * g_ii * dr))
    integral_ee = float(np.sum(4.0 * math.pi * (r**2) * n_density * g_ee * dr))

    # Direct count of neighbors within sphere of radius r_cutoff
    inv_box = 1.0 / box
    direct_ie = 0.0
    direct_ii = 0.0
    direct_ee = 0.0
    for f_idx in range(n_frames):
        pos_i = positions[f_idx, species == 0]
        pos_e = positions[f_idx, species == 1]

        d_ie = pos_i[:, None, :] - pos_e[None, :, :]
        d_ie -= box * np.round(d_ie * inv_box)
        direct_ie += float(np.sum(np.linalg.norm(d_ie, axis=-1) < r_cutoff)) / n_pairs

        d_ii = pos_i[:, None, :] - pos_i[None, :, :]
        d_ii -= box * np.round(d_ii * inv_box)
        dist_ii = np.linalg.norm(d_ii, axis=-1)
        direct_ii += float(np.sum((dist_ii > 0.0) & (dist_ii < r_cutoff))) / n_pairs

        d_ee = pos_e[:, None, :] - pos_e[None, :, :]
        d_ee -= box * np.round(d_ee * inv_box)
        dist_ee = np.linalg.norm(d_ee, axis=-1)
        direct_ee += float(np.sum((dist_ee > 0.0) & (dist_ee < r_cutoff))) / n_pairs

    direct_ie /= n_frames
    direct_ii /= n_frames
    direct_ee /= n_frames

    np.testing.assert_allclose(integral_ie, direct_ie, rtol=1e-12)
    np.testing.assert_allclose(integral_ii, direct_ii, rtol=1e-12)
    np.testing.assert_allclose(integral_ee, direct_ee, rtol=1e-12)


def test_md_diffusion_vacf_msd_and_hansen_mcdonald_table_iii(tmp_path: Path) -> None:
    """Criteria 2 & 3: D_e from VACF and MSD agree within error and D_e* = 7.2 +/- 0.7."""
    state = PlasmaState(gamma=0.5, r_s=1.0, n_pairs=125, mass_ratio=1836.15)
    dt = 0.03
    # Record every 5 steps (dt_frame = 0.15) to accurately resolve Z_e(t) integral and unwrapped MSD
    traj = run_simulation(
        state=state,
        n_equil_steps=5_000,
        n_prod_steps=25_000,
        dt=dt,
        tau_berendsen=100.0 * dt,
        min_dist=0.3,
        seed=1,
        energy_stride=50,
        frame_stride=5,
    )

    h5_path = tmp_path / "run.h5"
    json_path = tmp_path / "results.json"
    save_trajectory_h5(h5_path, state, traj, dt, 5_000, 25_000, 1)
    res = analyze_h5(h5_path, json_path)
    assert json_path.exists()

    d_vacf = float(res["d_e_star_vacf_block_mean"])
    d_vacf_err = float(res["d_e_star_vacf_block_err"])
    d_msd = float(res["d_e_star_msd_block_mean"])
    d_msd_err = float(res["d_e_star_msd_block_err"])
    combined_err = math.hypot(d_vacf_err, d_msd_err)

    print(
        f"\n[T3 Diffusion Summary Gamma=0.5, r_s=1.0]"
        f"\n  D_e* (VACF 5-block) = {d_vacf:.3f} +/- {d_vacf_err:.3f}"
        f"\n  D_e* (MSD  5-block) = {d_msd:.3f} +/- {d_msd_err:.3f}"
        f"\n  Table III reference = 7.2 +/- 0.7"
    )

    # Criterion 2: D_e from VACF and MSD agree within uncertainty (3 sigma / tolerance)
    assert abs(d_vacf - d_msd) <= max(3.0 * combined_err, 0.7)

    # Criterion 3: D_e* = 7.2 +/- 0.7 (Table III of Hansen & McDonald 1981)
    d_best = 0.5 * (d_vacf + d_msd)
    assert abs(d_best - 7.2) <= 0.7


def test_charge_dsf_synthetic_plasmon_oscillation() -> None:
    """Criterion 4: on synthetic trajectory oscillating at omega_0, charge_dsf peaks at omega_0."""
    n_pairs = 64
    n_total = 2 * n_pairs
    box = 10.0
    q_min = 2.0 * math.pi / box

    dt = 0.05
    n_frames = 800
    t = np.arange(n_frames, dtype=np.float64) * dt
    omega_0 = 1.25

    rng = np.random.default_rng(7)
    base_pos = rng.uniform(0.0, box, size=(n_total, 3))
    charge = np.array([1.0] * n_pairs + [-1.0] * n_pairs, dtype=np.float64)

    positions = np.empty((n_frames, n_total, 3), dtype=np.float64)
    amp = 0.08
    for idx in range(n_frames):
        shift = amp * math.cos(omega_0 * t[idx])
        positions[idx] = base_pos
        # Electrons oscillate along x/y/z to excite longitudinal mode at |k| = q_min
        positions[idx, n_pairs:, 0] = (base_pos[n_pairs:, 0] + shift) % box
        positions[idx, n_pairs:, 1] = (base_pos[n_pairs:, 1] + shift) % box
        positions[idx, n_pairs:, 2] = (base_pos[n_pairs:, 2] + shift) % box

    omega, s_zz, omega_peak = charge_dsf(positions, charge, box, dt=dt, q_target=q_min)
    freq_res = float(omega[1] - omega[0])
    assert abs(omega_peak - omega_0) <= 1.5 * freq_res


def test_flyvbjerg_petersen_block_average_ar1() -> None:
    """Verify block_average on 5e4-point AR(1) process against analytical stderr within 15%."""
    from plasmamd.observables import block_average

    n_points = 50_000
    tau_corr = 20.0
    phi = math.exp(-1.0 / tau_corr)
    sigma_eps = 1.0
    mu_true = -1.25

    # Exact stationary variance: sigma_x^2 = sigma_eps^2 / (1 - phi^2)
    sigma_x = sigma_eps / math.sqrt(1.0 - phi * phi)

    # Exact analytical standard error of the sample mean for finite N:
    # Var(x_bar) = (sigma_x^2 / N) * [ (1 + phi)/(1 - phi) - 2*phi*(1 - phi^N) / (N*(1 - phi)^2) ]
    tau_int_factor = (1.0 + phi) / (1.0 - phi) - (
        2.0 * phi * (1.0 - phi**n_points) / (float(n_points) * (1.0 - phi) ** 2)
    )
    stderr_analytical = sigma_x * math.sqrt(tau_int_factor / float(n_points))

    rng = np.random.default_rng(2026)
    x = np.empty(n_points, dtype=np.float64)
    x[0] = mu_true + rng.normal(0.0, sigma_x)
    eps = rng.normal(0.0, sigma_eps, size=n_points - 1)
    for t in range(1, n_points):
        x[t] = mu_true + phi * (x[t - 1] - mu_true) + eps[t - 1]

    mean_est, stderr_est, n_blocks_used = block_average(x)
    rel_err = abs(stderr_est - stderr_analytical) / stderr_analytical

    print(
        f"\n[Flyvbjerg-Petersen AR(1) N={n_points}, tau={tau_corr}]"
        f"\n  mean_est          = {mean_est:.6f} (true {mu_true:.6f})"
        f"\n  stderr_est        = {stderr_est:.6e}"
        f"\n  stderr_analytical = {stderr_analytical:.6e}"
        f"\n  relative_error    = {rel_err:.2%}"
        f"\n  n_blocks_used     = {n_blocks_used}"
    )

    assert n_blocks_used >= 16
    assert abs(mean_est - mu_true) <= 3.0 * stderr_analytical
    assert rel_err <= 0.15
