"""Tests for Ewald summation, thermodynamic derivatives, and full MD run (T2)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from plasmamd.ewald import (
    deutsch_temperature_derivative_energy,
    ewald_reciprocal,
    total_forces_energy_virial,
    tune_ewald_parameters,
)
from plasmamd.md import initialize_positions, run_simulation
from plasmamd.state import PlasmaState
from run_md import save_trajectory_h5


def test_ewald_alpha_invariance_and_reciprocal_finite_difference() -> None:
    """Verify Ewald energy invariance wrt alpha (< 1e-6) and reciprocal force vs FD."""
    state = PlasmaState(gamma=0.5, r_s=1.0, n_pairs=16)
    n_tot = state.n_total
    box = state.box
    species = np.array([0] * state.n_pairs + [1] * state.n_pairs, dtype=np.int8)
    charge = np.where(species == 0, 1.0, -1.0).astype(np.float64)
    pos = initialize_positions(n_tot, box, min_dist=0.3, seed=7)

    alpha1, r_cut1, k_max1 = tune_ewald_parameters(box, tol=1e-6)
    alpha2 = alpha1 * 1.15
    k_max2 = k_max1 + 2

    _, u_cfg1, _, _ = total_forces_energy_virial(
        pos, species, charge, state.lam, box, alpha1, r_cut1, k_max1
    )
    _, u_cfg2, _, _ = total_forces_energy_virial(
        pos, species, charge, state.lam, box, alpha2, r_cut1, k_max2
    )

    assert abs(u_cfg1 - u_cfg2) / abs(u_cfg1) < 1e-6

    f_k, _, _ = ewald_reciprocal(pos, charge, box, alpha1, k_max1)
    eps = 1e-6
    for i in (0, n_tot - 1):
        for d in range(3):
            p_plus = pos.copy()
            p_minus = pos.copy()
            p_plus[i, d] += eps
            p_minus[i, d] -= eps
            _, u_p, _ = ewald_reciprocal(p_plus, charge, box, alpha1, k_max1)
            _, u_m, _ = ewald_reciprocal(p_minus, charge, box, alpha1, k_max1)
            fd = -(u_p - u_m) / (2.0 * eps)
            np.testing.assert_allclose(f_k[i, d], fd, rtol=1e-6, atol=1e-10)


def test_deutsch_temperature_derivative_against_fd() -> None:
    """Verify -T dv/dT term [Eq. (19)] against finite difference in temperature T."""
    gamma0 = 0.5
    state0 = PlasmaState(gamma=gamma0, r_s=1.0, n_pairs=16)
    box = state0.box
    species = np.array([0] * state0.n_pairs + [1] * state0.n_pairs, dtype=np.int8)
    charge = np.where(species == 0, 1.0, -1.0).astype(np.float64)
    pos = initialize_positions(state0.n_total, box, min_dist=0.3, seed=11)
    alpha, r_cut, k_max = tune_ewald_parameters(box, tol=1e-6)

    u_td_analytic = deutsch_temperature_derivative_energy(
        pos, species, charge, state0.lam, box, r_cut
    )

    t0 = state0.k_t
    dt_temp = 1e-5 * t0
    state_plus = PlasmaState(gamma=state0.e_sq / (t0 + dt_temp), r_s=1.0, n_pairs=16)
    state_minus = PlasmaState(gamma=state0.e_sq / (t0 - dt_temp), r_s=1.0, n_pairs=16)

    _, u_plus, _, _ = total_forces_energy_virial(
        pos, species, charge, state_plus.lam, box, alpha, r_cut, k_max
    )
    _, u_minus, _, _ = total_forces_energy_virial(
        pos, species, charge, state_minus.lam, box, alpha, r_cut, k_max
    )
    u_td_fd = -t0 * (u_plus - u_minus) / (2.0 * dt_temp)

    np.testing.assert_allclose(u_td_analytic, u_td_fd, rtol=1e-6, atol=1e-10)


def test_full_md_hansen_mcdonald_state(tmp_path: Path) -> None:
    """Criteria 1-4: 10k Berendsen + 50k NVE for Gamma=0.5, r_s=1, N=125 pairs."""
    state = PlasmaState(gamma=0.5, r_s=1.0, n_pairs=125, mass_ratio=1836.15)
    dt = 0.03
    traj = run_simulation(
        state=state,
        n_equil_steps=10_000,
        n_prod_steps=50_000,
        dt=dt,
        tau_berendsen=100.0 * dt,
        min_dist=0.3,
        seed=1,
        energy_stride=10,
        frame_stride=100,
    )

    h5_file = tmp_path / "run.h5"
    save_trajectory_h5(h5_file, state, traj, dt, 10_000, 50_000, 1)
    assert h5_file.exists()

    with h5py.File(h5_file, "r") as f:
        assert f["total_energy"].shape == (5001,)
        assert f["positions"].shape == (501, 250, 3)

    e0 = traj.total_energy[0]
    rel_drift = float(np.max(np.abs(traj.total_energy - e0)) / abs(e0))
    max_momentum = float(np.max(np.linalg.norm(traj.momentum, axis=1)))
    t_i_after_eq = float(traj.temp_i[0])
    t_e_after_eq = float(traj.temp_e[0])
    rel_temp_diff_eq = abs(t_e_after_eq - t_i_after_eq) / state.k_t
    t_i_mean = float(np.mean(traj.temp_i))
    t_e_mean = float(np.mean(traj.temp_e))
    mean_virial = float(np.mean(traj.virial))
    compressibility = 1.0 + mean_virial / (6.0 * state.n_pairs * state.k_t)
    mean_u_over_2nkt = float(np.mean(traj.u_therm)) / (2.0 * state.n_pairs * state.k_t)

    print(
        f"\n[MD Summary Gamma=0.5, r_s=1.0]"
        f"\n  rel_energy_drift = {rel_drift:.6e}"
        f"\n  max_momentum     = {max_momentum:.6e}"
        f"\n  T_i(0), T_e(0)   = {t_i_after_eq:.5f}, {t_e_after_eq:.5f} (target {state.k_t:.5f})"
        f"\n  <T_i>, <T_e>     = {t_i_mean:.5f}, {t_e_mean:.5f}"
        f"\n  PV / (2 N k_B T) = {compressibility:.4f} (Table II MD: 0.89 +/- 0.03)"
        f"\n  U / (2 N k_B T)  = {mean_u_over_2nkt:.4f} (Table II HNC(1): -0.243)"
    )

    # Criterion 1: Relative total energy drift in NVE < 3e-3 over 50,000 steps
    assert rel_drift < 3e-3

    # Criterion 2: Total linear momentum < 1e-8
    assert max_momentum < 1e-8

    # Criterion 3: Electron and ion temperatures after equilibration differ by < 5%
    assert rel_temp_diff_eq < 0.05
    assert abs(t_e_mean - t_i_mean) / state.k_t < 0.05

    # Criterion 4: Compressibility factor PV / (2 N k_B T) = 0.89 +/- 0.03 (Table II)
    assert abs(compressibility - 0.89) <= 0.03
