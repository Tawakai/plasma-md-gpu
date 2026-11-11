"""Tests for GPU short-range Ewald + Deutsch pair forces with 256-shared-memory tiling (T1)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import erfc

from plasmamd.gpu.pair_forces import pair_forces, pair_forces_cpu


def _sample_lam(gamma: float = 0.5, r_s: float = 1.0, mass_ratio: float = 1836.0) -> np.ndarray:
    """Compute 2x2 Deutsch thermal wavelength matrix [a] for H plasma (Eq. 10, 15)."""
    mu_ii = 0.5 * mass_ratio
    mu_ie = mass_ratio / (mass_ratio + 1.0)
    mu_ee = 0.5
    factor = gamma / (2.0 * math.pi * r_s)
    lam_ii = math.sqrt(factor / mu_ii)
    lam_ie = math.sqrt(factor / mu_ie)
    lam_ee = math.sqrt(factor / mu_ee)
    return np.array([[lam_ii, lam_ie], [lam_ie, lam_ee]], dtype=np.float64)


def test_total_force_conservation_and_bit_reproducibility() -> None:
    """Criterion 1: sum of forces < 1e-12 and bit-for-bit reproducibility across runs."""
    rng = np.random.default_rng(42)
    n = 64
    box = 10.0
    alpha = 0.35
    r_cut = 4.5

    pos = rng.uniform(0.0, box, size=(n, 3)).astype(np.float64)
    species = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int8)
    charge = np.where(species == 0, 1.0, -1.0).astype(np.float64)
    lam = _sample_lam()

    f1, u1, w1 = pair_forces(pos, species, charge, lam, box, alpha, r_cut)
    f2, u2, w2 = pair_forces(pos, species, charge, lam, box, alpha, r_cut)

    total_force = np.sum(f1, axis=0)
    assert np.linalg.norm(total_force) < 1e-12

    assert np.array_equal(f1, f2)
    assert u1 == u2
    assert w1 == w2


def test_force_matches_finite_difference_energy() -> None:
    """Criterion 2: force on each particle matches central finite difference of U, rtol 1e-6."""
    rng = np.random.default_rng(123)
    n = 16
    box = 8.0
    alpha = 0.4
    r_cut = 3.8

    pos = rng.uniform(0.0, box, size=(n, 3)).astype(np.float64)
    species = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int8)
    charge = np.where(species == 0, 1.0, -1.0).astype(np.float64)
    lam = _sample_lam()

    forces, _, _ = pair_forces(pos, species, charge, lam, box, alpha, r_cut)

    eps = 1e-6
    fd_forces = np.zeros_like(forces)
    for i in range(n):
        for dim in range(3):
            pos_plus = pos.copy()
            pos_minus = pos.copy()
            pos_plus[i, dim] += eps
            pos_minus[i, dim] -= eps
            _, u_plus, _ = pair_forces(pos_plus, species, charge, lam, box, alpha, r_cut)
            _, u_minus, _ = pair_forces(pos_minus, species, charge, lam, box, alpha, r_cut)
            fd_forces[i, dim] = -(u_plus - u_minus) / (2.0 * eps)

    np.testing.assert_allclose(forces, fd_forces, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize(
    ("s_a", "s_b", "q_a", "q_b", "r"),
    [
        (0, 0, 1.0, 1.0, 0.8),
        (0, 1, 1.0, -1.0, 0.25),
        (1, 1, -1.0, -1.0, 1.4),
    ],
)
def test_two_particles_analytical_energy_and_virial(
    s_a: int, s_b: int, q_a: float, q_b: float, r: float
) -> None:
    """Criterion 3: for two particles at distance r, energy equals v_ab(r) analytically."""
    box = 10.0
    alpha = 0.5
    r_cut = 4.0
    lam = _sample_lam()
    e_sq = 1.0 / 3.0

    pos = np.array([[1.0, 2.0, 3.0], [1.0 + r, 2.0, 3.0]], dtype=np.float64)
    species = np.array([s_a, s_b], dtype=np.int8)
    charge = np.array([q_a, q_b], dtype=np.float64)

    forces, u_gpu, w_gpu = pair_forces(pos, species, charge, lam, box, alpha, r_cut)

    lam_ab = lam[s_a, s_b]
    v_exact = q_a * q_b * e_sq * (erfc(alpha * r) / r - math.exp(-r / lam_ab) / r)
    dv_dr = (
        q_a
        * q_b
        * e_sq
        * (
            (1.0 + r / lam_ab) * math.exp(-r / lam_ab)
            - erfc(alpha * r)
            - (2.0 * alpha * r / math.sqrt(math.pi)) * math.exp(-((alpha * r) ** 2))
        )
        / (r * r)
    )
    w_exact = -r * dv_dr

    np.testing.assert_allclose(u_gpu, v_exact, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(w_gpu, w_exact, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(forces[0, 0], dv_dr, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(forces[1, 0], -dv_dr, rtol=1e-14, atol=1e-15)


def test_minimum_image_periodic_boundary() -> None:
    """Criterion 4: particle near box boundary interacts with nearest periodic image."""
    box = 10.0
    alpha = 0.4
    r_cut = 4.0
    lam = _sample_lam()
    r_true = 0.6

    pos_boundary = np.array([[0.2, 5.0, 5.0], [box - 0.4, 5.0, 5.0]], dtype=np.float64)
    pos_direct = np.array([[5.0 + r_true, 5.0, 5.0], [5.0, 5.0, 5.0]], dtype=np.float64)

    species = np.array([0, 1], dtype=np.int8)
    charge = np.array([1.0, -1.0], dtype=np.float64)

    f_bnd, u_bnd, w_bnd = pair_forces(pos_boundary, species, charge, lam, box, alpha, r_cut)
    f_dir, u_dir, w_dir = pair_forces(pos_direct, species, charge, lam, box, alpha, r_cut)

    np.testing.assert_allclose(u_bnd, u_dir, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(w_bnd, w_dir, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(f_bnd, f_dir, rtol=1e-14, atol=1e-15)


def test_gpu_matches_cpu_n1000() -> None:
    """Criterion 5: GPU and CPU (numba.njit) forces match within atol=1e-12 for N = 1000."""
    rng = np.random.default_rng(2026)
    n = 1000
    n_pairs = n // 2
    box = ((4.0 * math.pi / 3.0) * float(n_pairs)) ** (1.0 / 3.0)
    alpha = 0.45
    r_cut = 0.48 * box

    pos = rng.uniform(0.0, box, size=(n, 3)).astype(np.float64)
    species = np.array([0] * n_pairs + [1] * n_pairs, dtype=np.int8)
    charge = np.where(species == 0, 1.0, -1.0).astype(np.float64)
    lam = _sample_lam()

    f_cpu, u_cpu, w_cpu = pair_forces_cpu(pos, species, charge, lam, box, alpha, r_cut)
    f_gpu, u_gpu, w_gpu = pair_forces(pos, species, charge, lam, box, alpha, r_cut)

    np.testing.assert_allclose(f_gpu, f_cpu, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(u_gpu, u_cpu, rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(w_gpu, w_cpu, rtol=1e-13, atol=1e-12)
