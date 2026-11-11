"""Verification tests for CPU/NumPy reference Ewald energy calculation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from plasmamd.ewald_numpy import ewald_energy

# In reduced units (length a = 1, mass m_e = 1, time 1/omega_pe = 1):
# omega_pe^2 = 4 * pi * n * e^2 / m_e = 3 * e^2 = 1  =>  e^2 = 1/3.
E_SQ: float = 1.0 / 3.0


def _make_bcc_lattice(n_cells_per_side: int, box: float) -> NDArray[np.float64]:
    """Generate a body-centered cubic (BCC) lattice of 2 * M^3 points in [0, box)^3.

    Parameters
    ----------
    n_cells_per_side : int
        Number of unit cells along each cubic axis M [dimensionless].
    box : float
        Simulation cubic box side length L [a].

    Returns
    -------
    pos : np.ndarray, shape (2 * M^3, 3), dtype float64
        Particle coordinates in reduced units [a].
    """
    cell_size = box / float(n_cells_per_side)
    grid = np.arange(n_cells_per_side, dtype=np.float64)
    gx, gy, gz = np.meshgrid(grid, grid, grid, indexing="ij")
    corners = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]) * cell_size
    centers = corners + 0.5 * cell_size
    return np.vstack([corners, centers]).astype(np.float64)


def test_bcc_madelung_energy_brush_sahlin_teller() -> None:
    """Check BCC lattice energy per particle against Brush, Sahlin & Teller (1966):

    U / N = -0.895929 e^2 / a.
    """
    n_cells = 4
    n_particles = 2 * (n_cells**3)  # N = 128 unit charges
    assert n_particles == 128

    # Volume V = (4*pi/3) * N * a^3 so that ion-sphere radius a = 1 [Eq. (1)]
    box = ((4.0 * math.pi / 3.0) * float(n_particles)) ** (1.0 / 3.0)
    pos = _make_bcc_lattice(n_cells, box)
    charge = np.ones(n_particles, dtype=np.float64)

    # Choose matched Ewald parameters inside minimum-image limit (r_cut < L / 2):
    # alpha * r_cut = 6.0  => erfc(6.0) ~ 2.2e-17 (real-space truncation negligible)
    # k_max / (2 * alpha) = 6.0 => exp(-36) ~ 2.3e-16 (reciprocal truncation negligible)
    r_cut = 0.49 * box
    alpha = 6.0 / r_cut
    k_max = 12.0 * alpha

    u_total = ewald_energy(pos, charge, box, alpha, r_cut, k_max)
    u_per_particle = u_total / float(n_particles)
    expected_u_per_particle = -0.895929 * E_SQ

    # Tolerance justification:
    # The literature constant -0.895929 from Brush, Sahlin & Teller (1966) is given to
    # 6 significant digits (exact value is -0.895929255682...), which introduces a
    # rounding floor of ~3e-7; rtol=1e-5 safely covers the 6-digit reference precision
    # while strictly catching any missing self-energy, k=0, or background (-pi/(2*alpha^2*V)) terms.
    np.testing.assert_allclose(
        u_per_particle,
        expected_u_per_particle,
        rtol=1e-5,
        atol=0.0,
    )


@pytest.mark.parametrize("alpha_factor", [4.0, 5.0, 6.0, 7.0, 8.0])
def test_ewald_energy_alpha_invariance(alpha_factor: float) -> None:
    """Check that ewald_energy is invariant wrt alpha in [4/L, 8/L] for matched (r_c, k_max)."""
    rng = np.random.default_rng(2026)
    n_pairs = 32
    n_particles = 2 * n_pairs
    box = ((4.0 * math.pi / 3.0) * float(n_pairs)) ** (1.0 / 3.0)

    charge = np.array([1.0] * n_pairs + [-1.0] * n_pairs, dtype=np.float64)

    # Place particles within a central sub-cube of side 0.25 * L so all pairwise distances
    # satisfy r_ij <= sqrt(3)*0.25*L = 0.433*L < r_cut = 0.49*L. This guarantees zero
    # real-space cutoff error under minimum-image convention even at alpha = 4 / L,
    # while k_max = 2 * alpha * 4.2 guarantees reciprocal truncation exp(-17.64) ~ 2.2e-8.
    pos_cluster = 0.35 * box + rng.uniform(0.0, 0.25 * box, size=(n_particles, 3)).astype(
        np.float64
    )

    r_cut_ref = 0.49 * box
    alpha_ref = 6.0 / box
    k_max_ref = 2.0 * alpha_ref * 4.2
    u_ref = ewald_energy(pos_cluster, charge, box, alpha_ref, r_cut_ref, k_max_ref)

    alpha = alpha_factor / box
    r_cut = 0.49 * box
    k_max = 2.0 * alpha * 4.2
    u_test = ewald_energy(pos_cluster, charge, box, alpha, r_cut, k_max)

    # Tolerance justification:
    # Real-space pairs are all strictly inside r_cut = 0.49*L, and reciprocal cutoff
    # k_max / (2*alpha) = 4.2 bounds the k-space tail to exp(-17.64) = 2.18e-8 << rtol=1e-6.
    np.testing.assert_allclose(u_test, u_ref, rtol=1e-6, atol=0.0)


@pytest.mark.parametrize("r_dist", [0.5, 1.0, 1.5])
def test_two_opposite_charges_large_box_limit(r_dist: float) -> None:
    """Check that for two charges +1 and -1 at distance r, Ewald energy -> -e^2 / r as L -> inf."""
    charge = np.array([1.0, -1.0], dtype=np.float64)
    u_exact_coulomb = -E_SQ / r_dist

    # Place pair near the center of a large cubic cell of side L = 200 * a (r / L <= 0.0075).
    box = 200.0
    center = 0.5 * box
    pos = np.array(
        [
            [center, center, center],
            [center + r_dist, center, center],
        ],
        dtype=np.float64,
    )

    r_cut = 0.49 * box
    alpha = 8.0 / box
    k_max = 2.0 * alpha * 4.5

    u_ewald = ewald_energy(pos, charge, box, alpha, r_cut, k_max)

    # Tolerance justification:
    # For a neutral dipole of separation r in a periodic cubic box of side L, the real-space
    # pair at distance r <= 1.5 is well inside r_cut = 98.0, and the leading periodic-image
    # interaction is the dipole-lattice term of order O(e^2 * r^2 / L^3) ~ (1/3)*(2.25 / 8e6)
    # ~ 1e-7 (relative error ~ 4e-7). Setting rtol=1e-5 and atol=1e-6 rigorously verifies
    # convergence to the bare Coulomb limit -e^2 / r while accounting for finite-L image effects.
    np.testing.assert_allclose(u_ewald, u_exact_coulomb, rtol=1e-5, atol=1e-6)
