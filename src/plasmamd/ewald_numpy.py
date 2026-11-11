"""CPU/NumPy reference Ewald summation module for point charges in a cubic box."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import erfc

if TYPE_CHECKING:
    from numpy.typing import NDArray

_DEFAULT_E_SQ = 1.0 / 3.0


def tune_ewald(
    box: float,
    tol: float = 1e-6,
    alpha: float | None = None,
) -> tuple[float, float, float]:
    """Select optimized Ewald parameters (alpha, r_cut, k_max) for target tolerance.

    Parameters
    ----------
    box : float
        Cubic box side length L [a].
    tol : float, optional
        Target relative energy error tolerance [dimensionless].
    alpha : float, optional
        Ewald screening parameter [1/a]. If None, estimated automatically.

    Returns
    -------
    alpha : float
        Ewald screening parameter [1/a].
    r_cut : float
        Real-space cutoff radius (< 0.5 * box) [a].
    k_max : float
        Reciprocal-space wavenumber cutoff magnitude [1/a].
    """
    if tol <= 0.0 or tol >= 1.0:
        raise ValueError(f"Tolerance must be in (0, 1), got {tol}.")
    r_cut = 0.49 * box
    s = math.sqrt(-math.log(tol)) + 0.25
    if alpha is None:
        alpha = s / r_cut
    k_max = 2.0 * alpha * s
    return float(alpha), float(r_cut), float(k_max)


def ewald_energy(
    pos: NDArray[np.float64],
    charge: NDArray[np.float64],
    box: float,
    alpha: float,
    r_cut: float,
    k_max: float,
    e_sq: float = _DEFAULT_E_SQ,
) -> float:
    """Compute total electrostatic potential energy using vectorized NumPy Ewald summation.

    Parameters
    ----------
    pos : np.ndarray, shape (N, 3), dtype float64
        Particle coordinates in reduced units [a].
    charge : np.ndarray, shape (N,), dtype float64
        Particle charge numbers q_i [e].
    box : float
        Cubic simulation box side length L [a].
    alpha : float
        Ewald screening parameter [1/a].
    r_cut : float
        Real-space cutoff radius [a].
    k_max : float
        Reciprocal-space cutoff magnitude [1/a].
    e_sq : float, optional
        Squared elementary charge e^2 [m_e a^3 omega_pe^2], default 1/3.

    Returns
    -------
    u_total : float
        Total electrostatic energy U in reduced units [m_e a^2 omega_pe^2].
    """
    pos_c = np.ascontiguousarray(pos, dtype=np.float64)
    q_c = np.ascontiguousarray(charge, dtype=np.float64)
    n_particles = pos_c.shape[0]
    if n_particles == 0:
        return 0.0

    inv_box = 1.0 / box
    volume = box * box * box

    # 1. Real-space sum (O(N^2) vectorization via numpy broadcasting, no Python loops)
    diff = pos_c[:, None, :] - pos_c[None, :, :]
    diff -= box * np.round(diff * inv_box)
    dist = np.linalg.norm(diff, axis=-1)

    mask = (dist > 0.0) & (dist < r_cut)
    if np.any(mask):
        r_valid = dist[mask]
        q_prod = (q_c[:, None] * q_c[None, :])[mask]
        erfc_val = erfc(alpha * r_valid)
        u_real = 0.5 * e_sq * float(np.sum(q_prod * erfc_val / r_valid))
    else:
        u_real = 0.0

    # 2. Reciprocal-space sum (O(N * N_k) vectorization, no Python loops over particles)
    dk = 2.0 * math.pi / box
    m_max = int(math.ceil(k_max / dk))
    mx = np.arange(-m_max, m_max + 1, dtype=np.float64)
    my = np.arange(-m_max, m_max + 1, dtype=np.float64)
    mz = np.arange(-m_max, m_max + 1, dtype=np.float64)

    kx, ky, kz = np.meshgrid(mx * dk, my * dk, mz * dk, indexing="ij")
    k_vecs = np.column_stack([kx.ravel(), ky.ravel(), kz.ravel()])
    k_sq = np.sum(k_vecs * k_vecs, axis=1)

    # Filter 0 < |k| <= k_max and enforce unique hemisphere
    valid_k = (k_sq > 0.0) & (k_sq <= k_max * k_max)
    vx = k_vecs[:, 0]
    vy = k_vecs[:, 1]
    vz = k_vecs[:, 2]
    hemi = (vx > 0.0) | ((vx == 0.0) & (vy > 0.0)) | ((vx == 0.0) & (vy == 0.0) & (vz > 0.0))
    k_sel = k_vecs[valid_k & hemi]
    k_sel_sq = k_sq[valid_k & hemi]

    if len(k_sel) > 0:
        kr = np.dot(k_sel, pos_c.T)
        c_sum = np.dot(q_c, np.cos(kr).T)
        s_sum = np.dot(q_c, np.sin(kr).T)
        rho_sq = c_sum * c_sum + s_sum * s_sum

        k_weights = (2.0 * math.pi / volume) * np.exp(-k_sel_sq / (4.0 * alpha * alpha)) / k_sel_sq
        u_recip = e_sq * float(np.sum(2.0 * k_weights * rho_sq))
    else:
        u_recip = 0.0

    # 3. Self-interaction term
    u_self = -e_sq * (alpha / math.sqrt(math.pi)) * float(np.sum(q_c * q_c))

    # 4. Uniform background neutralizing term
    q_total = float(np.sum(q_c))
    u_bg = -e_sq * (math.pi / (2.0 * alpha * alpha * volume)) * (q_total * q_total)

    return float(u_real + u_recip + u_self + u_bg)
