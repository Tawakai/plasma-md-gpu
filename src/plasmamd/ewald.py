"""Ewald summation parameters, reciprocal-space GPU/CPU kernels, and Deutsch T-derivative."""

from __future__ import annotations

import math

import numpy as np
from numba import cuda, float64, int8
from numpy.typing import NDArray

from plasmamd.gpu.pair_forces import pair_forces

_THREADS_PER_BLOCK = 128


def tune_ewald_parameters(
    box: float,
    tol: float = 1e-6,
) -> tuple[float, float, int]:
    """Select Ewald parameters (alpha, r_cut, k_max_int) for relative energy error < tol.

    Parameters
    ----------
    box : float
        Cubic box side length L [a].
    tol : float, optional
        Target relative energy error tolerance [dimensionless].

    Returns
    -------
    alpha : float
        Ewald screening parameter [1/a].
    r_cut : float
        Real-space cutoff radius (< 0.5 * box) [a].
    k_max_int : int
        Maximum integer wave-vector index m_max such that |m| <= m_max [dimensionless].
    """
    if tol <= 0.0 or tol >= 1.0:
        raise ValueError(f"Tolerance must be in (0, 1), got {tol}.")
    r_cut = 0.48 * box
    s = math.sqrt(-math.log(tol)) + 0.25
    alpha = s / r_cut
    k_max_int = max(4, int(math.ceil(s * alpha * box / math.pi)))
    return float(alpha), float(r_cut), int(k_max_int)


def generate_k_vectors(
    box: float,
    k_max_int: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate half-space reciprocal lattice vectors k != 0 and squared magnitudes.

    Parameters
    ----------
    box : float
        Cubic box side length L [a].
    k_max_int : int
        Maximum integer index along each axis [dimensionless].

    Returns
    -------
    k_vecs : np.ndarray, shape (M, 3), dtype float64
        Reciprocal wave vectors k = (2*pi/L) * m in units of [1/a].
    k_sq : np.ndarray, shape (M,), dtype float64
        Squared magnitudes |k|^2 in units of [1/a^2].
    """
    dk = 2.0 * math.pi / box
    m_max_sq = k_max_int * k_max_int
    vecs: list[tuple[float, float, float]] = []

    for mx in range(0, k_max_int + 1):
        for my in range(-k_max_int, k_max_int + 1):
            for mz in range(-k_max_int, k_max_int + 1):
                if mx == 0 and my < 0:
                    continue
                if mx == 0 and my == 0 and mz <= 0:
                    continue
                msq = mx * mx + my * my + mz * mz
                if 0 < msq <= m_max_sq:
                    vecs.append((mx * dk, my * dk, mz * dk))

    k_vecs = np.array(vecs, dtype=np.float64)
    k_sq = np.sum(k_vecs * k_vecs, axis=1)
    return k_vecs, k_sq


@cuda.jit(
    (
        float64[:, :],
        float64[:],
        float64[:, :],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
    ),
    fastmath=False,
)
def _ewald_structure_factor_kernel(
    pos: NDArray[np.float64],
    charge: NDArray[np.float64],
    k_vecs: NDArray[np.float64],
    k_weights: NDArray[np.float64],
    cos_sum: NDArray[np.float64],
    sin_sum: NDArray[np.float64],
    u_per_k: NDArray[np.float64],
) -> None:
    """One-thread-per-k-vector kernel computing C(k), S(k), and energy per k."""
    k_idx = cuda.grid(1)
    n_k = k_vecs.shape[0]
    if k_idx >= n_k:
        return

    kx = k_vecs[k_idx, 0]
    ky = k_vecs[k_idx, 1]
    kz = k_vecs[k_idx, 2]
    n_particles = pos.shape[0]

    c_val = 0.0
    s_val = 0.0
    for j in range(n_particles):
        kr = kx * pos[j, 0] + ky * pos[j, 1] + kz * pos[j, 2]
        qj = charge[j]
        c_val += qj * math.cos(kr)
        s_val += qj * math.sin(kr)

    cos_sum[k_idx] = c_val
    sin_sum[k_idx] = s_val
    u_per_k[k_idx] = k_weights[k_idx] * (c_val * c_val + s_val * s_val)


@cuda.jit(
    (
        float64[:, :],
        float64[:],
        float64[:, :],
        float64[:],
        float64[:],
        float64[:],
        float64[:, :],
    ),
    fastmath=False,
)
def _ewald_recip_force_kernel(
    pos: NDArray[np.float64],
    charge: NDArray[np.float64],
    k_vecs: NDArray[np.float64],
    k_weights: NDArray[np.float64],
    cos_sum: NDArray[np.float64],
    sin_sum: NDArray[np.float64],
    forces_out: NDArray[np.float64],
) -> None:
    """One-thread-per-particle kernel computing reciprocal-space Ewald forces."""
    i = cuda.grid(1)
    n_particles = pos.shape[0]
    if i >= n_particles:
        return

    xi = pos[i, 0]
    yi = pos[i, 1]
    zi = pos[i, 2]
    qi = charge[i]
    n_k = k_vecs.shape[0]

    fx = 0.0
    fy = 0.0
    fz = 0.0
    for k_idx in range(n_k):
        kx = k_vecs[k_idx, 0]
        ky = k_vecs[k_idx, 1]
        kz = k_vecs[k_idx, 2]
        kr = kx * xi + ky * yi + kz * zi
        coeff = (
            2.0
            * qi
            * k_weights[k_idx]
            * (math.sin(kr) * cos_sum[k_idx] - math.cos(kr) * sin_sum[k_idx])
        )
        fx += coeff * kx
        fy += coeff * ky
        fz += coeff * kz

    forces_out[i, 0] = fx
    forces_out[i, 1] = fy
    forces_out[i, 2] = fz


@cuda.jit(
    (float64[:], float64[:]),
    fastmath=False,
)
def _deterministic_reduce_1d_kernel(
    arr: NDArray[np.float64],
    out_val: NDArray[np.float64],
) -> None:
    """Single-block deterministic float64 sum reduction on GPU."""
    s_data = cuda.shared.array(_THREADS_PER_BLOCK, dtype=float64)
    tid = cuda.threadIdx.x
    n = arr.shape[0]

    acc = 0.0
    idx = tid
    while idx < n:
        acc += arr[idx]
        idx += _THREADS_PER_BLOCK

    s_data[tid] = acc
    cuda.syncthreads()

    stride = _THREADS_PER_BLOCK // 2
    while stride > 0:
        if tid < stride:
            s_data[tid] += s_data[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        out_val[0] = s_data[0]


@cuda.jit(
    (
        float64[:, :],
        int8[:],
        float64[:],
        float64[:, :],
        float64,
        float64,
        float64,
        float64[:],
    ),
    fastmath=False,
)
def _deutsch_temp_deriv_kernel(
    pos: NDArray[np.float64],
    species: NDArray[np.int8],
    charge: NDArray[np.float64],
    lam: NDArray[np.float64],
    box: float,
    r_cut: float,
    e_sq: float,
    utd_per_particle: NDArray[np.float64],
) -> None:
    """Compute -T * dv_ab/dT contribution per particle [Eq. (19) of Hansen & McDonald]."""
    i = cuda.grid(1)
    n_particles = pos.shape[0]
    if i >= n_particles:
        return

    xi = pos[i, 0]
    yi = pos[i, 1]
    zi = pos[i, 2]
    si = species[i]
    qi = charge[i]
    inv_box = 1.0 / box
    r_cut_sq = r_cut * r_cut

    acc = 0.0
    for j in range(n_particles):
        if j == i:
            continue
        dx = xi - pos[j, 0]
        dy = yi - pos[j, 1]
        dz = zi - pos[j, 2]
        dx -= box * round(dx * inv_box)
        dy -= box * round(dy * inv_box)
        dz -= box * round(dz * inv_box)
        r2 = dx * dx + dy * dy + dz * dz
        if 0.0 < r2 < r_cut_sq:
            r = math.sqrt(r2)
            sj = species[j]
            qj = charge[j]
            lam_ij = lam[si, sj]
            acc += -0.5 * qi * qj * e_sq * math.exp(-r / lam_ij) / lam_ij

    utd_per_particle[i] = acc


def ewald_reciprocal(
    pos: NDArray[np.float64],
    charge: NDArray[np.float64],
    box: float,
    alpha: float,
    k_max_int: int,
    e_sq: float = 1.0 / 3.0,
    k_vecs: NDArray[np.float64] | None = None,
    k_sq: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], float, float]:
    """Compute Ewald reciprocal-space forces, energy (including self-energy), and virial.

    Parameters
    ----------
    pos : np.ndarray, shape (N, 3), dtype float64
        Particle coordinates [a].
    charge : np.ndarray, shape (N,), dtype float64
        Particle charge numbers q_j in units of [e].
    box : float
        Cubic box side length L [a].
    alpha : float
        Ewald screening parameter [1/a].
    k_max_int : int
        Reciprocal lattice spherical cutoff index m_max [dimensionless].
    e_sq : float, optional
        Squared elementary charge e^2 [m_e a^3 omega_pe^2].
    k_vecs : np.ndarray, optional
        Precomputed half-space k-vectors of shape (M, 3) [1/a].
    k_sq : np.ndarray, optional
        Precomputed |k|^2 of shape (M,) [1/a^2].

    Returns
    -------
    forces_k : np.ndarray, shape (N, 3), dtype float64
        Reciprocal-space forces on each particle [m_e a omega_pe^2].
    u_k : float
        Total reciprocal-space + self potential energy [m_e a^2 omega_pe^2].
    virial_k : float
        Reciprocal-space contribution to the virial (= u_k) [m_e a^2 omega_pe^2].
    """
    pos_c = np.ascontiguousarray(pos, dtype=np.float64)
    charge_c = np.ascontiguousarray(charge, dtype=np.float64)
    n_particles = pos_c.shape[0]
    if n_particles == 0:
        return np.zeros((0, 3), dtype=np.float64), 0.0, 0.0

    if k_vecs is None or k_sq is None:
        k_vecs, k_sq = generate_k_vectors(box, k_max_int)

    n_k = k_vecs.shape[0]
    volume = box * box * box
    k_weights = np.ascontiguousarray(
        2.0 * e_sq * (2.0 * math.pi / volume) * np.exp(-k_sq / (4.0 * alpha * alpha)) / k_sq,
        dtype=np.float64,
    )

    d_pos = cuda.to_device(pos_c)
    d_charge = cuda.to_device(charge_c)
    d_k_vecs = cuda.to_device(k_vecs)
    d_k_weights = cuda.to_device(k_weights)

    d_cos = cuda.device_array(n_k, dtype=np.float64)
    d_sin = cuda.device_array(n_k, dtype=np.float64)
    d_u_per_k = cuda.device_array(n_k, dtype=np.float64)
    d_u_rec = cuda.device_array(1, dtype=np.float64)
    d_forces = cuda.device_array((n_particles, 3), dtype=np.float64)

    blocks_k = (n_k + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
    _ewald_structure_factor_kernel[blocks_k, _THREADS_PER_BLOCK](
        d_pos,
        d_charge,
        d_k_vecs,
        d_k_weights,
        d_cos,
        d_sin,
        d_u_per_k,
    )
    _deterministic_reduce_1d_kernel[1, _THREADS_PER_BLOCK](d_u_per_k, d_u_rec)

    blocks_p = (n_particles + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
    _ewald_recip_force_kernel[blocks_p, _THREADS_PER_BLOCK](
        d_pos, d_charge, d_k_vecs, d_k_weights, d_cos, d_sin, d_forces
    )

    u_rec = float(d_u_rec.copy_to_host()[0])
    u_self = -e_sq * (alpha / math.sqrt(math.pi)) * float(np.sum(charge_c * charge_c))
    u_k = u_rec + u_self
    forces_k = d_forces.copy_to_host()
    return forces_k, u_k, u_k


def deutsch_temperature_derivative_energy(
    pos: NDArray[np.float64],
    species: NDArray[np.int8],
    charge: NDArray[np.float64],
    lam: NDArray[np.float64],
    box: float,
    r_cut: float,
    e_sq: float = 1.0 / 3.0,
) -> float:
    """Compute the -T * dv_ab/dT contribution to excess internal energy [Eq. (19)]."""
    pos_c = np.ascontiguousarray(pos, dtype=np.float64)
    species_c = np.ascontiguousarray(species, dtype=np.int8)
    charge_c = np.ascontiguousarray(charge, dtype=np.float64)
    lam_c = np.ascontiguousarray(lam, dtype=np.float64)
    n_particles = pos_c.shape[0]
    if n_particles == 0:
        return 0.0

    d_pos = cuda.to_device(pos_c)
    d_species = cuda.to_device(species_c)
    d_charge = cuda.to_device(charge_c)
    d_lam = cuda.to_device(lam_c)
    d_utd_part = cuda.device_array(n_particles, dtype=np.float64)
    d_utd_sum = cuda.device_array(1, dtype=np.float64)

    blocks = (n_particles + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
    _deutsch_temp_deriv_kernel[blocks, _THREADS_PER_BLOCK](
        d_pos, d_species, d_charge, d_lam, float(box), float(r_cut), float(e_sq), d_utd_part
    )
    _deterministic_reduce_1d_kernel[1, _THREADS_PER_BLOCK](d_utd_part, d_utd_sum)
    return 0.5 * float(d_utd_sum.copy_to_host()[0])


def total_forces_energy_virial(
    pos: NDArray[np.float64],
    species: NDArray[np.int8],
    charge: NDArray[np.float64],
    lam: NDArray[np.float64],
    box: float,
    alpha: float,
    r_cut: float,
    k_max_int: int,
    e_sq: float = 1.0 / 3.0,
    k_vecs: NDArray[np.float64] | None = None,
    k_sq: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], float, float, float]:
    """Compute total forces, configurational energy, thermal excess energy, and total virial."""
    f_sr, u_sr, w_sr = pair_forces(pos, species, charge, lam, box, alpha, r_cut, e_sq=e_sq)
    f_k, u_k, w_k = ewald_reciprocal(
        pos, charge, box, alpha, k_max_int, e_sq=e_sq, k_vecs=k_vecs, k_sq=k_sq
    )
    u_td = deutsch_temperature_derivative_energy(pos, species, charge, lam, box, r_cut, e_sq=e_sq)
    u_cfg = u_sr + u_k
    u_therm = u_cfg + u_td
    virial = w_sr + w_k
    return f_sr + f_k, u_cfg, u_therm, virial
