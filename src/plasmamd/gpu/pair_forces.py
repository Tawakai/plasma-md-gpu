"""CPU (numba.njit) and tiled shared-memory GPU (numba.cuda.jit) short-range pair forces."""

from __future__ import annotations

import math

import numpy as np
from numba import cuda, float64, int8, njit
from numpy.typing import NDArray

_TILE_SIZE = 256
_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)
_DEFAULT_E_SQ = 1.0 / 3.0


@njit(fastmath=False)
def _pair_forces_cpu_kernel(
    pos: NDArray[np.float64],
    species: NDArray[np.int8],
    charge: NDArray[np.float64],
    lam: NDArray[np.float64],
    box: float,
    alpha: float,
    r_cut: float,
    e_sq: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Reference CPU numba.njit kernel computing per-particle forces, energy, and virial."""
    n_particles = pos.shape[0]
    forces = np.zeros((n_particles, 3), dtype=np.float64)
    u_arr = np.zeros(n_particles, dtype=np.float64)
    w_arr = np.zeros(n_particles, dtype=np.float64)

    inv_box = 1.0 / box
    r_cut_sq = r_cut * r_cut

    for i in range(n_particles):
        xi = pos[i, 0]
        yi = pos[i, 1]
        zi = pos[i, 2]
        si = species[i]
        qi = charge[i]

        fx = 0.0
        fy = 0.0
        fz = 0.0
        ui = 0.0
        wi = 0.0

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
                inv_r = 1.0 / r
                inv_r2 = inv_r * inv_r
                inv_r3 = inv_r2 * inv_r

                sj = species[j]
                qj = charge[j]
                lam_ij = lam[si, sj]
                inv_lam = 1.0 / lam_ij

                ar = alpha * r
                rl = r * inv_lam

                erfc_ar = math.erfc(ar)
                exp_ar2 = math.exp(-ar * ar)
                exp_rl = math.exp(-rl)

                pref = qi * qj * e_sq
                v_ij = pref * (erfc_ar - exp_rl) * inv_r
                f_scalar = (
                    pref
                    * (erfc_ar + _TWO_OVER_SQRT_PI * ar * exp_ar2 - (1.0 + rl) * exp_rl)
                    * inv_r3
                )

                fx += f_scalar * dx
                fy += f_scalar * dy
                fz += f_scalar * dz
                ui += v_ij
                wi += f_scalar * r2

        forces[i, 0] = fx
        forces[i, 1] = fy
        forces[i, 2] = fz
        u_arr[i] = ui
        w_arr[i] = wi

    return forces, u_arr, w_arr


def pair_forces_cpu(
    pos: NDArray[np.float64],
    species: NDArray[np.int8],
    charge: NDArray[np.float64],
    lam: NDArray[np.float64],
    box: float,
    alpha: float,
    r_cut: float,
    e_sq: float = _DEFAULT_E_SQ,
) -> tuple[NDArray[np.float64], float, float]:
    """Compute short-range Ewald + Deutsch pair forces, energy, and virial on CPU (numba.njit).

    Parameters
    ----------
    pos : np.ndarray, shape (N, 3), dtype float64
        Particle coordinates in reduced units [a].
    species : np.ndarray, shape (N,), dtype int8
        Species index for each particle (0 for "i", 1 for "e") [dimensionless].
    charge : np.ndarray, shape (N,), dtype float64
        Particle charge numbers q_i in units of elementary charge e (±1.0) [dimensionless].
    lam : np.ndarray, shape (2, 2), dtype float64
        Thermal de Broglie wavelength matrix lambda_ab in reduced units [a].
    box : float
        Cubic simulation box side length L in reduced units [a].
    alpha : float
        Ewald screening parameter in reduced units [1/a].
    r_cut : float
        Real-space cutoff radius r_c in reduced units [a].
    e_sq : float, optional
        Squared elementary charge e^2 in reduced units [m_e a^3 omega_pe^2], default is 1/3.

    Returns
    -------
    forces : np.ndarray, shape (N, 3), dtype float64
        Total short-range pair force on each particle in reduced units [m_e a omega_pe^2].
    potential_energy : float
        Total short-range pair potential energy U in reduced units [m_e a^2 omega_pe^2].
    virial : float
        Total short-range pair virial W = sum_{i<j} r_ij . F_ij [m_e a^2 omega_pe^2].
    """
    pos_c = np.ascontiguousarray(pos, dtype=np.float64)
    species_c = np.ascontiguousarray(species, dtype=np.int8)
    charge_c = np.ascontiguousarray(charge, dtype=np.float64)
    lam_c = np.ascontiguousarray(lam, dtype=np.float64)

    n_particles = pos_c.shape[0]
    if n_particles == 0:
        return np.zeros((0, 3), dtype=np.float64), 0.0, 0.0

    forces, u_arr, w_arr = _pair_forces_cpu_kernel(
        pos_c, species_c, charge_c, lam_c, float(box), float(alpha), float(r_cut), float(e_sq)
    )
    potential_energy = 0.5 * float(np.sum(u_arr, dtype=np.float64))
    virial = 0.5 * float(np.sum(w_arr, dtype=np.float64))
    return forces, potential_energy, virial


@cuda.jit(
    (
        float64[:, :],
        int8[:],
        float64[:],
        float64[:, :],
        float64,
        float64,
        float64,
        float64,
        float64,
        float64[:, :],
        float64[:],
        float64[:],
    ),
    fastmath=False,
)
def _pair_forces_kernel(
    pos: NDArray[np.float64],
    species: NDArray[np.int8],
    charge: NDArray[np.float64],
    lam: NDArray[np.float64],
    box: float,
    alpha: float,
    r_cut: float,
    e_sq: float,
    two_over_sqrt_pi: float,
    forces_out: NDArray[np.float64],
    u_per_particle: NDArray[np.float64],
    w_per_particle: NDArray[np.float64],
) -> None:
    """Tiled shared-memory CUDA kernel (256 threads/block) for short-range pair interactions.

    Parameters are in reduced units: length [a], mass [m_e], time [1/omega_pe],
    energy [m_e a^2 omega_pe^2], force [m_e a omega_pe^2].
    """
    sh_pos = cuda.shared.array((_TILE_SIZE, 3), dtype=float64)
    sh_charge = cuda.shared.array(_TILE_SIZE, dtype=float64)
    sh_species = cuda.shared.array(_TILE_SIZE, dtype=int8)

    tid = cuda.threadIdx.x
    i = cuda.grid(1)
    n_particles = pos.shape[0]
    active = i < n_particles

    if active:
        xi = pos[i, 0]
        yi = pos[i, 1]
        zi = pos[i, 2]
        si = species[i]
        qi = charge[i]
    else:
        xi = 0.0
        yi = 0.0
        zi = 0.0
        si = int8(0)
        qi = 0.0

    inv_box = 1.0 / box
    r_cut_sq = r_cut * r_cut

    fx = 0.0
    fy = 0.0
    fz = 0.0
    ui = 0.0
    wi = 0.0

    n_tiles = (n_particles + _TILE_SIZE - 1) // _TILE_SIZE

    for tile_idx in range(n_tiles):
        j_load = tile_idx * _TILE_SIZE + tid
        if j_load < n_particles:
            sh_pos[tid, 0] = pos[j_load, 0]
            sh_pos[tid, 1] = pos[j_load, 1]
            sh_pos[tid, 2] = pos[j_load, 2]
            sh_charge[tid] = charge[j_load]
            sh_species[tid] = species[j_load]
        else:
            sh_pos[tid, 0] = 0.0
            sh_pos[tid, 1] = 0.0
            sh_pos[tid, 2] = 0.0
            sh_charge[tid] = 0.0
            sh_species[tid] = int8(0)

        cuda.syncthreads()

        if active:
            tile_limit = n_particles - tile_idx * _TILE_SIZE
            if tile_limit > _TILE_SIZE:
                tile_limit = _TILE_SIZE

            for k in range(tile_limit):
                j = tile_idx * _TILE_SIZE + k
                if j == i:
                    continue

                dx = xi - sh_pos[k, 0]
                dy = yi - sh_pos[k, 1]
                dz = zi - sh_pos[k, 2]

                dx -= box * round(dx * inv_box)
                dy -= box * round(dy * inv_box)
                dz -= box * round(dz * inv_box)

                r2 = dx * dx + dy * dy + dz * dz
                if 0.0 < r2 < r_cut_sq:
                    r = math.sqrt(r2)
                    inv_r = 1.0 / r
                    inv_r2 = inv_r * inv_r
                    inv_r3 = inv_r2 * inv_r

                    sj = sh_species[k]
                    qj = sh_charge[k]
                    lam_ij = lam[si, sj]
                    inv_lam = 1.0 / lam_ij

                    ar = alpha * r
                    rl = r * inv_lam

                    erfc_ar = math.erfc(ar)
                    exp_ar2 = math.exp(-ar * ar)
                    exp_rl = math.exp(-rl)

                    pref = qi * qj * e_sq

                    v_ij = pref * (erfc_ar - exp_rl) * inv_r
                    f_scalar = (
                        pref
                        * (erfc_ar + two_over_sqrt_pi * ar * exp_ar2 - (1.0 + rl) * exp_rl)
                        * inv_r3
                    )

                    fx += f_scalar * dx
                    fy += f_scalar * dy
                    fz += f_scalar * dz
                    ui += v_ij
                    wi += f_scalar * r2

        cuda.syncthreads()

    if active:
        forces_out[i, 0] = fx
        forces_out[i, 1] = fy
        forces_out[i, 2] = fz
        u_per_particle[i] = ui
        w_per_particle[i] = wi


@cuda.jit(
    (float64[:], float64[:], float64[:]),
    fastmath=False,
)
def _deterministic_reduce_pair_kernel(
    u_arr: NDArray[np.float64],
    w_arr: NDArray[np.float64],
    out_totals: NDArray[np.float64],
) -> None:
    """Single-block deterministic tree/serial reduction in float64 on GPU."""
    s_u = cuda.shared.array(128, dtype=float64)
    s_w = cuda.shared.array(128, dtype=float64)

    tid = cuda.threadIdx.x
    n = u_arr.shape[0]

    acc_u = 0.0
    acc_w = 0.0
    idx = tid
    while idx < n:
        acc_u += u_arr[idx]
        acc_w += w_arr[idx]
        idx += 128

    s_u[tid] = acc_u
    s_w[tid] = acc_w
    cuda.syncthreads()

    stride = 64
    while stride > 0:
        if tid < stride:
            s_u[tid] += s_u[tid + stride]
            s_w[tid] += s_w[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        out_totals[0] = 0.5 * s_u[0]
        out_totals[1] = 0.5 * s_w[0]


def pair_forces(
    pos: NDArray[np.float64],
    species: NDArray[np.int8],
    charge: NDArray[np.float64],
    lam: NDArray[np.float64],
    box: float,
    alpha: float,
    r_cut: float,
    e_sq: float = _DEFAULT_E_SQ,
) -> tuple[NDArray[np.float64], float, float]:
    """Compute short-range Ewald + Deutsch pair forces, energy, and virial on GPU.

    Uses 256-thread shared-memory tiling inside the CUDA kernel and deterministic
    host-side float64 summation for potential energy and virial.

    Parameters
    ----------
    pos : np.ndarray, shape (N, 3), dtype float64
        Particle coordinates in reduced units [a].
    species : np.ndarray, shape (N,), dtype int8
        Species index for each particle (0 for "i", 1 for "e") [dimensionless].
    charge : np.ndarray, shape (N,), dtype float64
        Particle charge numbers q_i in units of elementary charge e (±1.0) [dimensionless].
    lam : np.ndarray, shape (2, 2), dtype float64
        Thermal de Broglie wavelength matrix lambda_ab in reduced units [a].
    box : float
        Cubic simulation box side length L in reduced units [a].
    alpha : float
        Ewald screening parameter in reduced units [1/a].
    r_cut : float
        Real-space cutoff radius r_c in reduced units [a]. Requires r_cut < box / 2
        for strict minimum-image convention.
    e_sq : float, optional
        Squared elementary charge e^2 in reduced units [m_e a^3 omega_pe^2], default is 1/3.

    Returns
    -------
    forces : np.ndarray, shape (N, 3), dtype float64
        Total short-range pair force on each particle in reduced units [m_e a omega_pe^2].
    potential_energy : float
        Total short-range pair potential energy U in reduced units [m_e a^2 omega_pe^2].
    virial : float
        Total short-range pair virial W = sum_{i<j} r_ij . F_ij [m_e a^2 omega_pe^2].
    """
    pos_c = np.ascontiguousarray(pos, dtype=np.float64)
    species_c = np.ascontiguousarray(species, dtype=np.int8)
    charge_c = np.ascontiguousarray(charge, dtype=np.float64)
    lam_c = np.ascontiguousarray(lam, dtype=np.float64)

    n_particles = pos_c.shape[0]
    if pos_c.ndim != 2 or pos_c.shape[1] != 3:
        raise ValueError(f"Expected pos of shape (N, 3), got {pos_c.shape}.")
    if species_c.shape != (n_particles,):
        raise ValueError(f"Expected species of shape ({n_particles},), got {species_c.shape}.")
    if charge_c.shape != (n_particles,):
        raise ValueError(f"Expected charge of shape ({n_particles},), got {charge_c.shape}.")
    if lam_c.shape != (2, 2):
        raise ValueError(f"Expected lam of shape (2, 2), got {lam_c.shape}.")
    if n_particles == 0:
        return np.zeros((0, 3), dtype=np.float64), 0.0, 0.0

    d_pos = cuda.to_device(pos_c)
    d_species = cuda.to_device(species_c)
    d_charge = cuda.to_device(charge_c)
    d_lam = cuda.to_device(lam_c)

    d_forces = cuda.device_array((n_particles, 3), dtype=np.float64)
    d_u = cuda.device_array(n_particles, dtype=np.float64)
    d_w = cuda.device_array(n_particles, dtype=np.float64)

    blocks = (n_particles + _TILE_SIZE - 1) // _TILE_SIZE
    _pair_forces_kernel[blocks, _TILE_SIZE](
        d_pos,
        d_species,
        d_charge,
        d_lam,
        float(box),
        float(alpha),
        float(r_cut),
        float(e_sq),
        _TWO_OVER_SQRT_PI,
        d_forces,
        d_u,
        d_w,
    )

    forces = d_forces.copy_to_host()
    u_host = d_u.copy_to_host()
    w_host = d_w.copy_to_host()

    # Deterministic host-side reduction in float64
    potential_energy = 0.5 * float(np.sum(u_host, dtype=np.float64))
    virial = 0.5 * float(np.sum(w_host, dtype=np.float64))
    return forces, potential_energy, virial
