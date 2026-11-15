"""Short-range Ewald pair forces with Deutsch or Kelbg diffraction regularisation.

For every pair with ``0 < r < r_cut`` (``x = r / lambda_ab``, minimum image):

* Deutsch (``potential_type == 0``):
  ``v = q_i q_j e^2 / r * [erfc(alpha r) - exp(-x)]``
* Kelbg (``potential_type == 1``):
  ``v = q_i q_j e^2 / r * [erfc(alpha r) - exp(-x^2) + sqrt(pi) x erfc(x)]``

The pair formulas live in one plain-Python function, ``_pair_terms``, which is compiled
twice (``numba.njit`` and ``numba.cuda.jit(device=True)``), so the CPU reference and the
GPU kernel cannot drift apart.

Units: length ``a``, mass ``m_e``, time ``1 / omega_pe``; ``e^2 = 1/3``.
Species index: 0 = ion, 1 = electron.
"""

from __future__ import annotations

import math

import numba
import numpy as np
from numba import cuda, float64, int32
from numpy.typing import NDArray

TILE_SIZE = 256
POTENTIAL_DEUTSCH = 0
POTENTIAL_KELBG = 1
E_SQ = 1.0 / 3.0

_SQRT_PI = math.sqrt(math.pi)
_TWO_OVER_SQRT_PI = 2.0 / _SQRT_PI

FloatArray = NDArray[np.float64]
PairResult = tuple[FloatArray, float, float, float]


def _pair_terms(
    r: float, qq_e2: float, lam: float, alpha: float, potential_type: int
) -> tuple[float, float, float]:
    """Return ``(v_sr, F_r, T dv_sr/dT)`` for one pair at distance ``r``.

    ``F_r = -dv_sr/dr`` is the radial force (positive = repulsive).  The temperature
    derivative uses ``lambda ~ T^(-1/2)``, i.e. ``T d/dT = -(lambda / 2) d/dlambda``.
    """
    inv_r = 1.0 / r
    ar = alpha * r
    x = r / lam
    erfc_ar = math.erfc(ar)
    gauss_ar = _TWO_OVER_SQRT_PI * alpha * math.exp(-ar * ar)
    if potential_type == 0:
        exp_x = math.exp(-x)
        v = qq_e2 * inv_r * (erfc_ar - exp_x)
        f_r = qq_e2 * inv_r * (inv_r * (erfc_ar - exp_x) + gauss_ar - exp_x / lam)
        # d/dlambda[-exp(-r/lambda)/r] = -exp(-x)/lambda^2, hence the plus sign below.
        tdv = qq_e2 * exp_x / (2.0 * lam)
    else:
        exp_x2 = math.exp(-x * x)
        erfc_x = math.erfc(x)
        v = qq_e2 * inv_r * (erfc_ar - exp_x2 + _SQRT_PI * x * erfc_x)
        # The sqrt(pi) x erfc(x) / r^2 term cancels exactly against -g'(x) / (lambda r).
        f_r = qq_e2 * inv_r * (inv_r * (erfc_ar - exp_x2) + gauss_ar)
        tdv = qq_e2 * _SQRT_PI * erfc_x / (2.0 * lam)
    return v, f_r, tdv


_pair_terms_cpu = numba.njit(inline="always")(_pair_terms)
_pair_terms_gpu = cuda.jit(device=True)(_pair_terms)


def pair_terms(
    r: float, qq_e2: float, lam: float, alpha: float, potential_type: int
) -> tuple[float, float, float]:
    """Pure-Python access to the pair formulas (for tests and plots)."""
    return _pair_terms(r, qq_e2, lam, alpha, potential_type)


# --------------------------------------------------------------------------------------
# CPU reference
# --------------------------------------------------------------------------------------


@numba.njit(parallel=True, cache=True)
def _pair_forces_cpu_kernel(
    pos: FloatArray,
    species: NDArray[np.int32],
    charge: FloatArray,
    lam: FloatArray,
    box: float,
    alpha: float,
    r_cut: float,
    potential_type: int,
    forces: FloatArray,
    u_i: FloatArray,
    w_i: FloatArray,
    tdv_i: FloatArray,
) -> None:
    """Fill per-particle forces and half-weighted energy, virial and ``T dv/dT``."""
    n = pos.shape[0]
    inv_box = 1.0 / box
    r_cut_sq = r_cut * r_cut
    for i in numba.prange(n):
        fx = 0.0
        fy = 0.0
        fz = 0.0
        ui = 0.0
        wi = 0.0
        ti = 0.0
        for j in range(n):
            if j == i:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            dx -= box * math.floor(dx * inv_box + 0.5)
            dy -= box * math.floor(dy * inv_box + 0.5)
            dz -= box * math.floor(dz * inv_box + 0.5)
            r2 = dx * dx + dy * dy + dz * dz
            if r2 <= 0.0 or r2 >= r_cut_sq:
                continue
            r = math.sqrt(r2)
            v, f_r, tdv = _pair_terms_cpu(
                r, charge[i] * charge[j] * E_SQ, lam[species[i], species[j]], alpha, potential_type
            )
            f_over_r = f_r / r
            fx += f_over_r * dx
            fy += f_over_r * dy
            fz += f_over_r * dz
            ui += 0.5 * v
            wi += 0.5 * f_r * r
            ti += 0.5 * tdv
        forces[i, 0] = fx
        forces[i, 1] = fy
        forces[i, 2] = fz
        u_i[i] = ui
        w_i[i] = wi
        tdv_i[i] = ti


def pair_forces_kelbg_cpu_per_particle(
    pos: FloatArray,
    species: NDArray[np.integer],
    charge: FloatArray,
    lam: FloatArray,
    box: float,
    alpha: float,
    r_cut: float,
    potential_type: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """CPU forces ``(N, 3)`` and per-particle ``u_i``, ``w_i``, ``tdv_i`` (each with factor 1/2)."""
    _validate(pos, lam, box, r_cut, potential_type)
    n = pos.shape[0]
    forces = np.empty((n, 3), dtype=np.float64)
    u_i = np.empty(n, dtype=np.float64)
    w_i = np.empty(n, dtype=np.float64)
    tdv_i = np.empty(n, dtype=np.float64)
    _pair_forces_cpu_kernel(
        np.ascontiguousarray(pos, dtype=np.float64),
        np.ascontiguousarray(species, dtype=np.int32),
        np.ascontiguousarray(charge, dtype=np.float64),
        np.ascontiguousarray(lam, dtype=np.float64),
        float(box),
        float(alpha),
        float(r_cut),
        int(potential_type),
        forces,
        u_i,
        w_i,
        tdv_i,
    )
    return forces, u_i, w_i, tdv_i


def pair_forces_kelbg_cpu(
    pos: FloatArray,
    species: NDArray[np.integer],
    charge: FloatArray,
    lam: FloatArray,
    box: float,
    alpha: float,
    r_cut: float,
    potential_type: int = POTENTIAL_KELBG,
) -> PairResult:
    """Short-range forces on the CPU.

    Parameters
    ----------
    pos
        Positions ``(N, 3)`` in units of ``a``.
    species
        Species index per particle (0 = ion, 1 = electron).
    charge
        Charge numbers ``(N,)`` (+1 / -1).
    lam
        ``(2, 2)`` matrix of thermal lengths ``lambda_ab`` in units of ``a``.
    box, alpha, r_cut
        Cubic box edge, Ewald splitting parameter and real-space cutoff (``r_cut <= box / 2``).
    potential_type
        0 = Deutsch, 1 = Kelbg.

    Returns
    -------
    forces, u_sr, w_sr, tdv_dt
        Forces ``(N, 3)``, energy ``sum_{i<j} v``, virial ``sum_{i<j} r F_r`` and
        ``sum_{i<j} T dv/dT``.
    """
    forces, u_i, w_i, tdv_i = pair_forces_kelbg_cpu_per_particle(
        pos, species, charge, lam, box, alpha, r_cut, potential_type
    )
    return forces, float(u_i.sum()), float(w_i.sum()), float(tdv_i.sum())


# --------------------------------------------------------------------------------------
# GPU kernel (tiled, shared memory, no atomics)
# --------------------------------------------------------------------------------------


@cuda.jit
def pair_forces_kelbg_kernel(
    pos, species, charge, lam, box, alpha, r_cut, potential_type, forces, u_i, w_i, tdv_i
):
    """One thread per particle; neighbours are streamed through shared memory tile by tile.

    Must be launched with ``TILE_SIZE`` threads per block.  Each thread accumulates the
    force on its own particle and half of every pair's energy, virial and ``T dv/dT``,
    so no atomic operations are needed and the result is deterministic.
    """
    sh_pos = cuda.shared.array((TILE_SIZE, 3), float64)
    sh_charge = cuda.shared.array(TILE_SIZE, float64)
    sh_species = cuda.shared.array(TILE_SIZE, int32)

    n = pos.shape[0]
    i = cuda.grid(1)
    tx = cuda.threadIdx.x
    active = i < n

    xi = 0.0
    yi = 0.0
    zi = 0.0
    qi = 0.0
    si = 0
    if active:
        xi = pos[i, 0]
        yi = pos[i, 1]
        zi = pos[i, 2]
        qi = charge[i] * E_SQ
        si = species[i]

    inv_box = 1.0 / box
    r_cut_sq = r_cut * r_cut
    fx = 0.0
    fy = 0.0
    fz = 0.0
    ui = 0.0
    wi = 0.0
    ti = 0.0

    n_tiles = (n + TILE_SIZE - 1) // TILE_SIZE
    for tile in range(n_tiles):
        j_load = tile * TILE_SIZE + tx
        if j_load < n:
            sh_pos[tx, 0] = pos[j_load, 0]
            sh_pos[tx, 1] = pos[j_load, 1]
            sh_pos[tx, 2] = pos[j_load, 2]
            sh_charge[tx] = charge[j_load]
            sh_species[tx] = species[j_load]
        cuda.syncthreads()

        if active:
            rem = n - tile * TILE_SIZE
            n_in_tile = TILE_SIZE if rem > TILE_SIZE else rem
            for k in range(n_in_tile):
                if tile * TILE_SIZE + k == i:
                    continue
                dx = xi - sh_pos[k, 0]
                dy = yi - sh_pos[k, 1]
                dz = zi - sh_pos[k, 2]
                dx -= box * math.floor(dx * inv_box + 0.5)
                dy -= box * math.floor(dy * inv_box + 0.5)
                dz -= box * math.floor(dz * inv_box + 0.5)
                r2 = dx * dx + dy * dy + dz * dz
                if r2 <= 0.0 or r2 >= r_cut_sq:
                    continue
                r = math.sqrt(r2)
                v, f_r, tdv = _pair_terms_gpu(
                    r, qi * sh_charge[k], lam[si, sh_species[k]], alpha, potential_type
                )
                f_over_r = f_r / r
                fx += f_over_r * dx
                fy += f_over_r * dy
                fz += f_over_r * dz
                ui += 0.5 * v
                wi += 0.5 * f_r * r
                ti += 0.5 * tdv
        cuda.syncthreads()

    if active:
        forces[i, 0] = fx
        forces[i, 1] = fy
        forces[i, 2] = fz
        u_i[i] = ui
        w_i[i] = wi
        tdv_i[i] = ti


def launch_pair_forces(
    d_pos, d_species, d_charge, d_lam, box, alpha, r_cut, potential_type, d_forces, d_u, d_w, d_tdv
) -> None:
    """Launch the tiled kernel on device arrays (no host transfers)."""
    n = d_pos.shape[0]
    blocks = (n + TILE_SIZE - 1) // TILE_SIZE
    pair_forces_kelbg_kernel[blocks, TILE_SIZE](
        d_pos,
        d_species,
        d_charge,
        d_lam,
        float(box),
        float(alpha),
        float(r_cut),
        int(potential_type),
        d_forces,
        d_u,
        d_w,
        d_tdv,
    )


def pair_forces_kelbg_gpu(
    pos: FloatArray,
    species: NDArray[np.integer],
    charge: FloatArray,
    lam: FloatArray,
    box: float,
    alpha: float,
    r_cut: float,
    potential_type: int = POTENTIAL_KELBG,
) -> PairResult:
    """Short-range forces on the GPU; same contract as :func:`pair_forces_kelbg_cpu`.

    Host arrays in, host arrays out.  Per-particle energies are summed on the host in
    ``float64`` (deterministic), the kernel itself uses no atomics.
    """
    _validate(pos, lam, box, r_cut, potential_type)
    n = pos.shape[0]
    d_pos = cuda.to_device(np.ascontiguousarray(pos, dtype=np.float64))
    d_species = cuda.to_device(np.ascontiguousarray(species, dtype=np.int32))
    d_charge = cuda.to_device(np.ascontiguousarray(charge, dtype=np.float64))
    d_lam = cuda.to_device(np.ascontiguousarray(lam, dtype=np.float64))
    d_forces = cuda.device_array((n, 3), dtype=np.float64)
    d_u = cuda.device_array(n, dtype=np.float64)
    d_w = cuda.device_array(n, dtype=np.float64)
    d_tdv = cuda.device_array(n, dtype=np.float64)
    launch_pair_forces(
        d_pos, d_species, d_charge, d_lam, box, alpha, r_cut, potential_type,
        d_forces, d_u, d_w, d_tdv,
    )
    return (
        d_forces.copy_to_host(),
        float(d_u.copy_to_host().sum()),
        float(d_w.copy_to_host().sum()),
        float(d_tdv.copy_to_host().sum()),
    )


def _validate(pos: FloatArray, lam: FloatArray, box: float, r_cut: float, ptype: int) -> None:
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must have shape (N, 3)")
    if np.shape(lam) != (2, 2):
        raise ValueError("lam must have shape (2, 2)")
    if r_cut > 0.5 * box * (1.0 + 1e-12):
        raise ValueError("r_cut must not exceed box / 2 (minimum-image convention)")
    if ptype not in (POTENTIAL_DEUTSCH, POTENTIAL_KELBG):
        raise ValueError("potential_type must be 0 (Deutsch) or 1 (Kelbg)")
