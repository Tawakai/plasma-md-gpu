"""Molecular dynamics initialization, GPU integrator, and Berendsen thermostat."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import cuda, float64, int8
from numpy.typing import NDArray

from plasmamd.ewald import (
    _deterministic_reduce_1d_kernel,
    _deutsch_temp_deriv_kernel,
    _ewald_recip_force_kernel,
    _ewald_structure_factor_kernel,
    generate_k_vectors,
    tune_ewald_parameters,
)
from plasmamd.gpu.pair_forces import (
    _TILE_SIZE,
    _TWO_OVER_SQRT_PI,
    _deterministic_reduce_pair_kernel,
    _pair_forces_kernel,
)
from plasmamd.state import PlasmaState

_THREADS_PER_BLOCK = 128


@dataclass
class MDTrajectoryData:
    """In-memory container for recorded MD observables and trajectory frames."""

    steps_energy: NDArray[np.int64]
    u_cfg: NDArray[np.float64]
    u_therm: NDArray[np.float64]
    kinetic_energy: NDArray[np.float64]
    total_energy: NDArray[np.float64]
    virial: NDArray[np.float64]
    temp_i: NDArray[np.float64]
    temp_e: NDArray[np.float64]
    momentum: NDArray[np.float64]
    steps_frames: NDArray[np.int64]
    positions: NDArray[np.float64]
    velocities: NDArray[np.float64]


def initialize_positions(
    n_total: int,
    box: float,
    min_dist: float = 0.3,
    seed: int = 1,
) -> NDArray[np.float64]:
    """Generate random positions in [0, box)^3 with minimum-image distance >= min_dist."""
    rng = np.random.default_rng(seed)
    pos = np.empty((n_total, 3), dtype=np.float64)
    min_dist_sq = min_dist * min_dist
    inv_box = 1.0 / box

    for i in range(n_total):
        for _ in range(100_000):
            candidate = rng.uniform(0.0, box, size=3)
            if i == 0:
                pos[0] = candidate
                break
            diff = pos[:i] - candidate
            diff -= box * np.round(diff * inv_box)
            dist_sq = np.sum(diff * diff, axis=1)
            if np.all(dist_sq >= min_dist_sq):
                pos[i] = candidate
                break
        else:
            raise RuntimeError(f"Failed to place particle {i} with min_dist={min_dist}.")

    return pos


def initialize_velocities(
    species: NDArray[np.int8],
    masses_by_species: NDArray[np.float64],
    k_t: float,
    seed: int = 1,
) -> NDArray[np.float64]:
    """Sample Maxwell-Boltzmann velocities with zero net momentum per species and exact T."""
    rng = np.random.default_rng(seed + 10_000)
    n_total = species.shape[0]
    vel = np.zeros((n_total, 3), dtype=np.float64)

    for s_idx in (0, 1):
        mask = species == s_idx
        n_s = int(np.sum(mask))
        if n_s == 0:
            continue
        mass = float(masses_by_species[s_idx])
        sigma = math.sqrt(k_t / mass)
        v_s = rng.normal(0.0, sigma, size=(n_s, 3))
        v_s -= np.mean(v_s, axis=0, keepdims=True)
        kin_s = 0.5 * mass * float(np.sum(v_s * v_s))
        t_current = (2.0 * kin_s) / (3.0 * n_s)
        v_s *= math.sqrt(k_t / t_current)
        vel[mask] = v_s

    return vel


def berendsen_thermostat_step(
    vel: NDArray[np.float64],
    species: NDArray[np.int8],
    masses_by_species: NDArray[np.float64],
    k_t_target: float,
    dt: float,
    tau: float,
) -> NDArray[np.float64]:
    """Apply pure-function per-species Berendsen rescaling with zero species drift."""
    vel_out = vel.copy()
    for s_idx in (0, 1):
        mask = species == s_idx
        n_s = int(np.sum(mask))
        if n_s == 0:
            continue
        mass = float(masses_by_species[s_idx])
        v_s = vel_out[mask]
        v_s = v_s - np.mean(v_s, axis=0, keepdims=True)
        kin_s = 0.5 * mass * float(np.sum(v_s * v_s))
        t_s = (2.0 * kin_s) / (3.0 * n_s)
        if t_s > 0.0:
            scale = math.sqrt(max(0.1, 1.0 + (dt / tau) * (k_t_target / t_s - 1.0)))
            vel_out[mask] = v_s * scale
    return vel_out


@cuda.jit(
    (
        float64[:, :],
        float64[:, :],
        float64[:, :],
        int8[:],
        float64[:],
        float64,
        float64,
    ),
    fastmath=False,
)
def _verlet_first_half_kernel(
    pos: NDArray[np.float64],
    vel: NDArray[np.float64],
    forces: NDArray[np.float64],
    species: NDArray[np.int8],
    masses_by_species: NDArray[np.float64],
    dt: float,
    box: float,
) -> None:
    """First half-kick of velocity and position drift with periodic wrapping."""
    i = cuda.grid(1)
    n_particles = pos.shape[0]
    if i >= n_particles:
        return

    half_dt_over_m = 0.5 * dt / masses_by_species[species[i]]
    vx = vel[i, 0] + half_dt_over_m * forces[i, 0]
    vy = vel[i, 1] + half_dt_over_m * forces[i, 1]
    vz = vel[i, 2] + half_dt_over_m * forces[i, 2]

    vel[i, 0] = vx
    vel[i, 1] = vy
    vel[i, 2] = vz

    x = pos[i, 0] + dt * vx
    y = pos[i, 1] + dt * vy
    z = pos[i, 2] + dt * vz

    x -= box * math.floor(x / box)
    y -= box * math.floor(y / box)
    z -= box * math.floor(z / box)

    pos[i, 0] = x
    pos[i, 1] = y
    pos[i, 2] = z


@cuda.jit(
    (
        float64[:, :],
        float64[:, :],
        float64[:, :],
        int8[:],
        float64[:],
        float64,
    ),
    fastmath=False,
)
def _verlet_second_half_kernel(
    vel: NDArray[np.float64],
    forces_sr: NDArray[np.float64],
    forces_k: NDArray[np.float64],
    species: NDArray[np.int8],
    masses_by_species: NDArray[np.float64],
    dt: float,
) -> None:
    """Combine short-range and reciprocal forces and perform second velocity half-kick."""
    i = cuda.grid(1)
    n_particles = vel.shape[0]
    if i >= n_particles:
        return

    fx = forces_sr[i, 0] + forces_k[i, 0]
    fy = forces_sr[i, 1] + forces_k[i, 1]
    fz = forces_sr[i, 2] + forces_k[i, 2]
    forces_sr[i, 0] = fx
    forces_sr[i, 1] = fy
    forces_sr[i, 2] = fz

    half_dt_over_m = 0.5 * dt / masses_by_species[species[i]]
    vel[i, 0] += half_dt_over_m * fx
    vel[i, 1] += half_dt_over_m * fy
    vel[i, 2] += half_dt_over_m * fz


@cuda.jit(
    (
        float64[:, :],
        int8[:],
        float64[:],
        float64[:],
    ),
    fastmath=False,
)
def _species_kinetic_reduce_kernel(
    vel: NDArray[np.float64],
    species: NDArray[np.int8],
    masses_by_species: NDArray[np.float64],
    kin_out: NDArray[np.float64],
) -> None:
    """Single-block deterministic reduction of ion and electron kinetic energies."""
    s_ki = cuda.shared.array(_THREADS_PER_BLOCK, dtype=float64)
    s_ke = cuda.shared.array(_THREADS_PER_BLOCK, dtype=float64)

    tid = cuda.threadIdx.x
    n = vel.shape[0]
    acc_i = 0.0
    acc_e = 0.0

    idx = tid
    while idx < n:
        vx = vel[idx, 0]
        vy = vel[idx, 1]
        vz = vel[idx, 2]
        v2 = vx * vx + vy * vy + vz * vz
        sp = species[idx]
        ke_p = 0.5 * masses_by_species[sp] * v2
        if sp == 0:
            acc_i += ke_p
        else:
            acc_e += ke_p
        idx += _THREADS_PER_BLOCK

    s_ki[tid] = acc_i
    s_ke[tid] = acc_e
    cuda.syncthreads()

    stride = _THREADS_PER_BLOCK // 2
    while stride > 0:
        if tid < stride:
            s_ki[tid] += s_ki[tid + stride]
            s_ke[tid] += s_ke[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        kin_out[0] = s_ki[0]
        kin_out[1] = s_ke[0]


@cuda.jit(
    (
        float64[:, :],
        int8[:],
        float64[:],
        float64,
        float64,
        float64,
        int8,
    ),
    fastmath=False,
)
def _berendsen_zero_momentum_kernel(
    vel: NDArray[np.float64],
    species: NDArray[np.int8],
    masses_by_species: NDArray[np.float64],
    k_t_target: float,
    dt: float,
    tau: float,
    n_pairs: int,
) -> None:
    """Single-block GPU kernel removing species drift and applying Berendsen thermostat."""
    s_vix = cuda.shared.array(_THREADS_PER_BLOCK, dtype=float64)
    s_viy = cuda.shared.array(_THREADS_PER_BLOCK, dtype=float64)
    s_viz = cuda.shared.array(_THREADS_PER_BLOCK, dtype=float64)
    s_vex = cuda.shared.array(_THREADS_PER_BLOCK, dtype=float64)
    s_vey = cuda.shared.array(_THREADS_PER_BLOCK, dtype=float64)
    s_vez = cuda.shared.array(_THREADS_PER_BLOCK, dtype=float64)

    tid = cuda.threadIdx.x
    n = vel.shape[0]
    inv_n_pairs = 1.0 / float(n_pairs)

    vix = 0.0
    viy = 0.0
    viz = 0.0
    vex = 0.0
    vey = 0.0
    vez = 0.0

    idx = tid
    while idx < n:
        sp = species[idx]
        vx = vel[idx, 0]
        vy = vel[idx, 1]
        vz = vel[idx, 2]
        if sp == 0:
            vix += vx
            viy += vy
            viz += vz
        else:
            vex += vx
            vey += vy
            vez += vz
        idx += _THREADS_PER_BLOCK

    s_vix[tid] = vix
    s_viy[tid] = viy
    s_viz[tid] = viz
    s_vex[tid] = vex
    s_vey[tid] = vey
    s_vez[tid] = vez
    cuda.syncthreads()

    stride = _THREADS_PER_BLOCK // 2
    while stride > 0:
        if tid < stride:
            s_vix[tid] += s_vix[tid + stride]
            s_viy[tid] += s_viy[tid + stride]
            s_viz[tid] += s_viz[tid + stride]
            s_vex[tid] += s_vex[tid + stride]
            s_vey[tid] += s_vey[tid + stride]
            s_vez[tid] += s_vez[tid + stride]
        cuda.syncthreads()
        stride //= 2

    mean_ix = s_vix[0] * inv_n_pairs
    mean_iy = s_viy[0] * inv_n_pairs
    mean_iz = s_viz[0] * inv_n_pairs
    mean_ex = s_vex[0] * inv_n_pairs
    mean_ey = s_vey[0] * inv_n_pairs
    mean_ez = s_vez[0] * inv_n_pairs
    cuda.syncthreads()

    acc_ki = 0.0
    acc_ke = 0.0
    idx = tid
    while idx < n:
        sp = species[idx]
        if sp == 0:
            dvx = vel[idx, 0] - mean_ix
            dvy = vel[idx, 1] - mean_iy
            dvz = vel[idx, 2] - mean_iz
            vel[idx, 0] = dvx
            vel[idx, 1] = dvy
            vel[idx, 2] = dvz
            acc_ki += 0.5 * masses_by_species[0] * (dvx * dvx + dvy * dvy + dvz * dvz)
        else:
            dvx = vel[idx, 0] - mean_ex
            dvy = vel[idx, 1] - mean_ey
            dvz = vel[idx, 2] - mean_ez
            vel[idx, 0] = dvx
            vel[idx, 1] = dvy
            vel[idx, 2] = dvz
            acc_ke += 0.5 * masses_by_species[1] * (dvx * dvx + dvy * dvy + dvz * dvz)
        idx += _THREADS_PER_BLOCK

    s_vix[tid] = acc_ki
    s_vex[tid] = acc_ke
    cuda.syncthreads()

    stride = _THREADS_PER_BLOCK // 2
    while stride > 0:
        if tid < stride:
            s_vix[tid] += s_vix[tid + stride]
            s_vex[tid] += s_vex[tid + stride]
        cuda.syncthreads()
        stride //= 2

    t_i = (2.0 * s_vix[0]) / (3.0 * float(n_pairs))
    t_e = (2.0 * s_vex[0]) / (3.0 * float(n_pairs))

    arg_i = 1.0 + (dt / tau) * (k_t_target / t_i - 1.0)
    if arg_i < 0.1:
        arg_i = 0.1
    scale_i = math.sqrt(arg_i)

    arg_e = 1.0 + (dt / tau) * (k_t_target / t_e - 1.0)
    if arg_e < 0.1:
        arg_e = 0.1
    scale_e = math.sqrt(arg_e)

    idx = tid
    while idx < n:
        sp = species[idx]
        sc = scale_i if sp == 0 else scale_e
        vel[idx, 0] *= sc
        vel[idx, 1] *= sc
        vel[idx, 2] *= sc
        idx += _THREADS_PER_BLOCK


def run_simulation(
    state: PlasmaState,
    n_equil_steps: int = 10_000,
    n_prod_steps: int = 50_000,
    dt: float = 0.03,
    tau_berendsen: float = 3.0,
    min_dist: float = 0.3,
    seed: int = 1,
    energy_stride: int = 10,
    frame_stride: int = 100,
) -> MDTrajectoryData:
    """Execute full GPU molecular dynamics simulation (Berendsen equilibration + NVE production)."""
    n_pairs = state.n_pairs
    n_total = state.n_total
    box = state.box
    e_sq = state.e_sq
    k_t = state.k_t
    lam = state.lam
    masses_by_species = state.masses_by_species

    species = np.array([0] * n_pairs + [1] * n_pairs, dtype=np.int8)
    charge = np.where(species == 0, 1.0, -1.0).astype(np.float64)

    pos = initialize_positions(n_total, box, min_dist=min_dist, seed=seed)
    vel = initialize_velocities(species, masses_by_species, k_t, seed=seed)

    alpha, r_cut, k_max_int = tune_ewald_parameters(box, tol=1e-6)
    k_vecs, k_sq = generate_k_vectors(box, k_max_int)
    n_k = k_vecs.shape[0]
    k_weights = np.ascontiguousarray(
        2.0 * e_sq * (2.0 * math.pi / state.volume) * np.exp(-k_sq / (4.0 * alpha * alpha)) / k_sq,
        dtype=np.float64,
    )
    u_self = -e_sq * (alpha / math.sqrt(math.pi)) * float(np.sum(charge * charge))

    d_pos = cuda.to_device(pos)
    d_vel = cuda.to_device(vel)
    d_species = cuda.to_device(species)
    d_charge = cuda.to_device(charge)
    d_lam = cuda.to_device(lam)
    d_masses = cuda.to_device(masses_by_species)
    d_k_vecs = cuda.to_device(k_vecs)
    d_k_weights = cuda.to_device(k_weights)

    d_forces_sr = cuda.device_array((n_total, 3), dtype=np.float64)
    d_forces_k = cuda.device_array((n_total, 3), dtype=np.float64)
    d_u_sr_part = cuda.device_array(n_total, dtype=np.float64)
    d_w_sr_part = cuda.device_array(n_total, dtype=np.float64)
    d_sr_totals = cuda.device_array(2, dtype=np.float64)
    d_utd_part = cuda.device_array(n_total, dtype=np.float64)
    d_utd_sum = cuda.device_array(1, dtype=np.float64)

    d_cos = cuda.device_array(n_k, dtype=np.float64)
    d_sin = cuda.device_array(n_k, dtype=np.float64)
    d_u_per_k = cuda.device_array(n_k, dtype=np.float64)
    d_u_rec = cuda.device_array(1, dtype=np.float64)
    d_kin_species = cuda.device_array(2, dtype=np.float64)

    blocks_p = (n_total + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
    blocks_k = (n_k + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK

    def _compute_forces_on_gpu() -> None:
        _pair_forces_kernel[(n_total + _TILE_SIZE - 1) // _TILE_SIZE, _TILE_SIZE](
            d_pos,
            d_species,
            d_charge,
            d_lam,
            float(box),
            float(alpha),
            float(r_cut),
            float(e_sq),
            _TWO_OVER_SQRT_PI,
            d_forces_sr,
            d_u_sr_part,
            d_w_sr_part,
        )
        _ewald_structure_factor_kernel[blocks_k, _THREADS_PER_BLOCK](
            d_pos,
            d_charge,
            d_k_vecs,
            d_k_weights,
            d_cos,
            d_sin,
            d_u_per_k,
        )
        _ewald_recip_force_kernel[blocks_p, _THREADS_PER_BLOCK](
            d_pos, d_charge, d_k_vecs, d_k_weights, d_cos, d_sin, d_forces_k
        )

    _compute_forces_on_gpu()
    _verlet_second_half_kernel[blocks_p, _THREADS_PER_BLOCK](
        d_vel, d_forces_sr, d_forces_k, d_species, d_masses, 0.0
    )

    for _ in range(n_equil_steps):
        _verlet_first_half_kernel[blocks_p, _THREADS_PER_BLOCK](
            d_pos, d_vel, d_forces_sr, d_species, d_masses, float(dt), float(box)
        )
        _compute_forces_on_gpu()
        _verlet_second_half_kernel[blocks_p, _THREADS_PER_BLOCK](
            d_vel, d_forces_sr, d_forces_k, d_species, d_masses, float(dt)
        )
        _berendsen_zero_momentum_kernel[1, _THREADS_PER_BLOCK](
            d_vel,
            d_species,
            d_masses,
            float(k_t),
            float(dt),
            float(tau_berendsen),
            int(n_pairs),
        )

    steps_energy: list[int] = []
    u_cfg_list: list[float] = []
    u_therm_list: list[float] = []
    kin_list: list[float] = []
    tot_list: list[float] = []
    vir_list: list[float] = []
    temp_i_list: list[float] = []
    temp_e_list: list[float] = []
    mom_list: list[NDArray[np.float64]] = []

    steps_frames: list[int] = []
    pos_frames: list[NDArray[np.float64]] = []
    vel_frames: list[NDArray[np.float64]] = []

    particle_masses = masses_by_species[species][:, None]

    def _record_step(step_idx: int) -> None:
        if step_idx % energy_stride == 0:
            _deterministic_reduce_pair_kernel[1, _THREADS_PER_BLOCK](
                d_u_sr_part, d_w_sr_part, d_sr_totals
            )
            _deterministic_reduce_1d_kernel[1, _THREADS_PER_BLOCK](d_u_per_k, d_u_rec)
            _deutsch_temp_deriv_kernel[blocks_p, _THREADS_PER_BLOCK](
                d_pos,
                d_species,
                d_charge,
                d_lam,
                float(box),
                float(r_cut),
                float(e_sq),
                d_utd_part,
            )
            _deterministic_reduce_1d_kernel[1, _THREADS_PER_BLOCK](d_utd_part, d_utd_sum)
            _species_kinetic_reduce_kernel[1, _THREADS_PER_BLOCK](
                d_vel, d_species, d_masses, d_kin_species
            )

            sr_tot = d_sr_totals.copy_to_host()
            u_rec_val = float(d_u_rec.copy_to_host()[0])
            u_td_val = 0.5 * float(d_utd_sum.copy_to_host()[0])
            kin_sp = d_kin_species.copy_to_host()
            vel_host = d_vel.copy_to_host()

            u_k_val = u_rec_val + u_self
            u_cfg_val = float(sr_tot[0]) + u_k_val
            u_therm_val = u_cfg_val + u_td_val
            w_val = float(sr_tot[1]) + u_k_val
            k_i = float(kin_sp[0])
            k_e = float(kin_sp[1])
            k_tot = k_i + k_e

            steps_energy.append(step_idx)
            u_cfg_list.append(u_cfg_val)
            u_therm_list.append(u_therm_val)
            kin_list.append(k_tot)
            tot_list.append(k_tot + u_cfg_val)
            vir_list.append(w_val)
            temp_i_list.append((2.0 * k_i) / (3.0 * n_pairs))
            temp_e_list.append((2.0 * k_e) / (3.0 * n_pairs))
            mom_list.append(np.sum(particle_masses * vel_host, axis=0))

        if step_idx % frame_stride == 0:
            steps_frames.append(step_idx)
            pos_frames.append(d_pos.copy_to_host())
            vel_frames.append(d_vel.copy_to_host())

    _record_step(0)

    for step in range(1, n_prod_steps + 1):
        _verlet_first_half_kernel[blocks_p, _THREADS_PER_BLOCK](
            d_pos, d_vel, d_forces_sr, d_species, d_masses, float(dt), float(box)
        )
        _compute_forces_on_gpu()
        _verlet_second_half_kernel[blocks_p, _THREADS_PER_BLOCK](
            d_vel, d_forces_sr, d_forces_k, d_species, d_masses, float(dt)
        )
        if step % energy_stride == 0 or step % frame_stride == 0:
            _record_step(step)

    return MDTrajectoryData(
        steps_energy=np.array(steps_energy, dtype=np.int64),
        u_cfg=np.array(u_cfg_list, dtype=np.float64),
        u_therm=np.array(u_therm_list, dtype=np.float64),
        kinetic_energy=np.array(kin_list, dtype=np.float64),
        total_energy=np.array(tot_list, dtype=np.float64),
        virial=np.array(vir_list, dtype=np.float64),
        temp_i=np.array(temp_i_list, dtype=np.float64),
        temp_e=np.array(temp_e_list, dtype=np.float64),
        momentum=np.array(mom_list, dtype=np.float64),
        steps_frames=np.array(steps_frames, dtype=np.int64),
        positions=np.array(pos_frames, dtype=np.float64),
        velocities=np.array(vel_frames, dtype=np.float64),
    )
