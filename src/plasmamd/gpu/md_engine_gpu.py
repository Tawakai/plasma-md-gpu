"""GPU-resident velocity-Verlet MD engine for the two-component plasma.

Positions, unwrapped positions, velocities and forces stay in GPU memory for the whole
run.  Only small per-particle scalar arrays (energies, squared displacements) are copied
to the host, and only on thermostat and sampling steps.

Total energy (conserved in NVE): ``E_tot = E_kin + U_sr + U_recip + U_self``.
Internal energy with temperature-dependent potentials (Hansen & McDonald 1981, Eq. 19):
``U_int = E_tot - T dU_sr/dT``.

A CPU backend (``backend="cpu"``) runs the same driver with ``numba.njit`` twins of the
kernels; it is the reference the GPU path is tested against.

Units: length ``a``, mass ``m_e``, time ``1 / omega_pe``; ``e^2 = 1/3``, ``k_B T = 1/(3 Gamma)``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Protocol

import numba
import numpy as np
from numba import cuda
from numpy.typing import NDArray

from plasmamd.gpu.kelbg_forces import (
    E_SQ,
    POTENTIAL_KELBG,
    launch_pair_forces,
    pair_forces_kelbg_cpu_per_particle,
)
from plasmamd.state import PlasmaState

FloatArray = NDArray[np.float64]
THREADS_PER_BLOCK = 128
ION = 0
ELECTRON = 1
# Columns of the per-particle statistics array.
_KE, _PX, _PY, _PZ, _SQ_GLOBAL, _SQ_BLOCK = range(6)


# --------------------------------------------------------------------------------------
# Configuration and results
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MDConfig:
    """Parameters of one MD run.

    ``n_prod`` is split into ``n_blocks`` equal segments; each segment has its own time
    origin for the electron MSD and yields one independent estimate of ``D_e``.
    ``ion_dynamic_mass`` overrides the ion mass in the equations of motion only (the
    thermal lengths still use ``mass_ratio``); static properties do not depend on it,
    which makes ``ion_dynamic_mass=1`` the efficient choice for ``g(r)``.
    """

    n_pairs: int
    gamma: float = 0.5
    r_s: float = 1.0
    potential_type: int = POTENTIAL_KELBG
    mass_ratio: float = 1836.0
    ion_dynamic_mass: float | None = None
    dt: float = 0.03
    n_equil: int = 2000
    n_prod: int = 20000
    sample_stride: int = 10
    thermostat_stride: int = 10
    n_blocks: int = 5
    fit_start_frac: float = 0.5
    alpha_rcut: float = 3.2
    rdf_stride: int = 0
    rdf_bins: int = 100
    seed: int = 1
    backend: str = "gpu"

    def __post_init__(self) -> None:
        if self.n_pairs < 2:
            raise ValueError("n_pairs must be at least 2")
        if self.n_blocks < 1 or self.n_prod % self.n_blocks != 0:
            raise ValueError("n_prod must be a positive multiple of n_blocks")
        block = self.n_prod // self.n_blocks
        if block % self.sample_stride != 0 or block // self.sample_stride < 4:
            raise ValueError("each block must hold at least 4 samples and a whole number of them")
        if not 0.0 <= self.fit_start_frac < 1.0:
            raise ValueError("fit_start_frac must be in [0, 1)")
        if self.backend not in ("gpu", "cpu"):
            raise ValueError("backend must be 'gpu' or 'cpu'")

    @property
    def state(self) -> PlasmaState:
        """Thermodynamic state (box, ``k_B T``, thermal lengths)."""
        return PlasmaState(self.gamma, self.r_s, n_pairs=self.n_pairs, mass_ratio=self.mass_ratio)


@dataclass
class MDResult:
    """Output of :func:`run_md`.  Times are in ``1 / omega_pe``, lengths in ``a``."""

    config: MDConfig
    time: FloatArray  # sample times of the production stage, starting at 0
    msd_e: FloatArray  # electron MSD relative to the start of production
    e_total: FloatArray  # E_kin + U_sr + U_recip + U_self
    u_internal: FloatArray  # e_total - T dU_sr/dT
    kt_ion: FloatArray  # instantaneous 2 E_kin,i / (3 N_i)
    kt_electron: FloatArray
    momentum: FloatArray  # total momentum (n_samples, 3)
    block_tau: FloatArray  # lag times inside a block
    block_msd_e: FloatArray  # (n_blocks, n_tau) electron MSD with per-block origin
    block_slopes: FloatArray  # fitted d MSD / d tau per block
    d_e: float  # D_e* = mean(slope) / 6, units a^2 omega_pe
    d_e_err: float  # standard error over blocks (nan for a single block)
    steps_per_second: float
    rdf_r: FloatArray = field(default_factory=lambda: np.empty(0))
    g_ii: FloatArray = field(default_factory=lambda: np.empty(0))
    g_ie: FloatArray = field(default_factory=lambda: np.empty(0))
    g_ee: FloatArray = field(default_factory=lambda: np.empty(0))


# --------------------------------------------------------------------------------------
# Ewald set-up (host, once per run)
# --------------------------------------------------------------------------------------


def ewald_parameters(box: float, alpha_rcut: float = 3.2) -> tuple[float, float, int]:
    """Return ``(r_cut, alpha, n_max)`` with ``r_cut = box / 2``.

    Both sums are truncated where their Gaussian factor equals ``exp(-alpha_rcut^2)``
    (3.6e-5 for the default 3.2): ``alpha = alpha_rcut / r_cut`` and
    ``k_max = 2 alpha alpha_rcut``.
    """
    r_cut = 0.5 * box
    alpha = alpha_rcut / r_cut
    n_max = math.ceil(2.0 * alpha * alpha_rcut * box / (2.0 * math.pi))
    return r_cut, alpha, n_max


def half_sphere_k_vectors(box: float, alpha: float, n_max: int) -> tuple[FloatArray, FloatArray]:
    """Wave vectors of one half-sphere and weights ``(4 pi / V) exp(-k^2 / 4 alpha^2) / k^2``.

    The factor 2 for the omitted ``-k`` partner is already in the weight, so
    ``U_recip = e^2 sum_k w_k |S(k)|^2`` with ``S(k) = sum_j q_j exp(i k r_j)``.
    """
    rng = np.arange(-n_max, n_max + 1)
    nx, ny, nz = np.meshgrid(rng, rng, rng, indexing="ij")
    n = np.stack([nx.ravel(), ny.ravel(), nz.ravel()], axis=1)
    n_sq = (n * n).sum(axis=1)
    half = (n[:, 0] > 0) | ((n[:, 0] == 0) & (n[:, 1] > 0))
    half |= (n[:, 0] == 0) & (n[:, 1] == 0) & (n[:, 2] > 0)
    n = n[half & (n_sq <= n_max * n_max)]
    k_vecs = (2.0 * math.pi / box) * n.astype(np.float64)
    k_sq = (k_vecs * k_vecs).sum(axis=1)
    weights = (4.0 * math.pi / box**3) * np.exp(-k_sq / (4.0 * alpha * alpha)) / k_sq
    return np.ascontiguousarray(k_vecs), np.ascontiguousarray(weights)


def self_energy(charge: FloatArray, alpha: float) -> float:
    """Ewald self-interaction ``U_self = -e^2 alpha / sqrt(pi) sum_i q_i^2``."""
    return float(-E_SQ * alpha / math.sqrt(math.pi) * np.sum(charge * charge))


# --------------------------------------------------------------------------------------
# GPU kernels
# --------------------------------------------------------------------------------------


@cuda.jit
def ewald_structure_factor_gpu(pos, charge, k_vecs, k_weights, s_k, k_energy):
    """One thread per wave vector: ``S(k)`` and its energy term ``e^2 w_k |S(k)|^2``."""
    m = cuda.grid(1)
    if m >= k_vecs.shape[0]:
        return
    kx = k_vecs[m, 0]
    ky = k_vecs[m, 1]
    kz = k_vecs[m, 2]
    re = 0.0
    im = 0.0
    for j in range(pos.shape[0]):
        phase = kx * pos[j, 0] + ky * pos[j, 1] + kz * pos[j, 2]
        re += charge[j] * math.cos(phase)
        im += charge[j] * math.sin(phase)
    s_k[m, 0] = re
    s_k[m, 1] = im
    k_energy[m] = E_SQ * k_weights[m] * (re * re + im * im)


@cuda.jit
def ewald_recip_forces_gpu(pos, charge, k_vecs, k_weights, s_k, forces):
    """One thread per particle: add the reciprocal-space force to ``forces``.

    ``F_i = 2 e^2 q_i sum_k w_k k [sin(k r_i) Re S(k) - cos(k r_i) Im S(k)]`` over the
    half-sphere; ``s_k`` must come from :func:`ewald_structure_factor_gpu`.
    """
    i = cuda.grid(1)
    if i >= pos.shape[0]:
        return
    x = pos[i, 0]
    y = pos[i, 1]
    z = pos[i, 2]
    fx = 0.0
    fy = 0.0
    fz = 0.0
    for m in range(k_vecs.shape[0]):
        kx = k_vecs[m, 0]
        ky = k_vecs[m, 1]
        kz = k_vecs[m, 2]
        phase = kx * x + ky * y + kz * z
        amp = k_weights[m] * (math.sin(phase) * s_k[m, 0] - math.cos(phase) * s_k[m, 1])
        fx += amp * kx
        fy += amp * ky
        fz += amp * kz
    pref = 2.0 * E_SQ * charge[i]
    forces[i, 0] += pref * fx
    forces[i, 1] += pref * fy
    forces[i, 2] += pref * fz


@cuda.jit
def verlet_first_half_gpu(pos, pos_unwrapped, vel, forces, mass, dt, box):
    """Half kick, then drift of wrapped and unwrapped positions."""
    i = cuda.grid(1)
    if i >= pos.shape[0]:
        return
    half_dt_over_m = 0.5 * dt / mass[i]
    for d in range(3):
        v = vel[i, d] + half_dt_over_m * forces[i, d]
        vel[i, d] = v
        step = dt * v
        pos_unwrapped[i, d] += step
        x = pos[i, d] + step
        pos[i, d] = x - box * math.floor(x / box)


@cuda.jit
def verlet_second_half_gpu(vel, forces, mass, dt):
    """Second half kick with the forces at the new positions."""
    i = cuda.grid(1)
    if i >= vel.shape[0]:
        return
    half_dt_over_m = 0.5 * dt / mass[i]
    for d in range(3):
        vel[i, d] += half_dt_over_m * forces[i, d]


@cuda.jit
def particle_stats_gpu(vel, mass, pos_unwrapped, origin_global, origin_block, stats):
    """Per-particle kinetic energy, momentum and squared displacements from both origins."""
    i = cuda.grid(1)
    if i >= vel.shape[0]:
        return
    v_sq = 0.0
    sq_global = 0.0
    sq_block = 0.0
    for d in range(3):
        v = vel[i, d]
        v_sq += v * v
        stats[i, 1 + d] = mass[i] * v
        dg = pos_unwrapped[i, d] - origin_global[i, d]
        db = pos_unwrapped[i, d] - origin_block[i, d]
        sq_global += dg * dg
        sq_block += db * db
    stats[i, 0] = 0.5 * mass[i] * v_sq
    stats[i, 4] = sq_global
    stats[i, 5] = sq_block


@cuda.jit
def rescale_velocities_gpu(vel, species, scale_ion, scale_electron, vcm_x, vcm_y, vcm_z):
    """Scale each species to the target temperature and remove the centre-of-mass drift."""
    i = cuda.grid(1)
    if i >= vel.shape[0]:
        return
    s = scale_ion if species[i] == 0 else scale_electron
    vel[i, 0] = s * vel[i, 0] - vcm_x
    vel[i, 1] = s * vel[i, 1] - vcm_y
    vel[i, 2] = s * vel[i, 2] - vcm_z


@cuda.jit
def copy_positions_gpu(src, dst):
    """Device-to-device copy of an ``(N, 3)`` array."""
    i = cuda.grid(1)
    if i >= src.shape[0]:
        return
    for d in range(3):
        dst[i, d] = src[i, d]


# --------------------------------------------------------------------------------------
# CPU twins of the kernels (reference backend)
# --------------------------------------------------------------------------------------


@numba.njit(parallel=True, cache=True)
def ewald_recip_cpu(
    pos: FloatArray, charge: FloatArray, k_vecs: FloatArray, k_weights: FloatArray,
    forces: FloatArray,
) -> float:
    """Add reciprocal-space forces to ``forces`` and return ``U_recip`` (same formulas as GPU)."""
    n = pos.shape[0]
    n_k = k_vecs.shape[0]
    s_k = np.empty((n_k, 2))
    u_recip = 0.0
    for m in numba.prange(n_k):
        re = 0.0
        im = 0.0
        for j in range(n):
            phase = k_vecs[m, 0] * pos[j, 0] + k_vecs[m, 1] * pos[j, 1] + k_vecs[m, 2] * pos[j, 2]
            re += charge[j] * math.cos(phase)
            im += charge[j] * math.sin(phase)
        s_k[m, 0] = re
        s_k[m, 1] = im
        u_recip += E_SQ * k_weights[m] * (re * re + im * im)
    for i in numba.prange(n):
        fx = 0.0
        fy = 0.0
        fz = 0.0
        for m in range(n_k):
            phase = k_vecs[m, 0] * pos[i, 0] + k_vecs[m, 1] * pos[i, 1] + k_vecs[m, 2] * pos[i, 2]
            amp = k_weights[m] * (math.sin(phase) * s_k[m, 0] - math.cos(phase) * s_k[m, 1])
            fx += amp * k_vecs[m, 0]
            fy += amp * k_vecs[m, 1]
            fz += amp * k_vecs[m, 2]
        pref = 2.0 * E_SQ * charge[i]
        forces[i, 0] += pref * fx
        forces[i, 1] += pref * fy
        forces[i, 2] += pref * fz
    return u_recip


@numba.njit(cache=True)
def _rdf_accumulate(
    pos: FloatArray, species: NDArray[np.int32], box: float, r_max: float, hist: FloatArray
) -> None:
    """Add all pair distances below ``r_max`` to ``hist[pair_kind, bin]`` (ii=0, ie=1, ee=2)."""
    n = pos.shape[0]
    n_bins = hist.shape[1]
    inv_dr = n_bins / r_max
    for i in range(n - 1):
        for j in range(i + 1, n):
            r_sq = 0.0
            for d in range(3):
                dx = pos[i, d] - pos[j, d]
                dx -= box * math.floor(dx / box + 0.5)
                r_sq += dx * dx
            r = math.sqrt(r_sq)
            if r < r_max:
                hist[species[i] + species[j], int(r * inv_dr)] += 1.0


# --------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _System:
    """Host-side description of the simulated system."""

    pos: FloatArray
    vel: FloatArray
    species: NDArray[np.int32]
    charge: FloatArray
    mass: FloatArray
    lam: FloatArray
    box: float
    r_cut: float
    alpha: float
    k_vecs: FloatArray
    k_weights: FloatArray
    potential_type: int
    dt: float


class _Backend(Protocol):
    def compute_forces(self) -> None: ...
    def first_half(self) -> None: ...
    def second_half(self) -> None: ...
    def stats(self) -> FloatArray: ...
    def potential_energies(self) -> tuple[float, float, float]: ...
    def rescale(self, scale_ion: float, scale_electron: float, vcm: FloatArray) -> None: ...
    def set_origin(self, *, global_origin: bool) -> None: ...
    def positions(self) -> FloatArray: ...
    def synchronize(self) -> None: ...


class _GpuBackend:
    """All dynamic arrays live on the device; see the module docstring for what is copied."""

    def __init__(self, system: _System) -> None:
        self._s = system
        n = system.pos.shape[0]
        n_k = system.k_vecs.shape[0]
        self.d_pos = cuda.to_device(system.pos)
        self.d_pos_unwrapped = cuda.to_device(system.pos)
        self.d_vel = cuda.to_device(system.vel)
        self.d_forces = cuda.device_array((n, 3), dtype=np.float64)
        self._d_origin_global = cuda.to_device(system.pos)
        self._d_origin_block = cuda.to_device(system.pos)
        self._d_species = cuda.to_device(system.species)
        self._d_charge = cuda.to_device(system.charge)
        self._d_mass = cuda.to_device(system.mass)
        self._d_lam = cuda.to_device(system.lam)
        self.d_k_vecs = cuda.to_device(system.k_vecs)
        self.d_k_weights = cuda.to_device(system.k_weights)
        self._d_s_k = cuda.device_array((n_k, 2), dtype=np.float64)
        self._d_k_energy = cuda.device_array(n_k, dtype=np.float64)
        self._d_u = cuda.device_array(n, dtype=np.float64)
        self._d_w = cuda.device_array(n, dtype=np.float64)
        self._d_tdv = cuda.device_array(n, dtype=np.float64)
        self._d_stats = cuda.device_array((n, 6), dtype=np.float64)
        self._grid_n = ((n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK, THREADS_PER_BLOCK)
        self._grid_k = ((n_k + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK, THREADS_PER_BLOCK)

    def compute_forces(self) -> None:
        s = self._s
        launch_pair_forces(
            self.d_pos, self._d_species, self._d_charge, self._d_lam, s.box, s.alpha, s.r_cut,
            s.potential_type, self.d_forces, self._d_u, self._d_w, self._d_tdv,
        )
        ewald_structure_factor_gpu[self._grid_k](
            self.d_pos, self._d_charge, self.d_k_vecs, self.d_k_weights, self._d_s_k,
            self._d_k_energy,
        )
        ewald_recip_forces_gpu[self._grid_n](
            self.d_pos, self._d_charge, self.d_k_vecs, self.d_k_weights, self._d_s_k, self.d_forces
        )

    def first_half(self) -> None:
        verlet_first_half_gpu[self._grid_n](
            self.d_pos, self.d_pos_unwrapped, self.d_vel, self.d_forces, self._d_mass,
            self._s.dt, self._s.box,
        )

    def second_half(self) -> None:
        verlet_second_half_gpu[self._grid_n](self.d_vel, self.d_forces, self._d_mass, self._s.dt)

    def stats(self) -> FloatArray:
        particle_stats_gpu[self._grid_n](
            self.d_vel, self._d_mass, self.d_pos_unwrapped, self._d_origin_global,
            self._d_origin_block, self._d_stats,
        )
        return self._d_stats.copy_to_host()

    def potential_energies(self) -> tuple[float, float, float]:
        u_sr = float(self._d_u.copy_to_host().sum())
        tdv = float(self._d_tdv.copy_to_host().sum())
        u_recip = float(self._d_k_energy.copy_to_host().sum())
        return u_sr, u_recip, tdv

    def rescale(self, scale_ion: float, scale_electron: float, vcm: FloatArray) -> None:
        rescale_velocities_gpu[self._grid_n](
            self.d_vel, self._d_species, scale_ion, scale_electron,
            float(vcm[0]), float(vcm[1]), float(vcm[2]),
        )

    def set_origin(self, *, global_origin: bool) -> None:
        copy_positions_gpu[self._grid_n](self.d_pos_unwrapped, self._d_origin_block)
        if global_origin:
            copy_positions_gpu[self._grid_n](self.d_pos_unwrapped, self._d_origin_global)

    def positions(self) -> FloatArray:
        return self.d_pos.copy_to_host()

    def synchronize(self) -> None:
        cuda.synchronize()


class _CpuBackend:
    """Reference implementation of the same operations with NumPy / ``numba.njit``."""

    def __init__(self, system: _System) -> None:
        self._s = system
        self._pos = system.pos.copy()
        self._unwrapped = system.pos.copy()
        self._vel = system.vel.copy()
        self._origin_global = system.pos.copy()
        self._origin_block = system.pos.copy()
        self._forces = np.zeros_like(self._pos)
        self._inv_mass = (1.0 / system.mass)[:, None]
        self._energies = (0.0, 0.0, 0.0)

    def compute_forces(self) -> None:
        s = self._s
        self._forces, u_i, _, tdv_i = pair_forces_kelbg_cpu_per_particle(
            self._pos, s.species, s.charge, s.lam, s.box, s.alpha, s.r_cut, s.potential_type
        )
        u_recip = ewald_recip_cpu(self._pos, s.charge, s.k_vecs, s.k_weights, self._forces)
        self._energies = (float(u_i.sum()), float(u_recip), float(tdv_i.sum()))

    def first_half(self) -> None:
        s = self._s
        self._vel += 0.5 * s.dt * self._forces * self._inv_mass
        step = s.dt * self._vel
        self._unwrapped += step
        self._pos += step
        self._pos -= s.box * np.floor(self._pos / s.box)

    def second_half(self) -> None:
        self._vel += 0.5 * self._s.dt * self._forces * self._inv_mass

    def stats(self) -> FloatArray:
        out = np.empty((self._pos.shape[0], 6))
        mass = self._s.mass
        out[:, _KE] = 0.5 * mass * (self._vel**2).sum(axis=1)
        out[:, _PX : _PZ + 1] = mass[:, None] * self._vel
        out[:, _SQ_GLOBAL] = ((self._unwrapped - self._origin_global) ** 2).sum(axis=1)
        out[:, _SQ_BLOCK] = ((self._unwrapped - self._origin_block) ** 2).sum(axis=1)
        return out

    def potential_energies(self) -> tuple[float, float, float]:
        return self._energies

    def rescale(self, scale_ion: float, scale_electron: float, vcm: FloatArray) -> None:
        scale = np.where(self._s.species == ION, scale_ion, scale_electron)[:, None]
        self._vel = scale * self._vel - vcm

    def set_origin(self, *, global_origin: bool) -> None:
        self._origin_block = self._unwrapped.copy()
        if global_origin:
            self._origin_global = self._unwrapped.copy()

    def positions(self) -> FloatArray:
        return self._pos

    def synchronize(self) -> None:
        return None


# --------------------------------------------------------------------------------------
# Initial conditions
# --------------------------------------------------------------------------------------


def initial_positions(
    n_pairs: int, box: float, rng: np.random.Generator, min_ion_distance: float = 0.8
) -> FloatArray:
    """Ions by random sequential insertion (no close pairs), electrons uniformly.

    The effective electron potentials are finite at ``r = 0``, so only the nearly
    Coulombic ion-ion pairs need a minimum separation.
    """
    ions = np.empty((n_pairs, 3))
    placed = 0
    while placed < n_pairs:
        trial = rng.random(3) * box
        delta = ions[:placed] - trial
        delta -= box * np.round(delta / box)
        if placed == 0 or (delta * delta).sum(axis=1).min() >= min_ion_distance**2:
            ions[placed] = trial
            placed += 1
    electrons = rng.random((n_pairs, 3)) * box
    return np.ascontiguousarray(np.vstack([ions, electrons]))


def initial_velocities(
    species: NDArray[np.int32], mass: FloatArray, k_t: float, rng: np.random.Generator
) -> FloatArray:
    """Maxwellian velocities; each species has zero momentum and exactly ``k_B T``."""
    vel = rng.normal(size=(mass.size, 3)) * np.sqrt(k_t / mass)[:, None]
    for kind in (ION, ELECTRON):
        sel = species == kind
        vel[sel] -= vel[sel].mean(axis=0)
        kinetic = 0.5 * (mass[sel, None] * vel[sel] ** 2).sum()
        vel[sel] *= math.sqrt(1.5 * sel.sum() * k_t / kinetic)
    return np.ascontiguousarray(vel)


def _build_system(config: MDConfig) -> _System:
    state = config.state
    rng = np.random.default_rng(config.seed)
    n_pairs = config.n_pairs
    species = np.repeat(np.array([ION, ELECTRON], dtype=np.int32), n_pairs)
    charge = np.where(species == ION, 1.0, -1.0)
    ion_mass = config.mass_ratio if config.ion_dynamic_mass is None else config.ion_dynamic_mass
    mass = np.where(species == ION, ion_mass, 1.0)
    r_cut, alpha, n_max = ewald_parameters(state.box, config.alpha_rcut)
    k_vecs, k_weights = half_sphere_k_vectors(state.box, alpha, n_max)
    pos = initial_positions(n_pairs, state.box, rng)
    vel = initial_velocities(species, mass, state.k_t, rng)
    return _System(
        pos=pos, vel=vel, species=species, charge=charge, mass=mass,
        lam=np.ascontiguousarray(state.lam), box=state.box, r_cut=r_cut, alpha=alpha,
        k_vecs=k_vecs, k_weights=k_weights, potential_type=config.potential_type, dt=config.dt,
    )


# --------------------------------------------------------------------------------------
# Analysis helpers
# --------------------------------------------------------------------------------------


def diffusion_from_blocks(
    tau: FloatArray, block_msd: FloatArray, fit_start_frac: float = 0.5
) -> tuple[float, float, FloatArray]:
    """Self-diffusion coefficient from independent MSD segments.

    A straight line is fitted to each segment's ``MSD(tau)`` for
    ``tau >= fit_start_frac * tau_max`` (the ballistic part is skipped);
    ``D = mean(slope) / 6`` and the standard error is ``std(slope, ddof=1) / (6 sqrt(n))``.

    Returns
    -------
    d, d_err, slopes
        ``d_err`` is ``nan`` when there is a single segment.
    """
    tau = np.asarray(tau, dtype=np.float64)
    block_msd = np.atleast_2d(np.asarray(block_msd, dtype=np.float64))
    sel = tau >= fit_start_frac * tau[-1]
    if sel.sum() < 2:
        raise ValueError("need at least two samples in the fit window")
    slopes = np.array([np.polyfit(tau[sel], msd[sel], 1)[0] for msd in block_msd])
    n = slopes.size
    d_err = float(slopes.std(ddof=1) / (6.0 * math.sqrt(n))) if n > 1 else float("nan")
    return float(slopes.mean() / 6.0), d_err, slopes


def _normalise_rdf(
    hist: FloatArray, n_frames: int, n_pairs: int, box: float, r_max: float
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    edges = np.linspace(0.0, r_max, hist.shape[1] + 1)
    shell = (4.0 / 3.0) * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    volume = box**3
    like = 0.5 * n_pairs * (n_pairs - 1) * shell / volume * n_frames  # ideal-gas pair counts
    unlike = n_pairs * n_pairs * shell / volume * n_frames
    r = 0.5 * (edges[1:] + edges[:-1])
    return r, hist[0] / like, hist[1] / unlike, hist[2] / like


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def _species_summary(
    stats: FloatArray, n_pairs: int
) -> tuple[float, float, FloatArray, float, float]:
    """Return ``(E_kin_ion, E_kin_electron, P_total, MSD_e_global, MSD_e_block)``."""
    ions = slice(0, n_pairs)
    electrons = slice(n_pairs, 2 * n_pairs)
    return (
        float(stats[ions, _KE].sum()),
        float(stats[electrons, _KE].sum()),
        stats[:, _PX : _PZ + 1].sum(axis=0),
        float(stats[electrons, _SQ_GLOBAL].mean()),
        float(stats[electrons, _SQ_BLOCK].mean()),
    )


def _thermostat(backend: _Backend, system: _System, n_pairs: int, k_t: float) -> None:
    """Rescale ions and electrons separately to ``k_B T`` and zero the total momentum."""
    stats = backend.stats()
    ke_ion, ke_electron, _, _, _ = _species_summary(stats, n_pairs)
    target = 1.5 * n_pairs * k_t
    scale_ion = math.sqrt(target / ke_ion)
    scale_electron = math.sqrt(target / ke_electron)
    p_ion = stats[:n_pairs, _PX : _PZ + 1].sum(axis=0)
    p_electron = stats[n_pairs:, _PX : _PZ + 1].sum(axis=0)
    vcm = (scale_ion * p_ion + scale_electron * p_electron) / system.mass.sum()
    backend.rescale(scale_ion, scale_electron, vcm)


def _step(backend: _Backend) -> None:
    backend.first_half()
    backend.compute_forces()
    backend.second_half()


def run_md(config: MDConfig) -> MDResult:
    """Equilibrate with velocity rescaling, then run NVE and measure ``D_e``.

    Parameters
    ----------
    config
        Run parameters; ``config.backend`` selects ``"gpu"`` (device-resident) or ``"cpu"``.

    Returns
    -------
    MDResult
        Time series sampled every ``sample_stride`` steps and the block estimate of ``D_e*``.
    """
    system = _build_system(config)
    state = config.state
    n_pairs = config.n_pairs
    backend: _Backend = _GpuBackend(system) if config.backend == "gpu" else _CpuBackend(system)
    u_self = self_energy(system.charge, system.alpha)

    backend.compute_forces()
    for step in range(1, config.n_equil + 1):
        _step(backend)
        if step % config.thermostat_stride == 0:
            _thermostat(backend, system, n_pairs, state.k_t)

    block_len = config.n_prod // config.n_blocks
    samples_per_block = block_len // config.sample_stride
    n_samples = config.n_prod // config.sample_stride + 1
    series = np.empty((n_samples, 5))  # e_total, u_internal, kt_ion, kt_electron, msd_e
    momentum = np.empty((n_samples, 3))
    block_msd = np.empty((config.n_blocks, samples_per_block))
    r_max = 0.5 * system.box
    rdf_hist = np.zeros((3, config.rdf_bins))
    rdf_frames = 0

    def sample(index: int) -> float:
        ke_ion, ke_electron, p_total, msd_global, msd_block = _species_summary(
            backend.stats(), n_pairs
        )
        u_sr, u_recip, tdv = backend.potential_energies()
        e_total = ke_ion + ke_electron + u_sr + u_recip + u_self
        to_kt = 2.0 / (3.0 * n_pairs)
        series[index] = (e_total, e_total - tdv, to_kt * ke_ion, to_kt * ke_electron, msd_global)
        momentum[index] = p_total
        return msd_block

    backend.set_origin(global_origin=True)
    sample(0)
    backend.synchronize()
    started = time.perf_counter()
    for step in range(1, config.n_prod + 1):
        _step(backend)
        if step % config.sample_stride == 0:
            block, in_block = divmod(step - 1, block_len)
            slot = in_block // config.sample_stride
            block_msd[block, slot] = sample(step // config.sample_stride)
        if config.rdf_stride and step % config.rdf_stride == 0:
            _rdf_accumulate(backend.positions(), system.species, system.box, r_max, rdf_hist)
            rdf_frames += 1
        if step % block_len == 0:
            backend.set_origin(global_origin=False)
    backend.synchronize()
    elapsed = time.perf_counter() - started

    block_tau = config.dt * config.sample_stride * np.arange(1, samples_per_block + 1)
    d_e, d_e_err, slopes = diffusion_from_blocks(block_tau, block_msd, config.fit_start_frac)
    result = MDResult(
        config=config,
        time=config.dt * config.sample_stride * np.arange(n_samples),
        msd_e=series[:, 4].copy(),
        e_total=series[:, 0].copy(),
        u_internal=series[:, 1].copy(),
        kt_ion=series[:, 2].copy(),
        kt_electron=series[:, 3].copy(),
        momentum=momentum,
        block_tau=block_tau,
        block_msd_e=block_msd,
        block_slopes=slopes,
        d_e=d_e,
        d_e_err=d_e_err,
        steps_per_second=config.n_prod / elapsed,
    )
    if rdf_frames:
        result.rdf_r, result.g_ii, result.g_ie, result.g_ee = _normalise_rdf(
            rdf_hist, rdf_frames, n_pairs, system.box, r_max
        )
    return result


def run_md_gpu(config: MDConfig) -> MDResult:
    """Run :func:`run_md` on the GPU regardless of ``config.backend``."""
    if config.backend != "gpu":
        config = MDConfig(**{**config.__dict__, "backend": "gpu"})
    return run_md(config)
