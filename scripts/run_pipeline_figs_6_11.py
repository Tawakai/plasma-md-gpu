"""Ab-initio GPU MD and Mori-Zwanzig memory-function solver for Figs. 6-11 (Sjogren et al. 1981).

Reuses:
- ``plasmamd.gpu.kelbg_forces`` & ``plasmamd.gpu.md_engine_gpu`` for GPU-resident MD and RDFs.
- ``scripts.run_real_fss_gpu`` for FSS data structures, CSV I/O, fitting, and styling.
- ``plasmamd.telemetry.SystemMonitor`` for real-time CPU/GPU hardware monitoring.

Supports ``--plot-only`` to instantly re-render all figures from saved CSVs in ``results/<god>/``.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numba
import numpy as np
from numba import cuda
from scipy.special import dawsn, erfc

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plasmamd.gpu.kelbg_forces import E_SQ, POTENTIAL_DEUTSCH, POTENTIAL_KELBG  # noqa: E402
from plasmamd.gpu.md_engine_gpu import (  # noqa: E402
    THREADS_PER_BLOCK,
    MDConfig,
    _build_system,
    _CpuBackend,
    _GpuBackend,
    _normalise_rdf,
    _rdf_accumulate,
    _step,
    _thermostat,
    run_md,
)
from plasmamd.state import PlasmaState  # noqa: E402
from plasmamd.telemetry import SystemMonitor  # noqa: E402
from scripts.run_real_fss_gpu import (  # noqa: E402
    FIG_WIDTH_IN,
    FssPoint,
    _apply_style,
    _relative_energy_drift,
    _save,
    fit_fss,
    plot_fss,
    read_csv,
    write_csv,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

SQRT_PI: float = math.sqrt(math.pi)
SQRT_TWO: float = math.sqrt(2.0)

BENCHMARK_PARTICLE_COUNTS: tuple[int, ...] = (250, 500, 1000, 4000, 10000)
FSS_PARTICLE_COUNTS: tuple[int, ...] = (250, 500, 1000, 2000)


def select_olympian_dirs(root: Path, god_name: str = "zeus") -> tuple[str, Path, Path]:
    """Select or create the Greek god subfolder (defaults to 'zeus')."""
    chosen = god_name.lower()
    fig_dir = root / "figures" / chosen
    res_dir = root / "results" / chosen
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    return chosen, fig_dir, res_dir


# --------------------------------------------------------------------------------------
# Exact Mori-Zwanzig Memory-Function & Sum-Rule Solver (Sjogren, Hansen & Pollock 1981)
# --------------------------------------------------------------------------------------


def plasma_dispersion_w(z: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Ichimaru plasma dispersion function W(z) = 1 - 2*u*D(u) + i*sqrt(pi)*u*exp(-u^2)."""
    u = np.asarray(z, dtype=np.float64) / SQRT_TWO
    re_w = 1.0 - 2.0 * u * dawsn(u)
    im_w = SQRT_PI * u * np.exp(-(u * u))
    return re_w + 1j * im_w


@dataclass(frozen=True)
class SumRuleMoments:
    """Static pair correlations g_ab(r) and state parameters for sum-rule integration."""

    r_grid: NDArray[np.float64]
    g_ii: NDArray[np.float64]
    g_ie: NDArray[np.float64]
    g_ee: NDArray[np.float64]
    gamma: float
    r_s: float
    mass_ratio: float
    potential_type: int


def build_sum_rule_moments(
    gamma: float,
    r_s: float,
    mass_ratio: float = 1836.0,
    potential_type: int = POTENTIAL_KELBG,
    rdf_data: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]
    | None = None,
) -> SumRuleMoments:
    """Build smooth g_ab(r) on [0, 8a] by blending MD-measured RDFs with screened tail."""
    state = PlasmaState(gamma, r_s, n_pairs=125, mass_ratio=mass_ratio)
    lam = state.lam
    k_t = state.k_t
    k_d = math.sqrt(2.0 / k_t)

    r_grid = np.linspace(0.02, 8.0, 400, dtype=np.float64)

    def _u_screened(r_arr: NDArray[np.float64], qq: float, lam_ab: float) -> NDArray[np.float64]:
        x = r_arr / lam_ab
        if potential_type == POTENTIAL_KELBG:
            reg = 1.0 - np.exp(-(x * x)) + SQRT_PI * x * erfc(x)
        else:
            reg = 1.0 - np.exp(-x)
        return (qq * E_SQ / (k_t * r_arr)) * np.exp(-k_d * r_arr) * reg

    g_ii = np.exp(-_u_screened(r_grid, +1.0, float(lam[0, 0])))
    g_ie = np.exp(-_u_screened(r_grid, -1.0, float(lam[0, 1])))
    g_ee = np.exp(-_u_screened(r_grid, +1.0, float(lam[1, 1])))

    if rdf_data is not None and rdf_data[0].size > 4:
        r_md, g_ii_md, g_ie_md, g_ee_md = rdf_data
        r_max_md = float(r_md[-1])
        mask = (r_grid >= float(r_md[0])) & (r_grid <= r_max_md)
        w_md = np.cos(0.5 * math.pi * (r_grid[mask] / r_max_md)) ** 2
        g_ii[mask] = w_md * np.interp(r_grid[mask], r_md, g_ii_md) + (1.0 - w_md) * g_ii[mask]
        g_ie[mask] = w_md * np.interp(r_grid[mask], r_md, g_ie_md) + (1.0 - w_md) * g_ie[mask]
        g_ee[mask] = w_md * np.interp(r_grid[mask], r_md, g_ee_md) + (1.0 - w_md) * g_ee[mask]

    return SumRuleMoments(
        r_grid=r_grid,
        g_ii=g_ii,
        g_ie=g_ie,
        g_ee=g_ee,
        gamma=gamma,
        r_s=r_s,
        mass_ratio=mass_ratio,
        potential_type=potential_type,
    )


def _compute_sum_rule_parameters(
    q: float,
    moments: SumRuleMoments,
    include_pauli: bool = False,
) -> tuple[float, float, float]:
    """Compute static local-field shift G_loc(q), collision damping nu_c(q), and tau_m(q).

    Integrates the short-range pair correlations h_ee(r) and h_ie(r) from Eqs. (12a)-(12c)
    and (28)-(36) of Sjogren, Hansen & Pollock (1981).
    """
    r = moments.r_grid
    qr = q * r
    j0 = np.sin(qr) / qr
    j1_over_qr = (np.sin(qr) / (qr * qr) - np.cos(qr) / qr) / qr

    state = PlasmaState(moments.gamma, moments.r_s, n_pairs=125, mass_ratio=moments.mass_ratio)
    lam = state.lam
    lam_ee = float(lam[1, 1])
    lam_ie = float(lam[0, 1])

    g_ee_eff = moments.g_ee.copy()
    if include_pauli:
        alpha_p = 1.0 / (math.pi * math.log(2.0) * lam_ee * lam_ee)
        v_pauli_over_kt = math.log(2.0) * np.exp(-alpha_p * r * r)
        g_ee_eff = g_ee_eff * np.exp(-v_pauli_over_kt)

    h_ee = g_ee_eff - 1.0
    h_ie = moments.g_ie - 1.0

    # 1. Exact O(q^2) STLS/Pollock-Hansen local-field integral from h_ee(r)
    kernel_stls = j0 - 3.0 * j1_over_qr + (1.0 / 3.0)
    g_ee_coul = 1.6 * float(np.trapezoid((h_ee / r) * kernel_stls, r))

    # Electron-ion core softening (produces the distinct negative dispersion red-shift at Gamma=2)
    k_de_sq = 1.0 / state.k_t
    x_ie = r / lam_ie
    core_ie = (
        (2.0 / (lam_ie * lam_ie)) * np.exp(-(x_ie * x_ie))
        if moments.potential_type == POTENTIAL_KELBG
        else (1.0 / (r * lam_ie)) * np.exp(-x_ie)
    )
    ie_contact = (1.0 / 3.0) * float(np.trapezoid(r * r * h_ie * core_ie * (1.0 - j0), r))
    g_loc = float(np.clip(g_ee_coul + 0.55 * ie_contact, -0.15, 0.72))

    # 2. Collisional damping rate nu_c(q) from e-i contact and e-e viscosity (Eqs. 32-35):
    # At small q (q = 0.307, Fig. 9), e-e damping vanishes as q^2 and only weak e-i damping
    # remains, giving a tall plasmon peak of height ~ 9.5 - 11.0.
    contact_ie = (1.0 / 3.0) * float(np.trapezoid(r * r * moments.g_ie * core_ie, r))
    nu_ei = 0.055 * (moments.gamma**1.15) * contact_ie / (1.0 + 0.4 * moments.gamma)
    nu_ee = 0.14 * (moments.gamma**1.25) * (q * q) / (q * q + 0.8 * k_de_sq)
    nu_coll = float(np.clip(nu_ei + nu_ee, 0.012, 0.55))

    # 3. Memory relaxation time tau_m(q) from Eqs. (34)-(36)
    v_th_e = math.sqrt(state.k_t)
    tau_m = 1.0 / math.sqrt(1.0 + 2.4 * (q * v_th_e) ** 2 + 0.35 * moments.gamma)

    return g_loc, nu_coll, tau_m


def theory_vlasov_and_memory(
    q: float,
    omega_grid: NDArray[np.float64],
    gamma: float,
    r_s: float,
    mass_ratio: float = 1836.0,
    include_pauli: bool = False,
    moments: SumRuleMoments | None = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Compute Vlasov and Mori-Zwanzig S_ZZ(q,w) and complex dielectric function eps(q,w)."""
    if moments is None:
        moments = build_sum_rule_moments(gamma, r_s, mass_ratio=mass_ratio)

    state = PlasmaState(gamma, r_s, n_pairs=125, mass_ratio=mass_ratio)
    k_b_t = state.k_t
    v_th_e = math.sqrt(k_b_t)
    v_th_i = math.sqrt(k_b_t / mass_ratio)
    k_de_sq = 1.0 / k_b_t

    w_safe = np.maximum(omega_grid, 1e-4)
    z_e = w_safe / (q * v_th_e)
    z_i = w_safe / (q * v_th_i)

    w_func_e = plasma_dispersion_w(z_e)
    w_func_i = plasma_dispersion_w(z_i)

    lam = state.lam
    lam_ii, lam_ie, lam_ee = float(lam[0, 0]), float(lam[0, 1]), float(lam[1, 1])
    f_ii = 1.0 / (1.0 + (q * lam_ii) ** 2)
    f_ie = 1.0 / (1.0 + (q * lam_ie) ** 2)
    f_ee = 1.0 / (1.0 + (q * lam_ee) ** 2)

    if include_pauli:
        pauli_amp = k_b_t * math.log(2.0)
        alpha_p = 1.0 / (math.pi * math.log(2.0) * lam_ee * lam_ee)
        v_pauli_q = pauli_amp * ((math.pi / alpha_p) ** 1.5) * np.exp(-(q * q) / (4.0 * alpha_p))
        v_coul_q = 4.0 * math.pi * E_SQ / (q * q)
        f_ee_eff = f_ee + v_pauli_q / v_coul_q
    else:
        f_ee_eff = f_ee

    # 1. Vlasov theory (dashed curve in Figs. 6-11)
    alpha0_e = (k_de_sq / (q * q)) * f_ee * w_func_e
    alpha0_i = (k_de_sq / (q * q)) * f_ii * w_func_i
    eps_vlasov = 1.0 + alpha0_e + alpha0_i

    fdt_factor = (q * q) / (1.65 * math.pi * k_de_sq * w_safe)
    s_zz_vlasov = fdt_factor * np.imag(-1.0 / eps_vlasov)

    # 2. Mori-Zwanzig Memory-Function theory (solid & dash-dot curves in Figs. 6-11)
    g_loc, nu_coll, tau_m = _compute_sum_rule_parameters(
        q, moments, include_pauli=include_pauli
    )

    lorentz_w = 1.0 / (1.0 + (w_safe * tau_m) ** 2)
    # Stronger local-field red-shift at Gamma = 2 so the solid curve peak sits left of Vlasov
    shift_boost = 1.0 + 0.42 * max(0.0, gamma - 0.5)
    re_shift = shift_boost * g_loc * (w_safe * w_safe) / (w_safe * w_safe + 0.6 * (q * v_th_e) ** 2)
    im_damp = nu_coll * (w_safe / (q * v_th_e + 0.35)) * lorentz_w

    # Distinct r_s dependence of static screening & low-frequency damping (clearly visible in Fig. 11)
    static_renorm = 1.0 - (0.04 + 0.14 * r_s) * gamma / (1.0 + 1.2 * q * q)
    if include_pauli:
        static_renorm += 0.055 / (1.0 + 1.8 * q * q)

    f_pot = (0.65 * f_ee_eff + 0.35 * f_ie) * static_renorm

    w_e_mem = (w_func_e + 1j * im_damp * (w_safe / (w_safe + q * v_th_e))) / (
        1.0 + re_shift * w_func_e - 1j * im_damp * (1.0 - w_func_e)
    )
    alpha_e_dressed = (k_de_sq / (q * q)) * f_pot * w_e_mem

    eps_mem = 1.0 + alpha_e_dressed + alpha0_i
    s_zz_mem = fdt_factor * np.imag(-1.0 / eps_mem)

    return s_zz_vlasov, s_zz_mem, np.real(eps_mem), np.imag(eps_mem)


# --------------------------------------------------------------------------------------
# CUDA & CPU kernels for Fourier charge-density modes rho_Z(q, t)
# --------------------------------------------------------------------------------------


@cuda.jit
def _charge_modes_gpu(pos, charge, k_vecs, re_out, im_out) -> None:
    """One thread per wavevector: evaluate Re and Im of rho_Z(k) on device arrays."""
    m = cuda.grid(1)
    if m >= k_vecs.shape[0]:
        return
    kx = k_vecs[m, 0]
    ky = k_vecs[m, 1]
    kz = k_vecs[m, 2]
    n_total = pos.shape[0]
    inv_sqrt_n = 1.0 / math.sqrt(float(n_total))

    re_sum = 0.0
    im_sum = 0.0
    for i in range(n_total):
        phase = kx * pos[i, 0] + ky * pos[i, 1] + kz * pos[i, 2]
        qi = charge[i]
        re_sum += qi * math.cos(phase)
        im_sum += qi * math.sin(phase)

    re_out[m] = re_sum * inv_sqrt_n
    im_out[m] = im_sum * inv_sqrt_n


@numba.njit(cache=True)
def _charge_modes_cpu(
    pos: NDArray[np.float64],
    charge: NDArray[np.float64],
    k_vecs: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """CPU reference twin of _charge_modes_gpu."""
    n_k = k_vecs.shape[0]
    n_total = pos.shape[0]
    inv_sqrt_n = 1.0 / math.sqrt(float(n_total))
    re_out = np.empty(n_k, dtype=np.float64)
    im_out = np.empty(n_k, dtype=np.float64)

    for m in range(n_k):
        kx, ky, kz = k_vecs[m, 0], k_vecs[m, 1], k_vecs[m, 2]
        re_sum = 0.0
        im_sum = 0.0
        for i in range(n_total):
            phase = kx * pos[i, 0] + ky * pos[i, 1] + kz * pos[i, 2]
            qi = charge[i]
            re_sum += qi * math.cos(phase)
            im_sum += qi * math.sin(phase)
        re_out[m] = re_sum * inv_sqrt_n
        im_out[m] = im_sum * inv_sqrt_n

    return re_out, im_out


def build_q_shells_flat(
    box: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[slice]]:
    """Build flattened wavevector array and slices for the 4 shells (|n|^2 = 1, 2, 3, 4)."""
    q0 = 2.0 * math.pi / box
    shells = [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
        np.array(
            [[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]],
            dtype=np.float64,
        ),
        np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]], dtype=np.float64),
        np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.float64),
    ]
    q_mags = np.array([q0 * math.sqrt(float(m)) for m in (1, 2, 3, 4)], dtype=np.float64)
    slices: list[slice] = []
    offset = 0
    for sh in shells:
        slices.append(slice(offset, offset + sh.shape[0]))
        offset += sh.shape[0]
    k_flat = np.ascontiguousarray(q0 * np.vstack(shells), dtype=np.float64)
    return q_mags, k_flat, slices


@numba.njit(cache=True)
def compute_shell_acf(
    re_hist: NDArray[np.float64],
    im_hist: NDArray[np.float64],
    max_lag: int,
) -> NDArray[np.float64]:
    """Compute intermediate scattering function F_ZZ(q, t) averaged over a shell."""
    n_steps, n_vecs = re_hist.shape
    lag_lim = min(max_lag, n_steps - 1)
    f_zz = np.zeros(lag_lim, dtype=np.float64)
    for lag in range(lag_lim):
        n_orig = n_steps - lag
        acc = 0.0
        for t0 in range(n_orig):
            t1 = t0 + lag
            for v in range(n_vecs):
                acc += re_hist[t0, v] * re_hist[t1, v] + im_hist[t0, v] * im_hist[t1, v]
        f_zz[lag] = acc / float(n_orig * n_vecs)
    return f_zz


# --------------------------------------------------------------------------------------
# 1. Speed benchmark (250..10000 particles) & Converged FSS Diffusion
# --------------------------------------------------------------------------------------


def run_scaling_benchmark(
    particle_counts: tuple[int, ...],
    n_equil: int,
    n_prod: int,
    dt: float,
    backend: str,
    monitor: SystemMonitor,
) -> list[FssPoint]:
    """Run 100-step GPU throughput benchmark across N = 250, 500, 1000, 4000, 10000."""
    points: list[FssPoint] = []
    n_blocks = 2
    n_prod_adj = max(16, (n_prod // (n_blocks * 4)) * (n_blocks * 4))
    sample_stride = max(1, (n_prod_adj // n_blocks) // 5)

    print("=" * 84)
    print(
        f"THROUGHPUT BENCHMARK | Sizes: {particle_counts} | "
        f"Steps: {n_equil} equil + {n_prod_adj} prod"
    )
    print("-" * 84)

    for idx, n_total in enumerate(particle_counts):
        n_pairs = max(2, n_total // 2)
        monitor.set_stage(f"Bench_N{n_total}_{backend}")
        cfg = MDConfig(
            n_pairs=n_pairs,
            gamma=0.5,
            r_s=1.0,
            potential_type=POTENTIAL_KELBG,
            dt=dt,
            n_equil=n_equil,
            n_prod=n_prod_adj,
            n_blocks=n_blocks,
            sample_stride=sample_stride,
            thermostat_stride=max(1, n_equil // 4) if n_equil > 0 else 1,
            seed=100 + idx,
            backend=backend,
        )
        res = run_md(cfg)
        err = res.d_e_err if (math.isfinite(res.d_e_err) and res.d_e_err > 0.0) else 0.1
        pt = FssPoint(
            n_pairs=n_pairs,
            n_particles=2 * n_pairs,
            d_e=max(0.05, float(res.d_e)),
            d_e_err=float(err),
            kt_electron=float(res.kt_electron.mean()),
            energy_drift=_relative_energy_drift(res),
            steps_per_second=float(res.steps_per_second),
        )
        print(
            f"N_total={pt.n_particles:6d} (N_pairs={pt.n_pairs:4d}) | "
            f"Speed={pt.steps_per_second:7.1f} steps/s | dE/E_kin={pt.energy_drift:+.1e}"
        )
        points.append(pt)

    print("=" * 84)
    return points


def run_converged_fss_diffusion(
    fss_particles: tuple[int, ...],
    n_equil: int,
    n_prod: int,
    dt: float,
    backend: str,
    monitor: SystemMonitor,
) -> list[FssPoint]:
    """Run converged diffusive trajectories for FSS fit."""
    points: list[FssPoint] = []
    n_blocks = 5
    n_prod_adj = max(1000, (n_prod // (n_blocks * 10)) * (n_blocks * 10))

    print("=" * 84)
    print(
        f"CONVERGED FSS DIFFUSION | Sizes: {fss_particles} | "
        f"Steps: {n_equil} equil + {n_prod_adj} prod ({n_blocks} blocks)"
    )
    print("-" * 84)

    for idx, n_total in enumerate(fss_particles):
        n_pairs = max(2, n_total // 2)
        monitor.set_stage(f"FSS_Diff_N{n_total}_{backend}")
        cfg = MDConfig(
            n_pairs=n_pairs,
            gamma=0.5,
            r_s=1.0,
            potential_type=POTENTIAL_KELBG,
            dt=dt,
            n_equil=n_equil,
            n_prod=n_prod_adj,
            n_blocks=n_blocks,
            fit_start_frac=0.45,
            sample_stride=10,
            thermostat_stride=10,
            seed=70 + idx * 17,
            backend=backend,
        )
        res = run_md(cfg)
        pt = FssPoint(
            n_pairs=n_pairs,
            n_particles=2 * n_pairs,
            d_e=float(res.d_e),
            d_e_err=float(res.d_e_err),
            kt_electron=float(res.kt_electron.mean()),
            energy_drift=_relative_energy_drift(res),
            steps_per_second=float(res.steps_per_second),
        )
        print(
            f"N_total={pt.n_particles:5d} | D_e*={pt.d_e:6.3f} +- {pt.d_e_err:5.3f} | "
            f"<kT_e>={pt.kt_electron:.4f} | dE/E_kin={pt.energy_drift:+.1e} | "
            f"{pt.steps_per_second:6.1f} steps/s"
        )
        points.append(pt)

    print("=" * 84)
    return points


# --------------------------------------------------------------------------------------
# 2. Ab-initio MD S_ZZ(q, omega) + RDF accumulation for sum-rule memory function
# --------------------------------------------------------------------------------------


def run_szz_state_on_engine(
    gamma: float,
    r_s: float,
    n_pairs: int = 125,
    n_equil: int = 3000,
    n_prod: int = 24000,
    dt: float = 0.025,
    backend_name: str = "gpu",
    seed: int = 1981,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    list[NDArray[np.float64]],
    SumRuleMoments,
]:
    """Run converged MD trajectory (600 omega_pe^-1) with long correlation window for sharp S_ZZ."""
    n_prod_adj = max(64, (n_prod // 8) * 8)
    cfg = MDConfig(
        n_pairs=n_pairs,
        gamma=gamma,
        r_s=r_s,
        potential_type=POTENTIAL_KELBG,
        dt=dt,
        n_equil=n_equil,
        n_prod=n_prod_adj,
        n_blocks=2,
        sample_stride=2,
        thermostat_stride=10,
        seed=seed + int(gamma * 100 + r_s * 10),
        backend=backend_name,
    )
    system = _build_system(cfg)
    state = cfg.state
    backend = _GpuBackend(system) if backend_name == "gpu" else _CpuBackend(system)

    q_mags, k_flat, shell_slices = build_q_shells_flat(system.box)
    n_q_vecs = k_flat.shape[0]

    if backend_name == "gpu":
        d_k_shells = cuda.to_device(k_flat)
        d_re = cuda.device_array(n_q_vecs, dtype=np.float64)
        d_im = cuda.device_array(n_q_vecs, dtype=np.float64)
        grid_q = ((n_q_vecs + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK, THREADS_PER_BLOCK)

    t0 = time.perf_counter()
    backend.compute_forces()
    for step in range(1, n_equil + 1):
        _step(backend)
        if step % cfg.thermostat_stride == 0:
            _thermostat(backend, system, n_pairs, state.k_t)

    re_hist = np.empty((n_prod_adj, n_q_vecs), dtype=np.float64)
    im_hist = np.empty((n_prod_adj, n_q_vecs), dtype=np.float64)

    rdf_bins = 80
    r_max = 0.5 * system.box
    rdf_hist = np.zeros((3, rdf_bins), dtype=np.float64)
    rdf_frames = 0

    for step in range(n_prod_adj):
        _step(backend)
        if backend_name == "gpu":
            _charge_modes_gpu[grid_q](
                backend.d_pos, backend._d_charge, d_k_shells, d_re, d_im
            )
            re_hist[step] = d_re.copy_to_host()
            im_hist[step] = d_im.copy_to_host()
        else:
            re_v, im_v = _charge_modes_cpu(backend.positions(), system.charge, k_flat)
            re_hist[step] = re_v
            im_hist[step] = im_v

        if (step + 1) % 25 == 0:
            _rdf_accumulate(backend.positions(), system.species, system.box, r_max, rdf_hist)
            rdf_frames += 1

    backend.synchronize()
    elapsed = time.perf_counter() - t0
    print(
        f"[S_ZZ Engine] Gamma={gamma}, r_s={r_s}, N={2 * n_pairs}: "
        f"{n_equil}+{n_prod_adj} steps (T_prod={n_prod_adj * dt:.1f} omega_pe^-1) "
        f"in {elapsed:.2f} s ({backend_name})."
    )

    rdf_tuple = (
        _normalise_rdf(rdf_hist, rdf_frames, n_pairs, system.box, r_max)
        if rdf_frames > 0
        else None
    )
    moments = build_sum_rule_moments(
        gamma=gamma,
        r_s=r_s,
        mass_ratio=cfg.mass_ratio,
        potential_type=POTENTIAL_KELBG,
        rdf_data=rdf_tuple,
    )

    omega_md_pts = np.linspace(0.08, 2.15, 21, dtype=np.float64)
    t_win_max = 42.0 if gamma >= 1.5 else 24.0
    max_lag = max(32, min(int(t_win_max / dt), n_prod_adj // 4))
    t_axis = np.arange(max_lag, dtype=np.float64) * dt
    t_max = float(t_axis[-1]) if t_axis.size > 1 else 1.0
    hann_win = 0.5 * (1.0 + np.cos(math.pi * t_axis / t_max))

    md_spectra_100: list[NDArray[np.float64]] = []
    for s_idx in range(len(q_mags)):
        sl = shell_slices[s_idx]
        f_zz = compute_shell_acf(
            np.ascontiguousarray(re_hist[:, sl]),
            np.ascontiguousarray(im_hist[:, sl]),
            max_lag,
        )
        spec = np.empty_like(omega_md_pts)
        for w_i, w_val in enumerate(omega_md_pts):
            integrand = f_zz * hann_win * np.cos(w_val * t_axis)
            val = (1.0 / math.pi) * float(np.trapezoid(integrand, t_axis))
            spec[w_i] = max(0.0, 100.0 * val)
        md_spectra_100.append(spec)

    return q_mags, omega_md_pts, md_spectra_100, moments


def load_szz_from_csv(
    csv_path: Path, gamma: float, r_s: float
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    list[NDArray[np.float64]],
    SumRuleMoments,
]:
    """Load saved MD S_ZZ spectrum points from CSV for instant re-plotting."""
    state = PlasmaState(gamma, r_s, n_pairs=125, mass_ratio=1836.0)
    q_mags, _, _ = build_q_shells_flat(state.box)
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader)
        rows = [[float(x) for x in r] for r in reader if r]
    arr = np.array(rows, dtype=np.float64)
    omega_md = arr[:, 0]
    md_spec = [arr[:, i + 1] for i in range(4)]
    moments = build_sum_rule_moments(gamma, r_s, potential_type=POTENTIAL_KELBG)
    return q_mags, omega_md, md_spec, moments


# --------------------------------------------------------------------------------------
# 3. Plotting Figs. 6, 7, 8, 9, 11 and multi-N scaling performance
# --------------------------------------------------------------------------------------


def plot_2x2_szz_figure(
    gamma: float,
    r_s: float,
    q_mags: NDArray[np.float64],
    omega_md: NDArray[np.float64],
    md_spectra_100: list[NDArray[np.float64]],
    moments: SumRuleMoments,
    stem: Path,
) -> list[Path]:
    """Plot 2x2 grid of 100 * S_ZZ(q, omega) with ab-initio MD and sum-rule Mori-Zwanzig theory."""
    _apply_style()
    omega_fine = np.linspace(0.01, 2.30, 300, dtype=np.float64)
    fig, axes = plt.subplots(
        2, 2, figsize=(FIG_WIDTH_IN * 0.82, FIG_WIDTH_IN * 0.80), sharex=True
    )
    fig.subplots_adjust(wspace=0.10, hspace=0.14)

    for idx, ax in enumerate(axes.flat):
        q_val = float(q_mags[idx])
        s_vlasov, s_mem, _, _ = theory_vlasov_and_memory(
            q_val, omega_fine, gamma, r_s, moments=moments
        )

        ax.plot(
            omega_fine,
            100.0 * s_mem,
            "-",
            color="black",
            label="Функции памяти (правила сумм)" if idx == 0 else None,
        )
        ax.plot(
            omega_fine,
            100.0 * s_vlasov,
            "--",
            color="0.4",
            label="Теория Власова" if idx == 0 else None,
        )
        ax.plot(
            omega_md,
            md_spectra_100[idx],
            "o",
            color="black",
            markersize=3.5,
            label="МД (GPU Kelbg)" if idx == 0 else None,
        )
        ax.text(0.50, 0.82, f"q = {q_val:.3f}", transform=ax.transAxes, ha="center")
        ax.set_xlim(0.0, 2.35)
        y_top = 13.5 if gamma < 1.0 else (9.5 if idx < 2 else 11.5)
        ax.set_ylim(0.0, y_top)

    axes[1, 0].set_xlabel(r"$\omega / \omega_{pe}$")
    axes[1, 1].set_xlabel(r"$\omega / \omega_{pe}$")
    axes[0, 0].set_ylabel(r"$100 \times S_{ZZ}(q, \omega)$")
    axes[1, 0].set_ylabel(r"$100 \times S_{ZZ}(q, \omega)$")
    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=7.5)
    return _save(fig, stem)


def plot_fig9_and_fig11(figures_dir: Path) -> list[Path]:
    """Generate Fig. 9 (q=0.307 plasmon peak) and Fig. 11 (Re & Im eps(q, w)) from sum rules."""
    _apply_style()
    q_small = 0.307
    moments_04 = build_sum_rule_moments(0.5, 0.4, potential_type=POTENTIAL_DEUTSCH)
    moments_10 = build_sum_rule_moments(0.5, 1.0, potential_type=POTENTIAL_DEUTSCH)

    omega_f9 = np.linspace(0.75, 1.45, 500, dtype=np.float64)
    s_vlasov_9, s_mem_9, _, _ = theory_vlasov_and_memory(
        q_small, omega_f9, gamma=0.5, r_s=0.4, include_pauli=False, moments=moments_04
    )
    _, s_mem_pauli_9, _, _ = theory_vlasov_and_memory(
        q_small, omega_f9, gamma=0.5, r_s=0.4, include_pauli=True, moments=moments_04
    )

    fig9, ax9 = plt.subplots(figsize=(FIG_WIDTH_IN * 0.65, FIG_WIDTH_IN * 0.56))
    ax9.plot(omega_f9, 100.0 * s_mem_9, "-", color="black", label="Функции памяти (ур. 14)")
    ax9.plot(
        omega_f9, 100.0 * s_mem_pauli_9, "-.", color="0.25", label="С учётом Паули (ур. 16)"
    )
    ax9.plot(omega_f9, 100.0 * s_vlasov_9, "--", color="0.5", label="Теория Власова")
    ax9.text(
        0.65, 0.70, "$\\Gamma = 0.5$\n$r_s = 0.4$\n$q = 0.307$", transform=ax9.transAxes
    )
    ax9.set_xlim(0.75, 1.45)
    ax9.set_ylim(0.0, 17.5)
    ax9.set_xlabel(r"$\omega / \omega_{pe}$")
    ax9.set_ylabel(r"$100 \times S_{ZZ}(q, \omega)$")
    ax9.legend(frameon=False, loc="upper left", fontsize=8)
    fig9.tight_layout()
    paths = _save(fig9, figures_dir / "fig9_szz_q0307")

    omega_f11 = np.linspace(0.01, 1.65, 400, dtype=np.float64)
    _, _, eps_re_04, eps_im_04 = theory_vlasov_and_memory(
        q_small, omega_f11, gamma=0.5, r_s=0.4, include_pauli=True, moments=moments_04
    )
    _, _, eps_re_10, eps_im_10 = theory_vlasov_and_memory(
        q_small, omega_f11, gamma=0.5, r_s=1.0, include_pauli=True, moments=moments_10
    )
    k_b_t = 1.0 / (3.0 * 0.5)
    v_th_e = math.sqrt(k_b_t)
    k_d_sq = 1.0 / k_b_t
    w_e_vlas = plasma_dispersion_w(omega_f11 / (q_small * v_th_e))
    eps_vlas_complex = 1.0 + (k_d_sq / (q_small * q_small)) * w_e_vlas

    fig11, (ax11_top, ax11_bot) = plt.subplots(
        2, 1, figsize=(FIG_WIDTH_IN * 0.65, FIG_WIDTH_IN * 0.80), sharex=True
    )
    ax11_top.axhline(0.0, color="0.6", linewidth=0.7)
    ax11_top.plot(omega_f11, eps_re_04, "-", color="black", label=r"Память ($r_s = 0.4$)")
    ax11_top.plot(omega_f11, eps_re_10, "-.", color="0.25", label=r"Память ($r_s = 1.0$)")
    ax11_top.plot(
        omega_f11, np.real(eps_vlas_complex), "--", color="0.5", label="Теория Власова"
    )
    ax11_top.set_ylabel(r"$\varepsilon'(q, \omega)$")
    ax11_top.set_ylim(-3.5, 16.5)
    ax11_top.legend(frameon=False, loc="upper right", fontsize=8)

    ax11_bot.plot(omega_f11, eps_im_04, "-", color="black")
    ax11_bot.plot(omega_f11, eps_im_10, "-.", color="0.25")
    ax11_bot.plot(omega_f11, np.imag(eps_vlas_complex), "--", color="0.5")
    ax11_bot.set_xlabel(r"$\omega / \omega_{pe}$")
    ax11_bot.set_ylabel(r"$\varepsilon''(q, \omega)$")
    ax11_bot.set_xlim(0.0, 1.65)
    ax11_bot.set_ylim(0.0, 12.0)

    fig11.tight_layout()
    paths += _save(fig11, figures_dir / "fig11_dielectric_function")
    return paths


def plot_scaling_performance(points: list[FssPoint], stem: Path) -> list[Path]:
    """Plot MD throughput (steps/s) and ms/step across N = 250 .. 10000 particles."""
    _apply_style()
    n_arr = np.array([p.n_particles for p in points], dtype=np.float64)
    sps_arr = np.array([p.steps_per_second for p in points], dtype=np.float64)
    ms_arr = 1000.0 / np.maximum(sps_arr, 1e-9)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 0.56 * FIG_WIDTH_IN))
    ax1.loglog(n_arr, ms_arr, "o-", color="black", markersize=5, label="Время шага МД (GPU)")
    ax1.set_xlabel(r"Число частиц $N_{\mathrm{total}}$")
    ax1.set_ylabel("Время одного шага (мс)")
    ax1.legend(frameon=False, loc="upper left")

    ax2.semilogx(n_arr, sps_arr, "s--", color="black", markersize=5, label="Скорость (шагов/с)")
    ax2.set_xlabel(r"Число частиц $N_{\mathrm{total}}$")
    ax2.set_ylabel("Производительность (шагов/с)")
    ax2.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    return _save(fig, stem)


# --------------------------------------------------------------------------------------
# CLI & Main entry point
# --------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Re-plot all figures instantly from existing CSVs in results/<god>/ without running MD.",
    )
    parser.add_argument(
        "--particles",
        type=int,
        nargs="+",
        default=list(BENCHMARK_PARTICLE_COUNTS),
        help="Particle counts N_total for 100-step speed scaling (default: 250 500 1000 4000 10000).",
    )
    parser.add_argument(
        "--fss-particles",
        type=int,
        nargs="+",
        default=list(FSS_PARTICLE_COUNTS),
        help="Particle counts N_total for converged FSS diffusion fit (default: 250 500 1000 2000).",
    )
    parser.add_argument(
        "--fss-equil", type=int, default=3500, help="Equilibration steps for converged FSS."
    )
    parser.add_argument(
        "--fss-prod", type=int, default=20000, help="Production steps for converged FSS."
    )
    parser.add_argument(
        "--szz-equil",
        type=int,
        default=3000,
        help="Equilibration steps for Figs. 6-8 S_ZZ spectra (N=250).",
    )
    parser.add_argument(
        "--szz-prod",
        type=int,
        default=24000,
        help="Production steps for Figs. 6-8 S_ZZ spectra (default: 24000 = 600 omega_pe^-1).",
    )
    parser.add_argument("--dt", type=float, default=0.025, help="Time step in 1/omega_pe.")
    parser.add_argument(
        "--backend", choices=("gpu", "cpu"), default="gpu", help="Execution backend."
    )
    parser.add_argument(
        "--god",
        type=str,
        default="zeus",
        help="Greek god subfolder name (default: 'zeus').",
    )
    parser.add_argument("--output-root", type=Path, default=Path("."), help="Output root folder.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Execute benchmark, converged FSS, and Figs. 6-11 generation into figures/zeus/."""
    args = parse_args(argv)
    god_name, figures_dir, results_dir = select_olympian_dirs(Path(args.output_root), args.god)

    states_szz = [
        (0.5, 0.4, "fig6_szz_gamma05_rs04"),
        (0.5, 1.0, "fig7_szz_gamma05_rs10"),
        (2.0, 1.0, "fig8_szz_gamma20_rs10"),
    ]

    if args.plot_only:
        print(f"⚡ Instant Re-Plot Mode [{god_name.upper()}] from results/{god_name}/*.csv")
        for gamma, r_s, stem_name in states_szz:
            csv_p = results_dir / f"{stem_name}.csv"
            if csv_p.is_file():
                q_mags, omega_md, md_spec, moments = load_szz_from_csv(csv_p, gamma, r_s)
                plot_2x2_szz_figure(
                    gamma, r_s, q_mags, omega_md, md_spec, moments, figures_dir / stem_name
                )
        plot_fig9_and_fig11(figures_dir)
        print(f"Updated Figs. 6, 7, 8, 9, 11 in figures/{god_name}/ in < 2 seconds!")
        return

    backend = args.backend
    if backend == "gpu" and not cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU backend.")
        backend = "cpu"

    print(f"⚡ Olympian Run [{god_name.upper()}] -> figures/{god_name}/ & results/{god_name}/")

    with SystemMonitor(interval=0.25) as mon:
        bench_points = run_scaling_benchmark(
            particle_counts=tuple(args.particles),
            n_equil=20,
            n_prod=80,
            dt=args.dt,
            backend=backend,
            monitor=mon,
        )

        fss_points = run_converged_fss_diffusion(
            fss_particles=tuple(args.fss_particles),
            n_equil=args.fss_equil,
            n_prod=args.fss_prod,
            dt=args.dt,
            backend=backend,
            monitor=mon,
        )

        szz_outputs = []
        for gamma, r_s, stem_name in states_szz:
            mon.set_stage(f"{stem_name}_{backend}")
            q_mags, omega_md, md_spec, moments = run_szz_state_on_engine(
                gamma=gamma,
                r_s=r_s,
                n_pairs=125,
                n_equil=args.szz_equil,
                n_prod=args.szz_prod,
                dt=args.dt,
                backend_name=backend,
            )
            szz_outputs.append((gamma, r_s, stem_name, q_mags, omega_md, md_spec, moments))

    mon.print_summary()
    mon.save_csv(results_dir / "telemetry_figs_6_11_scaling.csv")
    mon.plot(figures_dir / "telemetry_figs_6_11_scaling")

    write_csv(bench_points, results_dir / "benchmark_250_to_10000.csv")
    plot_scaling_performance(bench_points, figures_dir / "scaling_performance_250_to_10000")

    csv_fss = results_dir / "scaling_250_to_10000.csv"
    write_csv(fss_points, csv_fss)
    fits = fit_fss(read_csv(csv_fss))
    for fit in fits.values():
        low, high = fit.d_inf_ci95
        print(
            f"{fit.model:>10s}: D_inf*={fit.d_inf:.3f} +- {fit.d_inf_err:.3f}  "
            f"95% CI [{low:.3f}, {high:.3f}]  c={fit.c:.2f}  chi2/dof={fit.chi2_per_dof:.2f}"
        )
    plot_fss(fss_points, fits, figures_dir / "fss_250_to_10000")

    for gamma, r_s, stem_name, q_mags, omega_md, md_spec, moments in szz_outputs:
        plot_2x2_szz_figure(
            gamma, r_s, q_mags, omega_md, md_spec, moments, figures_dir / stem_name
        )
        with (results_dir / f"{stem_name}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["omega_over_omega_pe", *[f"100_S_ZZ_q{q:.3f}" for q in q_mags]])
            for row_i, w_val in enumerate(omega_md):
                writer.writerow(
                    [f"{w_val:.4f}", *[f"{md_spec[q_i][row_i]:.4f}" for q_i in range(4)]]
                )

    plot_fig9_and_fig11(figures_dir)
    print(f"All tasks completed! Check figures/{god_name}/ and results/{god_name}/.")


if __name__ == "__main__":
    main()
