"""Hermes Multi-Scale Pipeline: separate tables (I-V) and plots (Figs. 6-11) for N = 250, 1000, 5000, 12500.

For each particle count N_total, runs GPU MD once per state and evaluates BOTH:
  - ``hephaestus`` (strict first-principles HNC + Resibois + Mori-Zwanzig),
  - ``apollo``     (earlier phenomenological / fitted memory-function & transport formulas).

Outputs are organized into:
  figures/hermes/N{250,1000,5000,12500}/{hephaestus,apollo}/
  results/hermes/N{250,1000,5000,12500}/{hephaestus,apollo}/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from numba import cuda

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plasmamd.gpu.kelbg_forces import POTENTIAL_DEUTSCH, POTENTIAL_KELBG  # noqa: E402
from plasmamd.gpu.md_engine_gpu import (  # noqa: E402
    THREADS_PER_BLOCK,
    MDConfig,
    _build_system,
    _CpuBackend,
    _GpuBackend,
    _species_summary,
    _step,
    _thermostat,
    diffusion_from_blocks,
    self_energy,
)
from plasmamd.state import PlasmaState  # noqa: E402
from plasmamd.telemetry import SystemMonitor  # noqa: E402
from scripts.run_hephaestus_figs_6_11 import (  # noqa: E402
    plasma_dispersion_w,
    plot_fig9_and_fig11,
    theory_vlasov_and_memory_ab_initio,
)
from scripts.run_hephaestus_tables import (  # noqa: E402
    PAPER_TABLE_I,
    PAPER_TABLE_II,
    PAPER_TABLE_III,
    PAPER_TABLE_IV,
    PAPER_TABLE_V,
    HNCSolution,
    compute_table_i_state,
    compute_table_iii_transport_ab_initio,
    compute_table_iv_ab_initio,
    compute_table_v_row_ab_initio,
    deutsch_wavelengths,
    render_table_figure,
    save_csv_table,
    solve_two_component_hnc,
)
from scripts.run_pipeline_figs_6_11 import (  # noqa: E402
    _charge_modes_cpu,
    _charge_modes_gpu,
    build_q_shells_flat,
    compute_shell_acf,
)
from scripts.run_real_fss_gpu import FIG_WIDTH_IN, _apply_style, _save  # noqa: E402

if TYPE_CHECKING:
    from numpy.typing import NDArray

DEFAULT_PARTICLES: tuple[int, ...] = (250, 1000, 5000, 12500)
STATES: tuple[tuple[float, float, str], ...] = (
    (0.5, 0.4, "fig6_szz_gamma05_rs04"),
    (0.5, 1.0, "fig7_szz_gamma05_rs10"),
    (2.0, 1.0, "fig8_szz_gamma20_rs10"),
)

# Optimal step budgets per N_total so N=12500 completes cleanly on a single GPU
STEP_BUDGETS: dict[int, tuple[int, int]] = {
    250: (2000, 10000),
    1000: (1500, 6000),
    5000: (600, 2000),
    12500: (300, 1000),
}


# --------------------------------------------------------------------------------------
# Apollo (Phenomenological / Fitted) Theoretical Formulations for Side-by-Side Comparison
# --------------------------------------------------------------------------------------


def apollo_theory_szz(
    q: float,
    omega_grid: NDArray[np.float64],
    gamma: float,
    r_s: float,
    hnc: HNCSolution,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Earlier phenomenological Mori-Zwanzig S_ZZ(q, w) curve (Apollo/Zeus approach)."""
    state = PlasmaState(gamma, r_s, n_pairs=125, mass_ratio=1836.0)
    k_b_t = state.k_t
    v_th_e = math.sqrt(k_b_t)
    v_th_i = math.sqrt(k_b_t / 1836.0)
    k_de_sq = 1.0 / k_b_t

    w_safe = np.maximum(omega_grid, 1e-4)
    w_func_e = plasma_dispersion_w(w_safe / (q * v_th_e))
    w_func_i = plasma_dispersion_w(w_safe / (q * v_th_i))

    lam_ii, lam_ie, lam_ee = deutsch_wavelengths(gamma, r_s, 1836.0)
    f_ii = 1.0 / (1.0 + (q * lam_ii) ** 2)
    f_ie = 1.0 / (1.0 + (q * lam_ie) ** 2)
    f_ee = 1.0 / (1.0 + (q * lam_ee) ** 2)

    alpha0_e = (k_de_sq / (q * q)) * f_ee * w_func_e
    alpha0_i = (k_de_sq / (q * q)) * f_ii * w_func_i
    eps_vlasov = 1.0 + alpha0_e + alpha0_i

    fdt_factor = (q * q) / (1.65 * math.pi * k_de_sq * w_safe)
    s_zz_vlasov = fdt_factor * np.imag(-1.0 / eps_vlasov)

    # Phenomenological local-field & collisional damping parameters from Apollo/Zeus
    g_loc = float(np.clip(0.22 * gamma * (q * q) / (q * q + k_de_sq), -0.15, 0.72))
    nu_coll = float(
        np.clip(
            0.055 * (gamma**1.15) + 0.14 * (gamma**1.25) * (q * q) / (q * q + 0.8 * k_de_sq),
            0.012,
            0.55,
        )
    )
    tau_m = 1.0 / math.sqrt(1.0 + 2.4 * (q * v_th_e) ** 2 + 0.35 * gamma)

    lorentz_w = 1.0 / (1.0 + (w_safe * tau_m) ** 2)
    shift_boost = 1.0 + 0.42 * max(0.0, gamma - 0.5)
    re_shift = shift_boost * g_loc * (w_safe * w_safe) / (w_safe * w_safe + 0.6 * (q * v_th_e) ** 2)
    im_damp = nu_coll * (w_safe / (q * v_th_e + 0.35)) * lorentz_w

    static_renorm = 1.0 - (0.04 + 0.14 * r_s) * gamma / (1.0 + 1.2 * q * q)
    f_pot = (0.65 * f_ee + 0.35 * f_ie) * static_renorm

    w_e_mem = (w_func_e + 1j * im_damp * (w_safe / (w_safe + q * v_th_e))) / (
        1.0 + re_shift * w_func_e - 1j * im_damp * (1.0 - w_func_e)
    )
    eps_mem = 1.0 + (k_de_sq / (q * q)) * f_pot * w_e_mem + alpha0_i
    s_zz_mem = fdt_factor * np.imag(-1.0 / eps_mem)
    return s_zz_vlasov, s_zz_mem


def apollo_table_iii_metrics(
    gamma: float,
    r_s: float,
    hnc: HNCSolution,
    md_deutsch: dict[str, Any],
    md_kelbg: dict[str, Any],
) -> dict[str, float]:
    """Earlier phenomenological Table III transport metrics (Apollo approach)."""
    base = compute_table_iii_transport_ab_initio(gamma, r_s, hnc, md_deutsch, md_kelbg)
    # Apollo phenomenological scaling factors
    d_e_d = base["D_e* (Deutsch)"] * (1.08 if gamma < 1.0 else 1.05)
    tau_se = base["tau_se*"] / max(
        0.5, 1.0 - 0.35 * (base["omega_se*"] ** 2) * (base["tau_se*"] ** 2)
    )
    tau_j = tau_se * ((base["omega_J*"] / max(1e-4, base["omega_se*"])) ** 2) * 0.76
    sigma_s = 1.0 / (4.0 * math.pi * math.sqrt(0.5 * math.pi) * (base["omega_J*"] ** 2) * tau_j)
    cgs = compute_table_i_state(gamma, r_s, 0.025, 10000)
    a2_wpe = (cgs["a_cm"] ** 2) * cgs["omega_pe_s"]
    out = dict(base)
    out["D_e* (Deutsch)"] = d_e_d
    out["D_e (cm^2/s) [D]"] = d_e_d * a2_wpe
    out["tau_se*"] = tau_se
    out["tau_J*"] = tau_j
    out["sigma*"] = sigma_s
    out["sigma (1e5 mho/cm)"] = (sigma_s * cgs["omega_pe_s"]) / 8.98755e16
    return out


# --------------------------------------------------------------------------------------
# Unified GPU MD Runner for a Given (N_total, Gamma, r_s, Potential)
# --------------------------------------------------------------------------------------


def run_md_for_size_and_state(
    n_total: int,
    gamma: float,
    r_s: float,
    potential_type: int,
    n_equil: int,
    n_prod: int,
    dt: float = 0.025,
    backend_name: str = "gpu",
    record_szz: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Run GPU/CPU MD for a given N_total, extracting thermodynamics, D_e*, speed, and optionally S_ZZ(q,w)."""
    n_pairs = max(2, n_total // 2)
    n_blocks = 4
    sample_stride = 5
    n_prod_adj = max(80, (n_prod // (n_blocks * sample_stride)) * (n_blocks * sample_stride))

    cfg = MDConfig(
        n_pairs=n_pairs,
        gamma=gamma,
        r_s=r_s,
        potential_type=potential_type,
        dt=dt,
        n_equil=n_equil,
        n_prod=n_prod_adj,
        n_blocks=n_blocks,
        sample_stride=sample_stride,
        thermostat_stride=10,
        seed=seed + n_pairs + int(gamma * 100 + r_s * 10),
        backend=backend_name,
    )
    system = _build_system(cfg)
    if potential_type == POTENTIAL_DEUTSCH:
        system.lam[:, :] = cfg.state.lam / math.sqrt(math.pi)

    state = cfg.state
    backend = _GpuBackend(system) if backend_name == "gpu" else _CpuBackend(system)
    u_self = self_energy(system.charge, system.alpha)

    backend.compute_forces()
    for step in range(1, n_equil + 1):
        _step(backend)
        if step % cfg.thermostat_stride == 0:
            _thermostat(backend, system, n_pairs, state.k_t)

    block_len = n_prod_adj // n_blocks
    samples_per_block = block_len // sample_stride
    block_msd_e = np.empty((n_blocks, samples_per_block), dtype=np.float64)

    u_excess_samples: list[float] = []
    pv_over_2nkt_samples: list[float] = []

    if record_szz:
        q_mags, k_flat, shell_slices = build_q_shells_flat(system.box)
        n_q_vecs = k_flat.shape[0]
        re_hist = np.empty((n_prod_adj, n_q_vecs), dtype=np.float64)
        im_hist = np.empty((n_prod_adj, n_q_vecs), dtype=np.float64)
        if backend_name == "gpu":
            d_k_shells = cuda.to_device(k_flat)
            d_re = cuda.device_array(n_q_vecs, dtype=np.float64)
            d_im = cuda.device_array(n_q_vecs, dtype=np.float64)
            grid_q = ((n_q_vecs + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK, THREADS_PER_BLOCK)

    backend.set_origin(global_origin=True)
    backend.synchronize()
    t0 = time.perf_counter()

    for step in range(1, n_prod_adj + 1):
        _step(backend)
        if record_szz:
            if backend_name == "gpu":
                _charge_modes_gpu[grid_q](backend.d_pos, backend._d_charge, d_k_shells, d_re, d_im)
                re_hist[step - 1] = d_re.copy_to_host()
                im_hist[step - 1] = d_im.copy_to_host()
            else:
                re_v, im_v = _charge_modes_cpu(backend.positions(), system.charge, k_flat)
                re_hist[step - 1] = re_v
                im_hist[step - 1] = im_v

        if step % sample_stride == 0:
            stats = backend.stats()
            _, _, _, _, msd_e_block = _species_summary(stats, n_pairs)
            u_sr, u_recip, tdv = backend.potential_energies()
            n_tot = 2.0 * n_pairs
            denom_2nkt = n_tot * state.k_t
            u_ex_red = (u_sr + u_recip + u_self - tdv) / denom_2nkt
            w_tot = u_sr + u_recip + u_self - 0.5 * tdv
            pv_red = 1.0 + w_tot / (3.0 * denom_2nkt)

            u_excess_samples.append(u_ex_red)
            pv_over_2nkt_samples.append(pv_red)

            block, in_block = divmod(step - 1, block_len)
            slot = in_block // sample_stride
            block_msd_e[block, slot] = msd_e_block

        if step % block_len == 0:
            backend.set_origin(global_origin=False)

    backend.synchronize()
    elapsed = time.perf_counter() - t0
    sps = n_prod_adj / max(elapsed, 1e-9)

    dt_sample = dt * sample_stride
    block_tau = dt_sample * np.arange(1, samples_per_block + 1)
    d_e, d_e_err, _ = diffusion_from_blocks(block_tau, block_msd_e, fit_start_frac=0.45)

    out: dict[str, Any] = {
        "N_total": n_total,
        "PV_over_2NkT": float(np.mean(pv_over_2nkt_samples)),
        "U_over_2NkT": float(np.mean(u_excess_samples)),
        "D_e*": float(d_e),
        "D_e_err": float(d_e_err),
        "steps_per_second": float(sps),
        "n_prod": n_prod_adj,
    }

    if record_szz:
        omega_md = np.linspace(0.08, 2.15, 21, dtype=np.float64)
        t_win_max = min(24.0, 0.25 * n_prod_adj * dt)
        max_lag = max(16, min(int(t_win_max / dt), n_prod_adj // 4))
        t_axis = np.arange(max_lag, dtype=np.float64) * dt
        t_max = float(t_axis[-1]) if t_axis.size > 1 else 1.0
        hann_win = 0.5 * (1.0 + np.cos(math.pi * t_axis / t_max))

        md_spec: list[list[float]] = []
        for s_idx in range(len(q_mags)):
            sl = shell_slices[s_idx]
            f_zz = compute_shell_acf(
                np.ascontiguousarray(re_hist[:, sl]),
                np.ascontiguousarray(im_hist[:, sl]),
                max_lag,
            )
            spec = []
            for w_val in omega_md:
                integrand = f_zz * hann_win * np.cos(w_val * t_axis)
                val = (1.0 / math.pi) * float(np.trapezoid(integrand, t_axis))
                spec.append(max(0.0, 100.0 * val))
            md_spec.append(spec)
        out["q_mags"] = q_mags.tolist()
        out["omega_md"] = omega_md.tolist()
        out["md_spec"] = md_spec

    return out


# --------------------------------------------------------------------------------------
# Plotting & Table Generation per N_total for BOTH Apollo and Hephaestus
# --------------------------------------------------------------------------------------


def plot_szz_comparison_for_approach(
    approach: str,
    n_total: int,
    gamma: float,
    r_s: float,
    q_mags: NDArray[np.float64],
    omega_md: NDArray[np.float64],
    md_spec: list[NDArray[np.float64]],
    hnc: HNCSolution,
    stem: Path,
) -> None:
    """Plot 2x2 S_ZZ(q, omega) for a given N_total under either 'hephaestus' or 'apollo'."""
    _apply_style()
    omega_fine = np.linspace(0.01, 2.30, 300, dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN * 0.82, FIG_WIDTH_IN * 0.80), sharex=True)
    fig.subplots_adjust(wspace=0.10, hspace=0.14)

    for idx, ax in enumerate(axes.flat):
        q_val = float(q_mags[idx])
        if approach == "hephaestus":
            s_vlasov, s_mem, _, _ = theory_vlasov_and_memory_ab_initio(
                q_val, omega_fine, gamma, r_s, hnc=hnc
            )
            mem_label = "Память (Hephaestus: HNC)"
        else:
            s_vlasov, s_mem = apollo_theory_szz(q_val, omega_fine, gamma, r_s, hnc)
            mem_label = "Память (Apollo: подгонка)"

        ax.plot(
            omega_fine, 100.0 * s_mem, "-", color="black", label=mem_label if idx == 0 else None
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
            md_spec[idx],
            "o",
            color="black",
            markersize=3.5,
            label=f"МД Kelbg (N={n_total})" if idx == 0 else None,
        )
        ax.text(0.50, 0.82, f"q = {q_val:.3f}", transform=ax.transAxes, ha="center")
        ax.set_xlim(0.0, 2.35)
        y_top = 13.5 if gamma < 1.0 else (9.5 if idx < 2 else 11.5)
        ax.set_ylim(0.0, y_top)

    axes[1, 0].set_xlabel(r"$\omega / \omega_{pe}$")
    axes[1, 1].set_xlabel(r"$\omega / \omega_{pe}$")
    axes[0, 0].set_ylabel(r"$100 \times S_{ZZ}(q, \omega)$")
    axes[1, 0].set_ylabel(r"$100 \times S_{ZZ}(q, \omega)$")
    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=6.8)
    _save(fig, stem)


def generate_tables_for_approach(
    approach: str,
    n_total: int,
    n_prod: int,
    hnc1_map: dict[tuple[float, float], HNCSolution],
    hnc2_map: dict[tuple[float, float], HNCSolution],
    md_d_map: dict[tuple[float, float], dict[str, Any]],
    md_k_map: dict[tuple[float, float], dict[str, Any]],
    fig_dir: Path,
    res_dir: Path,
) -> None:
    """Generate Tables I-V (CSV, PNG, PDF, Markdown) for a specific (N_total, approach)."""
    tag = f"[HERMES N={n_total} | {approach.upper()}]"
    states_key = [(0.5, 0.4), (0.5, 1.0), (2.0, 1.0)]

    # TABLE I
    t1_headers = [
        "Параметр",
        "Ст.(0.5,0.4)",
        "Наш(0.5,0.4)",
        "Ст.(0.5,1.0)",
        "Наш(0.5,1.0)",
        "Ст.(2.0,1.0)",
        "Наш(2.0,1.0)",
    ]
    c04 = compute_table_i_state(0.5, 0.4, 0.025, n_prod)
    c10 = compute_table_i_state(0.5, 1.0, 0.025, n_prod)
    c20 = compute_table_i_state(2.0, 1.0, 0.025, n_prod)
    t1_rows = [
        ["N_total", "250", f"{n_total}", "250", f"{n_total}", "250", f"{n_total}"],
        [
            "T (K)",
            f"{PAPER_TABLE_I[(0.5, 0.4)]['T_K']:.2e}",
            f"{c04['T_K']:.2e}",
            f"{PAPER_TABLE_I[(0.5, 1.0)]['T_K']:.2e}",
            f"{c10['T_K']:.2e}",
            f"{PAPER_TABLE_I[(2.0, 1.0)]['T_K']:.2e}",
            f"{c20['T_K']:.2e}",
        ],
        [
            "n (cm^-3)",
            f"{PAPER_TABLE_I[(0.5, 0.4)]['n_cm3']:.2e}",
            f"{c04['n_cm3']:.2e}",
            f"{PAPER_TABLE_I[(0.5, 1.0)]['n_cm3']:.2e}",
            f"{c10['n_cm3']:.2e}",
            f"{PAPER_TABLE_I[(2.0, 1.0)]['n_cm3']:.2e}",
            f"{c20['n_cm3']:.2e}",
        ],
        [
            "omega_pe (s^-1)",
            f"{PAPER_TABLE_I[(0.5, 0.4)]['omega_pe_s']:.2e}",
            f"{c04['omega_pe_s']:.2e}",
            f"{PAPER_TABLE_I[(0.5, 1.0)]['omega_pe_s']:.2e}",
            f"{c10['omega_pe_s']:.2e}",
            f"{PAPER_TABLE_I[(2.0, 1.0)]['omega_pe_s']:.2e}",
            f"{c20['omega_pe_s']:.2e}",
        ],
        [
            "N_t (шагов)",
            "5.60e+04",
            f"{n_prod:.2e}",
            "7.30e+04",
            f"{n_prod:.2e}",
            "4.80e+04",
            f"{n_prod:.2e}",
        ],
    ]
    save_csv_table(res_dir / "table1_comparison.csv", t1_headers, t1_rows)
    render_table_figure(
        f"TABLE I {tag}",
        t1_headers,
        t1_rows,
        fig_dir / "table1_comparison",
        fig_width_mult=1.35,
        fig_height_in=2.4,
    )

    # TABLE II
    t2_headers = [
        "Режим",
        "PV Ст.MD",
        "PV Наш D/K",
        "PV HNC1",
        "PV HNC2",
        "U HNC1",
        "U HNC2",
        "chi_T(1)",
        "chi_T(2)",
    ]
    t2_rows = []
    for st in states_key:
        p = PAPER_TABLE_II[st]
        h1, h2 = hnc1_map[st], hnc2_map[st]
        md_d, md_k = md_d_map[st], md_k_map[st]
        t2_rows.append(
            [
                f"G={st[0]},rs={st[1]}",
                f"{p['PV_MD']:.2f}",
                f"{md_d['PV_over_2NkT']:.2f}/{md_k['PV_over_2NkT']:.2f}",
                f"{p['PV_HNC1']:.3f}/{h1.pv_over_2nkt:.3f}",
                f"{p['PV_HNC2']:.3f}/{h2.pv_over_2nkt:.3f}",
                f"{p['U_HNC1']:+.3f}/{h1.u_over_2nkt:+.3f}",
                f"{p['U_HNC2']:+.3f}/{h2.u_over_2nkt:+.3f}",
                f"{p['chiT_HNC1']:.3f}/{h1.inv_chi_t:.3f}",
                f"{p['chiT_HNC2']:.3f}/{h2.inv_chi_t:.3f}",
            ]
        )
    save_csv_table(res_dir / "table2_comparison.csv", t2_headers, t2_rows)
    render_table_figure(
        f"TABLE II {tag}",
        t2_headers,
        t2_rows,
        fig_dir / "table2_comparison",
        fig_width_mult=1.70,
        fig_height_in=2.3,
        font_size=7.0,
    )

    # TABLE III
    tab3_calc = {
        st: (
            compute_table_iii_transport_ab_initio(
                st[0], st[1], hnc1_map[st], md_d_map[st], md_k_map[st]
            )
            if approach == "hephaestus"
            else apollo_table_iii_metrics(st[0], st[1], hnc1_map[st], md_d_map[st], md_k_map[st])
        )
        for st in states_key
    }
    t3_headers = [
        "Величина",
        "Ст.(0.5,0.4)",
        "Наш(0.5,0.4)",
        "Ст.(0.5,1.0)",
        "Наш(0.5,1.0)",
        "Ст.(2.0,1.0)",
        "Наш(2.0,1.0)",
    ]
    t3_map = [
        ("D_e* (МД Дойч)", "D_e*", "D_e* (Deutsch)"),
        ("D_e* (МД Кельбг)", "D_e*", "D_e* (Kelbg)"),
        ("D_e (cm^2/s)", "D_e (cm^2/s)", "D_e (cm^2/s) [D]"),
        ("tau_se*", "tau_se*", "tau_se*"),
        ("omega_se*", "omega_se*", "omega_se*"),
        ("omega_si*", "omega_si*", "omega_si*"),
        ("sigma*", "sigma*", "sigma*"),
        ("tau_J*", "tau_J*", "tau_J*"),
        ("omega_J*", "omega_J*", "omega_J*"),
    ]
    t3_rows = []
    for label, p_key, c_key in t3_map:
        p_vals = PAPER_TABLE_III[p_key]
        row = [label]
        for idx, st in enumerate(states_key):
            pv, cv = p_vals[idx], tab3_calc[st][c_key]
            row.append("-" if pv is None else f"{pv:.3g}")
            row.append("-" if not math.isfinite(cv) else f"{cv:.3g}")
        t3_rows.append(row)
    save_csv_table(res_dir / "table3_comparison.csv", t3_headers, t3_rows)
    render_table_figure(
        f"TABLE III {tag}",
        t3_headers,
        t3_rows,
        fig_dir / "table3_comparison",
        fig_width_mult=1.40,
        fig_height_in=3.6,
        font_size=7.4,
    )

    # TABLE IV
    t4_headers = [
        "q",
        "(0.5,0.4) Ст.",
        "(0.5,0.4) Наш",
        "(0.5,1.0) Ст.",
        "(0.5,1.0) Наш",
        "(2.0,1.0) Ст.",
        "(2.0,1.0) Наш",
    ]
    t4_rows = []
    for q_val, p04, p10, p20 in PAPER_TABLE_IV:
        m04, v04 = compute_table_iv_ab_initio(q_val, 0.5, 0.4, hnc1_map[(0.5, 0.4)])
        m10, v10 = compute_table_iv_ab_initio(q_val, 0.5, 1.0, hnc1_map[(0.5, 1.0)])
        m20, v20 = compute_table_iv_ab_initio(q_val, 2.0, 1.0, hnc1_map[(2.0, 1.0)])
        t4_rows.append(
            [
                f"{q_val:.3f}",
                f"{p04[0]:.2f}/{p04[1]:.2f}/{p04[2]:.2f}",
                f"{m04:.2f}/{v04:.2f}",
                f"{p10[0]:.2f}/{p10[1]:.2f}/{p10[2]:.2f}",
                f"{m10:.2f}/{v10:.2f}",
                f"{p20[0]:.2f}/{p20[1]:.3f}/{p20[2]:.3f}",
                f"{m20:.3f}/{v20:.3f}",
            ]
        )
    save_csv_table(res_dir / "table4_comparison.csv", t4_headers, t4_rows)
    render_table_figure(
        f"TABLE IV {tag}",
        t4_headers,
        t4_rows,
        fig_dir / "table4_comparison",
        fig_width_mult=1.55,
        fig_height_in=2.6,
        font_size=7.2,
    )

    # TABLE V
    t5_headers = [
        "Режим / q",
        "S_ii",
        "S_ie",
        "S_ee",
        "Omega4_ii*",
        "-Omega4_ie*",
        "Omega4_ee*",
        "tau_11*",
        "tau_22*",
    ]
    t5_rows = []
    for st in states_key:
        for p_row in PAPER_TABLE_V[st]:
            q_v = p_row[0]
            s_ii, s_ie, s_ee, w4_ii, w4_ie, w4_ee, t11, t22 = compute_table_v_row_ab_initio(
                q_v, st[0], st[1], hnc1_map[st]
            )
            t5_rows.append(
                [
                    f"({st[0]},{st[1]}) q={q_v:.3f}",
                    f"{p_row[1]:.4f}/{s_ii:.4f}",
                    f"{p_row[2]:.4f}/{s_ie:.4f}",
                    f"{p_row[3]:.4f}/{s_ee:.4f}",
                    f"{p_row[4]:.2e}/{w4_ii:.2e}",
                    f"{p_row[5]:.2e}/{w4_ie:.2e}",
                    f"{p_row[6]:.3g}/{w4_ee:.3g}",
                    f"{p_row[7]:.1f}/{t11:.1f}",
                    f"{p_row[8]:.2f}/{t22:.2f}",
                ]
            )
    save_csv_table(res_dir / "table5_comparison.csv", t5_headers, t5_rows)
    render_table_figure(
        f"TABLE V {tag}",
        t5_headers,
        t5_rows,
        fig_dir / "table5_comparison",
        fig_width_mult=1.75,
        fig_height_in=6.8,
        font_size=6.6,
    )

    md_lines = [f"# Таблицы I–V {tag}\n"]
    for title, headers, rows in [
        ("## TABLE I", t1_headers, t1_rows),
        ("## TABLE II", t2_headers, t2_rows),
        ("## TABLE III", t3_headers, t3_rows),
        ("## TABLE IV", t4_headers, t4_rows),
        ("## TABLE V", t5_headers, t5_rows),
    ]:
        md_lines.append(title + "\n")
        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in rows:
            md_lines.append("| " + " | ".join(r) + " |")
        md_lines.append("\n")
    (res_dir / f"TABLES_I_TO_V_{approach.upper()}.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )


def plot_hermes_scaling_summary(summary_rows: list[dict[str, Any]], stem: Path) -> None:
    """Plot summary of D_e*(N), PV/2NkT(N), and GPU throughput across N = 250, 1000, 5000, 12500."""
    _apply_style()
    n_arr = np.array([r["N_total"] for r in summary_rows], dtype=np.float64)
    de_k = np.array([r["D_e_Kelbg_05_10"] for r in summary_rows], dtype=np.float64)
    de_d = np.array([r["D_e_Deutsch_05_10"] for r in summary_rows], dtype=np.float64)
    sps = np.array([r["steps_per_second"] for r in summary_rows], dtype=np.float64)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 0.56 * FIG_WIDTH_IN))
    ax1.plot(n_arr ** (-1.0 / 3.0), de_d, "s-", color="0.45", label="Дойч (0.5, 1.0)")
    ax1.plot(n_arr ** (-1.0 / 3.0), de_k, "o-", color="black", label="Кельбг (0.5, 1.0)")
    ax1.set_xlabel(r"$N_{\mathrm{total}}^{-1/3}$")
    ax1.set_ylabel(r"$D_e^*$")
    ax1.legend(frameon=False)

    ax2.loglog(
        n_arr, 1000.0 / np.maximum(sps, 1e-9), "o-", color="black", label="Время шага GPU (мс)"
    )
    ax2.set_xlabel(r"Число частиц $N_{\mathrm{total}}$")
    ax2.set_ylabel("Время одного шага (мс)")
    ax2.legend(frameon=False)

    fig.tight_layout()
    _save(fig, stem)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--particles",
        type=int,
        nargs="+",
        default=list(DEFAULT_PARTICLES),
        help="Particle counts N_total (default: 250 1000 5000 12500).",
    )
    parser.add_argument(
        "--steps-scale",
        type=float,
        default=1.0,
        help="Multiplier for default step counts (e.g. 0.5 for faster run, 2.0 for longer).",
    )
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--rerun-md", action="store_true", help="Ignore cached MD JSONs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    backend = args.backend
    if backend == "gpu" and not cuda.is_available():
        backend = "cpu"

    root_fig = Path("figures/hermes")
    root_res = Path("results/hermes")
    root_fig.mkdir(parents=True, exist_ok=True)
    root_res.mkdir(parents=True, exist_ok=True)

    print(
        f"🪽 [HERMES] Запуск раздельного моделирования для N = {args.particles} (Apollo + Hephaestus)"
    )

    # Solve HNC once (independent of finite box size N)
    hnc1_map = {(g, rs): solve_two_component_hnc(g, rs, include_pauli=False) for g, rs, _ in STATES}
    hnc2_map = {(g, rs): solve_two_component_hnc(g, rs, include_pauli=True) for g, rs, _ in STATES}

    summary_rows: list[dict[str, Any]] = []

    with SystemMonitor(interval=0.5) as mon:
        for n_total in args.particles:
            base_eq, base_pr = STEP_BUDGETS.get(n_total, (600, 2000))
            n_equil = max(100, int(base_eq * args.steps_scale))
            n_prod = max(200, int(base_pr * args.steps_scale))

            n_fig_dir = root_fig / f"N{n_total}"
            n_res_dir = root_res / f"N{n_total}"
            for approach in ("hephaestus", "apollo"):
                (n_fig_dir / approach).mkdir(parents=True, exist_ok=True)
                (n_res_dir / approach).mkdir(parents=True, exist_ok=True)

            cache_path = n_res_dir / "md_cache.json"
            if cache_path.is_file() and not args.rerun_md:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                md_d_map = {(g, rs): cached["deutsch"][f"{g}_{rs}"] for g, rs, _ in STATES}
                md_k_map = {(g, rs): cached["kelbg"][f"{g}_{rs}"] for g, rs, _ in STATES}
                print(f"  [N={n_total:5d}] Загружен кэш из {cache_path}")
            else:
                md_d_map = {}
                md_k_map = {}
                t_start_n = time.perf_counter()
                for gamma, r_s, _stem in STATES:
                    mon.set_stage(f"N{n_total}_Deutsch_{gamma}_{r_s}")
                    md_d_map[(gamma, r_s)] = run_md_for_size_and_state(
                        n_total,
                        gamma,
                        r_s,
                        POTENTIAL_DEUTSCH,
                        n_equil,
                        n_prod,
                        backend_name=backend,
                        record_szz=False,
                    )
                    mon.set_stage(f"N{n_total}_Kelbg_{gamma}_{r_s}")
                    md_k_map[(gamma, r_s)] = run_md_for_size_and_state(
                        n_total,
                        gamma,
                        r_s,
                        POTENTIAL_KELBG,
                        n_equil,
                        n_prod,
                        backend_name=backend,
                        record_szz=True,
                    )
                cache_payload = {
                    "N_total": n_total,
                    "n_equil": n_equil,
                    "n_prod": n_prod,
                    "deutsch": {f"{k[0]}_{k[1]}": v for k, v in md_d_map.items()},
                    "kelbg": {f"{k[0]}_{k[1]}": v for k, v in md_k_map.items()},
                }
                cache_path.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
                print(
                    f"  [N={n_total:5d}] МД завершено ({n_equil}+{n_prod} шагов) за "
                    f"{time.perf_counter() - t_start_n:.1f} с | "
                    f"Speed={md_k_map[(0.5, 1.0)]['steps_per_second']:.1f} steps/s"
                )

            # Generate BOTH Hephaestus (ab-initio) and Apollo (fitted) tables and plots for this N_total
            for approach in ("hephaestus", "apollo"):
                generate_tables_for_approach(
                    approach,
                    n_total,
                    n_prod,
                    hnc1_map,
                    hnc2_map,
                    md_d_map,
                    md_k_map,
                    n_fig_dir / approach,
                    n_res_dir / approach,
                )
                for gamma, r_s, stem in STATES:
                    k_res = md_k_map[(gamma, r_s)]
                    plot_szz_comparison_for_approach(
                        approach,
                        n_total,
                        gamma,
                        r_s,
                        np.array(k_res["q_mags"], dtype=np.float64),
                        np.array(k_res["omega_md"], dtype=np.float64),
                        [np.array(s, dtype=np.float64) for s in k_res["md_spec"]],
                        hnc1_map[(gamma, r_s)],
                        n_fig_dir / approach / stem,
                    )
                if approach == "hephaestus":
                    plot_fig9_and_fig11(n_fig_dir / approach)

            summary_rows.append(
                {
                    "N_total": n_total,
                    "D_e_Deutsch_05_10": md_d_map[(0.5, 1.0)]["D_e*"],
                    "D_e_Kelbg_05_10": md_k_map[(0.5, 1.0)]["D_e*"],
                    "PV_Deutsch_05_10": md_d_map[(0.5, 1.0)]["PV_over_2NkT"],
                    "PV_Kelbg_05_10": md_k_map[(0.5, 1.0)]["PV_over_2NkT"],
                    "steps_per_second": md_k_map[(0.5, 1.0)]["steps_per_second"],
                }
            )

    mon.print_summary()
    mon.save_csv(root_res / "telemetry_hermes.csv")
    mon.plot(root_fig / "telemetry_hermes")

    with (root_res / "hermes_scaling_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    plot_hermes_scaling_summary(summary_rows, root_fig / "hermes_scaling_summary")
    print(
        "🪽 [HERMES] Все графики и таблицы для N = 250, 1000, 5000, 12500 сохранены в figures/hermes/ и results/hermes/!"
    )


if __name__ == "__main__":
    main()
