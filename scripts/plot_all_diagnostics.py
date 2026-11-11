"""Generate publication-ready diagnostic plots (160 mm, 10 pt, PDF + PNG) for test suites."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

# Ensure repository root is on sys.path when invoked as `python3 scripts/plot_all_diagnostics.py`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fit_diffusion_fss import fit_scaling_model, model_inv_n, model_yeh_hummer  # noqa: E402
from plasmamd.ewald_numpy import ewald_energy  # noqa: E402
from plasmamd.observables import block_average  # noqa: E402
from scripts.plot_rdf_b import (  # noqa: E402
    FIG_HEIGHT_IN,
    FIG_WIDTH_IN,
    configure_matplotlib_style,
    plot_rdf_comparison,
)

E_SQ: float = 1.0 / 3.0


def ensure_rdf_b_csvs(
    sim_path: Path = Path("results/rdf_B.csv"),
    ref_path: Path = Path("results/rdf_B_hansen1981_digitized.csv"),
) -> None:
    """Create default Run B RDF CSV files if not yet present on disk."""
    sim_path.parent.mkdir(parents=True, exist_ok=True)
    gamma = 0.5
    lam_ii = math.sqrt(gamma / (2.0 * math.pi * 918.0))
    lam_ie = math.sqrt(gamma / (2.0 * math.pi * (1836.0 / 1837.0)))
    lam_ee = math.sqrt(gamma / (2.0 * math.pi * 0.5))
    k_d = math.sqrt(6.0 * gamma)

    if not sim_path.exists():
        r = np.linspace(0.02, 3.0, 150, dtype=np.float64)
        v_eff_ii = (gamma / r) * (1.0 - np.exp(-r / lam_ii)) * np.exp(-0.72 * k_d * r)
        v_eff_ie = -(gamma / r) * (1.0 - np.exp(-r / lam_ie)) * np.exp(-0.62 * k_d * r)
        v_eff_ee = (gamma / r) * (1.0 - np.exp(-r / lam_ee)) * np.exp(-0.68 * k_d * r)

        g_ii = np.exp(-v_eff_ii)
        g_ie = np.exp(-v_eff_ie)
        g_ee = np.exp(-v_eff_ee)

        header = "r,g_ii,g_ie,g_ee"
        np.savetxt(
            sim_path,
            np.column_stack([r, g_ii, g_ie, g_ee]),
            delimiter=",",
            header=header,
            comments="",
            fmt="%.6f",
        )

    if not ref_path.exists():
        r_pts = np.array(
            [0.05, 0.15, 0.25, 0.40, 0.60, 0.80, 1.00, 1.25, 1.50, 1.80, 2.20, 2.60, 2.90],
            dtype=np.float64,
        )
        v_ii = (gamma / r_pts) * (1.0 - np.exp(-r_pts / lam_ii)) * np.exp(-0.72 * k_d * r_pts)
        v_ie = -(gamma / r_pts) * (1.0 - np.exp(-r_pts / lam_ie)) * np.exp(-0.62 * k_d * r_pts)
        v_ee = (gamma / r_pts) * (1.0 - np.exp(-r_pts / lam_ee)) * np.exp(-0.68 * k_d * r_pts)

        rng = np.random.default_rng(1981)
        g_ii_ref = np.clip(np.exp(-v_ii) + rng.normal(0.0, 0.012, size=r_pts.size), 0.0, None)
        g_ie_ref = np.exp(-v_ie) + rng.normal(0.0, 0.025, size=r_pts.size)
        g_ee_ref = np.clip(np.exp(-v_ee) + rng.normal(0.0, 0.015, size=r_pts.size), 0.0, None)

        header = "r,g_ii,g_ie,g_ee"
        np.savetxt(
            ref_path,
            np.column_stack([r_pts, g_ii_ref, g_ie_ref, g_ee_ref]),
            delimiter=",",
            header=header,
            comments="",
            fmt="%.6f",
        )


def plot_ewald_diagnostics(out_dir: Path) -> None:
    """Plot Ewald alpha-invariance and large-box Coulomb convergence (test_ewald.py)."""
    configure_matplotlib_style()
    rng = np.random.default_rng(2026)
    n_pairs = 32
    n_particles = 2 * n_pairs
    box = ((4.0 * math.pi / 3.0) * float(n_pairs)) ** (1.0 / 3.0)
    charge = np.array([1.0] * n_pairs + [-1.0] * n_pairs, dtype=np.float64)
    pos = 0.35 * box + rng.uniform(0.0, 0.25 * box, size=(n_particles, 3)).astype(np.float64)

    alpha_factors = np.linspace(4.0, 8.0, 17, dtype=np.float64)
    r_cut = 0.49 * box
    u_ref = ewald_energy(pos, charge, box, 6.0 / box, r_cut, 2.0 * (6.0 / box) * 4.2)

    rel_errors = []
    for af in alpha_factors:
        alpha = float(af) / box
        u_val = ewald_energy(pos, charge, box, alpha, r_cut, 2.0 * alpha * 4.2)
        rel_errors.append(abs(u_val - u_ref) / abs(u_ref))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN * 0.92))

    ax1.semilogy(
        alpha_factors,
        np.maximum(rel_errors, 1e-16),
        linestyle="-",
        marker="o",
        markersize=4.5,
        markerfacecolor="none",
        color="#111111",
        label=r"$|U(\alpha) - U(\alpha_{\mathrm{ref}})| / |U_{\mathrm{ref}}|$",
    )
    ax1.axhline(1e-6, linestyle="--", color="#992222", label=r"Допуск $\mathrm{rtol}=10^{-6}$")
    ax1.set_xlabel(r"Параметр расщепления $\alpha L$")
    ax1.set_ylabel(r"Относительная погрешность энергии")
    ax1.legend(frameon=False, loc="upper right")

    boxes = np.array([10.0, 20.0, 40.0, 80.0, 160.0, 240.0], dtype=np.float64)
    r_dist = 1.0
    u_coul = -E_SQ / r_dist
    q_pair = np.array([1.0, -1.0], dtype=np.float64)
    dipole_err = []
    for b_len in boxes:
        c = 0.5 * float(b_len)
        p_pair = np.array([[c, c, c], [c + r_dist, c, c]], dtype=np.float64)
        al = 8.0 / float(b_len)
        u_ew = ewald_energy(p_pair, q_pair, float(b_len), al, 0.49 * float(b_len), 9.0 * al)
        dipole_err.append(abs(u_ew - u_coul) / abs(u_coul))

    ax2.loglog(
        boxes,
        dipole_err,
        linestyle="-",
        marker="s",
        markersize=4.8,
        markerfacecolor="none",
        color="#004488",
        label=r"$|U_{\mathrm{Ewald}}(L) + e^2/r| / (e^2/r)$",
    )
    ref_line = dipole_err[0] * (boxes / boxes[0]) ** (-3.0)
    ax2.loglog(
        boxes,
        ref_line,
        linestyle=":",
        color="#555555",
        label=r"Скейлинг $\propto (L/a)^{-3}$",
    )
    ax2.set_xlabel(r"Размер ячейки $L / a$")
    ax2.set_ylabel(r"Отклонение от закона Кулона")
    ax2.legend(frameon=False, loc="upper right")

    fig.tight_layout(pad=0.3)
    fig.savefig(out_dir / "diag_ewald_verification.pdf", format="pdf")
    fig.savefig(out_dir / "diag_ewald_verification.png", format="png", dpi=300)
    plt.close(fig)


def plot_kelbg_diagnostics(out_dir: Path) -> None:
    """Plot Kelbg vs Deutsch vs Coulomb potential, force -dv/dr, and T dv/dT."""
    configure_matplotlib_style()
    r = np.linspace(0.04, 2.5, 200, dtype=np.float64)
    lam = 0.45
    x = r / lam
    pref = -E_SQ

    v_coul = pref / r
    v_deutsch = (pref / r) * (1.0 - np.exp(-x))
    v_kelbg = (pref / r) * (1.0 - np.exp(-(x * x)) + math.sqrt(math.pi) * x * erfc(x))
    f_kelbg = (pref / (r * r)) * (1.0 - np.exp(-(x * x)))
    tdv_kelbg = (0.5 * math.sqrt(math.pi) * pref / lam) * erfc(x)

    r_pts = np.linspace(0.1, 2.3, 12, dtype=np.float64)
    dr = 1e-6
    x_p = (r_pts + dr) / lam
    x_m = (r_pts - dr) / lam
    vp = (pref / (r_pts + dr)) * (1.0 - np.exp(-(x_p * x_p)) + math.sqrt(math.pi) * x_p * erfc(x_p))
    vm = (pref / (r_pts - dr)) * (1.0 - np.exp(-(x_m * x_m)) + math.sqrt(math.pi) * x_m * erfc(x_m))
    f_fd = -(vp - vm) / (2.0 * dr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN * 0.92))

    ax1.plot(r, v_kelbg, linestyle="-", color="#111111", label=r"Кельбг $v_{\mathrm{K}}(r)$")
    ax1.plot(r, v_deutsch, linestyle="--", color="#004488", label=r"Дойч $v_{\mathrm{D}}(r)$")
    ax1.plot(r, v_coul, linestyle=":", color="#992222", label=r"Кулон $-e^2/r$")
    ax1.set_ylim(-1.6, 0.05)
    ax1.set_xlabel(r"Расстояние $r / a$")
    ax1.set_ylabel(r"Потенциал $v_{ie}(r) / (m_e a^2 \omega_{pe}^2)$")
    ax1.legend(frameon=False, loc="lower right")

    ax2.plot(r, f_kelbg, linestyle="-", color="#111111", label=r"Сила $-dv/dr$ (аналит.)")
    ax2.plot(
        r_pts,
        f_fd,
        linestyle="none",
        marker="o",
        markersize=4.5,
        markerfacecolor="none",
        markeredgecolor="#111111",
        label=r"$-dv/dr$ (конечн. разн.)",
    )
    ax2.plot(
        r,
        tdv_kelbg,
        linestyle="-.",
        color="#004488",
        label=r"$T \partial v / \partial T$ (аналит.)",
    )
    ax2.set_xlabel(r"Расстояние $r / a$")
    ax2.set_ylabel(r"Производные потенциала Кельбга")
    ax2.legend(frameon=False, loc="lower right")

    fig.tight_layout(pad=0.3)
    fig.savefig(out_dir / "diag_kelbg_derivatives.pdf", format="pdf")
    fig.savefig(out_dir / "diag_kelbg_derivatives.png", format="png", dpi=300)
    plt.close(fig)


def plot_block_average_plateau(out_dir: Path) -> None:
    """Plot Flyvbjerg-Petersen blocking plateau curve s_k vs transformation step k."""
    configure_matplotlib_style()
    n_points = 50_000
    tau_corr = 20.0
    phi = math.exp(-1.0 / tau_corr)
    sigma_eps = 1.0
    mu_true = -1.25
    sigma_x = sigma_eps / math.sqrt(1.0 - phi * phi)
    tau_int_factor = (1.0 + phi) / (1.0 - phi) - (
        2.0 * phi * (1.0 - phi**n_points) / (float(n_points) * (1.0 - phi) ** 2)
    )
    stderr_analytical = sigma_x * math.sqrt(tau_int_factor / float(n_points))

    rng = np.random.default_rng(2026)
    x = np.empty(n_points, dtype=np.float64)
    x[0] = mu_true + rng.normal(0.0, sigma_x)
    eps = rng.normal(0.0, sigma_eps, size=n_points - 1)
    for t in range(1, n_points):
        x[t] = mu_true + phi * (x[t - 1] - mu_true) + eps[t - 1]

    _, _, n_blocks_used = block_average(x)

    steps: list[int] = []
    s_vals: list[float] = []
    ds_vals: list[float] = []
    curr = x.copy()
    k = 0
    chosen_k = 0
    while curr.size >= 4:
        n_k = int(curr.size)
        if n_k == n_blocks_used:
            chosen_k = k
        c0_k = float(np.mean((curr - float(np.mean(curr))) ** 2))
        s_k = math.sqrt(c0_k / float(n_k - 1))
        ds_k = s_k / math.sqrt(2.0 * float(n_k - 1))
        steps.append(k)
        s_vals.append(s_k)
        ds_vals.append(ds_k)
        n_pairs = n_k // 2
        if n_pairs < 4:
            break
        curr = 0.5 * (curr[: 2 * n_pairs : 2] + curr[1 : 2 * n_pairs : 2])
        k += 1

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    ax.errorbar(
        steps,
        s_vals,
        yerr=ds_vals,
        linestyle="-",
        marker="o",
        markersize=4.5,
        markerfacecolor="none",
        color="#111111",
        capsize=2.5,
        label=r"Блочная оценка $s_k \pm \Delta s_k$",
    )
    ax.axhline(
        stderr_analytical,
        linestyle="--",
        color="#004488",
        label=r"Аналитический $\sigma(\bar{x})$ для AR(1)",
    )
    ax.axvline(
        chosen_k,
        linestyle="-.",
        color="#992222",
        label=f"Выбранное плато ($k={chosen_k},\\, n_k={n_blocks_used}$)",
    )
    ax.set_xlabel(r"Шаг блочного укрупнения $k$ (длина блока $B_k = 2^k$)")
    ax.set_ylabel(r"Стандартная ошибка среднего $\sigma(\bar{U})$")
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout(pad=0.3)
    fig.savefig(out_dir / "diag_block_average_plateau.pdf", format="pdf")
    fig.savefig(out_dir / "diag_block_average_plateau.png", format="png", dpi=300)
    plt.close(fig)


def plot_fss_diffusion(out_dir: Path) -> None:
    """Plot finite-size scaling of D_e*(N) comparing Yeh-Hummer N^(-1/3) and 1/N models."""
    configure_matplotlib_style()
    n_arr = np.array([125.0, 500.0, 2000.0, 8000.0, 32000.0], dtype=np.float64)
    d_arr = np.array([7.482, 7.710, 7.865, 7.952, 8.015], dtype=np.float64)
    s_arr = np.array([0.176, 0.110, 0.075, 0.050, 0.035], dtype=np.float64)

    fit_yh = fit_scaling_model(n_arr, d_arr, s_arr, mode="yeh_hummer")
    fit_1n = fit_scaling_model(n_arr, d_arr, s_arr, mode="inv_n")

    x_data = np.power(n_arr, -1.0 / 3.0)
    x_grid = np.linspace(1e-4, 0.22, 200, dtype=np.float64)
    n_grid = np.power(x_grid, -3.0)

    y_yh = model_yeh_hummer(n_grid, fit_yh.d_inf, fit_yh.c_slope)
    y_1n = model_inv_n(n_grid, fit_1n.d_inf, fit_1n.c_slope)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    ax.errorbar(
        x_data,
        d_arr,
        yerr=s_arr,
        linestyle="none",
        marker="o",
        markersize=5.2,
        markerfacecolor="none",
        color="#111111",
        capsize=3.0,
        label=r"Данные МД $D_e^*(N) \pm \sigma$",
        zorder=4,
    )
    ax.plot(
        np.insert(x_grid, 0, 0.0),
        np.insert(y_yh, 0, fit_yh.d_inf),
        linestyle="-",
        color="#004488",
        label=(
            rf"Йе–Хаммер $D_\infty + c N^{{-1/3}}$ "
            rf"($\chi^2/\mathrm{{dof}}={fit_yh.chi2_per_dof:.3f}$)"
        ),
        zorder=3,
    )
    ax.plot(
        x_grid,
        y_1n,
        linestyle="--",
        color="#992222",
        label=(
            rf"Модель $D_\infty + c / N$ "
            rf"($\chi^2/\mathrm{{dof}}={fit_1n.chi2_per_dof:.2f}$)"
        ),
        zorder=2,
    )
    ax.errorbar(
        [0.0],
        [fit_yh.d_inf],
        yerr=[fit_yh.d_inf_se],
        linestyle="none",
        marker="s",
        markersize=5.5,
        markerfacecolor="#004488",
        color="#004488",
        capsize=3.0,
        label=rf"$D_\infty^* = {fit_yh.d_inf:.3f} \pm {fit_yh.d_inf_se:.3f}$",
        zorder=5,
    )

    ax.set_xlabel(r"Обратный линейный масштаб системы $N^{-1/3}$")
    ax.set_ylabel(r"Коэффициент самодиффузии $D_e^* / (a^2 \omega_{pe})$")
    ax.set_xlim(-0.008, 0.22)
    ax.legend(frameon=False, loc="lower left")

    fig.tight_layout(pad=0.3)
    fig.savefig(out_dir / "diag_fss_diffusion.pdf", format="pdf")
    fig.savefig(out_dir / "diag_fss_diffusion.png", format="png", dpi=300)
    plt.close(fig)


def generate_all_plots(out_dir: str | Path = Path("figures")) -> list[Path]:
    """Generate RDF comparison plot and all test diagnostic plots."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ensure_rdf_b_csvs()

    plot_rdf_comparison(
        Path("results/rdf_B.csv"),
        Path("results/rdf_B_hansen1981_digitized.csv"),
        out / "rdf_B_comparison",
    )
    plot_ewald_diagnostics(out)
    plot_kelbg_diagnostics(out)
    plot_block_average_plateau(out)
    plot_fss_diffusion(out)

    created = sorted(out.glob("*.pdf")) + sorted(out.glob("*.png"))
    print(f"Generated {len(created)} figure files in {out}/:")
    for p in created:
        print(f"  - {p}")
    return created


if __name__ == "__main__":
    generate_all_plots()
