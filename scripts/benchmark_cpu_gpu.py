"""Benchmark CPU (numba.njit) vs GPU (numba.cuda.jit) pair force execution times across N."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import cuda

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plasmamd.gpu.pair_forces import (  # noqa: E402
    _DEFAULT_E_SQ,
    _TILE_SIZE,
    _TWO_OVER_SQRT_PI,
    _pair_forces_kernel,
    pair_forces,
    pair_forces_cpu,
)
from scripts.plot_rdf_b import (  # noqa: E402
    FIG_HEIGHT_IN,
    FIG_WIDTH_IN,
    configure_matplotlib_style,
)


def _sample_lam(gamma: float = 0.5, r_s: float = 1.0, mass_ratio: float = 1836.0) -> np.ndarray:
    """Compute 2x2 Deutsch thermal wavelength matrix [a] for H plasma."""
    mu_ii = 0.5 * mass_ratio
    mu_ie = mass_ratio / (mass_ratio + 1.0)
    mu_ee = 0.5
    factor = gamma / (2.0 * math.pi * r_s)
    return np.array(
        [
            [math.sqrt(factor / mu_ii), math.sqrt(factor / mu_ie)],
            [math.sqrt(factor / mu_ie), math.sqrt(factor / mu_ee)],
        ],
        dtype=np.float64,
    )


def benchmark_single_n(
    n_total: int,
    n_repeats: int = 10,
    seed: int = 2026,
) -> tuple[float, float, float]:
    """Measure median execution time [ms] for CPU, GPU end-to-end (with PCIe), and GPU kernel-only.

    Parameters
    ----------
    n_total : int
        Total number of particles (e.g., 250 as in Hansen & McDonald 1981).
    n_repeats : int, optional
        Number of timed repetitions after JIT warm-up.
    seed : int, optional
        RNG seed for particle positions.

    Returns
    -------
    cpu_ms : float
        Median CPU execution time in milliseconds.
    gpu_total_ms : float
        Median GPU execution time including host-device memory transfers [ms].
    gpu_kernel_ms : float
        Median pure GPU kernel execution time on pre-allocated device arrays [ms].
    """
    rng = np.random.default_rng(seed + n_total)
    n_pairs = max(1, n_total // 2)
    box = ((4.0 * math.pi / 3.0) * float(n_pairs)) ** (1.0 / 3.0)
    alpha = 6.0 / (0.48 * box)
    r_cut = 0.48 * box

    pos = np.ascontiguousarray(rng.uniform(0.0, box, size=(n_total, 3)), dtype=np.float64)
    species = np.ascontiguousarray([0] * n_pairs + [1] * (n_total - n_pairs), dtype=np.int8)
    charge = np.ascontiguousarray(np.where(species == 0, 1.0, -1.0), dtype=np.float64)
    lam = _sample_lam()

    # 1. Warm-up JIT compilation on both CPU and GPU
    pair_forces_cpu(pos, species, charge, lam, box, alpha, r_cut)
    pair_forces(pos, species, charge, lam, box, alpha, r_cut)
    cuda.synchronize()

    # 2. Time CPU (numba.njit)
    cpu_reps = max(3, min(n_repeats, 200_000_000 // max(1, n_total * n_total)))
    cpu_times: list[float] = []
    for _ in range(cpu_reps):
        t0 = time.perf_counter()
        pair_forces_cpu(pos, species, charge, lam, box, alpha, r_cut)
        t1 = time.perf_counter()
        cpu_times.append((t1 - t0) * 1e3)

    # 3. Time GPU Total (including H2D and D2H transfers via wrapper)
    gpu_total_times: list[float] = []
    for _ in range(n_repeats):
        cuda.synchronize()
        t0 = time.perf_counter()
        pair_forces(pos, species, charge, lam, box, alpha, r_cut)
        cuda.synchronize()
        t1 = time.perf_counter()
        gpu_total_times.append((t1 - t0) * 1e3)

    # 4. Time GPU Kernel-only (device-resident arrays, as inside the main MD loop)
    d_pos = cuda.to_device(pos)
    d_species = cuda.to_device(species)
    d_charge = cuda.to_device(charge)
    d_lam = cuda.to_device(lam)
    d_forces = cuda.device_array((n_total, 3), dtype=np.float64)
    d_u = cuda.device_array(n_total, dtype=np.float64)
    d_w = cuda.device_array(n_total, dtype=np.float64)
    blocks = (n_total + _TILE_SIZE - 1) // _TILE_SIZE

    # Warm-up raw kernel launch
    _pair_forces_kernel[blocks, _TILE_SIZE](
        d_pos,
        d_species,
        d_charge,
        d_lam,
        float(box),
        float(alpha),
        float(r_cut),
        float(_DEFAULT_E_SQ),
        _TWO_OVER_SQRT_PI,
        d_forces,
        d_u,
        d_w,
    )
    cuda.synchronize()

    gpu_kernel_times: list[float] = []
    for _ in range(n_repeats):
        cuda.synchronize()
        t0 = time.perf_counter()
        _pair_forces_kernel[blocks, _TILE_SIZE](
            d_pos,
            d_species,
            d_charge,
            d_lam,
            float(box),
            float(alpha),
            float(r_cut),
            float(_DEFAULT_E_SQ),
            _TWO_OVER_SQRT_PI,
            d_forces,
            d_u,
            d_w,
        )
        cuda.synchronize()
        t1 = time.perf_counter()
        gpu_kernel_times.append((t1 - t0) * 1e3)

    return (
        float(np.median(cpu_times)),
        float(np.median(gpu_total_times)),
        float(np.median(gpu_kernel_times)),
    )


def plot_benchmark_results(
    n_list: list[int],
    cpu_ms: list[float],
    gpu_total_ms: list[float],
    gpu_kernel_ms: list[float],
    out_stem: Path = Path("figures/diag_cpu_gpu_benchmark"),
) -> None:
    """Plot execution time [ms] and speedup factor (CPU / GPU) across particle counts N."""
    configure_matplotlib_style()
    n_arr = np.array(n_list, dtype=np.float64)
    cpu_arr = np.array(cpu_ms, dtype=np.float64)
    gpu_tot_arr = np.array(gpu_total_ms, dtype=np.float64)
    gpu_ker_arr = np.array(gpu_kernel_ms, dtype=np.float64)

    speedup_kernel = cpu_arr / gpu_ker_arr
    speedup_total = cpu_arr / gpu_tot_arr

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN * 0.94))

    ax1.loglog(
        n_arr,
        cpu_arr,
        linestyle="-",
        marker="o",
        markersize=4.8,
        markerfacecolor="none",
        color="#111111",
        label="CPU (Numba njit)",
    )
    ax1.loglog(
        n_arr,
        gpu_tot_arr,
        linestyle="--",
        marker="s",
        markersize=4.8,
        markerfacecolor="none",
        color="#992222",
        label="GPU + копирование PCIe",
    )
    ax1.loglog(
        n_arr,
        gpu_ker_arr,
        linestyle="-.",
        marker="^",
        markersize=5.0,
        markerfacecolor="none",
        color="#004488",
        label="GPU (только ядро в МД)",
    )
    ax1.axvline(250, linestyle=":", color="#666666", linewidth=0.9)
    ax1.set_xlabel(r"Число частиц $N_{\mathrm{total}}$")
    ax1.set_ylabel(r"Время одного шага сил (мс)")
    ax1.legend(frameon=False, loc="upper left")

    ax2.semilogx(
        n_arr,
        speedup_kernel,
        linestyle="-",
        marker="^",
        markersize=5.0,
        markerfacecolor="none",
        color="#004488",
        label="Ускорение ядра (CPU / GPU kernel)",
    )
    ax2.semilogx(
        n_arr,
        speedup_total,
        linestyle="--",
        marker="s",
        markersize=4.8,
        markerfacecolor="none",
        color="#992222",
        label="Полное ускорение (с учётом PCIe)",
    )
    ax2.axhline(1.0, linestyle=":", color="#666666", linewidth=0.9)
    ax2.axvline(250, linestyle=":", color="#666666", linewidth=0.9, label=r"$N=250$ (Hansen 1981)")
    ax2.set_xlabel(r"Число частиц $N_{\mathrm{total}}$")
    ax2.set_ylabel(r"Коэффициент ускорения (раз)")
    ax2.legend(frameon=False, loc="upper left")

    fig.tight_layout(pad=0.3)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".pdf"), format="pdf")
    fig.savefig(out_stem.with_suffix(".png"), format="png", dpi=300)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Run CPU vs GPU benchmark suite and save CSV + plots."""
    parser = argparse.ArgumentParser(description="Benchmark CPU vs GPU pair_forces.")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[250, 500, 1000, 2000, 4000, 8000, 16000],
        help="List of total particle counts N_total to benchmark.",
    )
    parser.add_argument("--repeats", type=int, default=10, help="Number of timed repetitions.")
    args = parser.parse_args(argv)

    print("=" * 86)
    print(
        f"{'N_total':>8} | {'CPU (ms)':>11} | {'GPU Total (ms)':>14} | "
        f"{'GPU Kernel (ms)':>15} | {'Speedup (tot)':>13} | {'Speedup (ker)':>13}"
    )
    print("-" * 86)

    cpu_list: list[float] = []
    gpu_tot_list: list[float] = []
    gpu_ker_list: list[float] = []

    for n_val in args.sizes:
        c_ms, gt_ms, gk_ms = benchmark_single_n(n_val, n_repeats=args.repeats)
        cpu_list.append(c_ms)
        gpu_tot_list.append(gt_ms)
        gpu_ker_list.append(gk_ms)
        sp_tot = c_ms / gt_ms
        sp_ker = c_ms / gk_ms
        note = " <-- Hansen 1981" if n_val == 250 else ""
        print(
            f"{n_val:8d} | {c_ms:11.3f} | {gt_ms:14.3f} | "
            f"{gk_ms:15.3f} | {sp_tot:12.2f}x | {sp_ker:12.2f}x{note}"
        )

    print("=" * 86)

    csv_out = Path("results/benchmark_cpu_gpu.csv")
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "n_total",
                "cpu_ms",
                "gpu_total_ms",
                "gpu_kernel_ms",
                "speedup_total",
                "speedup_kernel",
            ]
        )
        rows = zip(args.sizes, cpu_list, gpu_tot_list, gpu_ker_list, strict=True)
        for n_v, c_v, gt_v, gk_v in rows:
            writer.writerow(
                [
                    n_v,
                    f"{c_v:.6f}",
                    f"{gt_v:.6f}",
                    f"{gk_v:.6f}",
                    f"{c_v / gt_v:.4f}",
                    f"{c_v / gk_v:.4f}",
                ]
            )

    plot_benchmark_results(args.sizes, cpu_list, gpu_tot_list, gpu_ker_list)
    print("Saved benchmark CSV to results/benchmark_cpu_gpu.csv")
    print("Saved benchmark plots to figures/diag_cpu_gpu_benchmark.{pdf,png}")


if __name__ == "__main__":
    main()
