"""Plot g_ii, g_ie, g_ee from results/rdf_B.csv against digitized Hansen & McDonald (1981) data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

MM_TO_INCH: float = 1.0 / 25.4
FIG_WIDTH_IN: float = 160.0 * MM_TO_INCH
FIG_HEIGHT_IN: float = 100.0 * MM_TO_INCH


def configure_matplotlib_style() -> None:
    """Configure publication-quality matplotlib parameters (160 mm width, 10 pt font)."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 1.4,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_rdf_csv(
    csv_path: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Load columns (r, g_ii, g_ie, g_ee) from a CSV file."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"RDF CSV file not found: {path}")

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    required = ("r", "g_ii", "g_ie", "g_ee")
    if data.dtype.names is not None and all(col in data.dtype.names for col in required):
        return (
            np.asarray(data["r"], dtype=np.float64),
            np.asarray(data["g_ii"], dtype=np.float64),
            np.asarray(data["g_ie"], dtype=np.float64),
            np.asarray(data["g_ee"], dtype=np.float64),
        )

    raw = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 4:
        raise ValueError(f"Expected 4 columns (r, g_ii, g_ie, g_ee) in {path}.")
    return raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]


def plot_rdf_comparison(
    sim_csv: str | Path = Path("results/rdf_B.csv"),
    ref_csv: str | Path = Path("results/rdf_B_hansen1981_digitized.csv"),
    out_stem: str | Path = Path("figures/rdf_B_comparison"),
) -> tuple[Path, Path]:
    """Generate grayscale-safe publication plot of g_ii, g_ie, g_ee and save PDF + 300 dpi PNG."""
    configure_matplotlib_style()

    r_sim, g_ii_sim, g_ie_sim, g_ee_sim = load_rdf_csv(sim_csv)
    r_ref, g_ii_ref, g_ie_ref, g_ee_ref = load_rdf_csv(ref_csv)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    # Reference line g(r) = 1
    ax.axhline(1.0, color="#888888", linestyle=":", linewidth=0.8, zorder=1)

    # Simulation curves: distinct linestyles for B/W print + high-contrast luminance colors
    ax.plot(
        r_sim,
        g_ie_sim,
        linestyle="--",
        color="#004488",
        linewidth=1.5,
        label=r"$g_{ie}(r)$ (МД)",
        zorder=3,
    )
    ax.plot(
        r_sim,
        g_ee_sim,
        linestyle="-.",
        color="#992222",
        linewidth=1.5,
        label=r"$g_{ee}(r)$ (МД)",
        zorder=3,
    )
    ax.plot(
        r_sim,
        g_ii_sim,
        linestyle="-",
        color="#111111",
        linewidth=1.5,
        label=r"$g_{ii}(r)$ (МД)",
        zorder=3,
    )

    # Digitized points from Hansen & McDonald (1981): open markers for B/W clarity
    ax.plot(
        r_ref,
        g_ie_ref,
        linestyle="none",
        marker="s",
        markersize=4.8,
        markerfacecolor="none",
        markeredgecolor="#004488",
        markeredgewidth=1.1,
        label=r"$g_{ie}(r)$ (Hansen & McDonald 1981)",
        zorder=4,
    )
    ax.plot(
        r_ref,
        g_ee_ref,
        linestyle="none",
        marker="^",
        markersize=5.2,
        markerfacecolor="none",
        markeredgecolor="#992222",
        markeredgewidth=1.1,
        label=r"$g_{ee}(r)$ (Hansen & McDonald 1981)",
        zorder=4,
    )
    ax.plot(
        r_ref,
        g_ii_ref,
        linestyle="none",
        marker="o",
        markersize=4.8,
        markerfacecolor="none",
        markeredgecolor="#111111",
        markeredgewidth=1.1,
        label=r"$g_{ii}(r)$ (Hansen & McDonald 1981)",
        zorder=4,
    )

    ax.set_xlabel(r"Межчастичное расстояние $r / a$")
    ax.set_ylabel(r"Парная корреляционная функция $g_{ab}(r)$")
    ax.set_xlim(0.0, float(max(np.max(r_sim), np.max(r_ref))))
    y_max = float(max(np.max(g_ie_sim), np.max(g_ie_ref))) * 1.08
    ax.set_ylim(0.0, max(2.5, y_max))

    # Legend without frame (frameon=False), no figure title
    ax.legend(frameon=False, loc="upper right", ncol=1, handlelength=2.6)

    fig.tight_layout(pad=0.3)

    stem = Path(out_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")

    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Plot RDF g_ii, g_ie, g_ee against digitized Hansen & McDonald (1981) data."
    )
    parser.add_argument(
        "--sim",
        type=Path,
        default=Path("results/rdf_B.csv"),
        help="Path to simulation RDF CSV (r, g_ii, g_ie, g_ee).",
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=Path("results/rdf_B_hansen1981_digitized.csv"),
        help="Path to digitized reference RDF CSV (r, g_ii, g_ie, g_ee).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures/rdf_B_comparison"),
        help="Output file stem (produces .pdf and 300 dpi .png).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    pdf_path, png_path = plot_rdf_comparison(args.sim, args.ref, args.out)
    print(f"Saved RDF figures: {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()
