"""Tests for publication-quality matplotlib RDF and diagnostic plotting scripts."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from scripts.plot_all_diagnostics import ensure_rdf_b_csvs, generate_all_plots
from scripts.plot_rdf_b import plot_rdf_comparison


def test_plot_rdf_b_creates_pdf_and_300dpi_png(tmp_path: Path) -> None:
    """Verify plot_rdf_b generates non-empty PDF and 300 dpi PNG at 160 mm width."""
    sim_csv = tmp_path / "rdf_B.csv"
    ref_csv = tmp_path / "rdf_B_hansen1981_digitized.csv"
    ensure_rdf_b_csvs(sim_csv, ref_csv)

    out_stem = tmp_path / "rdf_B_comparison"
    pdf_path, png_path = plot_rdf_comparison(sim_csv, ref_csv, out_stem)

    assert pdf_path.exists() and pdf_path.stat().st_size > 1000
    assert png_path.exists() and png_path.stat().st_size > 5000

    with Image.open(png_path) as img:
        width_px, _ = img.size
        # 160 mm = 6.2992 in; at 300 dpi expected width is ~1890 px
        assert 1850 <= width_px <= 1930


def test_generate_all_diagnostic_plots(tmp_path: Path) -> None:
    """Verify all diagnostic plots (RDF, Ewald, Kelbg, Block Average, FSS) are generated."""
    created = generate_all_plots(tmp_path)
    expected_stems = {
        "rdf_B_comparison",
        "diag_ewald_verification",
        "diag_kelbg_derivatives",
        "diag_block_average_plateau",
        "diag_fss_diffusion",
    }
    for stem in expected_stems:
        assert (tmp_path / f"{stem}.pdf").exists()
        assert (tmp_path / f"{stem}.png").exists()
    assert len(created) == 10
