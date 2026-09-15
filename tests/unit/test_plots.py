from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from pheno_rwe.plots import PlotInputError, kinds, render_plot


def digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_plot_registry_has_complete_catalog() -> None:
    assert set(kinds()) == {
        "km_curve",
        "forest",
        "love_plot",
        "umap_scatter",
        "trajectory",
        "attrition",
        "mapping_coverage",
        "incidence_bar",
        "pathway_bars",
    }


def test_km_plot_is_bit_stable_when_replayed_from_persisted_csv(tmp_path) -> None:
    source = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B"],
            "time": [0, 10, 20, 0, 10, 20],
            "survival": [1, 0.8, 0.6, 1, 0.9, 0.75],
            "lower": [1, 0.6, 0.4, 1, 0.7, 0.55],
            "upper": [1, 0.95, 0.8, 1, 0.98, 0.9],
            "at_risk": [10, 8, 6, 10, 9, 7],
        }
    ).to_csv(source, index=False)
    first = render_plot("km_curve", source, tmp_path / "first")
    second = render_plot("km_curve", first.data_path, tmp_path / "second")
    assert digest(first.png_path) == digest(second.png_path)
    assert digest(first.svg_path) == digest(second.svg_path)
    assert digest(first.data_path) == digest(second.data_path)
    assert digest(first.spec_path) == digest(second.spec_path)
    assert first.png_path.stat().st_size > 10_000
    assert first.svg_path.read_text().lstrip().startswith("<?xml")


def test_plots_refuse_non_csv_objects_and_validate_columns(tmp_path) -> None:
    with pytest.raises(PlotInputError, match="tidy CSV path only"):
        render_plot("forest", pd.DataFrame({"estimate": [1]}), tmp_path / "plot")
    source = tmp_path / "bad.csv"
    pd.DataFrame({"estimate": [1]}).to_csv(source, index=False)
    with pytest.raises(PlotInputError, match="missing required columns"):
        render_plot("forest", source, tmp_path / "plot")
