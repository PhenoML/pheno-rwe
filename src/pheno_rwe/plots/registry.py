"""CSV-only plot registry and deterministic PNG/SVG serialization."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

from pheno_rwe.plots.base import PlotInputError, PlotRenderer, PlotResult

_REGISTRY: dict[str, PlotRenderer] = {}
_LOADED = False


def register(renderer: PlotRenderer, *, replace: bool = False) -> PlotRenderer:
    kind = str(getattr(renderer, "kind", "")).strip()
    if not kind:
        raise ValueError("plot renderer must declare a kind")
    if kind in _REGISTRY and _REGISTRY[kind] is not renderer and not replace:
        raise ValueError(f"plot kind already registered: {kind}")
    _REGISTRY[kind] = renderer
    return renderer


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    from pheno_rwe.plots.renderers import BUILTIN_RENDERERS

    for renderer in BUILTIN_RENDERERS:
        register(renderer)
    _LOADED = True


def get(kind: str) -> PlotRenderer:
    _load()
    try:
        return _REGISTRY[kind]
    except KeyError as exc:
        raise PlotInputError(
            f"unknown plot kind '{kind}'; available kinds: {', '.join(sorted(_REGISTRY))}"
        ) from exc


def kinds() -> tuple[str, ...]:
    _load()
    return tuple(sorted(_REGISTRY))


def _style_path() -> Path:
    return Path(__file__).resolve().parent.parent / "pheno_rwe.mplstyle"


def render_plot(
    kind: str,
    source_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    csv_path: str | Path | None = None,
    params: dict[str, Any] | None = None,
    spec: Any | None = None,
    provenance: dict[str, Any] | None = None,
) -> PlotResult:
    """Render from a tidy CSV path; dataframe/model inputs are intentionally rejected."""

    source_value = source_csv if source_csv is not None else csv_path
    if (
        source_value is None
        or isinstance(source_value, (dict, list))
        or hasattr(source_value, "columns")
    ):
        raise PlotInputError(
            "plots consume a tidy CSV path only; model objects and dataframes are not accepted"
        )
    source = Path(source_value)
    if not source.is_file():
        raise PlotInputError(f"plot source CSV does not exist: {source}")
    if source.suffix.lower() != ".csv":
        raise PlotInputError("plot source must be a .csv file")
    if output_dir is None:
        raise PlotInputError("output_dir is required")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    merged_params: dict[str, Any] = {}
    if spec is not None:
        value = spec.model_dump(mode="python") if hasattr(spec, "model_dump") else dict(spec)
        merged_params.update(value.get("params", value))
        if value.get("title"):
            merged_params["title"] = value["title"]
    merged_params.update(params or {})
    # Source selection belongs to the CLI adapter, not the visual specification.
    merged_params.pop("source_csv", None)
    merged_params.pop("analysis_id", None)

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        matplotlib.rcParams["svg.hashsalt"] = "pheno-rwe-v1"
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        from pheno_rwe.analyses.base import OptionalDependencyError

        raise OptionalDependencyError("matplotlib", "plots") from exc

    try:
        frame = cast(Any, pd.read_csv)(source, float_precision="round_trip")
    except Exception as exc:
        raise PlotInputError(f"could not read plot CSV: {exc}") from exc
    if frame.columns.duplicated().any():
        raise PlotInputError("plot CSV contains duplicate column names")
    data_path = destination / "data.csv"
    temporary_data = destination / ".data.csv.tmp"
    frame.to_csv(temporary_data, index=False, lineterminator="\n", float_format="%.17g")
    os.replace(temporary_data, data_path)
    # Draw from the exact persisted representation so --from-csv is a true replay.
    frame = cast(Any, pd.read_csv)(data_path, float_precision="round_trip")

    renderer = get(kind)
    style = _style_path()
    if not style.is_file():
        raise RuntimeError(f"bundled matplotlib style is missing: {style}")
    with plt.style.context(str(style)):
        figure = renderer.draw(frame, merged_params)
        png_path = destination / "plot.png"
        svg_path = destination / "plot.svg"
        temporary_png = destination / ".plot.png.tmp"
        temporary_svg = destination / ".plot.svg.tmp"
        figure.savefig(
            temporary_png,
            format="png",
            dpi=300,
            bbox_inches="tight",
            metadata={"Software": "pheno-rwe"},
        )
        figure.savefig(
            temporary_svg,
            format="svg",
            bbox_inches="tight",
            metadata={"Date": None, "Creator": "pheno-rwe"},
        )
        plt.close(figure)
        os.replace(temporary_png, png_path)
        os.replace(temporary_svg, svg_path)

    spec_payload = {
        "schema_version": "1.0",
        "kind": kind,
        "params": merged_params,
        "columns": [str(column) for column in frame.columns],
        "renderer": "matplotlib",
        "dpi": 300,
        "style": "pheno_rwe.mplstyle",
    }
    if provenance is not None:
        spec_payload["provenance"] = provenance
    spec_path = destination / "spec.json"
    temporary_spec = destination / ".spec.json.tmp"
    temporary_spec.write_text(
        json.dumps(spec_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_spec, spec_path)
    return PlotResult(kind, destination, png_path, svg_path, data_path, spec_path)


render = render_plot
get_plot = get
list_plots = kinds
