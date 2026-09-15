"""Stable analysis protocol and serializable result envelope.

The CLI is intentionally a thin adapter around this module.  Analyses receive an
``AnalysisContext`` and return an ``AnalysisResult``; neither object knows about
Typer, manifests, or a frontend.
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from pydantic import BaseModel


class AnalysisError(ValueError):
    """An analysis could not be run because its deterministic contract failed."""


class AnalysisInputError(AnalysisError):
    """The supplied data or parameters are not suitable for the analysis."""


class OptionalDependencyError(AnalysisError, ImportError):
    """A requested method needs a package from the analysis extra."""

    def __init__(self, package: str, feature: str | None = None) -> None:
        detail = f" to run {feature}" if feature else ""
        super().__init__(
            f"Optional dependency '{package}' is required{detail}. "
            'Install it with `pip install "pheno-rwe[analysis]"`.'
        )
        self.package = package
        self.feature = feature


def require_dependency(
    import_name: str, *, package: str | None = None, feature: str | None = None
) -> Any:
    """Import an optional package with an actionable, stable error message."""

    try:
        return __import__(import_name, fromlist=["*"])
    except ImportError as exc:  # pragma: no cover - the error path is environment dependent
        raise OptionalDependencyError(package or import_name, feature) from exc


def _spec_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if is_dataclass(value):
        return asdict(cast(Any, value))
    if isinstance(value, Mapping):
        return dict(value)
    raise AnalysisInputError("analysis spec must be a mapping, dataclass, or pydantic model")


@dataclass(frozen=True, slots=True)
class AnalysisContext:
    """All deterministic inputs available to an analysis implementation."""

    db_path: Path | str | None
    spec: Mapping[str, Any] | BaseModel | Any
    analysis_id: str
    seed: int
    output_dir: Path | str
    guardrail_outcomes: tuple[Mapping[str, Any], ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.analysis_id or not str(self.analysis_id).strip():
            raise AnalysisInputError("analysis_id must be non-empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise AnalysisInputError("seed must be an integer")
        object.__setattr__(
            self, "db_path", Path(self.db_path) if self.db_path is not None else None
        )
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    @property
    def spec_dict(self) -> dict[str, Any]:
        return _spec_mapping(self.spec)

    @property
    def kind(self) -> str:
        value = self.spec_dict.get("kind", self.spec_dict.get("type", ""))
        return str(value)

    @property
    def params(self) -> dict[str, Any]:
        """Return params while accepting both nested and flat plan representations."""

        spec = self.spec_dict
        nested = spec.get("params")
        if nested is None:
            return {
                key: value
                for key, value in spec.items()
                if key not in {"id", "analysis_id", "kind", "type", "name"}
            }
        if not isinstance(nested, Mapping):
            raise AnalysisInputError("analysis spec 'params' must be an object")
        return dict(nested)


def _json_value(value: Any) -> Any:
    """Convert numpy/pandas/scalar values without importing those packages."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    return str(value)


def package_versions(names: tuple[str, ...] = ()) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("pheno-rwe", "pandas", *names):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return result


@dataclass(slots=True)
class AnalysisResult:
    """Uniform envelope written by every registered analysis.

    ``tables`` is deliberately excluded from ``result.json``.  It is serialized
    as tidy CSV files and represented in the envelope by ``output_tables``.
    """

    analysis_id: str
    kind: str
    n: dict[str, Any]
    estimates: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    assumptions_checked: list[dict[str, Any] | str] = field(default_factory=list)
    guardrail_outcomes: list[dict[str, Any]] = field(default_factory=list)
    power_note: str = ""
    seed: int = 0
    package_versions: dict[str, str] = field(default_factory=dict)
    tables: dict[str, Any] = field(default_factory=dict, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    output_tables: list[dict[str, Any]] = field(default_factory=list)
    output_dir: Path | str | None = field(default=None, repr=False)

    def model_dump(self) -> dict[str, Any]:
        return _json_value(
            {
                "schema_version": "1.0",
                "analysis_id": self.analysis_id,
                "kind": self.kind,
                "status": self.status,
                "n": self.n,
                "estimates": self.estimates,
                "provenance": self.provenance,
                "assumptions_checked": self.assumptions_checked,
                "guardrail_outcomes": self.guardrail_outcomes,
                "power_note": self.power_note,
                "seed": self.seed,
                "package_versions": self.package_versions,
                "output_tables": self.output_tables,
                "metadata": self.metadata,
            }
        )

    def _table_records(self, table: Any) -> tuple[list[str], list[dict[str, Any]]]:
        if hasattr(table, "to_dict") and hasattr(table, "columns"):
            columns = [str(column) for column in table.columns]
            records = table.to_dict(orient="records")
        elif isinstance(table, list):
            records = [dict(row) for row in table]
            columns = []
            for row in records:
                for column in row:
                    if column not in columns:
                        columns.append(str(column))
        else:
            raise TypeError("result tables must be pandas DataFrames or lists of row mappings")
        return columns, [_json_value(record) for record in records]

    def write(self, output_dir: Path | str | None = None) -> Path:
        """Write tidy CSV tables followed by a canonical result envelope."""

        destination = (
            Path(output_dir)
            if output_dir is not None
            else (Path(self.output_dir) if self.output_dir is not None else None)
        )
        if destination is None:
            raise AnalysisInputError("AnalysisResult.write requires output_dir")
        destination.mkdir(parents=True, exist_ok=True)
        descriptions: list[dict[str, Any]] = []
        for name in sorted(self.tables):
            if not name or any(
                char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for char in name
            ):
                raise AnalysisInputError(f"unsafe output table name: {name!r}")
            columns, records = self._table_records(self.tables[name])
            target = destination / f"{name}.csv"
            temporary = destination / f".{name}.csv.tmp"
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(records)
            os.replace(temporary, target)
            descriptions.append(
                {
                    "name": name,
                    "path": target.name,
                    "format": "text/csv",
                    "rows": len(records),
                    "columns": columns,
                }
            )
        self.output_tables = descriptions
        result_path = destination / "result.json"
        temporary_result = destination / ".result.json.tmp"
        temporary_result.write_text(
            json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_result, result_path)
        return result_path


@runtime_checkable
class Analysis(Protocol):
    @property
    def kind(self) -> str: ...

    @property
    def Params(self) -> type[BaseModel]: ...

    @property
    def requires(self) -> tuple[str, ...]: ...

    @property
    def rules(self) -> tuple[str, ...]: ...

    def run(self, ctx: AnalysisContext) -> AnalysisResult: ...
