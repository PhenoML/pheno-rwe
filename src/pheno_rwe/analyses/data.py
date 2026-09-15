"""Validated, read-only tabular input loading for analyses."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pheno_rwe.analyses.base import (
    AnalysisContext,
    AnalysisInputError,
    OptionalDependencyError,
)

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?$")


class DataSourceParams(BaseModel):
    """Common input fields accepted by every analysis.

    Inline ``data`` exists for library use and deterministic unit tests.  Study
    execution normally uses a read-only SELECT against ``deid.duckdb``.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    query: str | None = None
    table: str | None = None
    csv: Path | None = None
    data: Any | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def one_source(self) -> DataSourceParams:
        sources = [
            self.query is not None,
            self.table is not None,
            self.csv is not None,
            self.data is not None,
        ]
        if sum(sources) > 1:
            raise ValueError("provide only one of query, table, csv, or data")
        if self.query is not None:
            normalized = self.query.lstrip().lower()
            if not (normalized.startswith("select") or normalized.startswith("with")):
                raise ValueError("query must be a read-only SELECT or WITH statement")
            if ";" in self.query.rstrip().rstrip(";"):
                raise ValueError("query must contain one statement")
        if self.table is not None and not _IDENTIFIER.fullmatch(self.table):
            raise ValueError("table must be a simple schema-qualified identifier")
        return self


def _pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pandas is a core dependency
        raise OptionalDependencyError("pandas", "analyses") from exc
    return pd


def quote_table(table: str) -> str:
    if not _IDENTIFIER.fullmatch(table):
        raise AnalysisInputError("unsafe table identifier")
    return ".".join(f'"{part}"' for part in table.split("."))


def load_frame(
    ctx: AnalysisContext, params: DataSourceParams, *, default_table: str | None = None
) -> Any:
    pd = _pandas()
    if params.data is not None:
        frame = (
            params.data.copy()
            if isinstance(params.data, pd.DataFrame)
            else pd.DataFrame(params.data)
        )
    elif params.csv is not None:
        csv_path = Path(params.csv)
        if not csv_path.is_absolute():
            csv_path = Path(ctx.output_dir).parent.parent / csv_path
        if not csv_path.is_file():
            raise AnalysisInputError(f"input CSV does not exist: {csv_path}")
        frame = pd.read_csv(csv_path)
    else:
        table = params.table or default_table
        query = params.query or (f"SELECT * FROM {quote_table(table)}" if table else None)
        if query is None:
            raise AnalysisInputError("analysis requires query, table, csv, or inline data")
        if ctx.db_path is None or not Path(ctx.db_path).is_file():
            raise AnalysisInputError("de-identified DuckDB database does not exist")
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - duckdb is a core dependency
            raise OptionalDependencyError("duckdb", "database-backed analyses") from exc
        connection = duckdb.connect(str(ctx.db_path), read_only=True)
        try:
            frame = connection.execute(query).fetchdf()
        except Exception as exc:
            raise AnalysisInputError(f"analysis query failed: {exc}") from exc
        finally:
            connection.close()
    if not isinstance(frame, pd.DataFrame):  # defensive for dataframe-like objects
        frame = pd.DataFrame(frame)
    if frame.columns.duplicated().any():
        duplicates = sorted(set(frame.columns[frame.columns.duplicated()].astype(str)))
        raise AnalysisInputError(f"input contains duplicate columns: {', '.join(duplicates)}")
    return frame


def require_columns(frame: Any, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise AnalysisInputError(f"missing required columns: {', '.join(missing)}")


def two_groups(
    series: Any, requested: tuple[Any | None, Any | None] = (None, None)
) -> tuple[Any, Any]:
    values = [value for value in series.dropna().unique().tolist()]
    if requested[0] is not None or requested[1] is not None:
        exposed, comparator = requested
        if exposed is None or comparator is None or exposed == comparator:
            raise AnalysisInputError(
                "exposed_value and comparator_value must be distinct and supplied together"
            )
        missing = [value for value in (exposed, comparator) if value not in values]
        if missing:
            raise AnalysisInputError(f"requested group values not found: {missing}")
        return exposed, comparator
    if len(values) != 2:
        raise AnalysisInputError(
            f"exactly two non-missing groups are required; found {len(values)}"
        )
    return tuple(sorted(values, key=lambda value: str(value)))  # type: ignore[return-value]


def as_records(frame: Any) -> list[dict[str, Any]]:
    return frame.to_dict(orient="records")


def validate_finite_numeric(
    frame: Any, columns: Iterable[str], *, allow_missing: bool = True
) -> None:
    import numpy as np

    for column in columns:
        converted = _pandas().to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & converted.isna()
        if invalid.any():
            raise AnalysisInputError(f"column '{column}' contains non-numeric values")
        if not allow_missing and converted.isna().any():
            raise AnalysisInputError(f"column '{column}' contains missing values")
        if np.isinf(converted.dropna().to_numpy(dtype=float)).any():
            raise AnalysisInputError(f"column '{column}' contains infinite values")
