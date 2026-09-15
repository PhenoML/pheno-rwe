"""Descriptive baseline table with standardized mean differences."""

from __future__ import annotations

import math
from typing import Any

from pydantic import Field, model_validator

from pheno_rwe.analyses.base import (
    AnalysisContext,
    AnalysisInputError,
    AnalysisResult,
    package_versions,
)
from pheno_rwe.analyses.data import DataSourceParams, load_frame, require_columns
from pheno_rwe.analyses.provenance import summarize_provenance
from pheno_rwe.analyses.stats import standardized_mean_difference


class TableOneParams(DataSourceParams):
    group_column: str = "group"
    categorical: list[str] = Field(default_factory=list)
    continuous: list[str] = Field(default_factory=list)
    variables: list[str] = Field(default_factory=list)
    small_cell_threshold: int = Field(default=5, ge=1)
    include_overall: bool = True

    @model_validator(mode="after")
    def variables_are_disjoint(self) -> TableOneParams:
        overlap = set(self.categorical) & set(self.continuous)
        if overlap:
            raise ValueError(
                f"variables cannot be both categorical and continuous: {sorted(overlap)}"
            )
        return self


def _categorical_smd(group_a: Any, group_b: Any, level: Any) -> float | None:
    a = group_a.notna()
    b = group_b.notna()
    if not a.any() or not b.any():
        return None
    p_a = float((group_a[a] == level).mean())
    p_b = float((group_b[b] == level).mean())
    pooled = (p_a + p_b) / 2
    denominator = math.sqrt(pooled * (1 - pooled))
    if denominator == 0:
        return 0.0 if p_a == p_b else None
    return (p_a - p_b) / denominator


def _display_number(value: float | None, digits: int = 3) -> str:
    return "" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


class TableOneAnalysis:
    kind = "table_one"
    Params = TableOneParams
    requires = ("person-level analysis dataset",)
    rules = ("SMALL-CELL", "MISSINGNESS")

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        params = self.Params.model_validate(ctx.params)
        frame = load_frame(ctx, params)
        require_columns(frame, [params.group_column])
        if frame.empty:
            raise AnalysisInputError("table_one requires at least one row")
        if frame[params.group_column].isna().any():
            raise AnalysisInputError("group column contains missing values")

        requested = list(
            dict.fromkeys([*params.variables, *params.categorical, *params.continuous])
        )
        if requested:
            require_columns(frame, requested)
        excluded = {
            params.group_column,
            "person_id",
            "mapping_status",
            "origin",
            "domain",
        }
        if not requested:
            requested = [str(column) for column in frame.columns if column not in excluded]

        categorical = list(params.categorical)
        continuous = list(params.continuous)
        unassigned = [
            column for column in requested if column not in categorical and column not in continuous
        ]
        for column in unassigned:
            series = frame[column]
            if str(series.dtype) in {"object", "category", "bool", "boolean", "string"}:
                categorical.append(column)
            else:
                continuous.append(column)

        pd = __import__("pandas")
        for column in continuous:
            converted = pd.to_numeric(frame[column], errors="coerce")
            invalid = frame[column].notna() & converted.isna()
            if invalid.any():
                raise AnalysisInputError(
                    f"continuous variable '{column}' contains non-numeric values"
                )
            frame[column] = converted

        group_values = sorted(
            frame[params.group_column].unique().tolist(), key=lambda value: str(value)
        )
        if not group_values:
            raise AnalysisInputError("group column has no observed values")
        group_frames: list[tuple[str, Any]] = [
            (str(value), frame.loc[frame[params.group_column] == value]) for value in group_values
        ]
        if params.include_overall:
            group_frames.append(("Overall", frame))

        rows: list[dict[str, Any]] = []
        estimates: list[dict[str, Any]] = []
        two_group = len(group_values) == 2

        for variable in continuous:
            smd = None
            if two_group:
                first = frame.loc[frame[params.group_column] == group_values[0], variable].dropna()
                second = frame.loc[frame[params.group_column] == group_values[1], variable].dropna()
                smd = standardized_mean_difference(first, second)
                estimates.append(
                    {"variable": variable, "level": None, "estimand": "smd", "estimate": smd}
                )
            for group, subset in group_frames:
                values = subset[variable].dropna().astype(float)
                statistics = {
                    "n": int(values.size),
                    "missing": int(subset[variable].isna().sum()),
                    "mean": float(values.mean()) if len(values) else None,
                    "std": float(values.std(ddof=1)) if len(values) > 1 else None,
                    "median": float(values.median()) if len(values) else None,
                    "q1": float(values.quantile(0.25)) if len(values) else None,
                    "q3": float(values.quantile(0.75)) if len(values) else None,
                }
                for statistic, value in statistics.items():
                    rows.append(
                        {
                            "variable": variable,
                            "variable_type": "continuous",
                            "level": None,
                            "group": group,
                            "statistic": statistic,
                            "value": value,
                            "display": str(value)
                            if statistic in {"n", "missing"}
                            else _display_number(value),
                            "smd": smd,
                            "suppressed": False,
                        }
                    )

        for variable in categorical:
            levels = sorted(
                frame[variable].dropna().unique().tolist(), key=lambda value: str(value)
            )
            if not levels:
                levels = ["(no observed values)"]
            for level in levels:
                smd = None
                if two_group:
                    group_a = frame.loc[frame[params.group_column] == group_values[0], variable]
                    group_b = frame.loc[frame[params.group_column] == group_values[1], variable]
                    smd = _categorical_smd(group_a, group_b, level)
                    estimates.append(
                        {
                            "variable": variable,
                            "level": str(level),
                            "estimand": "smd",
                            "estimate": smd,
                        }
                    )
                for group, subset in group_frames:
                    denominator = int(subset[variable].notna().sum())
                    count = int((subset[variable] == level).sum()) if denominator else 0
                    suppressed = 0 < count < params.small_cell_threshold
                    percent = 100 * count / denominator if denominator else None
                    rows.append(
                        {
                            "variable": variable,
                            "variable_type": "categorical",
                            "level": str(level),
                            "group": group,
                            "statistic": "count_percent",
                            "value": None if suppressed else count,
                            "percent": None if suppressed else percent,
                            "display": f"<{params.small_cell_threshold}"
                            if suppressed
                            else (f"{count} ({percent:.1f}%)" if percent is not None else "0"),
                            "smd": smd,
                            "suppressed": suppressed,
                        }
                    )
            for group, subset in group_frames:
                missing = int(subset[variable].isna().sum())
                suppressed = 0 < missing < params.small_cell_threshold
                rows.append(
                    {
                        "variable": variable,
                        "variable_type": "categorical",
                        "level": "Missing",
                        "group": group,
                        "statistic": "missing",
                        "value": None if suppressed else missing,
                        "percent": None if suppressed else 100 * missing / len(subset),
                        "display": f"<{params.small_cell_threshold}"
                        if suppressed
                        else str(missing),
                        "smd": None,
                        "suppressed": suppressed,
                    }
                )

        provenance_columns = {"mapping_status", "origin", "domain"} & set(frame.columns)
        provenance = (
            summarize_provenance(frame)
            if provenance_columns
            else dict(ctx.provenance) or {"available": False}
        )
        per_group = {
            str(value): int((frame[params.group_column] == value).sum()) for value in group_values
        }
        missing_counts = {column: int(frame[column].isna().sum()) for column in requested}
        return AnalysisResult(
            analysis_id=ctx.analysis_id,
            kind=self.kind,
            n={"total": int(len(frame)), "per_group": per_group, "excluded": 0},
            estimates=estimates,
            provenance=provenance,
            assumptions_checked=[
                {"name": "group_nonmissing", "passed": True},
                {"name": "missingness_reported", "passed": True, "counts": missing_counts},
                {
                    "name": "small_cell_suppression",
                    "passed": True,
                    "threshold": params.small_cell_threshold,
                },
            ],
            guardrail_outcomes=[dict(item) for item in ctx.guardrail_outcomes],
            power_note="Descriptive analysis; no hypothesis-test power calculation was performed.",
            seed=ctx.seed,
            package_versions=package_versions(),
            tables={"data": rows, "smd": estimates},
            metadata={"continuous": continuous, "categorical": categorical},
            output_dir=ctx.output_dir,
        )


ANALYSIS = TableOneAnalysis()
