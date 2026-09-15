"""Longitudinal descriptive trajectories with patient-cluster bootstrap bands."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pheno_rwe.analyses.base import (
    AnalysisContext,
    AnalysisInputError,
    AnalysisResult,
    OptionalDependencyError,
    package_versions,
)
from pheno_rwe.analyses.data import DataSourceParams, load_frame, require_columns


class UnitConversion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_unit: str
    multiplier: float = 1.0
    offset: float = 0.0


class TrajectoryParams(DataSourceParams):
    cohort: str | None = None
    measurement: str | None = None
    patient_column: str = "person_id"
    time_column: str = "time"
    value_column: str = "value"
    group_column: str | None = None
    group_by: str | None = None
    unit_column: str | None = "unit"
    mixed_model: bool = False
    unit_harmonization: dict[str, str | float | UnitConversion] = Field(default_factory=dict)
    harmonization_map: dict[str, str | float | UnitConversion] = Field(default_factory=dict)
    bootstrap_iterations: int = Field(default=1_000, ge=100, le=100_000)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)

    @model_validator(mode="after")
    def normalize_aliases(self) -> TrajectoryParams:
        if self.group_by is not None:
            if self.group_column is not None and self.group_column != self.group_by:
                raise ValueError("group_column and group_by disagree")
            self.group_column = self.group_by
        if self.harmonization_map:
            if self.unit_harmonization and self.unit_harmonization != self.harmonization_map:
                raise ValueError("unit_harmonization and harmonization_map disagree")
            self.unit_harmonization = self.harmonization_map
        return self


def _harmonize_units(frame: Any, params: TrajectoryParams) -> tuple[Any, list[str]]:
    if not params.unit_column or params.unit_column not in frame:
        return frame, []
    units = sorted(frame[params.unit_column].dropna().astype(str).unique().tolist())
    if len(units) <= 1:
        return frame, units
    if not params.unit_harmonization:
        raise AnalysisInputError(
            f"mixed measurement units are refused without a harmonization map: {units}"
        )
    missing = [unit for unit in units if unit not in params.unit_harmonization]
    if missing:
        raise AnalysisInputError(f"harmonization map does not cover units: {missing}")
    result = frame.copy()
    targets: list[str] = []
    for unit in units:
        conversion = params.unit_harmonization[unit]
        if isinstance(conversion, UnitConversion):
            target, multiplier, offset = (
                conversion.target_unit,
                conversion.multiplier,
                conversion.offset,
            )
        elif isinstance(conversion, (int, float)):
            target, multiplier, offset = "harmonized", float(conversion), 0.0
        else:
            target, multiplier, offset = str(conversion), 1.0, 0.0
        mask = result[params.unit_column].astype(str) == unit
        result.loc[mask, params.value_column] = (
            result.loc[mask, params.value_column] * multiplier + offset
        )
        result.loc[mask, params.unit_column] = target
        targets.append(target)
    remaining = sorted(set(targets))
    if len(remaining) > 1:
        raise AnalysisInputError(f"harmonization map still yields mixed target units: {remaining}")
    return result, remaining


def _bootstrap_summary(frame: Any, params: TrajectoryParams, group_column: str, seed: int) -> Any:
    import numpy as np
    import pandas as pd

    keys = [group_column, params.time_column]
    observed = (
        frame.groupby(keys, dropna=False, sort=True)[params.value_column]
        .agg(["mean", "count", "std"])
        .reset_index()
        .rename(columns={"mean": "value", "count": "n_measurements", "std": "standard_deviation"})
    )
    patients = sorted(frame[params.patient_column].unique().tolist(), key=lambda value: str(value))
    rng = np.random.default_rng(seed)
    distributions: dict[tuple[Any, Any], list[float]] = {
        tuple(row): [] for row in observed[keys].itertuples(index=False, name=None)
    }
    indexed = {patient: frame.loc[frame[params.patient_column] == patient] for patient in patients}
    for _ in range(params.bootstrap_iterations):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        sample = pd.concat([indexed[patient] for patient in sampled], ignore_index=True)
        means = sample.groupby(keys, dropna=False)[params.value_column].mean()
        for key in distributions:
            if key in means.index:
                distributions[key].append(float(means.loc[key]))
    alpha = 1 - params.confidence_level
    lower: list[float | None] = []
    upper: list[float | None] = []
    for key in observed[keys].itertuples(index=False, name=None):
        values = distributions[tuple(key)]
        if values:
            bounds = np.quantile(values, [alpha / 2, 1 - alpha / 2])
            lower.append(float(bounds[0]))
            upper.append(float(bounds[1]))
        else:
            lower.append(None)
            upper.append(None)
    observed["lower"] = lower
    observed["upper"] = upper
    observed["n_patients"] = [
        int(
            frame.loc[
                (frame[group_column] == row[0]) & (frame[params.time_column] == row[1]),
                params.patient_column,
            ].nunique()
        )
        for row in observed[keys].itertuples(index=False, name=None)
    ]
    return observed


def _mixed_model(
    frame: Any, params: TrajectoryParams, group_column: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("statsmodels", "trajectory mixed-effects model") from exc
    patient_count = int(frame[params.patient_column].nunique())
    median_measurements = float(frame.groupby(params.patient_column).size().median())
    if patient_count < 30 or median_measurements < 3:
        raise AnalysisInputError(
            "mixed-effects trajectory requires at least 30 patients and a median of at least "
            "3 measurements"
        )
    working = frame.rename(
        columns={
            params.value_column: "_value",
            params.time_column: "_time",
            params.patient_column: "_person",
            group_column: "_group",
        }
    )
    formula = "_value ~ _time" if working["_group"].nunique() == 1 else "_value ~ _time * C(_group)"
    try:
        fit = smf.mixedlm(formula, working, groups=working["_person"]).fit(
            reml=True, method="lbfgs", disp=False
        )
    except Exception as exc:
        raise AnalysisInputError(f"mixed-effects trajectory model failed: {exc}") from exc
    intervals = fit.conf_int()
    rows = [
        {
            "term": str(term),
            "estimate": float(fit.params[term]),
            "lower": float(intervals.loc[term, 0]),
            "upper": float(intervals.loc[term, 1]),
            "p_value": float(fit.pvalues[term]),
        }
        for term in fit.params.index
    ]
    return rows, {"converged": bool(fit.converged), "formula": formula, "reml": True}


class TrajectoryAnalysis:
    kind = "trajectory"
    Params = TrajectoryParams
    requires = ("longitudinal measurements", "consistent units")
    rules = ("MEAS-UNITS", "TRAJ-MIXED-MIN")

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        import pandas as pd

        params = self.Params.model_validate(ctx.params)
        frame = load_frame(ctx, params)
        required = [params.patient_column, params.time_column, params.value_column]
        if params.group_column:
            required.append(params.group_column)
        if params.unit_column and params.unit_column in frame:
            required.append(params.unit_column)
        require_columns(frame, required)
        if frame.empty:
            raise AnalysisInputError("trajectory requires at least one measurement")
        working = (
            frame[required]
            .dropna(subset=[params.patient_column, params.time_column, params.value_column])
            .copy()
        )
        values: Any = pd.to_numeric(working[params.value_column], errors="coerce")
        times: Any = pd.to_numeric(working[params.time_column], errors="coerce")
        if values.isna().any() or times.isna().any():
            raise AnalysisInputError("trajectory time and value columns must be numeric")
        working[params.value_column] = values.astype(float)
        working[params.time_column] = times.astype(float)
        working, units = _harmonize_units(working, params)
        group_column = params.group_column or "_group"
        if params.group_column is None:
            working[group_column] = "Overall"
        if working[group_column].isna().any():
            raise AnalysisInputError("trajectory group column contains missing values")

        summary = _bootstrap_summary(working, params, group_column, ctx.seed)
        summary = summary.rename(columns={group_column: "group", params.time_column: "time"})
        raw = working.rename(
            columns={
                params.patient_column: "person_id",
                group_column: "group",
                params.time_column: "time",
                params.value_column: "value",
            }
        )[["person_id", "group", "time", "value"]]
        raw["row_type"] = "individual"
        summary["row_type"] = "summary"
        data = pd.concat([raw, summary], ignore_index=True, sort=False)

        coefficients: list[dict[str, Any]] = []
        mixed_metadata: dict[str, Any] = {"requested": params.mixed_model}
        if params.mixed_model:
            coefficients, fit_metadata = _mixed_model(working, params, group_column)
            mixed_metadata.update(fit_metadata)
        counts = working.groupby(group_column)[params.patient_column].nunique().sort_index()
        return AnalysisResult(
            analysis_id=ctx.analysis_id,
            kind=self.kind,
            n={
                "total": int(working[params.patient_column].nunique()),
                "per_group": {str(key): int(value) for key, value in counts.items()},
                "measurements": int(len(working)),
                "excluded": int(len(frame) - len(working)),
            },
            estimates=coefficients,
            provenance=dict(ctx.provenance) or {"available": False},
            assumptions_checked=[
                {"name": "measurement_units", "passed": True, "units": units},
                {
                    "name": "patient_cluster_bootstrap",
                    "passed": True,
                    "iterations": params.bootstrap_iterations,
                },
                {"name": "mixed_model_sample_size", "passed": True if params.mixed_model else None},
            ],
            guardrail_outcomes=[dict(item) for item in ctx.guardrail_outcomes],
            power_note=(
                "Trajectory confidence bands are descriptive patient-cluster bootstrap intervals; "
                "no post-hoc power was calculated."
            ),
            seed=ctx.seed,
            package_versions=package_versions(("statsmodels",) if params.mixed_model else ()),
            tables={
                "data": data,
                "summary": summary,
                **({"mixed_model": coefficients} if coefficients else {}),
            },
            metadata={"units": units, "mixed_model": mixed_metadata},
            output_dir=ctx.output_dir,
        )


ANALYSIS = TrajectoryAnalysis()
