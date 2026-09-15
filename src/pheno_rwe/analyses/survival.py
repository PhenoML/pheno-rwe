"""Kaplan-Meier, log-rank, RMST, and optional penalized Cox analysis."""

from __future__ import annotations

import math
from typing import Any

from pydantic import Field, model_validator

from pheno_rwe.analyses.base import (
    AnalysisContext,
    AnalysisInputError,
    AnalysisResult,
    OptionalDependencyError,
    package_versions,
)
from pheno_rwe.analyses.data import DataSourceParams, load_frame, require_columns, two_groups
from pheno_rwe.analyses.power import survival_log_hazard_mde


class SurvivalParams(DataSourceParams):
    duration_column: str = "duration"
    event_column: str = "event"
    group_column: str = "group"
    exposed_value: Any | None = None
    comparator_value: Any | None = None
    censor_rule: str = Field(min_length=1)
    rmst_horizon: float | None = Field(default=None, gt=0)
    horizon: float | None = Field(default=None, gt=0)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    bootstrap_iterations: int = Field(default=1_000, ge=100, le=100_000)
    cox: bool = False
    cox_covariates: list[str] = Field(default_factory=list)
    cox_penalizer: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def normalize_horizon(self) -> SurvivalParams:
        if (
            self.rmst_horizon is not None
            and self.horizon is not None
            and self.rmst_horizon != self.horizon
        ):
            raise ValueError("rmst_horizon and horizon disagree")
        if self.rmst_horizon is None:
            self.rmst_horizon = self.horizon
        if (self.exposed_value is None) != (self.comparator_value is None):
            raise ValueError("exposed_value and comparator_value must be supplied together")
        return self


def kaplan_meier(
    duration: Any, event: Any, *, confidence_level: float = 0.95
) -> list[dict[str, Any]]:
    """Compute a product-limit curve with Greenwood log-log intervals."""

    import numpy as np

    times = np.asarray(duration, dtype=float)
    events = np.asarray(event, dtype=int)
    if times.ndim != 1 or events.shape != times.shape or not len(times):
        raise AnalysisInputError("Kaplan-Meier inputs must be non-empty one-dimensional arrays")
    if not np.isfinite(times).all() or (times < 0).any():
        raise AnalysisInputError("survival durations must be finite and non-negative")
    if set(np.unique(events)) - {0, 1}:
        raise AnalysisInputError("survival event indicator must contain only 0 and 1")
    try:
        from scipy.stats import norm
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("scipy", "survival confidence intervals") from exc

    z = float(norm.ppf(0.5 + confidence_level / 2))
    survival = 1.0
    greenwood = 0.0
    rows: list[dict[str, Any]] = [
        {
            "time": 0.0,
            "survival": 1.0,
            "lower": 1.0,
            "upper": 1.0,
            "at_risk": int(len(times)),
            "events": 0,
            "censored": 0,
        }
    ]
    for time in sorted(np.unique(times)):
        at_risk = int((times >= time).sum())
        at_time = times == time
        deaths = int((at_time & (events == 1)).sum())
        censored = int((at_time & (events == 0)).sum())
        if deaths:
            survival *= 1 - deaths / at_risk
            if at_risk > deaths:
                greenwood += deaths / (at_risk * (at_risk - deaths))
            else:
                greenwood = math.inf
        if survival <= 0:
            lower = upper = 0.0
        elif survival >= 1 or greenwood == 0:
            lower = upper = survival
        elif math.isfinite(greenwood):
            standard_error = math.sqrt(greenwood) / abs(math.log(survival))
            log_minus_log = math.log(-math.log(survival))
            lower = math.exp(-math.exp(log_minus_log + z * standard_error))
            upper = math.exp(-math.exp(log_minus_log - z * standard_error))
        else:
            lower = upper = 0.0
        rows.append(
            {
                "time": float(time),
                "survival": float(survival),
                "lower": float(lower),
                "upper": float(upper),
                "at_risk": at_risk,
                "events": deaths,
                "censored": censored,
            }
        )
    return rows


def restricted_mean_survival(curve: list[dict[str, Any]], horizon: float) -> float:
    """Integrate a right-continuous KM step curve through a fixed horizon."""

    if horizon <= 0:
        raise AnalysisInputError("RMST horizon must be positive")
    area = 0.0
    for index, row in enumerate(curve):
        start = float(row["time"])
        if start >= horizon:
            break
        stop = horizon
        if index + 1 < len(curve):
            stop = min(horizon, float(curve[index + 1]["time"]))
        if stop > start:
            area += (stop - start) * float(row["survival"])
    return area


def _median(curve: list[dict[str, Any]], column: str = "survival") -> float | None:
    return next((float(row["time"]) for row in curve if float(row[column]) <= 0.5), None)


def _logrank(duration: Any, event: Any, group: Any, exposed: Any) -> tuple[float, float]:
    import numpy as np

    times = np.asarray(duration, dtype=float)
    events = np.asarray(event, dtype=int)
    groups = np.asarray(group)
    observed_1 = 0.0
    expected_1 = 0.0
    variance = 0.0
    for time in sorted(np.unique(times[events == 1])):
        risk = times >= time
        n = int(risk.sum())
        n_1 = int((risk & (groups == exposed)).sum())
        deaths = (times == time) & (events == 1)
        d = int(deaths.sum())
        d_1 = int((deaths & (groups == exposed)).sum())
        if n <= 0:
            continue
        observed_1 += d_1
        expected_1 += d * n_1 / n
        if n > 1:
            variance += n_1 * (n - n_1) * d * (n - d) / (n * n * (n - 1))
    statistic = (observed_1 - expected_1) ** 2 / variance if variance > 0 else 0.0
    try:
        from scipy.stats import chi2
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("scipy", "log-rank test") from exc
    return float(statistic), float(chi2.sf(statistic, 1))


def _bootstrap_rmst_difference(
    first: Any,
    second: Any,
    *,
    horizon: float,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    import numpy as np

    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=float)
    first = first.reset_index(drop=True)
    second = second.reset_index(drop=True)
    for index in range(iterations):
        sample_a = first.iloc[rng.integers(0, len(first), len(first))]
        sample_b = second.iloc[rng.integers(0, len(second), len(second))]
        rmst_a = restricted_mean_survival(
            kaplan_meier(sample_a["_duration"], sample_a["_event"]), horizon
        )
        rmst_b = restricted_mean_survival(
            kaplan_meier(sample_b["_duration"], sample_b["_event"]), horizon
        )
        values[index] = rmst_a - rmst_b
    alpha = 1 - confidence_level
    lower, upper = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)


def _cox_result(
    frame: Any, params: SurvivalParams, exposed: Any, confidence_level: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        import pandas as pd
        from lifelines import CoxPHFitter
        from lifelines.statistics import proportional_hazard_test
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("lifelines", "penalized Cox regression") from exc

    working = (
        frame[
            [
                params.duration_column,
                params.event_column,
                params.group_column,
                *params.cox_covariates,
            ]
        ]
        .dropna()
        .copy()
    )
    treatment = (working.pop(params.group_column) == exposed).astype(int).rename("exposed")
    working = working.rename(
        columns={params.duration_column: "_duration", params.event_column: "_event"}
    )
    covariates = pd.get_dummies(working[params.cox_covariates], drop_first=True, dtype=float)
    model_frame = pd.concat([working[["_duration", "_event"]], treatment, covariates], axis=1)
    penalizer = (
        params.cox_penalizer
        if params.cox_penalizer is not None
        else (0.1 if len(model_frame) < 200 else 0.0)
    )
    model = CoxPHFitter(penalizer=penalizer)
    try:
        model.fit(model_frame, duration_col="_duration", event_col="_event", show_progress=False)
    except Exception as exc:
        raise AnalysisInputError(f"penalized Cox model failed: {exc}") from exc
    summary = model.summary
    if "exposed" not in summary.index:
        raise AnalysisInputError("Cox model did not estimate the exposure coefficient")
    row = summary.loc["exposed"]
    alpha = 1 - confidence_level
    lower_column = next(
        column for column in summary.columns if str(column).startswith("exp(coef) lower")
    )
    upper_column = next(
        column for column in summary.columns if str(column).startswith("exp(coef) upper")
    )
    tests = proportional_hazard_test(model, model_frame, time_transform="rank")
    ph_rows = [
        {
            "covariate": str(index),
            "test_statistic": float(tests.summary.loc[index, "test_statistic"]),
            "p_value": float(tests.summary.loc[index, "p"]),
        }
        for index in tests.summary.index
    ]
    return (
        {
            "estimand": "hazard_ratio",
            "estimate": float(row["exp(coef)"]),
            "lower": float(row[lower_column]),
            "upper": float(row[upper_column]),
            "p_value": float(row["p"]),
            "method": "cox_proportional_hazards",
            "penalizer": penalizer,
            "alpha": alpha,
        },
        ph_rows,
    )


class SurvivalAnalysis:
    kind = "survival"
    Params = SurvivalParams
    requires = ("duration", "event indicator", "pre-specified censor rule")
    rules = ("SURV-CENSOR", "SURV-EVENTS", "COMP-MIN-ARM")

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        import pandas as pd

        params = self.Params.model_validate(ctx.params)
        frame = load_frame(ctx, params)
        required = [
            params.duration_column,
            params.event_column,
            params.group_column,
            *params.cox_covariates,
        ]
        require_columns(frame, required)
        exposed, comparator = two_groups(
            frame[params.group_column], (params.exposed_value, params.comparator_value)
        )
        selected = frame.loc[frame[params.group_column].isin([exposed, comparator])].copy()
        duration: Any = pd.to_numeric(selected[params.duration_column], errors="coerce")
        event: Any = pd.to_numeric(selected[params.event_column], errors="coerce")
        invalid = selected[params.duration_column].notna() & duration.isna()
        invalid |= selected[params.event_column].notna() & event.isna()
        if invalid.any():
            raise AnalysisInputError("duration and event columns must be numeric")
        complete = selected.assign(_duration=duration, _event=event).dropna(
            subset=["_duration", "_event"]
        )
        if (complete["_duration"] < 0).any():
            raise AnalysisInputError("survival durations must be non-negative")
        if not set(complete["_event"].unique()).issubset({0, 1}):
            raise AnalysisInputError("event indicator must contain only 0 and 1")
        complete["_event"] = complete["_event"].astype(int)
        first = complete.loc[complete[params.group_column] == exposed]
        second = complete.loc[complete[params.group_column] == comparator]
        if first.empty or second.empty:
            raise AnalysisInputError("both survival groups need complete observations")

        curves: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        curve_by_group: dict[Any, list[dict[str, Any]]] = {}
        for value, subset in ((exposed, first), (comparator, second)):
            curve = kaplan_meier(
                subset["_duration"], subset["_event"], confidence_level=params.confidence_level
            )
            curve_by_group[value] = curve
            curves.extend({"group": str(value), **row} for row in curve)
            summaries.append(
                {
                    "group": str(value),
                    "n": int(len(subset)),
                    "events": int(subset["_event"].sum()),
                    "censored": int((1 - subset["_event"]).sum()),
                    "median": _median(curve),
                    "median_lower": _median(curve, "lower"),
                    "median_upper": _median(curve, "upper"),
                }
            )

        observed_maximum = min(float(first["_duration"].max()), float(second["_duration"].max()))
        horizon = params.rmst_horizon if params.rmst_horizon is not None else observed_maximum
        if horizon > observed_maximum:
            raise AnalysisInputError(
                f"RMST horizon {horizon:g} exceeds follow-up available in both groups "
                f"({observed_maximum:g})"
            )
        rmst_first = restricted_mean_survival(curve_by_group[exposed], horizon)
        rmst_second = restricted_mean_survival(curve_by_group[comparator], horizon)
        rmst_lower, rmst_upper = _bootstrap_rmst_difference(
            first,
            second,
            horizon=horizon,
            iterations=params.bootstrap_iterations,
            confidence_level=params.confidence_level,
            seed=ctx.seed,
        )
        logrank_statistic, logrank_p = _logrank(
            complete["_duration"], complete["_event"], complete[params.group_column], exposed
        )
        estimates = [
            {
                "estimand": "rmst_difference",
                "estimate": rmst_first - rmst_second,
                "lower": rmst_lower,
                "upper": rmst_upper,
                "p_value": None,
                "method": "kaplan_meier_area_stratified_bootstrap_ci",
                "horizon": horizon,
                "exposed": str(exposed),
                "comparator": str(comparator),
            },
            {
                "estimand": "survival_curve_difference",
                "estimate": logrank_statistic,
                "lower": None,
                "upper": None,
                "p_value": logrank_p,
                "method": "logrank_chi_square",
                "horizon": None,
                "exposed": str(exposed),
                "comparator": str(comparator),
            },
        ]
        ph_rows: list[dict[str, Any]] = []
        if params.cox:
            cox, ph_rows = _cox_result(complete, params, exposed, params.confidence_level)
            estimates.append({**cox, "exposed": str(exposed), "comparator": str(comparator)})

        events_first = int(first["_event"].sum())
        events_second = int(second["_event"].sum())
        log_mde = survival_log_hazard_mde(events_first, events_second)
        if math.isfinite(log_mde):
            power_note = (
                "Design-stage precision: given the observed event counts, an 80%-power, two-sided "
                "alpha=0.05 design detects hazard ratios outside approximately "
                f"{math.exp(-log_mde):.3f}–"
                f"{math.exp(log_mde):.3f}. This is a minimum detectable effect, not post-hoc power."
            )
        else:
            power_note = (
                "A survival MDE cannot be estimated because one arm has no observed events."
            )

        return AnalysisResult(
            analysis_id=ctx.analysis_id,
            kind=self.kind,
            n={
                "total": int(len(complete)),
                "per_group": {str(exposed): int(len(first)), str(comparator): int(len(second))},
                "events_per_group": {str(exposed): events_first, str(comparator): events_second},
                "excluded": int(len(frame) - len(complete)),
            },
            estimates=estimates,
            provenance=dict(ctx.provenance) or {"available": False},
            assumptions_checked=[
                {"name": "censor_rule_pre_specified", "passed": True, "rule": params.censor_rule},
                {
                    "name": "noninformative_censoring",
                    "passed": None,
                    "note": "scientific assumption; not empirically testable",
                },
                {
                    "name": "proportional_hazards",
                    "passed": None
                    if not params.cox
                    else all(row["p_value"] >= 0.05 for row in ph_rows),
                    "tests": ph_rows,
                },
                {"name": "rmst_horizon_in_support", "passed": True, "horizon": horizon},
            ],
            guardrail_outcomes=[dict(item) for item in ctx.guardrail_outcomes],
            power_note=power_note,
            seed=ctx.seed,
            package_versions=package_versions(("scipy", "lifelines") if params.cox else ("scipy",)),
            tables={
                "data": curves,
                "summary": summaries,
                "estimates": estimates,
                **({"schoenfeld": ph_rows} if ph_rows else {}),
            },
            metadata={"censor_rule": params.censor_rule, "rmst_horizon": horizon},
            output_dir=ctx.output_dir,
        )


ANALYSIS = SurvivalAnalysis()
