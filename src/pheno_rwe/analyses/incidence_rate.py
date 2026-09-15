"""Incidence rates with exact Poisson intervals and rate-ratio contrasts."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import Field, model_validator

from pheno_rwe.analyses.base import (
    AnalysisContext,
    AnalysisInputError,
    AnalysisResult,
    OptionalDependencyError,
    package_versions,
)
from pheno_rwe.analyses.data import DataSourceParams, load_frame, require_columns, two_groups


class IncidenceRateParams(DataSourceParams):
    group_column: str = "group"
    event_count_column: str = "events"
    person_time_column: str | None = "person_time"
    observation_start_column: str | None = None
    observation_end_column: str | None = None
    exposed_value: Any | None = None
    comparator_value: Any | None = None
    rate_scale: float = Field(default=1_000.0, gt=0)
    person_time_unit: Literal["days", "person_years"] = "person_years"
    confidence_level: float = Field(default=0.95, gt=0, lt=1)

    @model_validator(mode="after")
    def person_time_source(self) -> IncidenceRateParams:
        dates = self.observation_start_column is not None or self.observation_end_column is not None
        if dates and not (self.observation_start_column and self.observation_end_column):
            raise ValueError("both observation start and end columns are required")
        if self.person_time_column is None and not dates:
            raise ValueError("person_time_column or observation start/end columns are required")
        if (self.exposed_value is None) != (self.comparator_value is None):
            raise ValueError("exposed_value and comparator_value must be supplied together")
        return self


def exact_poisson_interval(
    events: int, person_time: float, confidence_level: float = 0.95
) -> tuple[float, float]:
    if events < 0 or person_time <= 0:
        raise AnalysisInputError("events must be non-negative and person-time positive")
    try:
        from scipy.stats import chi2
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("scipy", "exact Poisson confidence intervals") from exc
    alpha = 1 - confidence_level
    lower = 0.0 if events == 0 else 0.5 * float(chi2.ppf(alpha / 2, 2 * events)) / person_time
    upper = 0.5 * float(chi2.ppf(1 - alpha / 2, 2 * (events + 1))) / person_time
    return lower, upper


def _exact_rate_ratio_interval(
    events_a: int,
    time_a: float,
    events_b: int,
    time_b: float,
    confidence_level: float,
) -> tuple[float, float, float, float]:
    try:
        from scipy.stats import beta, binomtest
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("scipy", "exact incidence rate ratio") from exc
    total = events_a + events_b
    if total == 0:
        return math.nan, 0.0, math.inf, 1.0
    estimate = (events_a / time_a) / (events_b / time_b) if events_b else math.inf
    alpha = 1 - confidence_level
    p_lower = 0.0 if events_a == 0 else float(beta.ppf(alpha / 2, events_a, events_b + 1))
    p_upper = 1.0 if events_b == 0 else float(beta.ppf(1 - alpha / 2, events_a + 1, events_b))

    def transform(probability: float) -> float:
        if probability <= 0:
            return 0.0
        if probability >= 1:
            return math.inf
        return probability / (1 - probability) * time_b / time_a

    null_probability = time_a / (time_a + time_b)
    p_value = float(binomtest(events_a, total, null_probability).pvalue)
    return estimate, transform(p_lower), transform(p_upper), p_value


class IncidenceRateAnalysis:
    kind = "incidence_rate"
    Params = IncidenceRateParams
    requires = ("event counts", "person-time")
    rules = ("GLOBAL-MIN-N", "COMP-MIN-ARM")

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        import pandas as pd

        params = self.Params.model_validate(ctx.params)
        frame = load_frame(ctx, params)
        required = [params.group_column, params.event_count_column]
        if params.person_time_column:
            required.append(params.person_time_column)
        else:
            assert params.observation_start_column is not None
            assert params.observation_end_column is not None
            required.extend([params.observation_start_column, params.observation_end_column])
        require_columns(frame, required)
        exposed, comparator = two_groups(
            frame[params.group_column], (params.exposed_value, params.comparator_value)
        )
        selected = frame.loc[frame[params.group_column].isin([exposed, comparator])].copy()
        events: Any = pd.to_numeric(selected[params.event_count_column], errors="coerce")
        if (
            selected[params.event_count_column].notna() & events.isna()
        ).any() or events.isna().any():
            raise AnalysisInputError("event counts must be complete numeric values")
        if (events < 0).any() or ((events % 1) != 0).any():
            raise AnalysisInputError("event counts must be non-negative integers")
        selected["_events"] = events.astype(int)
        if params.person_time_column:
            person_time: Any = pd.to_numeric(selected[params.person_time_column], errors="coerce")
        else:
            assert params.observation_start_column is not None
            assert params.observation_end_column is not None
            starts: Any = pd.to_datetime(selected[params.observation_start_column], errors="coerce")
            ends: Any = pd.to_datetime(selected[params.observation_end_column], errors="coerce")
            person_time = (ends - starts).dt.total_seconds() / (86_400 * 365.25)
        if person_time.isna().any() or (person_time <= 0).any():
            raise AnalysisInputError("person-time must be complete and strictly positive")
        selected["_person_time"] = person_time.astype(float)

        rates: list[dict[str, Any]] = []
        totals: dict[Any, tuple[int, float, int]] = {}
        for value in (exposed, comparator):
            group = selected.loc[selected[params.group_column] == value]
            count = int(group["_events"].sum())
            time = float(group["_person_time"].sum())
            lower, upper = exact_poisson_interval(count, time, params.confidence_level)
            totals[value] = count, time, int(len(group))
            rates.append(
                {
                    "group": str(value),
                    "n": int(len(group)),
                    "events": count,
                    "person_time": time,
                    "person_time_unit": params.person_time_unit,
                    "rate": count / time * params.rate_scale,
                    "lower": lower * params.rate_scale,
                    "upper": upper * params.rate_scale,
                    "rate_scale": params.rate_scale,
                    "method": "exact_poisson_garwood",
                }
            )
        events_a, time_a, n_a = totals[exposed]
        events_b, time_b, n_b = totals[comparator]
        estimate, lower, upper, p_value = _exact_rate_ratio_interval(
            events_a, time_a, events_b, time_b, params.confidence_level
        )
        contrast = {
            "estimand": "incidence_rate_ratio",
            "estimate": estimate,
            "lower": lower,
            "upper": upper,
            "p_value": p_value,
            "method": "conditional_exact_binomial",
            "exposed": str(exposed),
            "comparator": str(comparator),
        }
        if events_a and events_b:
            z = 1.959963984540054 + 0.8416212335729143
            log_mde = z * math.sqrt(1 / events_a + 1 / events_b)
            power_note = (
                "Design-stage precision: given observed person-time and event counts, "
                "an 80%-power, two-sided alpha=0.05 design detects rate ratios outside "
                f"approximately {math.exp(-log_mde):.3f}–"
                f"{math.exp(log_mde):.3f}. This is a minimum detectable effect, not post-hoc power."
            )
        else:
            power_note = "A rate-ratio MDE cannot be estimated because one arm has no events."
        return AnalysisResult(
            analysis_id=ctx.analysis_id,
            kind=self.kind,
            n={
                "total": int(len(selected)),
                "per_group": {str(exposed): n_a, str(comparator): n_b},
                "excluded": int(len(frame) - len(selected)),
            },
            estimates=[contrast],
            provenance=dict(ctx.provenance) or {"available": False},
            assumptions_checked=[
                {"name": "positive_person_time", "passed": True},
                {"name": "poisson_event_process", "passed": None, "note": "scientific assumption"},
            ],
            guardrail_outcomes=[dict(item) for item in ctx.guardrail_outcomes],
            power_note=power_note,
            seed=ctx.seed,
            package_versions=package_versions(("scipy",)),
            tables={"data": rates, "estimates": [contrast]},
            metadata={
                "rate_scale": params.rate_scale,
                "person_time_unit": params.person_time_unit,
            },
            output_dir=ctx.output_dir,
        )


ANALYSIS = IncidenceRateAnalysis()
