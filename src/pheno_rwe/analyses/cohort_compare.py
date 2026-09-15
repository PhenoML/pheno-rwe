"""Two-cohort binary and continuous comparisons with multiplicity control."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pheno_rwe.analyses._firth import fit_firth_logistic
from pheno_rwe.analyses.base import (
    AnalysisContext,
    AnalysisInputError,
    AnalysisResult,
    OptionalDependencyError,
    package_versions,
)
from pheno_rwe.analyses.data import DataSourceParams, load_frame, require_columns, two_groups
from pheno_rwe.analyses.power import comparative_power_note
from pheno_rwe.analyses.stats import benjamini_hochberg, newcombe_difference_interval


class OutcomeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    column: str
    name: str | None = None
    outcome_type: Literal["auto", "binary", "continuous"] = Field(default="auto", alias="type")
    event_value: Any = 1


class CovariateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    rationale: str = "pre-specified adjustment covariate"


class CohortCompareParams(DataSourceParams):
    group_column: str = "group"
    exposed_value: Any | None = None
    comparator_value: Any | None = None
    outcomes: list[str | OutcomeSpec] = Field(min_length=1)
    binary_test: Literal["fisher", "boschloo"] = "fisher"
    alternative: Literal["two-sided", "less", "greater"] = "two-sided"
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    permutations: int = Field(default=10_000, ge=99, le=1_000_000)
    bootstrap_iterations: int = Field(default=2_000, ge=100, le=100_000)
    multiplicity: Literal["auto_bh", "bh", "none"] = "auto_bh"
    adjusted_estimator: Literal["firth_logistic"] = "firth_logistic"
    adjustment_set: list[CovariateSpec] = Field(default_factory=list)
    adjustment_covariates: list[str] = Field(default_factory=list)
    covariates: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def group_values_are_paired(self) -> CohortCompareParams:
        if (self.exposed_value is None) != (self.comparator_value is None):
            raise ValueError("exposed_value and comparator_value must be supplied together")
        return self

    def normalized_outcomes(self) -> list[OutcomeSpec]:
        return [
            OutcomeSpec(column=value) if isinstance(value, str) else value
            for value in self.outcomes
        ]

    def adjusted_columns(self) -> list[str]:
        return list(
            dict.fromkeys(
                [item.column for item in self.adjustment_set]
                + self.adjustment_covariates
                + self.covariates
            )
        )


def _scipy_stats() -> Any:
    try:
        from scipy import stats
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("scipy", "cohort_compare") from exc
    return stats


def _conditional_odds_ratio(table: Any, confidence_level: float) -> tuple[float, float, float]:
    stats = _scipy_stats()
    try:
        result = stats.contingency.odds_ratio(table, kind="conditional")
        interval = result.confidence_interval(confidence_level=confidence_level)
        return float(result.statistic), float(interval.low), float(interval.high)
    except (AttributeError, ValueError):
        # Stable continuity-corrected fallback for old SciPy builds.
        import numpy as np

        cells = np.asarray(table, dtype=float)
        if (cells == 0).any():
            cells += 0.5
        estimate = float(cells[0, 0] * cells[1, 1] / (cells[0, 1] * cells[1, 0]))
        standard_error = math.sqrt(float((1 / cells).sum()))
        z = stats.norm.ppf(0.5 + confidence_level / 2)
        return (
            estimate,
            math.exp(math.log(estimate) - z * standard_error),
            math.exp(math.log(estimate) + z * standard_error),
        )


def _permutation_p_value(a: Any, b: Any, iterations: int, seed: int) -> float:
    import numpy as np

    first = np.asarray(a, dtype=float)
    second = np.asarray(b, dtype=float)
    observed = abs(float(first.mean() - second.mean()))
    combined = np.concatenate([first, second]).copy()
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(iterations):
        permuted = rng.permutation(combined)
        statistic = abs(float(permuted[: len(first)].mean() - permuted[len(first) :].mean()))
        extreme += statistic >= observed - 1e-15
    return (extreme + 1) / (iterations + 1)


def _hodges_lehmann(
    a: Any, b: Any, iterations: int, seed: int, confidence_level: float
) -> tuple[float, float, float]:
    import numpy as np

    first = np.asarray(a, dtype=float)
    second = np.asarray(b, dtype=float)
    estimate = float(np.median(first[:, None] - second[None, :]))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled_a = rng.choice(first, size=len(first), replace=True)
        sampled_b = rng.choice(second, size=len(second), replace=True)
        bootstrap[index] = np.median(sampled_a[:, None] - sampled_b[None, :])
    alpha = 1 - confidence_level
    lower, upper = np.quantile(bootstrap, [alpha / 2, 1 - alpha / 2])
    return estimate, float(lower), float(upper)


def _firth_estimate(
    frame: Any,
    *,
    outcome: str,
    event_value: Any,
    group: str,
    exposed: Any,
    covariates: list[str],
    confidence_level: float,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    columns = [outcome, group, *covariates]
    complete = frame[columns].dropna().copy()
    if complete.empty:
        raise AnalysisInputError("no complete rows remain for adjusted Firth regression")
    y = (complete[outcome] == event_value).astype(int).to_numpy()
    treatment = (complete[group] == exposed).astype(int).rename("exposed")
    covariate_frame = pd.get_dummies(complete[covariates], drop_first=True, dtype=float)
    constant = [
        column for column in covariate_frame if covariate_frame[column].nunique(dropna=False) <= 1
    ]
    covariate_frame = covariate_frame.drop(columns=constant)
    if covariate_frame.empty:
        raise AnalysisInputError("adjustment set has no varying covariates")
    design_frame = pd.concat([treatment, covariate_frame], axis=1)
    design = np.column_stack([np.ones(len(design_frame)), design_frame.to_numpy(dtype=float)])
    fit = fit_firth_logistic(design, y)
    coefficient = float(fit.coefficients[1])
    standard_error = float(fit.standard_errors[1])
    z = _scipy_stats().norm.ppf(0.5 + confidence_level / 2)
    return {
        "estimand": "adjusted_odds_ratio",
        "estimate": math.exp(coefficient),
        "lower": math.exp(coefficient - z * standard_error),
        "upper": math.exp(coefficient + z * standard_error),
        "p_value": float(2 * _scipy_stats().norm.sf(abs(coefficient / standard_error)))
        if standard_error
        else None,
        "method": "firth_penalized_logistic_wald_ci",
        "n": int(len(complete)),
        "converged": fit.converged,
        "iterations": fit.iterations,
    }


class CohortCompareAnalysis:
    kind = "cohort_compare"
    Params = CohortCompareParams
    requires = ("two patient-level cohorts", "one or more outcomes")
    rules = ("GLOBAL-MIN-N", "COMP-MIN-ARM", "EPV-LOGISTIC", "MULTIPLICITY")

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        params = self.Params.model_validate(ctx.params)
        if len(params.outcomes) > 1 and params.multiplicity == "none":
            raise AnalysisInputError(
                "multiple outcomes require Benjamini-Hochberg multiplicity control"
            )
        frame = load_frame(ctx, params)
        outcomes = params.normalized_outcomes()
        covariates = params.adjusted_columns()
        require_columns(
            frame, [params.group_column, *[item.column for item in outcomes], *covariates]
        )
        exposed, comparator = two_groups(
            frame[params.group_column], (params.exposed_value, params.comparator_value)
        )
        selected = frame.loc[frame[params.group_column].isin([exposed, comparator])].copy()
        n_exposed = int((selected[params.group_column] == exposed).sum())
        n_comparator = int((selected[params.group_column] == comparator).sum())
        if not n_exposed or not n_comparator:
            raise AnalysisInputError("both comparison groups must contain observations")

        estimates: list[dict[str, Any]] = []
        primary_p_values: list[float | None] = []
        primary_indexes: list[list[int]] = []
        observed_types: list[str] = []
        alpha = 1 - params.confidence_level

        for outcome_index, outcome in enumerate(outcomes):
            outcome_name = outcome.name or outcome.column
            working = selected[[params.group_column, outcome.column, *covariates]].dropna(
                subset=[outcome.column]
            )
            values = working[outcome.column].dropna()
            observed = set(values.unique().tolist())
            inferred_binary = observed.issubset({0, 1, outcome.event_value}) and len(observed) <= 2
            outcome_type = outcome.outcome_type
            if outcome_type == "auto":
                outcome_type = "binary" if inferred_binary else "continuous"
            observed_types.append(outcome_type)
            start_index = len(estimates)

            if outcome_type == "binary":
                a_frame = working.loc[working[params.group_column] == exposed, outcome.column]
                b_frame = working.loc[working[params.group_column] == comparator, outcome.column]
                if a_frame.empty or b_frame.empty:
                    raise AnalysisInputError(
                        f"outcome '{outcome_name}' has no observations in one arm"
                    )
                a_events = int((a_frame == outcome.event_value).sum())
                b_events = int((b_frame == outcome.event_value).sum())
                table = [[a_events, len(a_frame) - a_events], [b_events, len(b_frame) - b_events]]
                stats = _scipy_stats()
                if params.binary_test == "boschloo":
                    if not hasattr(stats, "boschloo_exact"):
                        raise OptionalDependencyError("scipy>=1.11", "Boschloo's exact test")
                    p_value = float(
                        stats.boschloo_exact(table, alternative=params.alternative).pvalue
                    )
                else:
                    p_value = float(
                        stats.fisher_exact(table, alternative=params.alternative).pvalue
                    )
                odds_ratio, or_lower, or_upper = _conditional_odds_ratio(
                    table, params.confidence_level
                )
                risk_a = a_events / len(a_frame)
                risk_b = b_events / len(b_frame)
                rd_lower, rd_upper = newcombe_difference_interval(
                    a_events, len(a_frame), b_events, len(b_frame), alpha
                )
                common = {
                    "outcome": outcome_name,
                    "outcome_type": "binary",
                    "exposed": str(exposed),
                    "comparator": str(comparator),
                    "n_exposed": int(len(a_frame)),
                    "n_comparator": int(len(b_frame)),
                    "events_exposed": a_events,
                    "events_comparator": b_events,
                }
                estimates.extend(
                    [
                        {
                            **common,
                            "estimand": "odds_ratio",
                            "estimate": odds_ratio,
                            "lower": or_lower,
                            "upper": or_upper,
                            "p_value": p_value,
                            "method": f"{params.binary_test}_exact_conditional_mle",
                        },
                        {
                            **common,
                            "estimand": "risk_difference",
                            "estimate": risk_a - risk_b,
                            "lower": rd_lower,
                            "upper": rd_upper,
                            "p_value": p_value,
                            "method": "newcombe_score_ci",
                        },
                    ]
                )
                if covariates:
                    estimates.append(
                        {
                            **common,
                            **_firth_estimate(
                                working,
                                outcome=outcome.column,
                                event_value=outcome.event_value,
                                group=params.group_column,
                                exposed=exposed,
                                covariates=covariates,
                                confidence_level=params.confidence_level,
                            ),
                            "adjustment_set": covariates,
                        }
                    )
                primary_p_values.append(p_value)
            else:
                import pandas as pd

                converted: Any = pd.to_numeric(working[outcome.column], errors="coerce")
                if (working[outcome.column].notna() & converted.isna()).any():
                    raise AnalysisInputError(
                        f"continuous outcome '{outcome_name}' contains non-numeric values"
                    )
                working = working.assign(_outcome=converted).dropna(subset=["_outcome"])
                first = working.loc[working[params.group_column] == exposed, "_outcome"].to_numpy(
                    dtype=float
                )
                second = working.loc[
                    working[params.group_column] == comparator, "_outcome"
                ].to_numpy(dtype=float)
                if not len(first) or not len(second):
                    raise AnalysisInputError(
                        f"outcome '{outcome_name}' has no observations in one arm"
                    )
                stats = _scipy_stats()
                mann_whitney = stats.mannwhitneyu(first, second, alternative=params.alternative)
                permutation_p = _permutation_p_value(
                    first, second, params.permutations, ctx.seed + outcome_index
                )
                estimate, lower, upper = _hodges_lehmann(
                    first,
                    second,
                    params.bootstrap_iterations,
                    ctx.seed + 100_003 + outcome_index,
                    params.confidence_level,
                )
                estimates.append(
                    {
                        "outcome": outcome_name,
                        "outcome_type": "continuous",
                        "estimand": "hodges_lehmann_shift",
                        "estimate": estimate,
                        "lower": lower,
                        "upper": upper,
                        "p_value": float(mann_whitney.pvalue),
                        "permutation_p_value": permutation_p,
                        "method": "mann_whitney_hodges_lehmann_bootstrap_ci",
                        "exposed": str(exposed),
                        "comparator": str(comparator),
                        "n_exposed": int(len(first)),
                        "n_comparator": int(len(second)),
                    }
                )
                primary_p_values.append(float(mann_whitney.pvalue))

            primary_indexes.append(list(range(start_index, len(estimates))))

        q_values = benjamini_hochberg(primary_p_values) if len(outcomes) > 1 else primary_p_values
        for indexes, q_value in zip(primary_indexes, q_values, strict=True):
            for index in indexes:
                estimates[index]["q_value"] = q_value

        note_type = (
            "continuous" if observed_types and set(observed_types) == {"continuous"} else "binary"
        )
        return AnalysisResult(
            analysis_id=ctx.analysis_id,
            kind=self.kind,
            n={
                "total": int(len(selected)),
                "per_group": {str(exposed): n_exposed, str(comparator): n_comparator},
                "excluded": int(len(frame) - len(selected)),
            },
            estimates=estimates,
            provenance=dict(ctx.provenance) or {"available": False},
            assumptions_checked=[
                {"name": "exactly_two_groups", "passed": True},
                {
                    "name": "multiplicity",
                    "passed": True,
                    "method": "benjamini_hochberg" if len(outcomes) > 1 else "not_needed",
                    "requested": params.multiplicity,
                },
                {
                    "name": "adjusted_estimator",
                    "passed": True,
                    "method": "firth" if covariates else "unadjusted",
                },
            ],
            guardrail_outcomes=[dict(item) for item in ctx.guardrail_outcomes],
            power_note=comparative_power_note(n_exposed, n_comparator, note_type),
            seed=ctx.seed,
            package_versions=package_versions(("scipy",)),
            tables={"data": estimates},
            metadata={
                "exposed": str(exposed),
                "comparator": str(comparator),
                "adjustment_set": [item.model_dump() for item in params.adjustment_set],
                "adjusted_estimator": params.adjusted_estimator if covariates else None,
            },
            output_dir=ctx.output_dir,
        )


ANALYSIS = CohortCompareAnalysis()
