"""Pre-declared propensity-score adjusted association estimators."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pheno_rwe.analyses.base import (
    AnalysisContext,
    AnalysisInputError,
    AnalysisResult,
    OptionalDependencyError,
    package_versions,
)
from pheno_rwe.analyses.data import DataSourceParams, load_frame, require_columns, two_groups
from pheno_rwe.analyses.power import comparative_power_note
from pheno_rwe.analyses.stats import e_value, weighted_smd


class AdjustmentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    column: str = Field(validation_alias="name")
    rationale: str
    code_set: str | None = None

    @model_validator(mode="after")
    def rationale_required(self) -> AdjustmentSpec:
        if not self.rationale.strip():
            raise ValueError("every adjustment covariate requires a non-empty rationale")
        return self


class CausalEffectParams(DataSourceParams):
    treatment_column: str = "group"
    outcome_column: str | None = None
    outcome: str | None = None
    exposed_value: Any | None = None
    comparator_value: Any | None = None
    exposed_cohort: str | None = None
    comparator_cohort: str | None = None
    event_value: Any = 1
    outcome_type: Literal["auto", "binary", "continuous"] = "auto"
    adjustment_set: list[AdjustmentSpec] = Field(min_length=1)
    method: Literal["iptw", "ps_matching"] = "iptw"
    estimand: Literal["ate", "att"] = "ate"
    weight_trim_quantiles: tuple[float, float] = (0.01, 0.99)
    matching_caliper: float | None = Field(default=None, gt=0)
    bootstrap_iterations: int = Field(default=500, ge=100, le=100_000)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)

    @model_validator(mode="after")
    def normalize_plan_fields(self) -> CausalEffectParams:
        if self.outcome_column is None:
            self.outcome_column = self.outcome
        if not self.outcome_column:
            raise ValueError("outcome_column or outcome is required")
        if self.exposed_value is None and self.exposed_cohort is not None:
            self.exposed_value = self.exposed_cohort
        if self.comparator_value is None and self.comparator_cohort is not None:
            self.comparator_value = self.comparator_cohort
        if (self.exposed_value is None) != (self.comparator_value is None):
            raise ValueError("exposed and comparator values must be supplied together")
        lower, upper = self.weight_trim_quantiles
        if not 0 <= lower < upper <= 1:
            raise ValueError("weight_trim_quantiles must satisfy 0 <= lower < upper <= 1")
        if self.method == "ps_matching" and self.matching_caliper is None:
            self.matching_caliper = 0.2
        return self


def _sklearn() -> tuple[Any, Any]:
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("scikit-learn", "propensity score estimation") from exc
    return np, LogisticRegression


def _design(frame: Any, covariates: list[str]) -> tuple[Any, list[str]]:
    import pandas as pd

    pieces: list[Any] = []
    for column in covariates:
        # pandas-stubs intentionally models ``DataFrame.__getitem__`` as a
        # Series/DataFrame/scalar union.  This branch is column-oriented by
        # construction, so keep that dynamic boundary explicit and local.
        series: Any = frame[column]
        numeric: Any = pd.to_numeric(series, errors="coerce")
        nonnumeric_observed = series.notna() & numeric.isna()
        if not nonnumeric_observed.any():
            if numeric.isna().any():
                pieces.append(numeric.isna().astype(float).rename(f"{column}__missing"))
            median = float(numeric.median()) if numeric.notna().any() else 0.0
            pieces.append(numeric.fillna(median).astype(float).rename(column))
        else:
            encoded = pd.get_dummies(
                series.fillna("Missing").astype(str), prefix=column, drop_first=True, dtype=float
            )
            if encoded.empty:
                encoded[f"{column}__constant"] = 0.0
            pieces.append(encoded)
    design = pd.concat(pieces, axis=1)
    constant = [column for column in design if design[column].nunique(dropna=False) <= 1]
    design = design.drop(columns=constant)
    if design.empty:
        raise AnalysisInputError("adjustment set has no varying covariates")
    return design.to_numpy(dtype=float), [str(column) for column in design.columns]


def _propensity(design: Any, treatment: Any, seed: int) -> Any:
    np, logistic_regression = _sklearn()
    model = logistic_regression(
        C=1.0,
        solver="lbfgs",
        max_iter=2_000,
        random_state=seed,
    )
    model.fit(design, treatment)
    return np.clip(model.predict_proba(design)[:, 1], 1e-6, 1 - 1e-6)


def _iptw(treatment: Any, propensity: Any, estimand: str, trim: tuple[float, float]) -> Any:
    np, _ = _sklearn()
    treatment = np.asarray(treatment, dtype=int)
    probability = float(treatment.mean())
    if estimand == "att":
        weights = np.where(
            treatment == 1, 1.0, probability / (1 - probability) * propensity / (1 - propensity)
        )
    else:
        weights = np.where(
            treatment == 1, probability / propensity, (1 - probability) / (1 - propensity)
        )
    lower, upper = np.quantile(weights, trim)
    return np.clip(weights, lower, upper)


def _match(treatment: Any, propensity: Any, caliper_scale: float) -> Any:
    np, _ = _sklearn()
    treatment = np.asarray(treatment, dtype=int)
    propensity = np.asarray(propensity, dtype=float)
    logits = np.log(propensity / (1 - propensity))
    caliper = caliper_scale * float(np.std(logits, ddof=1))
    exposed_indexes = sorted(
        np.flatnonzero(treatment == 1), key=lambda index: (propensity[index], int(index))
    )
    available = set(int(index) for index in np.flatnonzero(treatment == 0))
    chosen: list[int] = []
    for exposed in exposed_indexes:
        if not available:
            break
        comparator = min(available, key=lambda index: (abs(logits[exposed] - logits[index]), index))
        if abs(logits[exposed] - logits[comparator]) <= caliper:
            chosen.extend([int(exposed), comparator])
            available.remove(comparator)
    if not chosen:
        raise AnalysisInputError("propensity-score caliper produced no matched pairs")
    return np.asarray(sorted(chosen), dtype=int)


def _effect(outcome: Any, treatment: Any, weights: Any, binary: bool) -> dict[str, float]:
    np, _ = _sklearn()
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treatment, dtype=int)
    w = np.asarray(weights, dtype=float)
    mean_1 = float(np.average(y[t == 1], weights=w[t == 1]))
    mean_0 = float(np.average(y[t == 0], weights=w[t == 0]))
    result = {"mean_exposed": mean_1, "mean_comparator": mean_0, "difference": mean_1 - mean_0}
    if binary:
        result["risk_ratio"] = mean_1 / mean_0 if mean_0 > 0 else math.inf
    return result


def _fit_effect(
    frame: Any, params: CausalEffectParams, exposed: Any, seed: int
) -> tuple[dict[str, float], Any, Any, list[str]]:
    covariates = [item.column for item in params.adjustment_set]
    design, design_columns = _design(frame, covariates)
    treatment = (frame[params.treatment_column] == exposed).astype(int).to_numpy()
    propensity = _propensity(design, treatment, seed)
    outcome = frame[params.outcome_column].to_numpy(dtype=float)
    if params.method == "ps_matching":
        selected = _match(treatment, propensity, params.matching_caliper or 0.2)
        selected_treatment = treatment[selected]
        selected_outcome = outcome[selected]
        weights = _sklearn()[0].ones(len(selected), dtype=float)
        effect = _effect(
            selected_outcome, selected_treatment, weights, params.outcome_type == "binary"
        )
        full_weights = _sklearn()[0].zeros(len(frame), dtype=float)
        full_weights[selected] = 1.0
        return effect, propensity, full_weights, design_columns
    weights = _iptw(treatment, propensity, params.estimand, params.weight_trim_quantiles)
    return (
        _effect(outcome, treatment, weights, params.outcome_type == "binary"),
        propensity,
        weights,
        design_columns,
    )


class CausalEffectAnalysis:
    kind = "causal_effect"
    Params = CausalEffectParams
    requires = ("pre-declared adjustment set with rationales", "positivity")
    rules = ("CAUSAL-ADJUSTMENT", "CAUSAL-MIN", "UNCHECKED-FRACTION")

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        import numpy as np
        import pandas as pd

        params = self.Params.model_validate(ctx.params)
        frame = load_frame(ctx, params)
        outcome_column = params.outcome_column
        assert outcome_column is not None  # guaranteed by normalize_plan_fields
        covariates = [item.column for item in params.adjustment_set]
        require_columns(frame, [params.treatment_column, outcome_column, *covariates])
        exposed, comparator = two_groups(
            frame[params.treatment_column], (params.exposed_value, params.comparator_value)
        )
        selected = frame.loc[frame[params.treatment_column].isin([exposed, comparator])].copy()
        outcome = pd.to_numeric(selected[outcome_column], errors="coerce")
        selected = selected.assign(_outcome=outcome).dropna(
            subset=["_outcome", params.treatment_column]
        )
        selected[outcome_column] = selected["_outcome"].astype(float)
        observed = set(selected[outcome_column].unique().tolist())
        inferred_binary = observed.issubset({0.0, 1.0}) and len(observed) <= 2
        if params.outcome_type == "auto":
            params.outcome_type = "binary" if inferred_binary else "continuous"
        if params.outcome_type == "binary":
            selected[outcome_column] = (selected[outcome_column] == params.event_value).astype(
                float
            )
        treatment = (selected[params.treatment_column] == exposed).astype(int).to_numpy()
        if not ((treatment == 0).any() and (treatment == 1).any()):
            raise AnalysisInputError(
                "causal_effect requires both exposure groups after outcome exclusions"
            )

        effect, propensity, weights, design_columns = _fit_effect(
            selected, params, exposed, ctx.seed
        )
        design, _ = _design(selected, covariates)
        balance_rows: list[dict[str, Any]] = []
        for index, column in enumerate(design_columns):
            balance_rows.append(
                {
                    "covariate": column,
                    "smd_before": weighted_smd(design[:, index], treatment),
                    "smd_after": weighted_smd(design[:, index], treatment, weights),
                }
            )

        rng = np.random.default_rng(ctx.seed)
        bootstrap: dict[str, list[float]] = {
            key: [] for key in effect if key in {"difference", "risk_ratio"}
        }
        failures = 0
        for iteration in range(params.bootstrap_iterations):
            indexes = rng.integers(0, len(selected), len(selected))
            sample = selected.iloc[indexes].reset_index(drop=True)
            if sample[params.treatment_column].nunique() < 2:
                failures += 1
                continue
            try:
                sampled_effect, _, _, _ = _fit_effect(
                    sample, params, exposed, ctx.seed + iteration + 1
                )
            except (AnalysisInputError, ValueError):
                failures += 1
                continue
            for key in bootstrap:
                value = sampled_effect.get(key)
                if value is not None and math.isfinite(value):
                    bootstrap[key].append(float(value))
        if min((len(values) for values in bootstrap.values()), default=0) < max(
            50, params.bootstrap_iterations // 2
        ):
            raise AnalysisInputError("too many full-pipeline bootstrap replicates failed")
        alpha = 1 - params.confidence_level

        def interval(key: str) -> tuple[float | None, float | None]:
            values = bootstrap.get(key, [])
            if not values:
                return None, None
            bounds = np.quantile(values, [alpha / 2, 1 - alpha / 2])
            return float(bounds[0]), float(bounds[1])

        difference_lower, difference_upper = interval("difference")
        estimates: list[dict[str, Any]] = []
        if params.outcome_type == "binary":
            rr_lower, rr_upper = interval("risk_ratio")
            risk_ratio = effect["risk_ratio"]
            confidence_limit = rr_lower if risk_ratio >= 1 else rr_upper
            estimate_e, limit_e = e_value(risk_ratio, confidence_limit)
            estimates.extend(
                [
                    {
                        "estimand": "risk_difference",
                        "estimate": effect["difference"],
                        "lower": difference_lower,
                        "upper": difference_upper,
                        "method": params.method,
                        "e_value": estimate_e,
                        "e_value_ci": limit_e,
                        "e_value_basis": "companion_risk_ratio",
                    },
                    {
                        "estimand": "risk_ratio",
                        "estimate": risk_ratio,
                        "lower": rr_lower,
                        "upper": rr_upper,
                        "method": params.method,
                        "e_value": estimate_e,
                        "e_value_ci": limit_e,
                        "e_value_basis": "risk_ratio",
                    },
                ]
            )
        else:
            pooled_sd = float(selected[outcome_column].std(ddof=1))
            standardized = abs(effect["difference"] / pooled_sd) if pooled_sd > 0 else 0.0
            approximate_rr = math.exp(0.91 * standardized)
            estimate_e, _ = e_value(approximate_rr)
            estimates.append(
                {
                    "estimand": "mean_difference",
                    "estimate": effect["difference"],
                    "lower": difference_lower,
                    "upper": difference_upper,
                    "method": params.method,
                    "e_value": estimate_e,
                    "e_value_ci": None,
                    "e_value_basis": "approximate_rr_from_standardized_mean_difference",
                }
            )
        for row in estimates:
            row.update(
                {
                    "exposed": str(exposed),
                    "comparator": str(comparator),
                    "claim_level": "adjusted_association",
                }
            )

        propensity_rows = [
            {
                "person_id": str(selected.iloc[index].get("person_id", index)),
                "group": str(selected.iloc[index][params.treatment_column]),
                "propensity_score": float(propensity[index]),
                "weight": float(weights[index]),
            }
            for index in range(len(selected))
        ]
        n_exposed = int((treatment == 1).sum())
        n_comparator = int((treatment == 0).sum())
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
                {
                    "name": "adjustment_set_pre_declared",
                    "passed": True,
                    "covariates": [item.model_dump() for item in params.adjustment_set],
                },
                {
                    "name": "positivity",
                    "passed": bool((propensity > 0.01).all() and (propensity < 0.99).all()),
                    "minimum": float(propensity.min()),
                    "maximum": float(propensity.max()),
                },
                {
                    "name": "balance",
                    "passed": all(
                        row["smd_after"] is not None and abs(row["smd_after"]) < 0.1
                        for row in balance_rows
                    ),
                    "threshold": 0.1,
                },
                {
                    "name": "bootstrap",
                    "passed": True,
                    "replicates": params.bootstrap_iterations,
                    "failed": failures,
                },
            ],
            guardrail_outcomes=[dict(item) for item in ctx.guardrail_outcomes],
            power_note=comparative_power_note(n_exposed, n_comparator, params.outcome_type),
            seed=ctx.seed,
            package_versions=package_versions(("scikit-learn",)),
            tables={"data": estimates, "balance": balance_rows, "propensity": propensity_rows},
            metadata={
                "claim_level": "adjusted_association",
                "method": params.method,
                "estimand": params.estimand,
                "weight_trim_quantiles": list(params.weight_trim_quantiles),
            },
            output_dir=ctx.output_dir,
        )


ANALYSIS = CausalEffectAnalysis()
