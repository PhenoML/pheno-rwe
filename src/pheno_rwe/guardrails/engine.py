"""Pure STATIC- and DATA-phase guardrail evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, cast

from pheno_rwe.errors import GuardrailRefusal
from pheno_rwe.plan.io import code_set_review_hash, parse_plan, plan_hash
from pheno_rwe.plan.schema import (
    AnalysisSpec,
    CausalEffectAnalysis,
    CohortCompareAnalysis,
    PatientSignatureAnalysis,
    Plan,
    SurvivalAnalysis,
    TrajectoryAnalysis,
)

from . import catalog
from .models import DataSummary, GuardrailPhase, RuleOutcome, Severity, ValidationReport

PlanLike = Plan | Mapping[str, Any]
SummaryLike = DataSummary | Mapping[str, Any]

_COMPARATIVE_KINDS = {"cohort_compare", "survival", "incidence_rate", "causal_effect"}


def evaluate_static_guardrails(plan: PlanLike) -> tuple[RuleOutcome, ...]:
    """Evaluate plan-only rules in a stable, deterministic order."""

    normalized = parse_plan(plan)
    outcomes: list[RuleOutcome] = []
    for code_set in normalized.code_sets:
        approval = code_set.approval
        missing_mapping_statuses = sum(coding.mapping_status is None for coding in code_set.codings)
        expected_review_hash = code_set_review_hash(code_set)
        approval_is_current = (
            approval.status == "approved"
            and missing_mapping_statuses == 0
            and approval.review_hash == expected_review_hash
        )
        if approval_is_current:
            continue
        mapping_status_counts: dict[str, int] = {}
        for coding in code_set.codings:
            status = coding.mapping_status or "UNSPECIFIED"
            mapping_status_counts[status] = mapping_status_counts.get(status, 0) + 1
        if approval.status == "rejected":
            action = "has been rejected"
        elif approval.status == "approved":
            action = "has stale or incomplete review evidence"
        else:
            action = "is pending review"
        outcomes.append(
            RuleOutcome(
                rule_id=catalog.CODESET_APPROVAL,
                phase=GuardrailPhase.STATIC,
                severity=Severity.REFUSE,
                message=(
                    f"Code set '{code_set.name}' {action}; a researcher must review every "
                    "coding and mapping-status count and record an approved decision in "
                    "plan.json."
                ),
                details={
                    "code_set": code_set.name,
                    "approval_status": approval.status,
                    "mapping_status_counts": dict(sorted(mapping_status_counts.items())),
                    "mapping_statuses_missing": missing_mapping_statuses,
                    "review_hash_matches": approval.review_hash == expected_review_hash,
                },
            )
        )
    for analysis in normalized.analyses:
        if not analysis.enabled:
            continue
        if isinstance(analysis, SurvivalAnalysis) and not analysis.params.censor_rule:
            outcomes.append(
                _outcome(
                    catalog.SURV_CENSOR,
                    GuardrailPhase.STATIC,
                    Severity.REFUSE,
                    analysis,
                    "Survival analysis requires a pre-specified censor rule.",
                )
            )

        if isinstance(analysis, CausalEffectAnalysis):
            if not analysis.params.adjustment_set:
                outcomes.append(
                    _outcome(
                        catalog.CAUSAL_ADJUSTMENT,
                        GuardrailPhase.STATIC,
                        Severity.REFUSE,
                        analysis,
                        "Causal-effect analysis requires a pre-declared adjustment set with "
                        "a rationale for every covariate.",
                    )
                )
            else:
                missing_rationale = [
                    covariate.name
                    for covariate in analysis.params.adjustment_set
                    if not covariate.rationale.strip()
                ]
                if missing_rationale:
                    outcomes.append(
                        _outcome(
                            catalog.CAUSAL_ADJUSTMENT,
                            GuardrailPhase.STATIC,
                            Severity.REFUSE,
                            analysis,
                            "Every causal adjustment covariate requires a "
                            "researcher-supplied rationale.",
                            covariates=missing_rationale,
                        )
                    )

        if isinstance(analysis, CohortCompareAnalysis) and len(analysis.params.outcomes) > 1:
            if analysis.params.multiplicity == "none":
                outcomes.append(
                    _outcome(
                        catalog.MULTIPLICITY,
                        GuardrailPhase.STATIC,
                        Severity.REFUSE,
                        analysis,
                        "Multiple outcomes require Benjamini-Hochberg multiplicity control; "
                        "set multiplicity to 'auto_bh' or 'bh'.",
                        outcome_count=len(analysis.params.outcomes),
                    )
                )
            else:
                outcomes.append(
                    _outcome(
                        catalog.MULTIPLICITY,
                        GuardrailPhase.STATIC,
                        Severity.INFO,
                        analysis,
                        "Benjamini-Hochberg q-values will be reported for the "
                        "pre-specified outcomes.",
                        outcome_count=len(analysis.params.outcomes),
                    )
                )
    return tuple(outcomes)


def validate_plan_static(plan: PlanLike) -> ValidationReport:
    """Validate schema plus STATIC rules and return, rather than raise, refusals."""

    normalized = parse_plan(plan)
    return ValidationReport(
        plan_hash=plan_hash(normalized),
        phases=(GuardrailPhase.STATIC,),
        outcomes=evaluate_static_guardrails(normalized),
    )


def evaluate_data_guardrails(
    plan: PlanLike,
    analysis: AnalysisSpec | str,
    summary: SummaryLike,
) -> tuple[RuleOutcome, ...]:
    """Evaluate all DATA rules for one analysis from aggregate counts only."""

    normalized = parse_plan(plan)
    analysis_spec = _find_analysis(normalized, analysis)
    data = summary if isinstance(summary, DataSummary) else DataSummary.model_validate(summary)
    if data.analysis_id != analysis_spec.id:
        raise ValueError(
            f"data summary is for analysis '{data.analysis_id}', expected '{analysis_spec.id}'"
        )
    if not analysis_spec.enabled:
        return ()

    outcomes: list[RuleOutcome] = []
    total_n = data.effective_total_n

    if (
        analysis_spec.kind in _COMPARATIVE_KINDS
        and total_n is not None
        and total_n < catalog.COMPARATIVE_MIN_N
    ):
        outcomes.append(
            _outcome(
                catalog.GLOBAL_MIN_N,
                GuardrailPhase.DATA,
                Severity.REFUSE,
                analysis_spec,
                f"Comparative analysis has n={total_n}; at least "
                f"{catalog.COMPARATIVE_MIN_N} patients are required.",
                n=total_n,
                threshold=catalog.COMPARATIVE_MIN_N,
            )
        )

    if analysis_spec.kind in _COMPARATIVE_KINDS and data.arm_n:
        small_arms = {
            arm: count for arm, count in sorted(data.arm_n.items()) if count < catalog.MIN_ARM_N
        }
        if small_arms:
            outcomes.append(
                _outcome(
                    catalog.COMP_MIN_ARM,
                    GuardrailPhase.DATA,
                    Severity.REFUSE,
                    analysis_spec,
                    f"Comparative arm size is below {catalog.MIN_ARM_N}; redesign or combine "
                    "clinically defensible groups.",
                    arm_n=dict(sorted(data.arm_n.items())),
                    small_arms=small_arms,
                    threshold=catalog.MIN_ARM_N,
                )
            )

    if isinstance(analysis_spec, CohortCompareAnalysis):
        adjusted = analysis_spec.params.adjusted or bool(analysis_spec.params.adjustment_set)
        events = data.effective_events_total
        parameters = data.model_parameters
        if parameters is None and adjusted:
            parameters = len(analysis_spec.params.adjustment_set) + 1
        if adjusted and events is not None and parameters:
            epv = events / parameters
            if epv < catalog.EPV_REFUSE_BELOW:
                outcomes.append(
                    _outcome(
                        catalog.EPV_LOGISTIC,
                        GuardrailPhase.DATA,
                        Severity.REFUSE,
                        analysis_spec,
                        f"Adjusted logistic model has {epv:.1f} events per parameter; fewer "
                        f"than {catalog.EPV_REFUSE_BELOW:g} is refused.",
                        events=events,
                        parameters=parameters,
                        epv=round(epv, 4),
                    )
                )
            elif epv < catalog.EPV_WARN_BELOW:
                outcomes.append(
                    _outcome(
                        catalog.EPV_LOGISTIC,
                        GuardrailPhase.DATA,
                        Severity.WARN,
                        analysis_spec,
                        f"Adjusted logistic model has {epv:.1f} events per parameter; estimates "
                        f"may be unstable below {catalog.EPV_WARN_BELOW:g}.",
                        events=events,
                        parameters=parameters,
                        epv=round(epv, 4),
                    )
                )

    if isinstance(analysis_spec, SurvivalAnalysis) and analysis_spec.params.cox:
        low_event_arms = {
            arm: count
            for arm, count in sorted(data.events_by_arm.items())
            if count < catalog.SURVIVAL_MIN_EVENTS_PER_ARM
        }
        if low_event_arms:
            outcomes.append(
                _outcome(
                    catalog.SURV_EVENTS,
                    GuardrailPhase.DATA,
                    Severity.REFUSE,
                    analysis_spec,
                    "Cox modeling requires at least "
                    f"{catalog.SURVIVAL_MIN_EVENTS_PER_ARM} events in every arm.",
                    events_by_arm=dict(sorted(data.events_by_arm.items())),
                    low_event_arms=low_event_arms,
                    threshold=catalog.SURVIVAL_MIN_EVENTS_PER_ARM,
                )
            )

    if isinstance(analysis_spec, PatientSignatureAnalysis) and total_n is not None:
        if total_n < catalog.UMAP_REFUSE_BELOW:
            outcomes.append(
                _outcome(
                    catalog.UMAP_MIN_N,
                    GuardrailPhase.DATA,
                    Severity.REFUSE,
                    analysis_spec,
                    f"Patient-signature UMAP requires n>={catalog.UMAP_REFUSE_BELOW}; use the "
                    "offered PCA descriptive downgrade instead.",
                    n=total_n,
                    threshold=catalog.UMAP_REFUSE_BELOW,
                    downgrade="pca",
                )
            )
        elif total_n < catalog.UMAP_WARN_BELOW:
            outcomes.append(
                _outcome(
                    catalog.UMAP_MIN_N,
                    GuardrailPhase.DATA,
                    Severity.WARN,
                    analysis_spec,
                    f"Patient-signature analysis at n={total_n} is exploratory; stability "
                    "across seeds must be emphasized.",
                    n=total_n,
                    warning_below=catalog.UMAP_WARN_BELOW,
                )
            )

    _evaluate_unchecked_fraction(normalized, analysis_spec, data, outcomes)

    if isinstance(analysis_spec, TrajectoryAnalysis):
        distinct_units = sorted({unit.strip() for unit in data.measurement_units if unit.strip()})
        if len(distinct_units) > 1 and not analysis_spec.params.unit_harmonization:
            outcomes.append(
                _outcome(
                    catalog.MEAS_UNITS,
                    GuardrailPhase.DATA,
                    Severity.REFUSE,
                    analysis_spec,
                    "Measurement values contain heterogeneous units and no harmonization map "
                    "was pre-specified.",
                    units=distinct_units,
                )
            )
        if analysis_spec.params.mixed_model:
            too_few_patients = total_n is not None and total_n < catalog.MIXED_MODEL_MIN_PATIENTS
            too_few_measurements = (
                data.median_measurements_per_patient is not None
                and data.median_measurements_per_patient
                < catalog.MIXED_MODEL_MIN_MEDIAN_MEASUREMENTS
            )
            if too_few_patients or too_few_measurements:
                outcomes.append(
                    _outcome(
                        catalog.TRAJECTORY_MIXED_MIN,
                        GuardrailPhase.DATA,
                        Severity.REFUSE,
                        analysis_spec,
                        "Mixed-effects trajectory modeling requires at least 30 patients and a "
                        "median of at least 3 measurements per patient.",
                        n=total_n,
                        median_measurements_per_patient=data.median_measurements_per_patient,
                    )
                )

    if isinstance(analysis_spec, CausalEffectAnalysis):
        exposed_n = data.arm_n.get(analysis_spec.params.exposed_cohort)
        if exposed_n is None:
            exposed_n = data.arm_n.get("exposed")
        events = data.effective_events_total
        low_exposed = exposed_n is not None and exposed_n < catalog.CAUSAL_MIN_EXPOSED
        low_events = events is not None and events < catalog.CAUSAL_MIN_EVENTS
        if low_exposed or low_events:
            outcomes.append(
                _outcome(
                    catalog.CAUSAL_MIN,
                    GuardrailPhase.DATA,
                    Severity.REFUSE,
                    analysis_spec,
                    "Causal-effect estimation requires at least 20 exposed patients and at "
                    "least 10 outcome events.",
                    exposed_n=exposed_n,
                    events=events,
                    min_exposed=catalog.CAUSAL_MIN_EXPOSED,
                    min_events=catalog.CAUSAL_MIN_EVENTS,
                )
            )
    return tuple(outcomes)


def validate_analysis_data(
    plan: PlanLike,
    analysis: AnalysisSpec | str,
    summary: SummaryLike,
) -> ValidationReport:
    """Return the re-runnable DATA gate for a single analysis."""

    normalized = parse_plan(plan)
    analysis_spec = _find_analysis(normalized, analysis)
    data = summary if isinstance(summary, DataSummary) else DataSummary.model_validate(summary)
    power_note = minimum_detectable_effect_note(data)
    return ValidationReport(
        plan_hash=plan_hash(normalized),
        phases=(GuardrailPhase.DATA,),
        outcomes=evaluate_data_guardrails(normalized, analysis_spec, data),
        power_notes={analysis_spec.id: power_note} if power_note else {},
    )


def validate_plan_data(
    plan: PlanLike,
    summaries: Mapping[str, SummaryLike] | Iterable[SummaryLike],
) -> ValidationReport:
    """Evaluate DATA rules for supplied analyses.

    Supplying a subset is supported because ``analyze`` re-checks one analysis at
    a time immediately before execution.
    """

    normalized = parse_plan(plan)
    summary_map = _summary_map(summaries)
    outcomes: list[RuleOutcome] = []
    notes: dict[str, str] = {}
    for analysis in normalized.analyses:
        summary = summary_map.get(analysis.id)
        if summary is None:
            continue
        outcomes.extend(evaluate_data_guardrails(normalized, analysis, summary))
        note = minimum_detectable_effect_note(summary)
        if note:
            notes[analysis.id] = note
    unknown = sorted(set(summary_map) - {analysis.id for analysis in normalized.analyses})
    if unknown:
        raise ValueError("data summaries reference unknown analysis id(s): " + ", ".join(unknown))
    return ValidationReport(
        plan_hash=plan_hash(normalized),
        phases=(GuardrailPhase.DATA,),
        outcomes=tuple(outcomes),
        power_notes=notes,
    )


def validate_plan(
    plan: PlanLike,
    summaries: Mapping[str, SummaryLike] | Iterable[SummaryLike] | None = None,
) -> ValidationReport:
    """Run STATIC rules and, when provided, DATA rules as one report."""

    static = validate_plan_static(plan)
    if summaries is None:
        return static
    data = validate_plan_data(plan, summaries)
    return ValidationReport(
        plan_hash=static.plan_hash,
        phases=(GuardrailPhase.STATIC, GuardrailPhase.DATA),
        outcomes=(*static.outcomes, *data.outcomes),
        power_notes=data.power_notes,
    )


def minimum_detectable_effect_note(summary: DataSummary) -> str | None:
    """Return a design-stage MDE sentence, never a post-hoc power calculation.

    The standardized two-sample approximation assumes two-sided alpha=.05 and an
    80% design target.  It is intentionally labeled as an approximation.
    """

    positive_arms = sorted(count for count in summary.arm_n.values() if count > 0)
    if len(positive_arms) < 2:
        return None
    n1, n2 = positive_arms[0], positive_arms[1]
    z_alpha_plus_power = 1.959963984540054 + 0.8416212335729143
    standardized_mde = z_alpha_plus_power * math.sqrt((1 / n1) + (1 / n2))
    return (
        f"With arm sizes {n1} and {n2}, the approximate minimum detectable standardized "
        f"effect is {standardized_mde:.2f} at two-sided alpha 0.05 and an 80% design "
        "target; this is a design-stage benchmark, not post-hoc power."
    )


def raise_for_refusal(report: ValidationReport) -> None:
    """Raise the shared typed error when a frontend wants exception semantics."""

    if report.refused:
        messages = [outcome.message for outcome in report.outcomes if outcome.refused]
        raise GuardrailRefusal("; ".join(messages), list(report.outcomes))


def _evaluate_unchecked_fraction(
    plan: Plan,
    analysis: AnalysisSpec,
    data: DataSummary,
    outcomes: list[RuleOutcome],
) -> None:
    fraction = data.unchecked_fraction
    if fraction is None or fraction <= catalog.UNCHECKED_WARN_ABOVE:
        return
    override = analysis.provenance_override or plan.provenance_override
    percentage = fraction * 100
    if fraction > catalog.UNCHECKED_REFUSE_ABOVE and override is None:
        severity = Severity.REFUSE
        message = (
            f"{percentage:.1f}% of contributing rows are UNCHECKED; above 50% requires an "
            "auditable provenance_override with justification in plan.json."
        )
    elif fraction > catalog.UNCHECKED_REFUSE_ABOVE:
        severity = Severity.WARN
        message = (
            f"{percentage:.1f}% of contributing rows are UNCHECKED; the plan-hashed provenance "
            "override is active and this limitation must be reported."
        )
    else:
        severity = Severity.WARN
        message = (
            f"{percentage:.1f}% of contributing rows are UNCHECKED; interpret results cautiously."
        )
    details: dict[str, Any] = {
        "unchecked_fraction": fraction,
        "warn_above": catalog.UNCHECKED_WARN_ABOVE,
        "refuse_above": catalog.UNCHECKED_REFUSE_ABOVE,
        "override_applied": override is not None,
    }
    if override is not None:
        details["override_justification"] = override.justification
    outcomes.append(
        _outcome(
            catalog.UNCHECKED_FRACTION,
            GuardrailPhase.DATA,
            severity,
            analysis,
            message,
            **details,
        )
    )


def _find_analysis(plan: Plan, analysis: AnalysisSpec | str) -> AnalysisSpec:
    analysis_id = analysis if isinstance(analysis, str) else analysis.id
    for candidate in plan.analyses:
        if candidate.id == analysis_id:
            return candidate
    raise ValueError(f"unknown analysis id: {analysis_id}")


def _summary_map(
    summaries: Mapping[str, SummaryLike] | Iterable[SummaryLike],
) -> dict[str, DataSummary]:
    if isinstance(summaries, Mapping):
        summary_mapping = cast(Mapping[str, SummaryLike], summaries)
        result: dict[str, DataSummary] = {}
        for key, value in summary_mapping.items():
            if isinstance(value, DataSummary):
                summary = value
            else:
                payload = dict(value)
                payload.setdefault("analysis_id", key)
                summary = DataSummary.model_validate(payload)
            if summary.analysis_id != key:
                raise ValueError(
                    f"summary mapping key '{key}' does not match analysis_id "
                    f"'{summary.analysis_id}'"
                )
            result[key] = summary
        return result

    result = {}
    for raw in summaries:
        summary = raw if isinstance(raw, DataSummary) else DataSummary.model_validate(raw)
        if summary.analysis_id in result:
            raise ValueError(f"duplicate data summary for analysis '{summary.analysis_id}'")
        result[summary.analysis_id] = summary
    return result


def _outcome(
    rule_id: str,
    phase: GuardrailPhase,
    severity: Severity,
    analysis: AnalysisSpec,
    message: str,
    **details: Any,
) -> RuleOutcome:
    return RuleOutcome(
        rule_id=rule_id,
        phase=phase,
        severity=severity,
        analysis_id=analysis.id,
        message=message,
        details=details,
    )
