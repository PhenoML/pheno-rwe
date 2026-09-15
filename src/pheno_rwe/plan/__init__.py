"""Public study-plan models and serialization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .io import (
    analysis_data_hash,
    canonical_plan_json,
    code_set_review_hash,
    is_plan_current,
    json_schema,
    load_plan,
    normalized_plan_dict,
    parse_plan,
    plan_hash,
    plan_json_schema,
    write_plan,
)
from .schema import (
    AdjustmentCovariate,
    AnalysisSpec,
    CausalEffectAnalysis,
    CausalEffectParams,
    CensorRuleSpec,
    CodeSet,
    CodeSetApproval,
    Coding,
    CohortCompareAnalysis,
    CohortCompareParams,
    CohortDef,
    ConceptCriterion,
    DeidConfig,
    Demographics,
    IncidenceRateAnalysis,
    IncidenceRateParams,
    IndexDateRule,
    MappingStatus,
    OutcomeDefinition,
    OutcomeRiskWindow,
    PatientSignatureAnalysis,
    PatientSignatureParams,
    Plan,
    PlotSpec,
    ProvenanceOverride,
    RelativeWindow,
    StudySpec,
    SurvivalAnalysis,
    SurvivalParams,
    TableOneAnalysis,
    TableOneParams,
    TemporalWindow,
    TrajectoryAnalysis,
    TrajectoryParams,
    TreatmentPathwaysAnalysis,
    TreatmentPathwaysParams,
    UnitConversionSpec,
)

if TYPE_CHECKING:
    from pheno_rwe.guardrails import ValidationReport


def validate_plan_static(plan: Plan | dict[str, Any]) -> ValidationReport:
    """Run static guardrails without introducing a package import cycle."""

    from pheno_rwe.guardrails import validate_plan_static as _validate

    return _validate(plan)


__all__ = [
    "AdjustmentCovariate",
    "AnalysisSpec",
    "CausalEffectAnalysis",
    "CausalEffectParams",
    "CensorRuleSpec",
    "CodeSet",
    "CodeSetApproval",
    "Coding",
    "CohortCompareAnalysis",
    "CohortCompareParams",
    "CohortDef",
    "ConceptCriterion",
    "DeidConfig",
    "Demographics",
    "IncidenceRateAnalysis",
    "IncidenceRateParams",
    "IndexDateRule",
    "MappingStatus",
    "OutcomeDefinition",
    "OutcomeRiskWindow",
    "PatientSignatureAnalysis",
    "PatientSignatureParams",
    "Plan",
    "PlotSpec",
    "ProvenanceOverride",
    "RelativeWindow",
    "StudySpec",
    "SurvivalAnalysis",
    "SurvivalParams",
    "TableOneAnalysis",
    "TableOneParams",
    "TemporalWindow",
    "TrajectoryAnalysis",
    "TrajectoryParams",
    "TreatmentPathwaysAnalysis",
    "TreatmentPathwaysParams",
    "UnitConversionSpec",
    "analysis_data_hash",
    "canonical_plan_json",
    "code_set_review_hash",
    "is_plan_current",
    "json_schema",
    "load_plan",
    "normalized_plan_dict",
    "parse_plan",
    "plan_hash",
    "plan_json_schema",
    "validate_plan_static",
    "write_plan",
]
