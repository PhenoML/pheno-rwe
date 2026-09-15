"""Versioned, declarative study-plan schema.

The models in this module intentionally describe *what* should be run, not how it
is run.  Executors and frontends can therefore share the same JSON contract.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    ),
]


class PlanModel(BaseModel):
    """Base configuration shared by plan models.

    Unknown fields are rejected so misspelled scientific choices do not silently
    disappear from a pre-registered plan.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )


MappingStatus = Literal["ALREADY_STANDARD", "MAPPED", "UNCHECKED", "UNMAPPED"]


class CodeSetApproval(PlanModel):
    """Researcher review state for one exact, plan-hashed code set."""

    status: Literal["pending", "approved", "rejected"] = "pending"
    reviewed_by: NonEmptyStr | None = None
    reviewed_at: datetime | None = None
    notes: str | None = None
    review_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_auditable_review(self) -> CodeSetApproval:
        reviewed = self.status in {"approved", "rejected"}
        if reviewed and (self.reviewed_by is None or self.reviewed_at is None):
            raise ValueError(
                f"reviewed_by and reviewed_at are required when approval status='{self.status}'"
            )
        if self.reviewed_at is not None and self.reviewed_at.utcoffset() is None:
            raise ValueError("reviewed_at must include a UTC offset")
        if self.status == "pending" and (
            self.reviewed_by is not None
            or self.reviewed_at is not None
            or self.review_hash is not None
        ):
            raise ValueError(
                "pending code-set approval cannot include reviewed_by, reviewed_at, or review_hash"
            )
        return self


class Coding(PlanModel):
    system: NonEmptyStr
    code: NonEmptyStr
    display: str | None = None
    concept_id: int | None = Field(default=None, ge=0)
    mapping_status: MappingStatus | None = None
    accepted: bool = True

    @property
    def source_value(self) -> str:
        """The exact source-value representation written by fhir2omop."""

        return f"{self.system}#{self.code}"


class CodeSet(PlanModel):
    name: Identifier
    domain: NonEmptyStr
    codings: list[Coding] = Field(min_length=1)
    resolve: bool = True
    description: str | None = None
    approval: CodeSetApproval = Field(default_factory=CodeSetApproval)

    @model_validator(mode="after")
    def normalize_domain(self) -> CodeSet:
        object.__setattr__(self, "domain", _normalized_domain(self.domain))
        return self


class TemporalWindow(PlanModel):
    anchor: NonEmptyStr = "index_date"
    days_before: int | None = Field(default=None, ge=0)
    days_after: int | None = Field(default=None, ge=0)


class ConceptCriterion(PlanModel):
    code_set: Identifier
    min_count: int = Field(default=1, ge=1)
    window: TemporalWindow | None = None


class Demographics(PlanModel):
    min_age: int | None = Field(default=None, ge=0, le=130)
    max_age: int | None = Field(default=None, ge=0, le=130)
    sex: list[NonEmptyStr] = Field(default_factory=list)
    race: list[NonEmptyStr] = Field(default_factory=list)
    ethnicity: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_age_range(self) -> Demographics:
        if self.min_age is not None and self.max_age is not None and self.min_age > self.max_age:
            raise ValueError("min_age must be less than or equal to max_age")
        return self


IndexStrategy = Literal["first_occurrence", "last_occurrence", "fixed_date"]


class IndexDateRule(PlanModel):
    strategy: IndexStrategy
    code_set: Identifier | None = None
    fixed_date: date | None = None

    @model_validator(mode="after")
    def check_strategy_fields(self) -> IndexDateRule:
        if self.strategy == "fixed_date":
            if self.fixed_date is None:
                raise ValueError("fixed_date is required when strategy='fixed_date'")
            if self.code_set is not None:
                raise ValueError("code_set must be omitted when strategy='fixed_date'")
        elif self.code_set is None:
            raise ValueError(f"code_set is required when strategy='{self.strategy}'")
        elif self.fixed_date is not None:
            raise ValueError("fixed_date is only valid when strategy='fixed_date'")
        return self


class CohortDef(PlanModel):
    name: Identifier
    inclusion: list[ConceptCriterion] = Field(default_factory=list)
    exclusion: list[ConceptCriterion] = Field(default_factory=list)
    demographics: Demographics = Field(default_factory=Demographics)
    index_date_rule: IndexDateRule | None = None
    description: str | None = None


DeidTier = Literal["baseline", "date_shift"]


class DeidConfig(PlanModel):
    tier: DeidTier = "baseline"
    age_bucket_years: int = Field(default=5, ge=1, le=20)
    age_cap: int = Field(default=90, ge=18, le=130)
    date_shift_max_days: int = Field(default=180, ge=1, le=3650)


class ProvenanceOverride(PlanModel):
    """Auditable, plan-hashed exception to the UNCHECKED provenance rule."""

    justification: NonEmptyStr


class StudySpec(PlanModel):
    id: Identifier | None = Field(
        default=None,
        validation_alias=AliasChoices("id", "study_id"),
    )
    name: NonEmptyStr = Field(
        validation_alias=AliasChoices("name", "title"),
    )
    question: NonEmptyStr
    description: str | None = None
    design: str | None = None


class RelativeWindow(PlanModel):
    start_day: int
    end_day: int

    @model_validator(mode="after")
    def check_order(self) -> RelativeWindow:
        if self.start_day > self.end_day:
            raise ValueError("start_day must be less than or equal to end_day")
        return self


class OutcomeRiskWindow(PlanModel):
    """Finite incident-outcome window relative to cohort index.

    Both risk-window boundaries are inclusive. ``washout_days`` covers the
    immediately preceding interval ``[index - washout_days, index)``.  Outcome
    ascertainment is always censored at the end of the continuous observation
    period containing index; this is deliberately not switchable for v1 binary
    risk analyses.
    """

    start_day: int = Field(ge=0)
    end_day: int = Field(ge=0)
    washout_days: int = Field(ge=0)
    observation_censor: Literal["end_of_observation"] = "end_of_observation"

    @model_validator(mode="after")
    def check_order(self) -> OutcomeRiskWindow:
        if self.start_day > self.end_day:
            raise ValueError("start_day must be less than or equal to end_day")
        return self


class OutcomeDefinition(PlanModel):
    """A code-set outcome together with its complete time-at-risk contract."""

    code_set: Identifier
    risk_window: OutcomeRiskWindow


class AdjustmentCovariate(PlanModel):
    name: Identifier
    rationale: str = ""
    code_set: Identifier | None = None


class CensorRuleSpec(PlanModel):
    strategy: Literal[
        "administrative",
        "end_of_observation",
        "outcome",
        "competing_event",
    ] = Field(validation_alias=AliasChoices("strategy", "kind"))
    description: str | None = None
    code_set: Identifier | None = None
    max_followup_days: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def check_strategy_fields(self) -> CensorRuleSpec:
        if self.strategy == "administrative" and self.max_followup_days is None:
            raise ValueError("max_followup_days is required when censor strategy='administrative'")
        if self.strategy in {"outcome", "competing_event"} and self.code_set is None:
            raise ValueError(f"code_set is required when censor strategy='{self.strategy}'")
        if self.strategy in {"administrative", "end_of_observation"} and self.code_set:
            raise ValueError(f"code_set is not valid when censor strategy='{self.strategy}'")
        return self


CensorRule = str | CensorRuleSpec


# Analysis parameter models -------------------------------------------------


class TableOneParams(PlanModel):
    cohort: Identifier
    stratify_by: NonEmptyStr | None = Field(
        default=None,
        validation_alias=AliasChoices("stratify_by", "group_by"),
    )
    variables: list[NonEmptyStr] = Field(default_factory=list)
    small_cell_threshold: int = Field(default=5, ge=1)


class CohortCompareParams(PlanModel):
    exposed_cohort: Identifier
    comparator_cohort: Identifier
    outcomes: list[OutcomeDefinition] = Field(min_length=1)
    adjustment_set: list[AdjustmentCovariate] = Field(default_factory=list)
    adjusted: bool = False
    exact_test: Literal["fisher", "boschloo"] = "fisher"
    adjusted_estimator: Literal["firth_logistic"] = "firth_logistic"
    multiplicity: Literal["auto_bh", "bh", "none"] = "auto_bh"
    permutations: int = Field(default=10_000, ge=99, le=1_000_000)

    @field_validator("outcomes", mode="before")
    @classmethod
    def reject_legacy_outcomes_without_time_at_risk(cls, value: object) -> object:
        if isinstance(value, list) and any(isinstance(item, str) for item in value):
            raise ValueError(
                "each outcome must declare code_set and risk_window with start_day, "
                "end_day, and washout_days"
            )
        return value

    @model_validator(mode="after")
    def check_adjustment_request(self) -> CohortCompareParams:
        if self.adjusted and not self.adjustment_set:
            raise ValueError("adjusted=true requires a non-empty adjustment_set")
        if self.adjustment_set:
            # The adjustment set is itself the scientific request.  Normalize
            # the legacy switch so the canonical plan hash describes what the
            # executor will actually estimate.
            object.__setattr__(self, "adjusted", True)
        names = [outcome.code_set for outcome in self.outcomes]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError("duplicate outcome code set(s): " + ", ".join(duplicates))
        return self


class SurvivalParams(PlanModel):
    exposed_cohort: Identifier
    comparator_cohort: Identifier
    outcome: Identifier
    censor_rule: CensorRule | None = None
    horizon_days: int | None = Field(default=None, gt=0)
    cox: bool = True
    ridge_penalizer: float | None = Field(default=None, ge=0)
    adjustment_set: list[AdjustmentCovariate] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_censor_rule(self) -> SurvivalParams:
        if isinstance(self.censor_rule, str):
            object.__setattr__(self, "censor_rule", self.censor_rule.strip() or None)
        return self


class IncidenceRateParams(PlanModel):
    cohorts: list[Identifier] = Field(min_length=2, max_length=2)
    outcome: Identifier
    time_scale: Literal["days", "person_years"] = "person_years"


class TreatmentPathwaysParams(PlanModel):
    cohort: Identifier
    drug_classes: list[Identifier] = Field(min_length=1)
    permissible_gap_days: int = Field(default=30, ge=0)
    max_lines: int = Field(default=3, ge=1, le=20)


class UnitConversionSpec(PlanModel):
    target_unit: NonEmptyStr
    multiplier: float = 1.0
    offset: float = 0.0


class TrajectoryParams(PlanModel):
    cohort: Identifier
    measurement: Identifier
    group_by: NonEmptyStr | None = None
    mixed_model: bool = False
    unit_harmonization: dict[NonEmptyStr, NonEmptyStr | float | UnitConversionSpec] = Field(
        default_factory=dict
    )


class PatientSignatureParams(PlanModel):
    cohort: Identifier
    baseline_window: RelativeWindow = Field(
        default_factory=lambda: RelativeWindow(start_day=-365, end_day=-1)
    )
    post_window: RelativeWindow = Field(
        default_factory=lambda: RelativeWindow(start_day=0, end_day=90)
    )
    feature_domains: list[NonEmptyStr] = Field(default_factory=list)
    n_neighbors: int | None = Field(default=None, ge=2)
    min_cluster_size: int = Field(default=5, ge=2)


class CausalEffectParams(PlanModel):
    exposed_cohort: Identifier
    comparator_cohort: Identifier
    outcome: OutcomeDefinition
    adjustment_set: list[AdjustmentCovariate] = Field(default_factory=list)
    method: Literal["iptw", "ps_matching"] = "iptw"
    estimand: Literal["ate", "att"] = "ate"
    weight_trim_quantiles: tuple[float, float] = (0.01, 0.99)
    matching_caliper: float | None = Field(default=None, gt=0)

    @field_validator("outcome", mode="before")
    @classmethod
    def reject_legacy_outcome_without_time_at_risk(cls, value: object) -> object:
        if isinstance(value, str):
            raise ValueError(
                "outcome must declare code_set and risk_window with start_day, "
                "end_day, and washout_days"
            )
        return value

    @model_validator(mode="after")
    def check_trim_quantiles(self) -> CausalEffectParams:
        lower, upper = self.weight_trim_quantiles
        if not (0 <= lower < upper <= 1):
            raise ValueError("weight_trim_quantiles must satisfy 0 <= lower < upper <= 1")
        if self.method == "ps_matching" and self.matching_caliper is None:
            object.__setattr__(self, "matching_caliper", 0.2)
        if self.method == "ps_matching" and self.estimand != "att":
            raise ValueError("1:1 propensity-score matching currently estimates the ATT")
        return self


class AnalysisBase(PlanModel):
    id: Identifier
    kind: str
    title: str | None = None
    enabled: bool = True
    provenance_override: ProvenanceOverride | None = None


class TableOneAnalysis(AnalysisBase):
    kind: Literal["table_one"]
    params: TableOneParams


class CohortCompareAnalysis(AnalysisBase):
    kind: Literal["cohort_compare"]
    params: CohortCompareParams


class SurvivalAnalysis(AnalysisBase):
    kind: Literal["survival"]
    params: SurvivalParams


class IncidenceRateAnalysis(AnalysisBase):
    kind: Literal["incidence_rate"]
    params: IncidenceRateParams


class TreatmentPathwaysAnalysis(AnalysisBase):
    kind: Literal["treatment_pathways"]
    params: TreatmentPathwaysParams


class TrajectoryAnalysis(AnalysisBase):
    kind: Literal["trajectory"]
    params: TrajectoryParams


class PatientSignatureAnalysis(AnalysisBase):
    kind: Literal["patient_signature"]
    params: PatientSignatureParams


class CausalEffectAnalysis(AnalysisBase):
    kind: Literal["causal_effect"]
    params: CausalEffectParams


AnalysisSpec = Annotated[
    TableOneAnalysis
    | CohortCompareAnalysis
    | SurvivalAnalysis
    | IncidenceRateAnalysis
    | TreatmentPathwaysAnalysis
    | TrajectoryAnalysis
    | PatientSignatureAnalysis
    | CausalEffectAnalysis,
    Field(discriminator="kind"),
]


# Plot parameter models -----------------------------------------------------


class AnalysisPlotParams(PlanModel):
    analysis_id: Identifier
    source_csv: str | None = None


class KmCurvePlotParams(AnalysisPlotParams):
    show_at_risk: bool = True


class ForestPlotParams(AnalysisPlotParams):
    pass


class LovePlotParams(AnalysisPlotParams):
    threshold: float = Field(default=0.1, gt=0)


class UmapScatterPlotParams(AnalysisPlotParams):
    color_by: NonEmptyStr | None = None


class TrajectoryPlotParams(AnalysisPlotParams):
    show_individuals: bool = True


class IncidenceBarPlotParams(AnalysisPlotParams):
    pass


class PathwayBarsPlotParams(AnalysisPlotParams):
    max_pathways: int = Field(default=20, ge=1)


class AttritionPlotParams(PlanModel):
    cohort: Identifier | None = None
    source_csv: str | None = None


class MappingCoveragePlotParams(PlanModel):
    domain: NonEmptyStr | None = None
    source_csv: str | None = None


class PlotBase(PlanModel):
    id: Identifier
    kind: str
    title: str | None = None
    enabled: bool = True


class KmCurvePlot(PlotBase):
    kind: Literal["km_curve"]
    params: KmCurvePlotParams


class ForestPlot(PlotBase):
    kind: Literal["forest"]
    params: ForestPlotParams


class LovePlot(PlotBase):
    kind: Literal["love_plot"]
    params: LovePlotParams


class UmapScatterPlot(PlotBase):
    kind: Literal["umap_scatter"]
    params: UmapScatterPlotParams


class TrajectoryPlot(PlotBase):
    kind: Literal["trajectory"]
    params: TrajectoryPlotParams


class AttritionPlot(PlotBase):
    kind: Literal["attrition"]
    params: AttritionPlotParams = Field(default_factory=AttritionPlotParams)


class MappingCoveragePlot(PlotBase):
    kind: Literal["mapping_coverage"]
    params: MappingCoveragePlotParams = Field(default_factory=MappingCoveragePlotParams)


class IncidenceBarPlot(PlotBase):
    kind: Literal["incidence_bar"]
    params: IncidenceBarPlotParams


class PathwayBarsPlot(PlotBase):
    kind: Literal["pathway_bars"]
    params: PathwayBarsPlotParams


PlotSpec = Annotated[
    KmCurvePlot
    | ForestPlot
    | LovePlot
    | UmapScatterPlot
    | TrajectoryPlot
    | AttritionPlot
    | MappingCoveragePlot
    | IncidenceBarPlot
    | PathwayBarsPlot,
    Field(discriminator="kind"),
]


class Plan(PlanModel):
    schema_version: str = Field(default="1.0", pattern=r"^1(?:\.\d+)?$")
    study: StudySpec
    code_sets: list[CodeSet] = Field(min_length=1)
    cohorts: list[CohortDef] = Field(min_length=1)
    index_date_rule: IndexDateRule
    deid: DeidConfig = Field(default_factory=DeidConfig)
    seed: int = Field(default=2025, ge=0, le=4_294_967_295)
    analyses: list[AnalysisSpec] = Field(min_length=1)
    plots: list[PlotSpec] = Field(default_factory=list)
    provenance_override: ProvenanceOverride | None = None

    @model_validator(mode="after")
    def validate_references_and_ids(self) -> Plan:
        code_set_names = [code_set.name for code_set in self.code_sets]
        cohort_names = [cohort.name for cohort in self.cohorts]
        analysis_ids = [analysis.id for analysis in self.analyses]
        plot_ids = [plot.id for plot in self.plots]

        _require_unique(code_set_names, "code set name")
        _require_unique(cohort_names, "cohort name")
        _require_unique(analysis_ids, "analysis id")
        _require_unique(plot_ids, "plot id")

        known_code_sets = set(code_set_names)
        referenced_code_sets: list[tuple[str, str]] = []
        if self.index_date_rule.code_set:
            referenced_code_sets.append(("index_date_rule", self.index_date_rule.code_set))
        for cohort in self.cohorts:
            if cohort.index_date_rule and cohort.index_date_rule.code_set:
                referenced_code_sets.append(
                    (f"cohort '{cohort.name}' index_date_rule", cohort.index_date_rule.code_set)
                )
            for criterion in [*cohort.inclusion, *cohort.exclusion]:
                referenced_code_sets.append(
                    (f"cohort '{cohort.name}' criterion", criterion.code_set)
                )
        missing_code_sets = [
            f"{owner}: {name}"
            for owner, name in referenced_code_sets
            if name not in known_code_sets
        ]
        if missing_code_sets:
            raise ValueError("unknown code-set reference(s): " + ", ".join(missing_code_sets))

        known_cohorts = set(cohort_names)
        code_sets_by_name = {code_set.name: code_set for code_set in self.code_sets}
        analysis_reference_errors: list[str] = []

        def require_cohort(owner: str, name: str) -> None:
            if name not in known_cohorts:
                analysis_reference_errors.append(f"{owner}: unknown cohort '{name}'")

        def require_distinct_cohorts(owner: str, exposed: str, comparator: str) -> None:
            require_cohort(owner, exposed)
            require_cohort(owner, comparator)
            if exposed == comparator:
                analysis_reference_errors.append(
                    f"{owner}: exposed and comparator cohorts must be distinct"
                )

        def require_code_set(owner: str, name: str, domain: str | None = None) -> None:
            code_set = code_sets_by_name.get(name)
            if code_set is None:
                analysis_reference_errors.append(f"{owner}: unknown code set '{name}'")
                return
            if not any(coding.accepted for coding in code_set.codings):
                analysis_reference_errors.append(
                    f"{owner}: code set '{name}' has no accepted codings"
                )
            if domain is not None and _normalized_domain(code_set.domain) != domain:
                analysis_reference_errors.append(
                    f"{owner}: code set '{name}' must use the {domain} domain"
                )

        def require_variable(owner: str, name: str) -> None:
            if name not in _DEMOGRAPHIC_VARIABLES:
                require_code_set(owner, name)

        def require_adjustment(owner: str, covariate: AdjustmentCovariate) -> None:
            if covariate.code_set:
                if covariate.name in _DEMOGRAPHIC_VARIABLES:
                    analysis_reference_errors.append(
                        f"{owner}: covariate '{covariate.name}' shadows a built-in demographic"
                    )
                require_code_set(owner, covariate.code_set)
            else:
                require_variable(owner, covariate.name)

        def require_adjustments(
            owner: str,
            covariates: list[AdjustmentCovariate],
            *,
            protected: set[str] | None = None,
        ) -> None:
            names = [covariate.name for covariate in covariates]
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                analysis_reference_errors.append(
                    f"{owner}: duplicate adjustment covariate(s): {', '.join(duplicates)}"
                )
            collisions = sorted(
                set(names) & ({"person_id", "group", "index_date"} | (protected or set()))
            )
            if collisions:
                analysis_reference_errors.append(
                    f"{owner}: adjustment covariate name collision(s): {', '.join(collisions)}"
                )
            for covariate in covariates:
                require_adjustment(owner, covariate)

        for analysis in self.analyses:
            owner = f"analysis '{analysis.id}'"
            if isinstance(analysis, TableOneAnalysis):
                table_params = analysis.params
                require_cohort(owner, table_params.cohort)
                if table_params.stratify_by:
                    require_variable(owner, table_params.stratify_by)
                for variable in table_params.variables:
                    require_variable(owner, variable)
            elif isinstance(analysis, CohortCompareAnalysis):
                compare_params = analysis.params
                require_distinct_cohorts(
                    owner, compare_params.exposed_cohort, compare_params.comparator_cohort
                )
                for outcome in compare_params.outcomes:
                    require_code_set(owner, outcome.code_set)
                require_adjustments(
                    owner,
                    compare_params.adjustment_set,
                    protected={outcome.code_set for outcome in compare_params.outcomes},
                )
            elif isinstance(analysis, SurvivalAnalysis):
                survival_params = analysis.params
                require_distinct_cohorts(
                    owner,
                    survival_params.exposed_cohort,
                    survival_params.comparator_cohort,
                )
                require_code_set(owner, survival_params.outcome)
                censor_rule = survival_params.censor_rule
                if isinstance(censor_rule, CensorRuleSpec) and censor_rule.code_set:
                    require_code_set(owner, censor_rule.code_set)
                require_adjustments(
                    owner,
                    survival_params.adjustment_set,
                    protected={"duration", "event", "outcome_date", "observation_end"},
                )
            elif isinstance(analysis, IncidenceRateAnalysis):
                incidence_params = analysis.params
                first, second = incidence_params.cohorts
                require_distinct_cohorts(owner, first, second)
                require_code_set(owner, incidence_params.outcome)
            elif isinstance(analysis, TreatmentPathwaysAnalysis):
                pathway_params = analysis.params
                require_cohort(owner, pathway_params.cohort)
                for code_set in pathway_params.drug_classes:
                    require_code_set(owner, code_set, "drug")
            elif isinstance(analysis, TrajectoryAnalysis):
                trajectory_params = analysis.params
                require_cohort(owner, trajectory_params.cohort)
                require_code_set(owner, trajectory_params.measurement, "measurement")
                if trajectory_params.group_by:
                    require_variable(owner, trajectory_params.group_by)
            elif isinstance(analysis, PatientSignatureAnalysis):
                signature_params = analysis.params
                require_cohort(owner, signature_params.cohort)
                for domain in signature_params.feature_domains:
                    try:
                        _normalized_domain(domain)
                    except ValueError as exc:
                        analysis_reference_errors.append(f"{owner}: {exc}")
            elif isinstance(analysis, CausalEffectAnalysis):
                causal_params = analysis.params
                require_distinct_cohorts(
                    owner, causal_params.exposed_cohort, causal_params.comparator_cohort
                )
                require_code_set(owner, causal_params.outcome.code_set)
                require_adjustments(
                    owner,
                    causal_params.adjustment_set,
                    protected={causal_params.outcome.code_set},
                )

        if analysis_reference_errors:
            raise ValueError(
                "invalid analysis reference(s): " + "; ".join(analysis_reference_errors)
            )

        known_analyses = set(analysis_ids)
        missing_analyses: list[str] = []
        for plot in self.plots:
            analysis_id = getattr(plot.params, "analysis_id", None)
            if analysis_id is not None and analysis_id not in known_analyses:
                missing_analyses.append(f"plot '{plot.id}': {analysis_id}")
        if missing_analyses:
            raise ValueError("unknown analysis reference(s): " + ", ".join(missing_analyses))
        return self


def _require_unique(values: list[str], label: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate {label}(s): {', '.join(duplicates)}")


_DEMOGRAPHIC_VARIABLES = {
    "age",
    "age_bucket",
    "sex",
    "gender",
    "race",
    "ethnicity",
    "gender_concept_id",
    "race_concept_id",
    "ethnicity_concept_id",
}

_DOMAIN_ALIASES = {
    "condition occurrence": "condition",
    "condition_occurrence": "condition",
    "drug exposure": "drug",
    "drug_exposure": "drug",
    "procedure occurrence": "procedure",
    "procedure_occurrence": "procedure",
    "visit occurrence": "visit",
    "visit_occurrence": "visit",
}
_DOMAINS = {"condition", "drug", "procedure", "measurement", "observation", "visit"}


def _normalized_domain(value: str) -> str:
    normalized = _DOMAIN_ALIASES.get(value.strip().lower(), value.strip().lower())
    if normalized not in _DOMAINS:
        raise ValueError(
            f"unsupported domain {value!r}; expected one of {', '.join(sorted(_DOMAINS))}"
        )
    return normalized
