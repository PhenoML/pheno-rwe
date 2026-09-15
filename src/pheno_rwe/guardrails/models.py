"""Typed inputs and outputs for deterministic guardrail evaluation."""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, computed_field, model_validator


class GuardrailPhase(StrEnum):
    STATIC = "static"
    DATA = "data"


class Severity(StrEnum):
    INFO = "info"
    WARN = "warn"
    REFUSE = "refuse"


class ExitCode(IntEnum):
    """Process-independent status contract used by CLI adapters."""

    OK = 0
    FATAL = 1
    PARTIAL = 2
    REFUSED = 3
    STALE_PLAN = 4


class GuardrailModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RuleOutcome(GuardrailModel):
    rule_id: str = Field(min_length=1)
    phase: GuardrailPhase
    severity: Severity
    message: str = Field(min_length=1)
    analysis_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def refused(self) -> bool:
        return self.severity is Severity.REFUSE

    @property
    def level(self) -> Severity:
        """Alias useful to renderers that call the outcome field ``level``."""

        return self.severity


class DataSummary(GuardrailModel):
    """Cheap, aggregate-only inputs needed by DATA-phase rules.

    Callers may obtain these values from DuckDB, but evaluation itself remains a
    pure function and cannot access patient-level data.
    """

    analysis_id: str = Field(min_length=1)
    total_n: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("total_n", "n"),
    )
    arm_n: dict[str, int] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("arm_n", "arm_counts"),
    )
    events_total: int | None = Field(default=None, ge=0)
    events_by_arm: dict[str, int] = Field(default_factory=dict)
    model_parameters: int | None = Field(default=None, ge=1)
    unchecked_fraction: float | None = Field(default=None, ge=0, le=1)
    measurement_units: tuple[str, ...] = ()
    median_measurements_per_patient: float | None = Field(default=None, ge=0)
    outcome_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def check_aggregate_consistency(self) -> DataSummary:
        for arm, count in self.arm_n.items():
            if count < 0:
                raise ValueError(f"arm_n['{arm}'] cannot be negative")
        if self.total_n is not None and self.arm_n and sum(self.arm_n.values()) > self.total_n:
            raise ValueError("sum(arm_n) cannot exceed total_n")
        if (
            self.events_total is not None
            and self.total_n is not None
            and self.events_total > self.total_n
        ):
            raise ValueError("events_total cannot exceed total_n")
        for arm, events in self.events_by_arm.items():
            if events < 0:
                raise ValueError(f"events_by_arm['{arm}'] cannot be negative")
            if arm in self.arm_n and events > self.arm_n[arm]:
                raise ValueError(f"events_by_arm['{arm}'] cannot exceed arm_n['{arm}']")
        return self

    @property
    def effective_total_n(self) -> int | None:
        if self.total_n is not None:
            return self.total_n
        return sum(self.arm_n.values()) if self.arm_n else None

    @property
    def effective_events_total(self) -> int | None:
        if self.events_total is not None:
            return self.events_total
        return sum(self.events_by_arm.values()) if self.events_by_arm else None


class ValidationReport(GuardrailModel):
    plan_hash: str
    phases: tuple[GuardrailPhase, ...]
    outcomes: tuple[RuleOutcome, ...] = ()
    power_notes: dict[str, str] = Field(default_factory=dict)

    @computed_field
    @property
    def refused(self) -> bool:
        return any(outcome.refused for outcome in self.outcomes)

    @computed_field
    @property
    def warning_count(self) -> int:
        return sum(outcome.severity is Severity.WARN for outcome in self.outcomes)

    @computed_field
    @property
    def refusal_count(self) -> int:
        return sum(outcome.severity is Severity.REFUSE for outcome in self.outcomes)

    @computed_field
    @property
    def exit_code(self) -> ExitCode:
        return ExitCode.REFUSED if self.refused else ExitCode.OK

    def outcomes_for(self, analysis_id: str) -> tuple[RuleOutcome, ...]:
        return tuple(outcome for outcome in self.outcomes if outcome.analysis_id == analysis_id)


def exit_code_for(
    outcomes: list[RuleOutcome] | tuple[RuleOutcome, ...] = (),
    *,
    fatal: bool = False,
    partial: bool = False,
    stale_plan: bool = False,
) -> ExitCode:
    """Resolve the documented exit status without calling ``sys.exit``.

    Stale pre-registration and guardrail refusal are distinct, auditable states;
    adapters remain responsible for turning the returned enum into a process exit.
    """

    if stale_plan:
        return ExitCode.STALE_PLAN
    if any(outcome.severity is Severity.REFUSE for outcome in outcomes):
        return ExitCode.REFUSED
    if fatal:
        return ExitCode.FATAL
    if partial:
        return ExitCode.PARTIAL
    return ExitCode.OK
