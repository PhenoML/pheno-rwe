"""Compatibility facade for plan validation and stale-registration checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pheno_rwe.guardrails.engine import validate_plan_static
from pheno_rwe.guardrails.models import ExitCode, ValidationReport

from .io import is_plan_current, plan_hash
from .schema import Plan


def validation_exit_code(report: ValidationReport) -> ExitCode:
    """Map a validation report to the documented CLI-compatible exit code."""

    return report.exit_code


def current_plan_exit_code(plan: Plan | Mapping[str, Any], validated_hash: str | None) -> ExitCode:
    """Return ``STALE_PLAN`` when analysis is not bound to this exact plan."""

    return ExitCode.OK if is_plan_current(plan, validated_hash) else ExitCode.STALE_PLAN


__all__ = [
    "current_plan_exit_code",
    "is_plan_current",
    "plan_hash",
    "validate_plan_static",
    "validation_exit_code",
]
