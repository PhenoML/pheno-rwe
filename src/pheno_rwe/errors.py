"""Typed errors shared by library and CLI adapters."""

from __future__ import annotations


class PhenoRWEError(Exception):
    """Base class for expected engine failures."""


class StudyNotFound(PhenoRWEError):
    pass


class InvalidStudy(PhenoRWEError):
    pass


class StudyLocked(PhenoRWEError):
    pass


class ConfigurationError(PhenoRWEError):
    pass


class ConceptResolverUnavailable(PhenoRWEError):
    pass


class GuardrailRefusal(PhenoRWEError):
    """A deterministic rule refused execution."""

    def __init__(self, message: str, outcomes: list[object] | None = None) -> None:
        super().__init__(message)
        self.outcomes = outcomes or []


class StalePlanError(PhenoRWEError):
    pass


class StepPartialError(PhenoRWEError):
    pass


class CancelledError(PhenoRWEError):
    pass
