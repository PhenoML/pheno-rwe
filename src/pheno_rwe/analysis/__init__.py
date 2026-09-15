"""Compatibility facade for the original singular ``analysis`` package name."""

from pheno_rwe.analyses import *  # noqa: F403
from pheno_rwe.analyses.registry import get_analysis, list_analyses, run_analysis

__all__ = [  # noqa: F405
    "Analysis",
    "AnalysisContext",
    "AnalysisError",
    "AnalysisInputError",
    "AnalysisResult",
    "OptionalDependencyError",
    "get",
    "get_analysis",
    "kinds",
    "list_analyses",
    "register",
    "prepare_context_spec",
    "run",
    "run_analysis",
]
