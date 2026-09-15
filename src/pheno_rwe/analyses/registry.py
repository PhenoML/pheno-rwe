"""Lazy registry for deterministic analysis implementations."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pheno_rwe.analyses.base import Analysis, AnalysisContext, AnalysisInputError, AnalysisResult

_REGISTRY: dict[str, Analysis] = {}
_BUILTINS_LOADED = False


def register(analysis: Analysis, *, replace: bool = False) -> Analysis:
    kind = str(getattr(analysis, "kind", "")).strip()
    if not kind:
        raise ValueError("registered analysis must declare a non-empty kind")
    if not replace and kind in _REGISTRY and _REGISTRY[kind] is not analysis:
        raise ValueError(f"analysis kind already registered: {kind}")
    if not callable(getattr(analysis, "run", None)):
        raise TypeError(f"analysis '{kind}' does not implement run(ctx)")
    _REGISTRY[kind] = analysis
    return analysis


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from pheno_rwe.analyses.builtins import BUILTIN_ANALYSES

    for analysis in BUILTIN_ANALYSES:
        register(analysis)
    _BUILTINS_LOADED = True


def get(kind: str) -> Analysis:
    _load_builtins()
    try:
        return _REGISTRY[kind]
    except KeyError as exc:
        choices = ", ".join(sorted(_REGISTRY))
        raise AnalysisInputError(
            f"unknown analysis kind '{kind}'; available kinds: {choices}"
        ) from exc


def kinds() -> tuple[str, ...]:
    _load_builtins()
    return tuple(sorted(_REGISTRY))


def items() -> Iterator[tuple[str, Analysis]]:
    _load_builtins()
    yield from sorted(_REGISTRY.items())


def validate_params(kind: str, params: Any) -> Any:
    analysis = get(kind)
    return analysis.Params.model_validate(params)


def run(ctx: AnalysisContext) -> AnalysisResult:
    kind = ctx.kind
    if not kind:
        raise AnalysisInputError("analysis spec must include 'kind' or 'type'")
    return get(kind).run(ctx)


def clear_registry(*, include_builtins: bool = True) -> None:
    """Testing hook; normal callers should never mutate the global registry."""

    global _BUILTINS_LOADED
    _REGISTRY.clear()
    _BUILTINS_LOADED = not include_builtins


# Descriptive aliases used by adapters and older callers.
get_analysis = get
list_analyses = kinds
run_analysis = run
