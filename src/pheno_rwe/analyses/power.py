"""Precision-based minimum detectable effect helpers (never post-hoc power)."""

from __future__ import annotations

import math
from statistics import NormalDist


def _z(alpha: float, power: float) -> float:
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must lie between 0 and 1")
    return NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)


def binary_risk_difference_mde(
    n_exposed: int,
    n_comparator: int,
    *,
    baseline_risk: float = 0.5,
    alpha: float = 0.05,
    target_power: float = 0.8,
) -> float:
    if n_exposed <= 0 or n_comparator <= 0:
        return math.inf
    variance = baseline_risk * (1 - baseline_risk) * (1 / n_exposed + 1 / n_comparator)
    return min(1.0, _z(alpha, target_power) * math.sqrt(variance))


def continuous_standardized_mde(
    n_exposed: int,
    n_comparator: int,
    *,
    alpha: float = 0.05,
    target_power: float = 0.8,
) -> float:
    if n_exposed <= 0 or n_comparator <= 0:
        return math.inf
    return _z(alpha, target_power) * math.sqrt(1 / n_exposed + 1 / n_comparator)


def survival_log_hazard_mde(
    events_exposed: int,
    events_comparator: int,
    *,
    alpha: float = 0.05,
    target_power: float = 0.8,
) -> float:
    if events_exposed <= 0 or events_comparator <= 0:
        return math.inf
    return _z(alpha, target_power) * math.sqrt(1 / events_exposed + 1 / events_comparator)


def comparative_power_note(n_exposed: int, n_comparator: int, outcome_type: str = "binary") -> str:
    if outcome_type == "continuous":
        value = continuous_standardized_mde(n_exposed, n_comparator)
        return (
            "Design-stage precision: with the observed arm sizes, an 80%-power, two-sided "
            f"alpha=0.05 design detects an approximately {value:.3f}-SD difference. "
            "This is a minimum detectable effect, not post-hoc power."
        )
    value = binary_risk_difference_mde(n_exposed, n_comparator)
    return (
        "Design-stage precision: under a conservative 50% baseline risk, an 80%-power, "
        f"two-sided alpha=0.05 design detects an approximately {value:.3f} absolute risk "
        "difference. This is a minimum detectable effect, not post-hoc power."
    )
