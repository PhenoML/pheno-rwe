"""Small deterministic statistical helpers shared across analysis kinds."""

from __future__ import annotations

import math
from collections.abc import Iterable
from statistics import NormalDist


def benjamini_hochberg(p_values: Iterable[float | None]) -> list[float | None]:
    values = list(p_values)
    indexed = [
        (index, float(value))
        for index, value in enumerate(values)
        if value is not None and math.isfinite(value)
    ]
    indexed.sort(key=lambda item: item[1])
    adjusted: dict[int, float] = {}
    running = 1.0
    total = len(indexed)
    for reverse_rank, (index, value) in enumerate(reversed(indexed), 1):
        rank = total - reverse_rank + 1
        running = min(running, value * total / rank)
        adjusted[index] = min(1.0, running)
    return [adjusted.get(index) for index in range(len(values))]


def wilson_interval(successes: float, total: float, alpha: float = 0.05) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    proportion = successes / total
    z = NormalDist().inv_cdf(1 - alpha / 2)
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def newcombe_difference_interval(
    events_a: int,
    n_a: int,
    events_b: int,
    n_b: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Newcombe score interval for two independent proportions (method 10)."""

    p_a = events_a / n_a
    p_b = events_b / n_b
    lower_a, upper_a = wilson_interval(events_a, n_a, alpha)
    lower_b, upper_b = wilson_interval(events_b, n_b, alpha)
    difference = p_a - p_b
    lower = difference - math.sqrt((p_a - lower_a) ** 2 + (upper_b - p_b) ** 2)
    upper = difference + math.sqrt((upper_a - p_a) ** 2 + (p_b - lower_b) ** 2)
    return max(-1.0, lower), min(1.0, upper)


def standardized_mean_difference(a: object, b: object) -> float | None:
    import numpy as np

    first = np.asarray(a, dtype=float)
    second = np.asarray(b, dtype=float)
    first = first[np.isfinite(first)]
    second = second[np.isfinite(second)]
    if not len(first) or not len(second):
        return None
    variance = (
        (float(np.var(first, ddof=1)) + float(np.var(second, ddof=1))) / 2
        if len(first) > 1 and len(second) > 1
        else 0
    )
    if variance <= 0:
        return 0.0 if float(np.mean(first)) == float(np.mean(second)) else None
    return (float(np.mean(first)) - float(np.mean(second))) / math.sqrt(variance)


def weighted_smd(values: object, treatment: object, weights: object | None = None) -> float | None:
    import numpy as np

    x = np.asarray(values, dtype=float)
    t = np.asarray(treatment, dtype=int)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w)
    x, t, w = x[mask], t[mask], w[mask]
    if not ((t == 0).any() and (t == 1).any()):
        return None

    def moments(group: int) -> tuple[float, float]:
        selected = t == group
        group_x, group_w = x[selected], w[selected]
        mean = float(np.average(group_x, weights=group_w))
        variance = float(np.average((group_x - mean) ** 2, weights=group_w))
        return mean, variance

    mean_1, var_1 = moments(1)
    mean_0, var_0 = moments(0)
    pooled = math.sqrt((var_1 + var_0) / 2)
    if pooled == 0:
        return 0.0 if mean_1 == mean_0 else None
    return (mean_1 - mean_0) / pooled


def e_value(risk_ratio: float, confidence_limit: float | None = None) -> tuple[float, float | None]:
    """VanderWeele-Ding E-value for a risk-ratio-scale estimate."""

    if not math.isfinite(risk_ratio) or risk_ratio <= 0:
        return math.nan, None
    rr = risk_ratio if risk_ratio >= 1 else 1 / risk_ratio
    estimate = rr + math.sqrt(rr * (rr - 1))
    if confidence_limit is None or not math.isfinite(confidence_limit) or confidence_limit <= 0:
        return estimate, None
    limit = confidence_limit
    if risk_ratio < 1:
        limit = 1 / limit
    if limit < 1:
        return estimate, 1.0
    return estimate, limit + math.sqrt(limit * (limit - 1))
