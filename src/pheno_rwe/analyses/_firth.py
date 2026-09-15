"""Compact Firth bias-reduced logistic regression.

This implements maximization of the Jeffreys-prior penalized log likelihood.
It intentionally exposes a very small API so it can be replaced by a dedicated
package without changing result contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pheno_rwe.analyses.base import AnalysisInputError, OptionalDependencyError


@dataclass(frozen=True, slots=True)
class FirthFit:
    coefficients: Any
    standard_errors: Any
    converged: bool
    iterations: int


def fit_firth_logistic(x: Any, y: Any, *, max_iter: int = 500, tolerance: float = 1e-8) -> FirthFit:
    try:
        import numpy as np
        from scipy.optimize import minimize
        from scipy.special import expit
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("scipy", "Firth logistic regression") from exc

    design = np.asarray(x, dtype=float)
    outcome = np.asarray(y, dtype=float)
    if design.ndim != 2 or outcome.ndim != 1 or design.shape[0] != outcome.size:
        raise AnalysisInputError("invalid Firth logistic design dimensions")
    if design.shape[0] <= design.shape[1]:
        raise AnalysisInputError(
            "Firth logistic regression needs more complete rows than parameters"
        )
    if set(np.unique(outcome)) - {0.0, 1.0}:
        raise AnalysisInputError("Firth logistic outcome must be binary")
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise AnalysisInputError("Firth logistic design matrix is rank deficient")

    def objective(beta: Any) -> float:
        eta = design @ beta
        probability = expit(eta)
        log_likelihood = float(
            np.sum(outcome * np.logaddexp(0, -eta) * -1 + (1 - outcome) * np.logaddexp(0, eta) * -1)
        )
        weights = np.clip(probability * (1 - probability), 1e-12, None)
        information = design.T @ (design * weights[:, None])
        sign, log_determinant = np.linalg.slogdet(information)
        if sign <= 0:
            return float("inf")
        return -(log_likelihood + 0.5 * log_determinant)

    result = minimize(
        objective,
        np.zeros(design.shape[1], dtype=float),
        method="BFGS",
        options={"maxiter": max_iter, "gtol": tolerance},
    )
    beta = np.asarray(result.x, dtype=float)
    probability = expit(design @ beta)
    weights = np.clip(probability * (1 - probability), 1e-12, None)
    covariance = np.linalg.pinv(design.T @ (design * weights[:, None]))
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
    # SciPy occasionally reports precision loss after reaching a stable optimum;
    # the explicit finite check keeps that behavior auditable without discarding it.
    converged = bool(
        result.success or (np.isfinite(beta).all() and float(np.linalg.norm(result.jac)) < 1e-4)
    )
    return FirthFit(beta, standard_errors, converged, int(result.nit))
