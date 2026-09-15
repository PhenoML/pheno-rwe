"""Seeded SVD/UMAP patient signatures with density clusters in SVD space."""

from __future__ import annotations

import itertools
from typing import Any

from pydantic import Field

from pheno_rwe.analyses.base import (
    AnalysisContext,
    AnalysisInputError,
    AnalysisResult,
    OptionalDependencyError,
    package_versions,
)
from pheno_rwe.analyses.data import DataSourceParams, load_frame, require_columns
from pheno_rwe.analyses.stats import benjamini_hochberg


class PatientSignatureParams(DataSourceParams):
    cohort: str | None = None
    patient_column: str = "person_id"
    feature_columns: list[str] = Field(default_factory=list)
    feature_domains: list[str] = Field(default_factory=list)
    n_components: int = Field(default=30, ge=1, le=100)
    n_neighbors: int | None = Field(default=None, ge=2)
    min_cluster_size: int = Field(default=5, ge=2)
    stability_runs: int = Field(default=3, ge=2, le=20)
    characterization_data: Any | None = Field(default=None, exclude=True)
    characterization_columns: list[str] = Field(default_factory=list)
    feature_provenance: list[dict[str, Any]] = Field(default_factory=list)
    dropped_features: list[dict[str, Any]] = Field(default_factory=list)
    matrix_hash: str | None = None
    feature_seed: int | None = None


def _dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import numpy as np
        import umap
        from sklearn.cluster import HDBSCAN
        from sklearn.decomposition import TruncatedSVD
        from sklearn.metrics import adjusted_rand_score
        from sklearn.preprocessing import RobustScaler
    except ImportError as exc:  # pragma: no cover
        package = "umap-learn" if "umap" in str(exc).lower() else "scikit-learn>=1.4"
        raise OptionalDependencyError(package, "patient_signature") from exc
    return np, umap, HDBSCAN, TruncatedSVD, (adjusted_rand_score, RobustScaler)


def _cluster(svd_values: Any, hdbscan_class: Any, min_cluster_size: int) -> Any:
    model = hdbscan_class(
        min_cluster_size=min_cluster_size,
        allow_single_cluster=False,
        copy=True,
    )
    return model.fit_predict(svd_values)


class PatientSignatureAnalysis:
    kind = "patient_signature"
    Params = PatientSignatureParams
    requires = ("patient feature matrix", "seed")
    rules = ("UMAP-MIN-N", "UNCHECKED-FRACTION")

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        import pandas as pd

        params = self.Params.model_validate(ctx.params)
        frame = load_frame(ctx, params)
        require_columns(frame, [params.patient_column, *params.feature_columns])
        if (
            frame[params.patient_column].isna().any()
            or frame[params.patient_column].duplicated().any()
        ):
            raise AnalysisInputError("patient_signature requires one non-missing row per patient")
        feature_columns = list(params.feature_columns)
        if not feature_columns:
            feature_columns = [
                str(column)
                for column in frame.select_dtypes(include="number").columns
                if column != params.patient_column
            ]
        if not feature_columns:
            raise AnalysisInputError("patient_signature found no numeric feature columns")
        values = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
        if values.isna().any().any():
            bad = values.columns[values.isna().any()].tolist()
            raise AnalysisInputError(
                f"patient_signature features must be complete numeric values: {bad}"
            )
        n_patients, n_features = values.shape
        if n_patients < 3:
            raise AnalysisInputError("patient_signature requires at least three patients")
        if params.min_cluster_size > n_patients:
            raise AnalysisInputError("min_cluster_size cannot exceed patient count")
        if params.feature_seed is not None and params.feature_seed != ctx.seed:
            raise AnalysisInputError(
                "feature-matrix seed does not match the registered analysis seed"
            )

        characterization_columns = list(params.characterization_columns)
        if params.characterization_data is None:
            characterization_frame: Any = frame[[params.patient_column, *feature_columns]].copy()
            characterization_columns = feature_columns
        else:
            characterization_frame = (
                params.characterization_data.copy()
                if isinstance(params.characterization_data, pd.DataFrame)
                else pd.DataFrame(params.characterization_data)
            )
            require_columns(
                characterization_frame,
                [params.patient_column, *characterization_columns],
            )
        if (
            characterization_frame[params.patient_column].isna().any()
            or characterization_frame[params.patient_column].duplicated().any()
        ):
            raise AnalysisInputError(
                "cluster characterization requires one non-missing row per patient"
            )
        characterization_frame = characterization_frame.set_index(params.patient_column).reindex(
            frame[params.patient_column]
        )
        if characterization_frame.index.isna().any():
            raise AnalysisInputError("cluster characterization is missing patients")
        characterization_values = characterization_frame[characterization_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        if characterization_values.isna().any().any():
            raise AnalysisInputError("cluster characterization features must be complete")
        if characterization_columns:
            observed_characterization = set(
                characterization_values.to_numpy(dtype=float).ravel().tolist()
            )
            if not observed_characterization.issubset({0.0, 1.0}):
                raise AnalysisInputError(
                    "cluster characterization features must be binary presence indicators"
                )

        np, umap_module, hdbscan_class, svd_class, helpers = _dependencies()
        adjusted_rand_score, robust_scaler = helpers
        scaled = robust_scaler().fit_transform(values.to_numpy(dtype=float))
        components = min(params.n_components, max(1, n_patients - 1), n_features)
        svd_model = svd_class(n_components=components, random_state=ctx.seed)
        svd_values = svd_model.fit_transform(scaled)
        n_neighbors = min(params.n_neighbors or 15, n_patients - 1)
        n_neighbors = max(2, n_neighbors)
        reducer = umap_module.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            metric="euclidean",
            random_state=ctx.seed,
            transform_seed=ctx.seed,
            n_jobs=1,
        )
        embedding = reducer.fit_transform(svd_values)
        labels = _cluster(svd_values, hdbscan_class, params.min_cluster_size)

        stability_labels = [labels]
        for offset in range(1, params.stability_runs):
            alternate_svd = svd_class(n_components=components, random_state=ctx.seed + offset)
            alternate_values = alternate_svd.fit_transform(scaled)
            stability_labels.append(
                _cluster(alternate_values, hdbscan_class, params.min_cluster_size)
            )
        ari_values = [
            float(adjusted_rand_score(first, second))
            for first, second in itertools.combinations(stability_labels, 2)
        ]
        stability_ari = float(np.mean(ari_values)) if ari_values else 1.0

        patients = frame[params.patient_column].astype(str).tolist()
        embedding_rows = [
            {
                "person_id": patients[index],
                "umap_1": float(embedding[index, 0]),
                "umap_2": float(embedding[index, 1]),
                "cluster": int(labels[index]),
            }
            for index in range(n_patients)
        ]
        medoid_rows: list[dict[str, Any]] = []
        for cluster in sorted(set(labels) - {-1}):
            member_indexes = np.flatnonzero(labels == cluster)
            centroid = svd_values[member_indexes].mean(axis=0)
            distance = ((svd_values[member_indexes] - centroid) ** 2).sum(axis=1)
            chosen = int(member_indexes[int(np.argmin(distance))])
            medoid_rows.append(
                {
                    "cluster": int(cluster),
                    "person_id": patients[chosen],
                    "cluster_size": int(len(member_indexes)),
                }
            )

        try:
            from scipy.stats import fisher_exact
        except ImportError as exc:  # pragma: no cover
            raise OptionalDependencyError("scipy", "patient cluster characterization") from exc
        characterization_rows: list[dict[str, Any]] = []
        raw = characterization_values.to_numpy(dtype=float)
        provenance_by_feature: dict[str, dict[str, float]] = {}
        for item in params.feature_provenance:
            feature = str(item.get("feature") or "")
            status = str(item.get("mapping_status") or "").upper()
            if feature and status:
                provenance_by_feature.setdefault(feature, {})[status] = float(
                    item.get("fraction") or 0.0
                )
        for cluster in sorted(set(labels) - {-1}):
            member = labels == cluster
            cluster_rows: list[dict[str, Any]] = []
            p_values: list[float] = []
            for feature_index, feature in enumerate(characterization_columns):
                present = raw[:, feature_index] != 0
                table = [
                    [int((member & present).sum()), int((member & ~present).sum())],
                    [int((~member & present).sum()), int((~member & ~present).sum())],
                ]
                result: Any = fisher_exact(table)
                row = {
                    "cluster": int(cluster),
                    "feature": feature,
                    "prevalence_cluster": float(present[member].mean()),
                    "prevalence_other": float(present[~member].mean()),
                    "odds_ratio": float(result.statistic),
                    "p_value": float(result.pvalue),
                }
                mix = provenance_by_feature.get(feature, {})
                for status in ("ALREADY_STANDARD", "MAPPED", "UNCHECKED", "UNMAPPED"):
                    row[f"mapping_{status.lower()}_fraction"] = mix.get(status)
                cluster_rows.append(row)
                p_values.append(float(result.pvalue))
            for row, q_value in zip(cluster_rows, benjamini_hochberg(p_values), strict=True):
                row["q_value"] = q_value
            characterization_rows.extend(cluster_rows)

        counts = pd.Series(labels).value_counts().sort_index()
        result_provenance = dict(ctx.provenance) or {"available": False}
        result_provenance["feature_matrix"] = {
            "matrix_hash": params.matrix_hash,
            "mapping_rows": len(params.feature_provenance),
        }
        return AnalysisResult(
            analysis_id=ctx.analysis_id,
            kind=self.kind,
            n={
                "total": n_patients,
                "per_group": {str(cluster): int(count) for cluster, count in counts.items()},
                "excluded": 0,
            },
            estimates=[],
            provenance=result_provenance,
            assumptions_checked=[
                {"name": "umap_sample_size", "passed": n_patients >= 30, "n": n_patients},
                {"name": "clustering_space", "passed": True, "space": "svd"},
                {
                    "name": "stability",
                    "passed": stability_ari >= 0.7,
                    "stability_ari": stability_ari,
                },
            ],
            guardrail_outcomes=[dict(item) for item in ctx.guardrail_outcomes],
            power_note=(
                "Patient signatures are hypothesis-generating; no inferential power claim is made."
            ),
            seed=ctx.seed,
            package_versions=package_versions(("scikit-learn", "umap-learn", "scipy")),
            tables={
                "data": embedding_rows,
                "characterization": characterization_rows,
                "medoids": medoid_rows,
                "feature_provenance": params.feature_provenance,
            },
            metadata={
                "interpretation": "hypothesis_generating",
                "matrix_hash": params.matrix_hash,
                "feature_seed": params.feature_seed,
                "dropped_features": params.dropped_features,
                "svd_components": components,
                "explained_variance_ratio": float(svd_model.explained_variance_ratio_.sum()),
                "n_neighbors": n_neighbors,
                "stability_ari": stability_ari,
                "stability_runs": params.stability_runs,
            },
            output_dir=ctx.output_dir,
        )


ANALYSIS = PatientSignatureAnalysis()
