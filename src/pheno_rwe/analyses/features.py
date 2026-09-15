"""Deterministic patient-feature matrix construction with provenance."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from pheno_rwe.analyses.base import AnalysisInputError
from pheno_rwe.hashing import hash_json


def derive_analysis_seed(plan_seed: int | str, analysis_id: str) -> int:
    if not analysis_id:
        raise AnalysisInputError("analysis_id must be non-empty when deriving a seed")
    return int(hash_json({"seed": plan_seed, "analysis_id": analysis_id})[:8], 16)


@dataclass(slots=True)
class FeatureMatrixResult:
    matrix: Any
    patients: Any
    feature_provenance: Any
    presence_matrix: Any
    dropped_features: list[dict[str, Any]] = field(default_factory=list)
    matrix_hash: str = ""
    seed: int = 0

    def model_dump(self) -> dict[str, Any]:
        return {
            "matrix_hash": self.matrix_hash,
            "seed": self.seed,
            "patient_count": int(len(self.matrix)),
            "feature_count": int(len(self.matrix.columns)),
            "features": [str(column) for column in self.matrix.columns],
            "dropped_features": self.dropped_features,
            "feature_provenance": self.feature_provenance.to_dict(orient="records"),
        }


def _window_tuple(value: Any) -> tuple[int, int]:
    if hasattr(value, "start_day") and hasattr(value, "end_day"):
        return int(value.start_day), int(value.end_day)
    if isinstance(value, Mapping):
        return int(value.get("start_day", value.get("start"))), int(
            value.get("end_day", value.get("end"))
        )
    start, end = value
    return int(start), int(end)


def _code_set_indexes(code_sets: Mapping[str, Any] | None) -> tuple[dict[int, str], dict[str, str]]:
    concepts: dict[int, str] = {}
    sources: dict[str, str] = {}
    for name, definition in sorted((code_sets or {}).items()):
        if hasattr(definition, "model_dump"):
            definition = definition.model_dump()
        if isinstance(definition, Mapping):
            codings = definition.get("codings", [])
            concept_ids = list(definition.get("concept_ids", []))
            source_values = list(definition.get("source_values", []))
            for coding in codings:
                if hasattr(coding, "model_dump"):
                    coding = coding.model_dump()
                concept_id = coding.get("concept_id")
                if concept_id:
                    concept_ids.append(concept_id)
                source = coding.get("source_value")
                if not source and coding.get("code"):
                    source = f"{coding.get('system', '')}#{coding['code']}"
                if source:
                    source_values.append(source)
        else:
            concept_ids, source_values = list(definition), []
        for concept_id in concept_ids:
            if int(concept_id) > 0:
                concepts.setdefault(int(concept_id), str(name))
        for source in source_values:
            sources.setdefault(str(source), str(name))
    return concepts, sources


def _event_key(row: Any, concept_sets: dict[int, str], source_sets: dict[str, str]) -> str | None:
    concept_id = (
        int(row.concept_id)
        if row.concept_id is not None and not math.isnan(float(row.concept_id))
        else 0
    )
    source_value = None if row.source_value is None else str(row.source_value)
    if concept_id > 0 and concept_id in concept_sets:
        return f"codeset:{concept_sets[concept_id]}"
    if source_value and source_value in source_sets:
        return f"codeset:{source_sets[source_value]}"
    if concept_id > 0:
        return f"concept:{concept_id}"
    if source_value and source_value.lower() not in {"nan", "none", ""}:
        return f"source:{source_value}"
    return None


def build_feature_matrix(
    events: Any,
    cohort: Any,
    *,
    analysis_id: str,
    plan_seed: int | str,
    windows: Mapping[str, tuple[int, int] | Mapping[str, int] | Any] | None = None,
    code_sets: Mapping[str, Any] | None = None,
    person_column: str = "person_id",
    index_date_column: str = "index_date",
    event_date_column: str = "event_date",
    concept_column: str = "concept_id",
    source_value_column: str = "source_value",
    domain_column: str = "domain",
    mapping_status_column: str = "mapping_status",
    prevalence_floor: int | None = None,
    include_presence: bool = True,
    include_counts: bool = True,
    demographic_columns: list[str] | None = None,
) -> FeatureMatrixResult:
    """Build a robust-scaled matrix without discarding concept_id=0 rows.

    Events with ``concept_id=0`` use ``source_value`` as their stable key.  The
    mapping-status mixture of every retained feature is returned separately.
    """

    import numpy as np
    import pandas as pd

    event_frame = events.copy() if isinstance(events, pd.DataFrame) else pd.DataFrame(events)
    cohort_frame = cohort.copy() if isinstance(cohort, pd.DataFrame) else pd.DataFrame(cohort)
    event_required = [person_column, event_date_column, concept_column, source_value_column]
    cohort_required = [person_column, index_date_column]
    missing = [column for column in event_required if column not in event_frame]
    missing += [column for column in cohort_required if column not in cohort_frame]
    if missing:
        raise AnalysisInputError(
            f"feature matrix is missing required columns: {sorted(set(missing))}"
        )
    duplicated: Any = cohort_frame[person_column].duplicated()
    if bool(duplicated.any()):
        raise AnalysisInputError("cohort must contain exactly one index date per patient")
    if not include_presence and not include_counts:
        raise AnalysisInputError("feature matrix must include presence, counts, or both")
    cohort_frame[index_date_column] = pd.to_datetime(
        cohort_frame[index_date_column], errors="coerce"
    )
    event_frame[event_date_column] = pd.to_datetime(event_frame[event_date_column], errors="coerce")
    cohort_index_dates: Any = cohort_frame[index_date_column]
    if bool(cohort_index_dates.isna().any()):
        raise AnalysisInputError("cohort index dates must be parseable")
    event_frame = event_frame.merge(
        cohort_frame[[person_column, index_date_column]],
        on=person_column,
        how="inner",
        validate="many_to_one",
    )
    event_frame = event_frame.dropna(subset=[event_date_column])
    event_frame["_relative_day"] = (
        event_frame[event_date_column] - event_frame[index_date_column]
    ).dt.total_seconds() / 86_400
    if domain_column not in event_frame:
        event_frame[domain_column] = "unknown"
    if mapping_status_column not in event_frame:
        event_frame[mapping_status_column] = "UNKNOWN"
    event_frame = event_frame.rename(
        columns={concept_column: "concept_id", source_value_column: "source_value"}
    )
    concept_sets, source_sets = _code_set_indexes(code_sets)
    event_frame["_key"] = [
        _event_key(row, concept_sets, source_sets)
        for row in event_frame[["concept_id", "source_value"]].itertuples(index=False)
    ]
    missing_key_count = int(event_frame["_key"].isna().sum())
    event_frame = event_frame.dropna(subset=["_key"])
    window_values = windows or {"baseline": (-365, -1), "post": (0, 90)}
    normalized_windows = {name: _window_tuple(value) for name, value in window_values.items()}
    for name, (start, end) in normalized_windows.items():
        if start > end:
            raise AnalysisInputError(f"feature window '{name}' has start after end")

    patients = cohort_frame[person_column].tolist()
    raw = pd.DataFrame(index=pd.Index(patients, name=person_column), dtype=float)
    provenance_rows: list[dict[str, Any]] = []
    candidate_prevalence: dict[str, int] = {}
    aggregate_columns: list[str] = []
    for window_name, (start, end) in normalized_windows.items():
        subset = event_frame.loc[
            event_frame["_relative_day"].between(start, end, inclusive="both")
        ].copy()
        if subset.empty:
            continue
        subset["_base"] = (
            window_name
            + "|"
            + subset[domain_column].fillna("unknown").astype(str)
            + "|"
            + subset["_key"].astype(str)
        )
        counts = (
            subset.groupby([person_column, "_base"], sort=True).size().rename("count").reset_index()
        )
        for base in sorted(counts["_base"].unique()):
            base_counts = counts.loc[counts["_base"] == base].set_index(person_column)["count"]
            prevalence = int((base_counts > 0).sum())
            candidate_prevalence[base] = prevalence
            final_names: list[str] = []
            if include_presence:
                name = f"{base}|presence"
                raw[name] = base_counts.reindex(raw.index).fillna(0).gt(0).astype(float)
                final_names.append(name)
            if include_counts:
                name = f"{base}|log1p_count"
                raw[name] = np.log1p(base_counts.reindex(raw.index).fillna(0).astype(float))
                final_names.append(name)
            parts = base.split("|", 2)
            feature_events = subset.loc[subset["_base"] == base]
            statuses = (
                feature_events[mapping_status_column]
                .fillna("UNKNOWN")
                .astype(str)
                .str.upper()
                .value_counts()
            )
            for name in final_names:
                for status, count in statuses.sort_index().items():
                    provenance_rows.append(
                        {
                            "feature": name,
                            "mapping_status": str(status),
                            "rows": int(count),
                            "fraction": float(count / len(feature_events)),
                            "domain": parts[1],
                        }
                    )

    # Numeric measurement summaries add clinically useful magnitude and trend
    # while preserving the same source-value fallback and provenance contract.
    if "value" in event_frame:
        measurements = event_frame.loc[
            event_frame[domain_column].astype(str).str.lower().eq("measurement")
            & (event_frame["_relative_day"] < 0)
        ].copy()
        measurements["_value"] = pd.to_numeric(measurements["value"], errors="coerce")
        measurements = measurements.dropna(subset=["_value"])
        for key in sorted(measurements["_key"].unique(), key=str):
            feature_events = measurements.loc[measurements["_key"] == key].copy()
            base = f"baseline|measurement|{key}"
            grouped = feature_events.groupby(person_column, sort=True)
            latest = grouped.apply(
                lambda group: group.sort_values("_relative_day", kind="mergesort").iloc[-1][
                    "_value"
                ]
            )
            median = grouped["_value"].median()

            def patient_slope(group: Any) -> float:
                if len(group) < 3 or group["_relative_day"].nunique() < 2:
                    return math.nan
                return float(
                    np.polyfit(
                        group["_relative_day"].to_numpy(dtype=float),
                        group["_value"].to_numpy(dtype=float),
                        1,
                    )[0]
                )

            slope = grouped.apply(patient_slope)
            aggregates = {
                f"{base}|latest_before_index": latest,
                f"{base}|median": median,
            }
            if int(slope.notna().sum()) >= (
                prevalence_floor
                if prevalence_floor is not None
                else max(3, math.ceil(0.02 * len(patients)))
            ):
                aggregates[f"{base}|slope"] = slope
            for name, values in aggregates.items():
                raw[name] = values.reindex(raw.index)
                aggregate_columns.append(name)
            candidate_prevalence[base] = int(latest.notna().sum())
            statuses = (
                feature_events[mapping_status_column]
                .fillna("UNKNOWN")
                .astype(str)
                .str.upper()
                .value_counts()
            )
            for name in aggregates:
                for status, count in statuses.sort_index().items():
                    provenance_rows.append(
                        {
                            "feature": name,
                            "mapping_status": str(status),
                            "rows": int(count),
                            "fraction": float(count / len(feature_events)),
                            "domain": "measurement",
                        }
                    )

    minimum = (
        prevalence_floor
        if prevalence_floor is not None
        else max(3, math.ceil(0.02 * len(patients)))
    )
    if minimum < 1:
        raise AnalysisInputError("prevalence_floor must be at least one")
    dropped: list[dict[str, Any]] = []
    for base, prevalence in candidate_prevalence.items():
        if prevalence < minimum:
            matching = [column for column in raw if column.startswith(f"{base}|")]
            raw = raw.drop(columns=matching)
            dropped.append(
                {
                    "feature": base,
                    "reason": "prevalence_below_floor",
                    "prevalence": prevalence,
                    "floor": minimum,
                }
            )
    if missing_key_count:
        dropped.append(
            {
                "feature": None,
                "reason": "missing_concept_and_source_value",
                "rows": missing_key_count,
            }
        )

    demographics = demographic_columns
    if demographics is None:
        demographics = [
            column
            for column in (
                "age",
                "age_bucket",
                "gender_concept_id",
                "race_concept_id",
                "ethnicity_concept_id",
            )
            if column in cohort_frame
        ]
    missing_demographics = [column for column in demographics if column not in cohort_frame]
    if missing_demographics:
        raise AnalysisInputError(f"demographic columns not found: {missing_demographics}")
    if demographics:
        demo = cohort_frame.set_index(person_column)[demographics].reindex(raw.index)
        numeric = demo.select_dtypes(include="number").columns.tolist()
        categorical = [column for column in demographics if column not in numeric]
        for column in numeric:
            values: Any = pd.to_numeric(demo[column], errors="coerce")
            raw[f"demographic|{column}"] = values.fillna(values.median()).astype(float)
        if categorical:
            categorical_demo: Any = demo[categorical]
            encoded = pd.get_dummies(
                categorical_demo.fillna("Missing"),
                prefix=[f"demographic|{column}" for column in categorical],
                dtype=float,
            )
            raw = raw.join(encoded)

    for column in aggregate_columns:
        if column in raw:
            aggregate_values: Any = raw[column]
            raw[column] = aggregate_values.fillna(aggregate_values.median())
    raw = raw.fillna(0.0).sort_index(axis=1)
    presence_columns = [column for column in raw if column.endswith("|presence")]
    presence_matrix = raw[presence_columns].copy()
    scaled = raw.copy()
    for column in scaled:
        median = float(scaled[column].median())
        q1 = float(scaled[column].quantile(0.25))
        q3 = float(scaled[column].quantile(0.75))
        iqr = q3 - q1
        scaled[column] = (scaled[column] - median) / iqr if iqr > 0 else scaled[column] - median
    scaled = scaled.sort_index()
    stable_rows = [
        {
            person_column: str(index),
            **{column: round(float(value), 12) for column, value in row.items()},
        }
        for index, row in scaled.iterrows()
    ]
    matrix_hash = hash_json({"columns": list(scaled.columns), "rows": stable_rows})
    provenance = cast(Any, pd.DataFrame)(
        provenance_rows,
        columns=["feature", "mapping_status", "rows", "fraction", "domain"],
    )
    if not provenance.empty:
        provenance = provenance.loc[provenance["feature"].isin(list(scaled.columns))].sort_values(
            ["feature", "mapping_status"], kind="mergesort"
        )
    return FeatureMatrixResult(
        matrix=scaled,
        patients=cohort_frame.set_index(person_column).reindex(scaled.index),
        feature_provenance=provenance,
        presence_matrix=presence_matrix,
        dropped_features=sorted(dropped, key=lambda row: str(row.get("feature"))),
        matrix_hash=matrix_hash,
        seed=derive_analysis_seed(plan_seed, analysis_id),
    )
