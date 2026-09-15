"""OHDSI-style drug eras and exhaustive first/second/third-line pathways."""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from pheno_rwe.analyses.base import (
    AnalysisContext,
    AnalysisInputError,
    AnalysisResult,
    package_versions,
)
from pheno_rwe.analyses.data import DataSourceParams, load_frame, require_columns


class TreatmentPathwaysParams(DataSourceParams):
    cohort: str | None = None
    patient_column: str = "person_id"
    drug_class_column: str = "drug_class"
    drug_code_column: str | None = None
    start_column: str = "start_date"
    end_column: str | None = "end_date"
    drug_classes: list[str] = Field(default_factory=list)
    class_mapping: dict[str, list[Any]] = Field(default_factory=dict)
    permissible_gap_days: int = Field(default=30, ge=0)
    gap_days: int | None = Field(default=None, ge=0)
    max_lines: int = Field(default=3, ge=1, le=20)

    @model_validator(mode="after")
    def normalize_gap(self) -> TreatmentPathwaysParams:
        if self.gap_days is not None:
            self.permissible_gap_days = self.gap_days
        if self.class_mapping and not self.drug_code_column:
            raise ValueError("drug_code_column is required with class_mapping")
        return self


def _assign_classes(frame: Any, params: TreatmentPathwaysParams) -> Any:
    if params.drug_class_column in frame.columns:
        result = frame[params.drug_class_column].astype("string")
    elif params.class_mapping and params.drug_code_column:
        reverse: dict[Any, str] = {}
        for class_name, codes in sorted(params.class_mapping.items()):
            for code in codes:
                if code in reverse and reverse[code] != class_name:
                    raise AnalysisInputError(f"drug code {code!r} maps to more than one class")
                reverse[code] = class_name
        result = frame[params.drug_code_column].map(reverse).astype("string")
    else:
        raise AnalysisInputError(
            f"input requires '{params.drug_class_column}' or a declared class_mapping"
        )
    if params.drug_classes:
        result = result.where(result.isin(params.drug_classes))
    return result


class TreatmentPathwaysAnalysis:
    kind = "treatment_pathways"
    Params = TreatmentPathwaysParams
    requires = ("drug exposure dates", "researcher-declared drug classes")
    rules = ("SMALL-CELL",)

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        import pandas as pd

        params = self.Params.model_validate(ctx.params)
        frame = load_frame(ctx, params)
        required = [params.patient_column, params.start_column]
        if params.end_column:
            required.append(params.end_column)
        if params.drug_class_column not in frame and params.drug_code_column:
            required.append(params.drug_code_column)
        require_columns(frame, required)
        if frame.empty:
            raise AnalysisInputError("treatment_pathways requires at least one drug exposure")
        working = frame.copy()
        working["_class"] = _assign_classes(working, params)
        working["_start"] = pd.to_datetime(working[params.start_column], errors="coerce")
        if params.end_column:
            working["_end"] = pd.to_datetime(working[params.end_column], errors="coerce")
            working["_end"] = working["_end"].fillna(working["_start"])
        else:
            working["_end"] = working["_start"]
        invalid_dates = working["_start"].isna() | working["_end"].isna()
        if invalid_dates.any():
            raise AnalysisInputError("drug exposure dates must be parseable")
        if (working["_end"] < working["_start"]).any():
            raise AnalysisInputError("drug exposure end dates cannot precede start dates")
        excluded_unclassified = int(working["_class"].isna().sum())
        working = working.dropna(subset=[params.patient_column, "_class"])
        if working.empty:
            raise AnalysisInputError("no drug exposures matched the declared drug classes")

        gap = pd.to_timedelta(params.permissible_gap_days, unit="D")
        eras: list[dict[str, Any]] = []
        for (patient, drug_class), records in working.groupby(
            [params.patient_column, "_class"], sort=True, dropna=False
        ):
            records = records.sort_values(["_start", "_end"], kind="mergesort")
            era_start = None
            era_end: Any = None
            exposure_count = 0
            for start, end in records[["_start", "_end"]].itertuples(index=False, name=None):
                if era_start is None:
                    era_start, era_end, exposure_count = start, end, 1
                elif start <= era_end + gap:
                    era_end = max(era_end, end)
                    exposure_count += 1
                else:
                    eras.append(
                        {
                            "person_id": patient,
                            "drug_class": str(drug_class),
                            "era_start": era_start,
                            "era_end": era_end,
                            "exposure_count": exposure_count,
                        }
                    )
                    era_start, era_end, exposure_count = start, end, 1
            eras.append(
                {
                    "person_id": patient,
                    "drug_class": str(drug_class),
                    "era_start": era_start,
                    "era_end": era_end,
                    "exposure_count": exposure_count,
                }
            )
        era_frame = pd.DataFrame(eras).sort_values(
            ["person_id", "era_start", "drug_class"], kind="mergesort"
        )

        patient_rows: list[dict[str, Any]] = []
        for patient, records in era_frame.groupby("person_id", sort=True):
            sequence: list[str] = []
            starts: list[Any] = []
            for _, class_value, start_value, _, _ in records.itertuples(index=False, name=None):
                if not sequence or sequence[-1] != class_value:
                    sequence.append(str(class_value))
                    starts.append(start_value)
            sequence = sequence[: params.max_lines]
            starts = starts[: params.max_lines]
            row: dict[str, Any] = {
                "person_id": patient,
                "sequence": " → ".join(sequence),
                "line_count": len(sequence),
            }
            for index in range(params.max_lines):
                row[f"line_{index + 1}"] = sequence[index] if index < len(sequence) else None
                if 0 < index < len(starts):
                    row[f"days_to_line_{index + 1}"] = int((starts[index] - starts[index - 1]).days)
                else:
                    row[f"days_to_line_{index + 1}"] = None
            patient_rows.append(row)
        patient_frame = pd.DataFrame(patient_rows)
        pathway_counts: Any = patient_frame.groupby("sequence", sort=True).size()
        pathway_frame = (
            pathway_counts.rename("count")
            .reset_index()
            .sort_values(["count", "sequence"], ascending=[False, True], kind="mergesort")
        )
        pathway_frame["percent"] = 100 * pathway_frame["count"] / len(patient_frame)

        switch_rows: list[dict[str, Any]] = []
        for line in range(2, params.max_lines + 1):
            column = f"days_to_line_{line}"
            values = patient_frame[column].dropna()
            switch_rows.append(
                {
                    "line": line,
                    "patients_switching": int(len(values)),
                    "median_days": float(values.median()) if len(values) else None,
                    "q1_days": float(values.quantile(0.25)) if len(values) else None,
                    "q3_days": float(values.quantile(0.75)) if len(values) else None,
                }
            )

        return AnalysisResult(
            analysis_id=ctx.analysis_id,
            kind=self.kind,
            n={
                "total": int(patient_frame["person_id"].nunique()),
                "per_group": {},
                "excluded": excluded_unclassified,
            },
            estimates=[],
            provenance=dict(ctx.provenance) or {"available": False},
            assumptions_checked=[
                {
                    "name": "drug_classes_pre_declared",
                    "passed": bool(
                        params.drug_classes
                        or params.class_mapping
                        or params.drug_class_column in frame
                    ),
                },
                {"name": "permissible_gap", "passed": True, "days": params.permissible_gap_days},
                {"name": "exhaustive_enumeration", "passed": True, "pattern_mining": False},
            ],
            guardrail_outcomes=[dict(item) for item in ctx.guardrail_outcomes],
            power_note=(
                "Descriptive pathway enumeration; no hypothesis-test power calculation was "
                "performed."
            ),
            seed=ctx.seed,
            package_versions=package_versions(),
            tables={
                "data": pathway_frame,
                "eras": era_frame,
                "patient_sequences": patient_frame,
                "time_to_switch": switch_rows,
            },
            metadata={
                "permissible_gap_days": params.permissible_gap_days,
                "max_lines": params.max_lines,
            },
            output_dir=ctx.output_dir,
        )


ANALYSIS = TreatmentPathwaysAnalysis()
