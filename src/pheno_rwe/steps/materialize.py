"""Per-patient fhir2omop materialization with stable IDs and resumability."""

from __future__ import annotations

import json
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pheno_rwe.client import AuditedTransport, PhenoTransport
from pheno_rwe.fhirtools.bundles import patient_ids_in_resource
from pheno_rwe.hashing import canonical_json, hash_file, hash_json
from pheno_rwe.manifest import record_step
from pheno_rwe.omop.loader import OmopLoader
from pheno_rwe.runtime import CancellationToken, ProgressEvent, ProgressSink, null_progress
from pheno_rwe.serialization import to_data
from pheno_rwe.steps.common import StepResult
from pheno_rwe.workspace import StudyWorkspace, find_study, patient_token, study_lock


def _bundle_patient_id(bundle: dict[str, Any], path: Path) -> str:
    patient_ids: set[str] = set()
    for entry in bundle.get("entry") or []:
        resource = entry.get("resource", {}) if isinstance(entry, dict) else {}
        if isinstance(resource, dict) and resource.get("resourceType") == "Patient":
            patient_ids.update(patient_ids_in_resource(resource))
    if len(patient_ids) != 1:
        raise ValueError(f"{path} must contain exactly one Patient resource")
    return next(iter(patient_ids))


def _patient_bundles(workspace: StudyWorkspace) -> list[Path]:
    selected: dict[str, Path] = {}
    for directory in (
        workspace.root / "raw" / "patients",
        workspace.root / "enriched" / "patients",
    ):
        if directory.exists():
            for path in sorted(directory.glob("*.bundle.json")):
                selected[path.name] = path
    return [selected[name] for name in sorted(selected)]


def coverage_report(connection: Any) -> dict[str, Any]:
    statuses = {
        str(status or "UNKNOWN"): int(count)
        for status, count in connection.execute(
            "SELECT mapping_status, COUNT(*) FROM meta.mapping GROUP BY mapping_status"
        ).fetchall()
    }
    per_table = [
        {"table": table, "mapping_status": status or "UNKNOWN", "count": int(count)}
        for table, status, count in connection.execute(
            """SELECT omop_table, mapping_status, COUNT(*)
               FROM meta.mapping GROUP BY omop_table, mapping_status
               ORDER BY omop_table, mapping_status"""
        ).fetchall()
    ]
    coverage = connection.execute(
        """SELECT COUNT(*), AVG(off_vocab_rate), MIN(off_vocab_rate), MAX(off_vocab_rate)
           FROM meta.coverage"""
    ).fetchone()
    return {
        "mapping_status_counts": statuses,
        "by_table": per_table,
        "patient_count": int(coverage[0] or 0),
        "mean_off_vocab_rate": float(coverage[1]) if coverage[1] is not None else None,
        "min_off_vocab_rate": float(coverage[2]) if coverage[2] is not None else None,
        "max_off_vocab_rate": float(coverage[3]) if coverage[3] is not None else None,
    }


def _coverage_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Mapping coverage",
        "",
        f"Patients: {report['patient_count']}",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {status} | {count} |"
        for status, count in sorted(report["mapping_status_counts"].items())
    )
    lines.extend(["", "| OMOP table | Status | Count |", "|---|---|---:|"])
    lines.extend(
        f"| {item['table']} | {item['mapping_status']} | {item['count']} |"
        for item in report["by_table"]
    )
    return "\n".join(lines) + "\n"


def materialize(
    study: StudyWorkspace | str | Path,
    transport: PhenoTransport,
    *,
    progress: ProgressSink = null_progress,
    cancellation: CancellationToken | None = None,
    force: bool = False,
) -> StepResult:
    workspace = study if isinstance(study, StudyWorkspace) else find_study(study)
    started = datetime.now(UTC)
    bundle_paths = _patient_bundles(workspace)
    token = cancellation or CancellationToken()
    items: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}
    failures = 0
    with study_lock(workspace):
        with OmopLoader(workspace.identified_db) as loader:
            for current, path in enumerate(bundle_paths, 1):
                token.checkpoint()
                bundle = json.loads(path.read_text(encoding="utf-8"))
                patient_id = _bundle_patient_id(bundle, path)
                audit_id = patient_token(workspace, patient_id)
                bundle_hash = hash_file(path)
                progress(
                    ProgressEvent(
                        "materialize",
                        "running",
                        f"Materializing patient {current} of {len(bundle_paths)}",
                        current,
                        len(bundle_paths),
                        audit_id,
                    )
                )
                if not force and loader.is_current(patient_id, bundle_hash):
                    items.append({"patient_token": audit_id, "status": "skipped"})
                    continue
                try:
                    response = to_data(transport.fhir2omop(bundle))
                    if not isinstance(response, dict):
                        raise TypeError("fhir2omop did not return an object")
                    loaded = loader.materialize_patient(
                        patient_id, response, bundle_hash=bundle_hash, bundle=bundle
                    )
                    items.append(
                        {
                            "patient_token": audit_id,
                            "status": "success",
                            "person_id": loaded.person_id,
                            "id_offset": loaded.id_offset,
                            "rows": loaded.row_counts,
                            "mappings": loaded.mapping_count,
                            "dropped": loaded.dropped_count,
                            "invalid_dates": loaded.invalid_date_count,
                        }
                    )
                except Exception as exc:  # item-level isolation is deliberate
                    failures += 1
                    error = str(exc).replace(patient_id, audit_id)
                    with suppress(Exception):
                        loader.mark_failed(patient_id, error)
                    items.append(
                        {
                            "patient_token": audit_id,
                            "status": "failed",
                            "error": error,
                        }
                    )
            report = coverage_report(loader.connection)

        json_report = workspace.root / "reports" / "mapping_coverage.json"
        markdown_report = workspace.root / "reports" / "mapping_coverage.md"
        json_report.write_text(canonical_json(report) + "\n", encoding="utf-8")
        markdown_report.write_text(_coverage_markdown(report), encoding="utf-8")
        for path in (workspace.identified_db, json_report, markdown_report):
            outputs[workspace.relative(path)] = hash_file(path)

    status = "partial" if failures else "success"
    audited = transport if isinstance(transport, AuditedTransport) else None
    record_step(
        workspace.manifest_path,
        step="materialize",
        status=status,
        started_at=started,
        params={"force": force},
        input_signature=hash_json(
            {workspace.relative(path): hash_file(path) for path in bundle_paths}
        ),
        inputs={workspace.relative(path): hash_file(path) for path in bundle_paths},
        outputs=outputs,
        items=items,
        api_calls=audited.manifest_calls() if audited else (),
    )
    return StepResult(
        "materialize",
        status,
        f"Materialized {len(bundle_paths) - failures} patient bundles; {failures} failed.",
        outputs,
        items,
    )


materialize_bundles = materialize
