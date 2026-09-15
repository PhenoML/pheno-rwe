"""Local bulk-FHIR ingest into canonical one-patient collection Bundles."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pheno_rwe.fhirtools.ndjson import group_ndjson
from pheno_rwe.hashing import canonical_json, hash_file, hash_paths
from pheno_rwe.manifest import record_step
from pheno_rwe.runtime import CancellationToken, ProgressEvent, ProgressSink, null_progress
from pheno_rwe.steps.common import StepResult, safe_identifier
from pheno_rwe.workspace import StudyWorkspace, find_study, patient_token, study_lock


def ingest_local(
    study: StudyWorkspace | str | Path,
    sources: str | Path | list[str | Path],
    *,
    progress: ProgressSink = null_progress,
    cancellation: CancellationToken | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> StepResult:
    workspace = study if isinstance(study, StudyWorkspace) else find_study(study)
    started = datetime.now(UTC)
    source_values = [sources] if isinstance(sources, (str, Path)) else list(sources)
    source_paths: list[Path] = []
    for source in source_values:
        path = Path(source).expanduser().resolve()
        if path.is_dir():
            source_paths.extend(
                child
                for child in sorted(path.rglob("*"))
                if child.is_file() and child.suffix.lower() in {".json", ".jsonl", ".ndjson"}
            )
        else:
            source_paths.append(path)
    grouped = group_ndjson(source_paths)
    token = cancellation or CancellationToken()
    outputs: dict[str, str] = {}
    items: list[dict[str, object]] = []
    with study_lock(workspace):
        for current, (patient_id, bundle) in enumerate(grouped.bundles.items(), 1):
            token.checkpoint()
            audit_id = patient_token(workspace, patient_id)
            progress(
                ProgressEvent(
                    "ingest",
                    "running",
                    f"Writing patient bundle {current} of {grouped.patient_count}",
                    current,
                    grouped.patient_count,
                    audit_id,
                )
            )
            target = (
                workspace.root / "raw" / "patients" / f"{safe_identifier(audit_id)}.bundle.json"
            )
            encoded = canonical_json(bundle) + "\n"
            status = "planned" if dry_run else "success"
            if target.exists() and not force:
                existing = target.read_text(encoding="utf-8")
                if existing == encoded:
                    status = "skipped"
            if not dry_run and status != "skipped":
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(encoded, encoding="utf-8")
            relative = workspace.relative(target)
            if target.exists():
                outputs[relative] = hash_file(target)
            items.append(
                {
                    "patient_token": audit_id,
                    "status": status,
                    "path": relative,
                    "resources": len(bundle.get("entry") or []),
                }
            )

        orphan_report = workspace.root / "reports" / "ingest_orphans.json"
        orphan_payload = {
            "orphan_count": len(grouped.orphans),
            "ambiguous_count": grouped.ambiguous_count,
            "duplicates_ignored": grouped.duplicate_count,
            "orphans": [
                {
                    "resource_type": resource.get("resourceType"),
                    "resource_id": resource.get("id"),
                }
                for resource in grouped.orphans
            ],
        }
        if not dry_run:
            orphan_report.parent.mkdir(parents=True, exist_ok=True)
            orphan_report.write_text(canonical_json(orphan_payload) + "\n", encoding="utf-8")
            outputs[workspace.relative(orphan_report)] = hash_file(orphan_report)

    status = "partial" if grouped.orphans or grouped.ambiguous_count else "success"
    if dry_run:
        return StepResult(
            "ingest",
            status,
            (
                f"Would write {grouped.patient_count} patient bundles; "
                f"{len(grouped.orphans)} orphan resources."
            ),
            outputs,
            items,
        )
    record_step(
        workspace.manifest_path,
        step="ingest",
        status=status,
        started_at=started,
        params={"sources": [str(path) for path in source_paths], "force": force},
        input_signature=hash_paths(source_paths),
        inputs={str(path): hash_file(path) for path in source_paths},
        outputs=outputs,
        items=items,
    )
    return StepResult(
        "ingest",
        status,
        f"Wrote {grouped.patient_count} patient bundles; {len(grouped.orphans)} orphan resources.",
        outputs,
        items,
        ["Some FHIR resources could not be assigned to one patient."] if grouped.orphans else [],
    )


ingest = ingest_local
ingest_ndjson = ingest_local
