"""Resolve researcher concepts through Construe and the same fhir2omop resolver as study data."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pheno_rwe.client import AuditedTransport, PhenoTransport
from pheno_rwe.hashing import canonical_json, hash_file, hash_json
from pheno_rwe.manifest import record_step
from pheno_rwe.serialization import to_data
from pheno_rwe.steps.common import StepResult, safe_identifier
from pheno_rwe.workspace import StudyWorkspace, find_study, study_lock


def _coding_values(response: Any) -> list[dict[str, Any]]:
    data = to_data(response)
    candidates: Any = data
    response_system = ""
    if isinstance(data, dict):
        candidates = (
            data.get("codings")
            or data.get("results")
            or data.get("concepts")
            or data.get("codes")
            or []
        )
        system_value = data.get("system") or {}
        response_system = str(
            system_value.get("url")
            or system_value.get("uri")
            or system_value.get("name")
            or system_value
            or ""
        )
    if not isinstance(candidates, list):
        candidates = [candidates]
    codings: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        coding = candidate.get("coding", candidate)
        system_value = coding.get("system") or response_system
        if isinstance(system_value, dict):
            system_value = (
                system_value.get("url") or system_value.get("uri") or system_value.get("name") or ""
            )
        system = str(system_value)
        code = str(coding.get("code") or "")
        if not code or (system, code) in seen:
            continue
        seen.add((system, code))
        codings.append(
            {
                "system": system,
                "code": code,
                "display": coding.get("display") or coding.get("description"),
            }
        )
    return codings


def _probe_bundle(codings: list[dict[str, Any]]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = [
        {"resource": {"resourceType": "Patient", "id": "codeset-probe-patient"}}
    ]
    for index, coding in enumerate(codings, 1):
        entries.append(
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": f"codeset-probe-{index}",
                    "subject": {"reference": "Patient/codeset-probe-patient"},
                    "code": {"coding": [coding]},
                }
            }
        )
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


def _mapping_index(response: Any) -> dict[tuple[str, str], dict[str, Any]]:
    data = to_data(response)
    mappings = data.get("mappings", []) if isinstance(data, dict) else []
    tables = data.get("tables", {}) if isinstance(data, dict) else {}
    concept_columns = {
        "condition_occurrence": ("condition_occurrence_id", "condition_concept_id"),
        "drug_exposure": ("drug_exposure_id", "drug_concept_id"),
        "procedure_occurrence": ("procedure_occurrence_id", "procedure_concept_id"),
        "measurement": ("measurement_id", "measurement_concept_id"),
        "observation": ("observation_id", "observation_concept_id"),
    }
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for mapping in mappings:
        if not isinstance(mapping, dict):
            continue
        coding = (
            mapping.get("source_coding")
            or mapping.get("sourceCoding")
            or mapping.get("coding")
            or {}
        )
        system = str(
            coding.get("system") or mapping.get("source_system") or mapping.get("system") or ""
        )
        code = str(coding.get("code") or mapping.get("source_code") or mapping.get("code") or "")
        if code:
            enriched = dict(mapping)
            table = mapping.get("omop_table")
            omop_id = mapping.get("omop_id")
            if table in concept_columns and omop_id is not None:
                id_column, concept_column = concept_columns[table]
                row = next(
                    (
                        candidate
                        for candidate in tables.get(table, []) or []
                        if candidate.get(id_column) == omop_id
                    ),
                    None,
                )
                if row:
                    enriched["concept_id"] = row.get(concept_column)
            index[(system, code)] = enriched
    return index


def resolve_code_set(
    study: StudyWorkspace | str | Path,
    transport: PhenoTransport,
    *,
    name: str,
    text: str,
    domain: str | None = None,
    codings: list[dict[str, Any]] | None = None,
    force: bool = False,
) -> StepResult:
    workspace = study if isinstance(study, StudyWorkspace) else find_study(study)
    started = datetime.now(UTC)
    resolved = codings or _coding_values(transport.resolve_codings(text, domain))
    response = transport.fhir2omop(_probe_bundle(resolved)) if resolved else {"mappings": []}
    mapping_index = _mapping_index(response)
    records: list[dict[str, Any]] = []
    for coding in resolved:
        mapping = mapping_index.get((coding.get("system", ""), coding["code"]), {})
        records.append(
            {
                **coding,
                "source_value": f"{coding.get('system', '')}#{coding['code']}",
                "concept_id": mapping.get("concept_id") or mapping.get("conceptId") or 0,
                "mapping_status": mapping.get("mapping_status")
                or mapping.get("mappingStatus")
                or "UNCHECKED",
                "accepted": True,
            }
        )
    counts: dict[str, int] = {}
    for record in records:
        status = str(record["mapping_status"])
        counts[status] = counts.get(status, 0) + 1
    payload = {
        "schema_version": "1.0",
        "name": name,
        "description": text,
        "domain": domain,
        "codings": records,
        "mapping_status_counts": counts,
        "approval": {
            "status": "pending",
            "reviewed_by": None,
            "reviewed_at": None,
            "notes": None,
            "review_hash": None,
        },
    }
    target = workspace.root / "codesets" / f"{safe_identifier(name)}.codeset.json"
    with study_lock(workspace):
        if (
            target.exists()
            and not force
            and json.loads(target.read_text(encoding="utf-8")) == payload
        ):
            status = "skipped"
        else:
            target.write_text(canonical_json(payload) + "\n", encoding="utf-8")
            status = "success"
    audited = transport if isinstance(transport, AuditedTransport) else None
    record_step(
        workspace.manifest_path,
        step="resolve-codes",
        status=status,
        started_at=started,
        params={"name": name, "text": text, "domain": domain},
        input_signature=hash_json(
            {"name": name, "text": text, "domain": domain, "codings": resolved}
        ),
        outputs={workspace.relative(target): hash_file(target)},
        items=records,
        api_calls=audited.manifest_calls() if audited else (),
    )
    return StepResult(
        "resolve-codes",
        status,
        f"Resolved {len(records)} codings. Researcher review is required before analysis.",
        {workspace.relative(target): hash_file(target)},
        records,
        ["Code sets are unapproved until the researcher reviews mapping statuses."],
    )
