"""DocumentReference enrichment with content-addressed caching and provenance."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pheno_rwe.client import AuditedTransport, PhenoTransport
from pheno_rwe.hashing import canonical_json, hash_bytes, hash_file, hash_json
from pheno_rwe.manifest import record_step
from pheno_rwe.runtime import CancellationToken, ProgressEvent, ProgressSink, null_progress
from pheno_rwe.serialization import to_data
from pheno_rwe.steps.common import StepResult, safe_identifier
from pheno_rwe.workspace import StudyWorkspace, audit_token, find_study, study_lock

MAX_DOCUMENT_BYTES = 20 * 1024 * 1024
SUPPORTED_BINARY_TYPES = {"application/pdf", "image/png", "image/jpeg", "image/tiff"}


def _resources(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    if bundle.get("resourceType") and bundle.get("resourceType") != "Bundle":
        return [bundle]
    return [
        entry["resource"]
        for entry in bundle.get("entry") or []
        if isinstance(entry, dict) and isinstance(entry.get("resource"), dict)
    ]


def _attachments(document_reference: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item["attachment"]
        for item in document_reference.get("content") or []
        if isinstance(item, dict) and isinstance(item.get("attachment"), dict)
    ]


def _attachment_bytes(
    attachment: dict[str, Any], transport: PhenoTransport, provider_id: str | None
) -> tuple[bytes, str]:
    mime_type = str(attachment.get("contentType") or "application/octet-stream").split(";")[0]
    if attachment.get("data"):
        return base64.b64decode(attachment["data"], validate=True), mime_type
    url = str(attachment.get("url") or "")
    if not url or not provider_id:
        raise ValueError("Document attachment has neither inline data nor a resolvable Binary URL")
    response = to_data(transport.fhir_search(provider_id, url))
    if isinstance(response, dict) and response.get("data"):
        return base64.b64decode(response["data"], validate=True), str(
            response.get("contentType") or mime_type
        )
    raise ValueError(f"Binary response for {url} did not contain data")


def _tag_resource(
    resource: dict[str, Any],
    doc_id: str,
    digest: str,
    ordinal: int,
) -> dict[str, Any]:
    result = dict(resource)
    original_id = str(result.get("id") or ordinal)
    result["id"] = f"enr-{digest[:12]}-{safe_identifier(original_id)}"
    meta = dict(result.get("meta") or {})
    tags = list(meta.get("tag") or [])
    tags.extend(
        [
            {"system": "https://phenoml.com/pheno-rwe/origin", "code": "enriched"},
            {"system": "https://phenoml.com/pheno-rwe/document-hash", "code": digest},
            {"system": "https://phenoml.com/pheno-rwe/document-reference", "code": doc_id},
        ]
    )
    meta["tag"] = tags
    result["meta"] = meta
    return result


def enrich_documents(
    study: StudyWorkspace | str | Path,
    transport: PhenoTransport,
    *,
    provider_id: str | None = None,
    detection_effort: str = "standard",
    validation_method: str = "check",
    progress: ProgressSink = null_progress,
    cancellation: CancellationToken | None = None,
    force: bool = False,
) -> StepResult:
    workspace = study if isinstance(study, StudyWorkspace) else find_study(study)
    started = datetime.now(UTC)
    source_files = sorted((workspace.root / "raw" / "patients").glob("*.bundle.json"))
    token = cancellation or CancellationToken()
    items: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}
    failures = 0
    document_number = 0
    with study_lock(workspace):
        for patient_number, source in enumerate(source_files, 1):
            token.checkpoint()
            bundle = json.loads(source.read_text(encoding="utf-8"))
            additions: list[dict[str, Any]] = []
            documents = [
                resource
                for resource in _resources(bundle)
                if resource.get("resourceType") == "DocumentReference"
            ]
            for document in documents:
                doc_id = str(document.get("id") or f"document-{document_number + 1}")
                doc_token = audit_token(workspace, doc_id, prefix="d")
                for attachment in _attachments(document):
                    document_number += 1
                    token.checkpoint()
                    progress(
                        ProgressEvent(
                            "enrich",
                            "running",
                            f"Processing document {doc_token}",
                            patient_number,
                            len(source_files),
                            doc_token,
                        )
                    )
                    try:
                        content, mime_type = _attachment_bytes(attachment, transport, provider_id)
                        if len(content) > MAX_DOCUMENT_BYTES:
                            raise ValueError("document exceeds the 20 MB client-side limit")
                        digest = hash_bytes(content)
                        cache = (
                            workspace.root
                            / "enriched"
                            / "documents"
                            / f"{digest[:12]}.response.json"
                        )
                        if cache.exists() and not force:
                            response = json.loads(cache.read_text(encoding="utf-8"))
                            cache_status = "cached"
                        else:
                            response = to_data(
                                transport.document(
                                    content,
                                    mime_type,
                                    detection_effort=detection_effort,
                                    validation_method=validation_method,
                                )
                            )
                            cache.write_text(canonical_json(response) + "\n", encoding="utf-8")
                            cache_status = "success"
                        returned_bundle = (
                            response.get("bundle", response) if isinstance(response, dict) else {}
                        )
                        for ordinal, resource in enumerate(_resources(returned_bundle), 1):
                            tagged = _tag_resource(resource, doc_id, digest, ordinal)
                            additions.append(
                                {
                                    "fullUrl": f"urn:uuid:{tagged['id']}",
                                    "resource": tagged,
                                }
                            )
                        outputs[workspace.relative(cache)] = hash_file(cache)
                        items.append(
                            {
                                "patient": source.stem,
                                "document_token": doc_token,
                                "document_hash": digest,
                                "status": cache_status,
                                "resources_added": len(additions),
                            }
                        )
                    except Exception as exc:
                        failures += 1
                        items.append(
                            {
                                "patient": source.stem,
                                "document_token": doc_token,
                                "status": "failed",
                                "error": str(exc).replace(doc_id, doc_token),
                            }
                        )
            enriched = dict(bundle)
            enriched["entry"] = list(bundle.get("entry") or []) + additions
            target = workspace.root / "enriched" / "patients" / source.name
            target.write_text(canonical_json(enriched) + "\n", encoding="utf-8")
            outputs[workspace.relative(target)] = hash_file(target)
    status = "partial" if failures else "success"
    audited = transport if isinstance(transport, AuditedTransport) else None
    record_step(
        workspace.manifest_path,
        step="enrich",
        status=status,
        started_at=started,
        params={"detection_effort": detection_effort, "validation_method": validation_method},
        input_signature=hash_json({str(p): hash_file(p) for p in source_files}),
        outputs=outputs,
        items=items,
        api_calls=audited.manifest_calls() if audited else (),
    )
    return StepResult(
        "enrich",
        status,
        f"Processed {document_number} documents; {failures} failed.",
        outputs,
        items,
    )
