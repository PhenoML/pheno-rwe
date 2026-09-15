"""Live FHIR pull exclusively through the PhenoML provider proxy."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from pheno_rwe.client import AuditedTransport, PhenoTransport
from pheno_rwe.hashing import canonical_json, hash_file, hash_json
from pheno_rwe.manifest import record_step
from pheno_rwe.runtime import CancellationToken, ProgressEvent, ProgressSink, null_progress
from pheno_rwe.serialization import to_data
from pheno_rwe.steps.common import StepResult, safe_identifier
from pheno_rwe.workspace import StudyWorkspace, find_study, patient_token, study_lock


@dataclass(slots=True)
class PullPreview:
    patient_ids: list[str]
    queries: list[Any] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def preview_live_cohort(
    transport: PhenoTransport,
    text: str,
    provider_id: str,
    *,
    queries_only: bool = False,
) -> PullPreview:
    response = (
        transport.cohort_queries(text, provider_id)
        if queries_only
        else transport.analyze_cohort(text, provider_id)
    )
    data = to_data(response)
    if not isinstance(data, dict):
        data = {"result": data}
    ids = data.get("patientIds") or data.get("patient_ids") or data.get("patients") or []
    if isinstance(ids, dict):
        ids = list(ids)
    patient_ids = [str(item.get("id") if isinstance(item, dict) else item) for item in ids]
    queries = data.get("queries") or data.get("fhirQueries") or []
    return PullPreview(patient_ids[:1000], list(queries), data)


def preview_from_ids(ids: Iterable[str]) -> PullPreview:
    """Build a pull preview from an explicit patient-ID set (no cohort-service call)."""

    return PullPreview(patient_ids=list(ids))


def _strip_narrative(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_narrative(item) for key, item in value.items() if key != "text"}
    if isinstance(value, list):
        return [_strip_narrative(item) for item in value]
    return value


def _merge_page(bundle: dict[str, Any], page: dict[str, Any]) -> None:
    bundle.setdefault("entry", []).extend(page.get("entry") or [])
    bundle["total"] = len(bundle["entry"])


def _next_link(bundle: dict[str, Any]) -> str | None:
    for link in bundle.get("link") or []:
        if link.get("relation") == "next":
            return link.get("url")
    return None


def _next_page_params(next_url: str) -> dict[str, str]:
    """Query parameters to re-issue against the original path for the next page.

    FHIR servers return absolute ``next`` links (for example Medplum's
    ``.../Patient?_count=20&_offset=20``). The PhenoML proxy resolves a base-relative
    path plus query parameters, not an absolute URL, so forward only the next page's
    query and re-issue it against the same resource path as the first page.
    """
    return dict(parse_qsl(urlsplit(next_url).query, keep_blank_values=True))


def pull_live_cohort(
    study: StudyWorkspace | str | Path,
    transport: PhenoTransport,
    *,
    text: str,
    provider_id: str,
    approved: bool = False,
    preview: PullPreview | None = None,
    confirm: Callable[[PullPreview], bool] | None = None,
    progress: ProgressSink = null_progress,
    cancellation: CancellationToken | None = None,
    force: bool = False,
) -> StepResult:
    workspace = study if isinstance(study, StudyWorkspace) else find_study(study)
    started = datetime.now(UTC)
    preview = preview or preview_live_cohort(transport, text, provider_id)
    if not approved and not (confirm and confirm(preview)):
        return StepResult(
            step="pull",
            status="preview",
            message="FHIR queries require researcher approval before execution.",
            items=[{"queries": preview.queries, "patient_count": len(preview.patient_ids)}],
        )
    items: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}
    failures = 0
    token = cancellation or CancellationToken()
    with study_lock(workspace):
        for index, patient_id in enumerate(preview.patient_ids, 1):
            token.checkpoint()
            audit_id = patient_token(workspace, patient_id)
            progress(
                ProgressEvent(
                    "pull",
                    "running",
                    f"Fetching patient {index} of {len(preview.patient_ids)}",
                    index,
                    len(preview.patient_ids),
                    audit_id,
                )
            )
            target = (
                workspace.root / "raw" / "patients" / f"{safe_identifier(audit_id)}.bundle.json"
            )
            if target.exists() and not force:
                items.append(
                    {
                        "patient_token": audit_id,
                        "status": "skipped",
                        "path": workspace.relative(target),
                    }
                )
                outputs[workspace.relative(target)] = hash_file(target)
                continue
            try:
                everything_path = f"Patient/{patient_id}/$everything"
                first = to_data(transport.fhir_search(provider_id, everything_path))
                if not isinstance(first, dict):
                    raise TypeError("FHIR proxy did not return a Bundle object")
                bundle = {"resourceType": "Bundle", "type": "collection", "entry": []}
                page = first
                while True:
                    _merge_page(bundle, page)
                    next_url = _next_link(page)
                    if not next_url:
                        break
                    next_params = _next_page_params(next_url)
                    if not next_params:
                        break
                    page_value = to_data(
                        transport.fhir_search(provider_id, everything_path, **next_params)
                    )
                    if not isinstance(page_value, dict):
                        break
                    page = page_value
                target.write_text(canonical_json(_strip_narrative(bundle)) + "\n", encoding="utf-8")
                outputs[workspace.relative(target)] = hash_file(target)
                items.append(
                    {
                        "patient_token": audit_id,
                        "status": "success",
                        "path": workspace.relative(target),
                    }
                )
            except Exception as exc:  # one patient's failure does not erase earlier work
                failures += 1
                error = str(exc).replace(patient_id, audit_id)
                items.append(
                    {
                        "patient_token": audit_id,
                        "status": "failed",
                        "error": error,
                    }
                )
    status = "partial" if failures else "success"
    audited = transport if isinstance(transport, AuditedTransport) else None
    record_step(
        workspace.manifest_path,
        step="pull",
        status=status,
        started_at=started,
        params={"text": text, "provider_id": provider_id, "approved": True},
        input_signature=hash_json(
            {"text": text, "provider": provider_id, "patients": preview.patient_ids}
        ),
        outputs=outputs,
        items=items,
        api_calls=audited.manifest_calls() if audited else (),
    )
    return StepResult(
        step="pull",
        status=status,
        message=(
            f"Fetched {len(preview.patient_ids) - failures} patient bundles; {failures} failed."
        ),
        outputs=outputs,
        items=items,
    )
