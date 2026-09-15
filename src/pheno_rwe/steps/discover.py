"""Stateless discovery primitives an agent orchestrates before ``pull``.

Discovery decides only *what* to pull (a deliberate superset). These commands are
query primitives that print JSON and deliberately do **not** call ``record_step`` --
provenance begins at ``pull`` and the deterministic pipeline that follows it. Each
function returns a :class:`StepResult` so the CLI can render it pretty or as ``--json``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pheno_rwe.client import PhenoTransport
from pheno_rwe.serialization import to_data
from pheno_rwe.steps.common import StepResult
from pheno_rwe.steps.pull import _next_link, _next_page_params
from pheno_rwe.steps.resolve_codes import _coding_values


def extract_codes(
    transport: PhenoTransport,
    text: str,
    domain: str | None = None,
) -> StepResult:
    """Resolve a natural-language concept to source codes (Construe).

    A lightweight sibling of ``resolve-codes`` with no artifact or fhir2omop probe.
    """

    codings = _coding_values(transport.resolve_codings(text, domain))
    return StepResult(
        step="extract-codes",
        message=f"Extracted {len(codings)} codings for {text!r}.",
        items=codings,
    )


def crosswalk_code(
    transport: PhenoTransport,
    *,
    system: str,
    code: str,
    targets: Sequence[str],
) -> StepResult:
    """Map one source coding to the target vocabularies present in the data (Crosswalk)."""

    codings = _crosswalk_codings(transport.crosswalk(system, code, targets))
    return StepResult(
        step="crosswalk",
        message=f"Matched {len(codings)} target codings for {system}#{code}.",
        items=codings,
    )


def _crosswalk_codings(response: Any) -> list[dict[str, Any]]:
    # A CrosswalkResponse groups matches under ``targets`` (its top-level system/code
    # are the *source*); flatten to the target codings the agent will search on.
    data = to_data(response)
    if not isinstance(data, dict):
        return []
    codings: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for target in data.get("targets") or []:
        if not isinstance(target, dict):
            continue
        target_system = str(target.get("system") or "")
        for match in target.get("matches") or []:
            if not isinstance(match, dict):
                continue
            code = str(match.get("code") or "")
            if not code or (target_system, code) in seen:
                continue
            seen.add((target_system, code))
            codings.append(
                {
                    "system": target_system,
                    "code": code,
                    "display": match.get("display"),
                    "cui": match.get("cui"),
                }
            )
    return codings


def search_fhir(
    transport: PhenoTransport,
    provider_id: str,
    path: str,
    params: dict[str, Any],
    *,
    count: bool = False,
    patients: bool = False,
) -> StepResult:
    """Run a FHIR-path search as a count, a patient-ID extraction, or a page preview."""

    if count and patients:
        raise ValueError("Choose either --count or --patients for fhir-search, not both.")
    if count:
        counted = transport.fhir_search(provider_id, path, **{**params, "_summary": "count"})
        total = _bundle_total(to_data(counted))
        return StepResult(
            step="fhir-search",
            message=f"{path}: {total} matching resources.",
            items=[{"total": total}],
        )
    if patients:
        ids = _collect_patient_ids(transport, provider_id, path, params)
        return StepResult(
            step="fhir-search",
            message=f"{path}: {len(ids)} distinct patients.",
            items=[{"id": patient_id} for patient_id in ids],
        )
    summary = _bundle_summary(to_data(transport.fhir_search(provider_id, path, **params)))
    return StepResult(
        step="fhir-search",
        message=f"{path}: first page of {summary['entry_count']} resources.",
        items=[summary],
    )


def _bundle_total(bundle: Any) -> int:
    if isinstance(bundle, dict):
        total = bundle.get("total")
        if isinstance(total, bool):  # bool is an int subclass; a FHIR total is never boolean.
            return 0
        if isinstance(total, int):
            return total
        if isinstance(total, str) and total.strip().isdigit():
            return int(total.strip())
        entries = bundle.get("entry")
        if isinstance(entries, list):
            return len(entries)
    return 0


def _bundle_summary(bundle: Any) -> dict[str, Any]:
    if not isinstance(bundle, dict):
        return {
            "resourceType": None,
            "type": None,
            "total": None,
            "entry_count": 0,
            "has_next": False,
        }
    entries = bundle.get("entry")
    return {
        "resourceType": bundle.get("resourceType"),
        "type": bundle.get("type"),
        "total": bundle.get("total"),
        "entry_count": len(entries) if isinstance(entries, list) else 0,
        "has_next": _next_link(bundle) is not None,
    }


def _collect_patient_ids(
    transport: PhenoTransport,
    provider_id: str,
    path: str,
    params: dict[str, Any],
) -> list[str]:
    # A dict preserves first-seen order while collapsing duplicates across pages.
    seen: dict[str, None] = {}
    page = to_data(transport.fhir_search(provider_id, path, **params))
    while isinstance(page, dict):
        for entry in page.get("entry") or []:
            if not isinstance(entry, dict):
                continue
            resource = entry.get("resource")
            if not isinstance(resource, dict):
                continue
            patient_id = _patient_id_from_resource(resource)
            if patient_id and patient_id not in seen:
                seen[patient_id] = None
        next_url = _next_link(page)
        if not next_url:
            break
        next_params = _next_page_params(next_url)
        if not next_params:
            break
        page = to_data(transport.fhir_search(provider_id, path, **next_params))
    return list(seen)


def _patient_id_from_resource(resource: dict[str, Any]) -> str | None:
    if resource.get("resourceType") == "Patient":
        identifier = resource.get("id")
        return str(identifier) if identifier else None
    for key in ("patient", "subject"):
        reference = resource.get(key)
        if isinstance(reference, dict):
            patient_id = _reference_to_patient_id(reference.get("reference"))
            if patient_id:
                return patient_id
    return None


def _reference_to_patient_id(reference: Any) -> str | None:
    if not isinstance(reference, str):
        return None
    marker = "Patient/"
    index = reference.rfind(marker)
    if index == -1:
        return None
    tail = reference[index + len(marker) :]
    identifier = tail.split("/", 1)[0].split("?", 1)[0]
    return identifier or None
