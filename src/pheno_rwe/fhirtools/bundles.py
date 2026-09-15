"""Small, deterministic FHIR Bundle helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any


def bundle_entries(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    if value.get("resourceType") != "Bundle":
        return [dict(value)]
    result: list[dict[str, Any]] = []
    for entry in value.get("entry") or []:
        if not isinstance(entry, Mapping):
            continue
        resource = entry.get("resource")
        if isinstance(resource, Mapping):
            result.append(dict(resource))
    return result


def make_collection_bundle(resources: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        (dict(resource) for resource in resources),
        key=lambda resource: (
            0 if resource.get("resourceType") == "Patient" else 1,
            str(resource.get("resourceType") or ""),
            str(resource.get("id") or ""),
            resource_digest(resource),
        ),
    )
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": resource} for resource in ordered],
    }


def normalize_reference(reference: str) -> str:
    """Normalize absolute, relative, and versioned references to ``Type/id``."""
    value = reference.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    parts = [part for part in value.split("/") if part]
    if "_history" in parts:
        parts = parts[: parts.index("_history")]
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return value


def patient_id_from_reference(reference: str | None) -> str | None:
    if not reference:
        return None
    normalized = normalize_reference(reference)
    if normalized.startswith("Patient/"):
        patient_id = normalized.partition("/")[2]
        return patient_id or None
    return None


def iter_references(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        reference = value.get("reference")
        if isinstance(reference, str):
            yield reference
        for child in value.values():
            yield from iter_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_references(child)


def resource_digest(resource: Mapping[str, Any]) -> str:
    encoded = json.dumps(resource, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resource_key(resource: Mapping[str, Any]) -> str:
    resource_type = str(resource.get("resourceType") or "")
    resource_id = str(resource.get("id") or "")
    if resource_type and resource_id:
        return f"{resource_type}/{resource_id}"
    return f"sha256:{resource_digest(resource)}"


def patient_ids_in_resource(resource: Mapping[str, Any]) -> set[str]:
    if resource.get("resourceType") == "Patient" and resource.get("id"):
        return {str(resource["id"])}
    return {
        patient_id
        for reference in iter_references(resource)
        if (patient_id := patient_id_from_reference(reference)) is not None
    }
