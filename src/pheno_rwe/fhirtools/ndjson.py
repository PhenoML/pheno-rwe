"""Streaming readers and patient grouping for local bulk-FHIR data."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pheno_rwe.fhirtools.bundles import (
    bundle_entries,
    iter_references,
    make_collection_bundle,
    normalize_reference,
    patient_ids_in_resource,
    resource_digest,
    resource_key,
)


class NDJSONError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GroupedBundles:
    bundles: dict[str, dict[str, Any]]
    orphans: tuple[dict[str, Any], ...]
    resource_count: int
    duplicate_count: int = 0
    ambiguous_count: int = 0

    @property
    def patient_count(self) -> int:
        return len(self.bundles)

    def __getitem__(self, patient_id: str) -> dict[str, Any]:
        return self.bundles[patient_id]

    def items(self):  # type intentionally mirrors dict.items for convenient callers
        return self.bundles.items()


def _expand_paths(sources: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(sources, (str, Path)):
        values = [Path(sources)]
    else:
        values = [Path(source) for source in sources]
    result: list[Path] = []
    for value in values:
        if value.is_dir():
            result.extend(
                path
                for path in sorted(value.rglob("*"))
                if path.is_file() and path.suffix.lower() in {".ndjson", ".json", ".jsonl"}
            )
        elif value.is_file():
            result.append(value)
        else:
            raise FileNotFoundError(value)
    if not result:
        raise NDJSONError("No .ndjson, .jsonl, or .json FHIR files were found")
    return result


def iter_file_resources(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise NDJSONError(f"Invalid JSON in {path}: {exc}") from exc
        values = value if isinstance(value, list) else [value]
        for item in values:
            if not isinstance(item, Mapping):
                raise NDJSONError(f"FHIR JSON value in {path} is not an object")
            yield from bundle_entries(item)
        return

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise NDJSONError(f"Invalid NDJSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, Mapping):
                raise NDJSONError(f"FHIR NDJSON value at {path}:{line_number} is not an object")
            yield from bundle_entries(value)


def iter_resources(sources: str | Path | Iterable[str | Path]) -> Iterator[dict[str, Any]]:
    for path in _expand_paths(sources):
        yield from iter_file_resources(path)


def group_resources_by_patient(resources: Iterable[Mapping[str, Any]]) -> GroupedBundles:
    """Group resources into deterministic per-patient collection Bundles.

    Direct Patient references establish ownership.  Encounter and other
    reference chains then propagate ownership backwards.  Finally, referenced
    helper resources (Medication, Practitioner, Binary, etc.) are copied into
    each bundle that needs them; a shared helper is deliberately duplicated
    rather than merging two patients into one component.
    """
    unique: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for raw in resources:
        resource = dict(raw)
        key = resource_key(resource)
        if key in unique:
            if resource_digest(unique[key]) != resource_digest(resource):
                raise NDJSONError(f"Conflicting duplicate FHIR resource {key}")
            duplicate_count += 1
            continue
        unique[key] = resource

    owners: dict[str, set[str]] = {
        key: patient_ids_in_resource(resource) for key, resource in unique.items()
    }
    known_patients = {
        str(resource["id"])
        for resource in unique.values()
        if resource.get("resourceType") == "Patient" and resource.get("id")
    }
    referenced_keys: dict[str, set[str]] = {}
    for key, resource in unique.items():
        references = {normalize_reference(value) for value in iter_references(resource)}
        referenced_keys[key] = {value for value in references if value in unique}

    # A resource such as ExplanationOfBenefit may point at an Encounter rather
    # than directly at the Patient.  Propagate only from the referenced object
    # to the referencing object, avoiding patient components joined by shared
    # organizations/providers.
    changed = True
    while changed:
        changed = False
        for key, references in referenced_keys.items():
            inferred = (
                set().union(*(owners[reference] for reference in references))
                if references
                else set()
            )
            inferred &= known_patients
            if inferred and not inferred.issubset(owners[key]):
                owners[key].update(inferred)
                changed = True

    grouped_keys: dict[str, set[str]] = defaultdict(set)
    ambiguous_count = 0
    for key, patient_owners in owners.items():
        valid = patient_owners & known_patients
        if len(valid) > 1:
            ambiguous_count += 1
            continue
        if len(valid) == 1:
            grouped_keys[next(iter(valid))].add(key)

    # Pull referenced helper resources into an already-owned patient's bundle,
    # recursively.  We don't assign global ownership to helpers because the
    # same Medication/Practitioner may legitimately be shared.
    for patient_id, keys in grouped_keys.items():
        pending = list(keys)
        while pending:
            key = pending.pop()
            for reference in referenced_keys.get(key, set()):
                reference_owners = owners[reference] & known_patients
                if reference_owners and reference_owners != {patient_id}:
                    continue
                if reference not in keys:
                    keys.add(reference)
                    pending.append(reference)

    included = set().union(*grouped_keys.values()) if grouped_keys else set()
    orphans = tuple(
        unique[key]
        for key in sorted(unique)
        if key not in included and unique[key].get("resourceType") != "Patient"
    )
    bundles = {
        patient_id: make_collection_bundle(unique[key] for key in keys)
        for patient_id, keys in sorted(grouped_keys.items())
    }
    # Preserve Patients even if they have no clinical resources.
    for patient_id in sorted(known_patients):
        patient_key = f"Patient/{patient_id}"
        bundles.setdefault(patient_id, make_collection_bundle([unique[patient_key]]))
    return GroupedBundles(
        bundles=bundles,
        orphans=orphans,
        resource_count=len(unique),
        duplicate_count=duplicate_count,
        ambiguous_count=ambiguous_count,
    )


def group_ndjson(sources: str | Path | Iterable[str | Path]) -> GroupedBundles:
    return group_resources_by_patient(iter_resources(sources))


def patient_bundle_filename(patient_id: str) -> str:
    return quote(patient_id, safe="-._~") + ".bundle.json"


def write_patient_bundles(grouped: GroupedBundles, directory: str | Path) -> dict[str, Path]:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for patient_id, bundle in grouped.bundles.items():
        path = destination / patient_bundle_filename(patient_id)
        path.write_text(
            json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        result[patient_id] = path
    return result


# Natural aliases for library users and older prototypes.
read_ndjson = iter_resources
group_by_patient = group_resources_by_patient
