from __future__ import annotations

import json

import pytest

from pheno_rwe.fhirtools.ndjson import NDJSONError, group_ndjson, group_resources_by_patient


def _ids(bundle: dict) -> set[str]:
    return {entry["resource"]["id"] for entry in bundle["entry"]}


def test_grouping_follows_encounters_and_pulls_reference_data() -> None:
    resources = [
        {"resourceType": "Patient", "id": "p1"},
        {"resourceType": "Patient", "id": "p2"},
        {"resourceType": "Encounter", "id": "e1", "subject": {"reference": "Patient/p1"}},
        {"resourceType": "Observation", "id": "o1", "encounter": {"reference": "Encounter/e1"}},
        {
            "resourceType": "MedicationRequest",
            "id": "rx2",
            "subject": {"reference": "Patient/p2"},
            "medicationReference": {"reference": "Medication/m1"},
        },
        {"resourceType": "Medication", "id": "m1", "code": {"text": "test"}},
        {"resourceType": "Condition", "id": "orphan"},
    ]

    grouped = group_resources_by_patient(resources)

    assert _ids(grouped["p1"]) == {"p1", "e1", "o1"}
    assert _ids(grouped["p2"]) == {"p2", "rx2", "m1"}
    assert [resource["id"] for resource in grouped.orphans] == ["orphan"]


def test_ndjson_reader_groups_a_directory(tmp_path) -> None:
    path = tmp_path / "patients.ndjson"
    path.write_text(
        "\n".join(
            json.dumps(value)
            for value in (
                {"resourceType": "Patient", "id": "p1"},
                {"resourceType": "Condition", "id": "c1", "subject": {"reference": "Patient/p1"}},
            )
        ),
        encoding="utf-8",
    )
    grouped = group_ndjson(tmp_path)
    assert grouped.patient_count == 1
    assert _ids(grouped["p1"]) == {"p1", "c1"}


def test_conflicting_duplicate_resource_is_refused() -> None:
    with pytest.raises(NDJSONError, match="Conflicting duplicate"):
        group_resources_by_patient(
            [
                {"resourceType": "Patient", "id": "p1", "gender": "female"},
                {"resourceType": "Patient", "id": "p1", "gender": "male"},
            ]
        )
