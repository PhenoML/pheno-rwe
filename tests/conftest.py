from __future__ import annotations

from typing import Any

import pytest

from pheno_rwe.omop.ddl import create_schema


@pytest.fixture
def duckdb_connection():
    import duckdb

    connection = duckdb.connect()
    create_schema(connection)
    try:
        yield connection
    finally:
        connection.close()


class FakePhenoClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def analyze_cohort(self, text: str, provider: str) -> dict[str, Any]:
        self.calls.append(("analyze_cohort", text))
        return {
            "queries": [{"resourceType": "Condition", "code": "44054006"}],
            "patientIds": ["patient-1"],
        }

    def cohort_queries(self, text: str, provider: str) -> dict[str, Any]:
        self.calls.append(("cohort_queries", text))
        return {"queries": [{"resourceType": "Condition", "code": "44054006"}]}

    def fhir_search(self, provider: str, fhir_path: str, **params: Any) -> dict[str, Any]:
        self.calls.append(("fhir_search", fhir_path))
        if fhir_path.startswith("Binary/"):
            return {"resourceType": "Binary", "contentType": "text/plain", "data": "Tm90ZQ=="}
        return {
            "resourceType": "Bundle",
            "type": "searchset",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "patient-1",
                        "text": {"status": "generated", "div": "private narrative"},
                    }
                },
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": "condition-1",
                        "subject": {"reference": "Patient/patient-1"},
                        "code": {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": "44054006",
                                    "display": "Type 2 diabetes mellitus",
                                }
                            ]
                        },
                    }
                },
            ],
        }

    def fhir2omop(self, bundle: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("fhir2omop", bundle.get("id")))
        codings = []
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            codings.extend(resource.get("code", {}).get("coding", []))
        mappings = [
            {
                "source_coding": coding,
                "source_system": coding.get("system"),
                "source_code": coding.get("code"),
                "mapping_status": "MAPPED",
                "concept_id": 201826,
            }
            for coding in codings
        ]
        return {
            "tables": {
                "person": [
                    {"person_id": 1, "person_source_value": "patient-1", "year_of_birth": 1970}
                ],
                "condition_occurrence": [
                    {
                        "condition_occurrence_id": 1,
                        "person_id": 1,
                        "condition_concept_id": 201826,
                        "condition_start_date": "2020-01-02",
                        "condition_source_value": "http://snomed.info/sct#44054006",
                    }
                ],
            },
            "mappings": mappings,
            "summary": {"total": len(mappings), "mapped": len(mappings)},
            "dropped": [],
            "vocab_version": "test-v1",
        }

    def document(self, content: bytes, mime_type: str, **options: Any) -> dict[str, Any]:
        self.calls.append(("document", mime_type))
        return {
            "resourceType": "Bundle",
            "type": "transaction",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": "temporary-1",
                        "code": {"text": "Hypertension"},
                    }
                }
            ],
        }

    def resolve_codings(self, text: str, domain: str | None = None) -> dict[str, Any]:
        self.calls.append(("resolve_codings", text))
        return {
            "codings": [
                {
                    "system": "http://snomed.info/sct",
                    "code": "44054006",
                    "display": "Type 2 diabetes mellitus",
                }
            ]
        }


@pytest.fixture
def fake_client() -> FakePhenoClient:
    return FakePhenoClient()
