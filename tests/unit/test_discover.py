from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from pheno_rwe.steps.discover import crosswalk_code, extract_codes, search_fhir
from pheno_rwe.steps.pull import preview_from_ids


class DiscoveryTransport:
    """Offline transport returning canned extract/crosswalk/search responses."""

    def __init__(self) -> None:
        self.searches: list[tuple[str, str, dict[str, Any]]] = []
        self.crosswalks: list[tuple[str, str, list[str]]] = []
        self.resolve_args: tuple[str, str | None] | None = None

    def resolve_codings(self, text: str, domain: str | None = None) -> dict[str, Any]:
        self.resolve_args = (text, domain)
        return {
            "codings": [
                {"system": "http://snomed.info/sct", "code": "44054006", "display": "T2DM"},
                # Duplicate (system, code) must collapse to a single coding.
                {"system": "http://snomed.info/sct", "code": "44054006", "display": "dup"},
                {"system": "http://loinc.org", "code": "1234-5", "display": "Lab"},
            ]
        }

    def crosswalk(self, system: str, code: str, targets: Sequence[str]) -> dict[str, Any]:
        # Mirror the real CrosswalkResponse: top-level system/code are the *source*,
        # and matches are nested under per-target-system groups.
        self.crosswalks.append((system, code, list(targets)))
        return {
            "resolved_umls_release": "2025AA",
            "system": system,
            "code": code,
            "targets": [
                {
                    "system": targets[0],
                    "matches": [
                        {"code": "rx-1", "display": "map 1", "cui": "C1"},
                        {"code": "rx-2", "display": "map 2", "cui": "C2"},
                        # Duplicate (system, code) must collapse.
                        {"code": "rx-1", "display": "dup", "cui": "C1"},
                    ],
                },
                {"system": targets[1], "reason_code": "NO_MATCH", "matches": []},
            ],
        }

    def fhir_search(self, provider: str, fhir_path: str, **params: Any) -> dict[str, Any]:
        self.searches.append((provider, fhir_path, params))
        if params.get("_summary") == "count":
            return {"resourceType": "Bundle", "type": "searchset", "total": 7}
        if fhir_path == "Patient":
            return {
                "resourceType": "Bundle",
                "entry": [
                    {"resource": {"resourceType": "Patient", "id": "P1"}},
                    {"resource": {"resourceType": "Patient", "id": "P3"}},
                ],
            }
        if fhir_path == "MedicationRequest":
            # Page 2 is reached by re-issuing the *original* path with the next link's
            # query (absolute next URLs from the server are not valid proxy paths).
            if params.get("_offset") == "20":
                return {
                    "resourceType": "Bundle",
                    "entry": [
                        _med("m3", subject="Patient/P2"),  # duplicate patient across pages
                        _med("m4", patient="Patient/P3"),  # alternate reference field
                    ],
                }
            return {
                "resourceType": "Bundle",
                "entry": [
                    _med("m1", subject="Patient/P1"),
                    _med("m2", subject="Patient/P2"),
                ],
                "link": [
                    {
                        "relation": "next",
                        "url": (
                            "https://fhir.example.test/fhir/R4/"
                            "MedicationRequest?code=x&_count=20&_offset=20"
                        ),
                    }
                ],
            }
        raise AssertionError(f"unexpected fhir_search: {fhir_path} {params}")


def _med(
    resource_id: str, *, subject: str | None = None, patient: str | None = None
) -> dict[str, Any]:
    resource: dict[str, Any] = {"resourceType": "MedicationRequest", "id": resource_id}
    if subject is not None:
        resource["subject"] = {"reference": subject}
    if patient is not None:
        resource["patient"] = {"reference": patient}
    return {"resource": resource}


def test_extract_codes_returns_deduped_codings() -> None:
    transport = DiscoveryTransport()

    result = extract_codes(transport, "type 2 diabetes", "condition")

    assert transport.resolve_args == ("type 2 diabetes", "condition")
    assert result.step == "extract-codes"
    assert result.status == "success"
    assert result.items == [
        {"system": "http://snomed.info/sct", "code": "44054006", "display": "T2DM"},
        {"system": "http://loinc.org", "code": "1234-5", "display": "Lab"},
    ]


def test_crosswalk_code_maps_to_each_requested_target() -> None:
    transport = DiscoveryTransport()
    targets = ["http://www.nlm.nih.gov/research/umls/rxnorm", "http://hl7.org/fhir/sid/ndc"]

    result = crosswalk_code(
        transport, system="http://snomed.info/sct", code="44054006", targets=targets
    )

    assert transport.crosswalks == [("http://snomed.info/sct", "44054006", targets)]
    assert result.step == "crosswalk"
    # Flattened target codings only, deduped; the unmatched second target drops out.
    assert result.items == [
        {"system": targets[0], "code": "rx-1", "display": "map 1", "cui": "C1"},
        {"system": targets[0], "code": "rx-2", "display": "map 2", "cui": "C2"},
    ]
    # The source coding must never leak into the target-coding list.
    assert all(coding["system"] != "http://snomed.info/sct" for coding in result.items)


def test_search_fhir_count_injects_summary_and_reports_total() -> None:
    transport = DiscoveryTransport()

    result = search_fhir(transport, "prov", "MedicationRequest", {"code": "x"}, count=True)

    assert transport.searches == [("prov", "MedicationRequest", {"code": "x", "_summary": "count"})]
    assert result.items == [{"total": 7}]
    assert "7" in result.message


def test_search_fhir_patients_dedupes_distinct_ids_across_pages() -> None:
    transport = DiscoveryTransport()

    result = search_fhir(transport, "prov", "MedicationRequest", {"code": "x"}, patients=True)

    # Both pages hit the original resource path; the second call carries the next-page
    # query (the absolute next URL must not be forwarded as a proxy path).
    assert [path for _, path, _ in transport.searches] == ["MedicationRequest", "MedicationRequest"]
    assert transport.searches[0][2] == {"code": "x"}
    assert transport.searches[1][2].get("_offset") == "20"
    assert result.items == [{"id": "P1"}, {"id": "P2"}, {"id": "P3"}]
    assert "3 distinct patients" in result.message


def test_search_fhir_patients_reads_patient_resource_ids() -> None:
    transport = DiscoveryTransport()

    result = search_fhir(transport, "prov", "Patient", {"gender": "male"}, patients=True)

    assert result.items == [{"id": "P1"}, {"id": "P3"}]


def test_search_fhir_defaults_to_first_page_summary() -> None:
    transport = DiscoveryTransport()

    result = search_fhir(transport, "prov", "MedicationRequest", {"code": "x"})

    # Summary mode never follows the next link.
    assert len(transport.searches) == 1
    summary = result.items[0]
    assert summary["resourceType"] == "Bundle"
    assert summary["entry_count"] == 2
    assert summary["has_next"] is True


def test_search_fhir_rejects_count_and_patients_together() -> None:
    transport = DiscoveryTransport()

    with pytest.raises(ValueError, match="not both"):
        search_fhir(transport, "prov", "Patient", {}, count=True, patients=True)


def test_preview_from_ids_builds_preview_without_a_service_call() -> None:
    preview = preview_from_ids(["a", "b", "c"])

    assert preview.patient_ids == ["a", "b", "c"]
    assert preview.queries == []
    assert preview.raw == {}
