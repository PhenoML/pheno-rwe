from __future__ import annotations

from pheno_rwe.omop.loader import OmopLoader


def _response(*, condition_date: str = "2024-01-10") -> dict:
    return {
        "tables": {
            "person": [
                {
                    "person_id": 1,
                    "gender_concept_id": 0,
                    "year_of_birth": 1980,
                    "race_concept_id": 0,
                    "ethnicity_concept_id": 0,
                    "person_source_value": "source-patient",
                }
            ],
            "condition_occurrence": [
                {
                    "condition_occurrence_id": 1,
                    "person_id": 1,
                    "condition_concept_id": 201826,
                    "condition_start_date": condition_date,
                    "condition_type_concept_id": 32817,
                    "condition_source_value": "http://snomed.info/sct#44054006",
                }
            ],
        },
        "mappings": [
            {
                "resource_type": "Condition",
                "resource_id": "enriched-condition",
                "omop_table": "condition_occurrence",
                "omop_id": 1,
                "source_system": "http://snomed.info/sct",
                "source_code": "44054006",
                "mapping_status": "MAPPED",
            }
        ],
        "summary": {
            "codes_already_standard": 0,
            "codes_normalized": 1,
            "codes_unmapped": 0,
            "off_vocab_rate": 1.0,
        },
        "dropped": [],
        "vocab_version": "test-vocab",
    }


def test_loader_reuses_block_and_replaces_patient_atomically(tmp_path) -> None:
    database = tmp_path / "omop.duckdb"
    with OmopLoader(database) as loader:
        first = loader.materialize_patient(
            "source-patient",
            _response(condition_date="not-a-date"),
            bundle_hash="one",
        )
        assert first.person_id == 1_000_001
        assert first.invalid_date_count == 1

        second = loader.materialize_patient(
            "source-patient", _response(condition_date="2024-01-11"), bundle_hash="two"
        )
        assert second.id_offset == first.id_offset
        assert second.person_id == first.person_id
        assert loader.connection.execute(
            "SELECT COUNT(*) FROM omop.condition_occurrence"
        ).fetchone() == (1,)
        assert loader.connection.execute("SELECT COUNT(*) FROM meta.invalid_date").fetchone() == (
            0,
        )
        mapping_id = loader.connection.execute("SELECT omop_id FROM meta.mapping").fetchone()[0]
        assert mapping_id == 1_000_001


def test_loader_records_enrichment_provenance(tmp_path) -> None:
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p1"}},
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "enriched-condition",
                    "meta": {
                        "tag": [
                            {
                                "system": "https://phenoml.com/pheno-rwe/origin",
                                "code": "enriched",
                            },
                            {
                                "system": "https://phenoml.com/pheno-rwe/document-reference",
                                "code": "doc-1",
                            },
                        ]
                    },
                }
            },
        ],
    }
    with OmopLoader(tmp_path / "data.duckdb") as loader:
        loader.materialize_patient("p1", _response(), bundle_hash="one", bundle=bundle)
        assert loader.connection.execute(
            "SELECT origin, doc_ref_id FROM meta.row_provenance"
        ).fetchone() == ("enriched", "doc-1")
