from __future__ import annotations

import pytest

from pheno_rwe.omop.ddl import TABLE_SPECS, registry_errors
from pheno_rwe.omop.rekey import BLOCK_SIZE, RekeyError, rekey_response


def test_registry_is_explicit_and_self_consistent() -> None:
    assert len(TABLE_SPECS) == 12
    assert registry_errors() == ()


def test_rekey_changes_row_ids_and_foreign_keys_but_not_concepts() -> None:
    response = {
        "tables": {
            "person": [{"person_id": 1, "gender_concept_id": 0, "location_id": 2}],
            "condition_occurrence": [
                {
                    "condition_occurrence_id": 7,
                    "person_id": 1,
                    "visit_occurrence_id": 3,
                    "condition_concept_id": 201826,
                    "condition_source_concept_id": 0,
                }
            ],
        },
        "mappings": [
            {"omop_table": "condition_occurrence", "omop_id": 7, "mapping_status": "MAPPED"}
        ],
    }

    result = rekey_response(response, BLOCK_SIZE)

    assert result["tables"]["person"][0] == {
        "person_id": BLOCK_SIZE + 1,
        "gender_concept_id": 0,
        "location_id": BLOCK_SIZE + 2,
    }
    condition = result["tables"]["condition_occurrence"][0]
    assert condition["condition_occurrence_id"] == BLOCK_SIZE + 7
    assert condition["person_id"] == BLOCK_SIZE + 1
    assert condition["visit_occurrence_id"] == BLOCK_SIZE + 3
    assert condition["condition_concept_id"] == 201826
    assert condition["condition_source_concept_id"] == 0
    assert result["mappings"][0]["omop_id"] == BLOCK_SIZE + 7


def test_rekey_refuses_ids_that_escape_the_patient_block() -> None:
    with pytest.raises(RekeyError, match="exceeds"):
        rekey_response(
            {"tables": {"person": [{"person_id": BLOCK_SIZE}]}, "mappings": []},
            BLOCK_SIZE,
        )
