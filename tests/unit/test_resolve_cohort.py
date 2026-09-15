from __future__ import annotations

from pheno_rwe.omop.ddl import create_schema
from pheno_rwe.steps.resolve_cohort import resolve_plan_cohorts


def _plan() -> dict:
    return {
        "study": {"name": "test", "question": "Who qualifies?"},
        "code_sets": [
            {
                "name": "index-condition",
                "domain": "condition",
                "codings": [
                    {
                        "system": "http://snomed.info/sct",
                        "code": "123",
                        "concept_id": 99,
                        "mapping_status": "MAPPED",
                    }
                ],
            },
            {
                "name": "excluded-drug",
                "domain": "drug",
                "codings": [
                    {
                        "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                        "code": "rx",
                        "concept_id": 0,
                        "mapping_status": "UNMAPPED",
                    }
                ],
            },
        ],
        "cohorts": [
            {
                "name": "cases",
                "inclusion": [{"code_set": "index-condition", "min_count": 1}],
                "exclusion": [
                    {
                        "code_set": "excluded-drug",
                        "window": {"anchor": "index_date", "days_before": 0, "days_after": 30},
                    }
                ],
            }
        ],
        "index_date_rule": {"strategy": "first_occurrence", "code_set": "index-condition"},
        "analyses": [{"id": "t1", "kind": "table_one", "params": {"cohort": "cases"}}],
    }


def test_concept_match_source_fallback_index_and_exclusion() -> None:
    import duckdb

    connection = duckdb.connect()
    create_schema(connection)
    connection.executemany(
        """INSERT INTO omop.person
           (person_id, gender_concept_id, year_of_birth, race_concept_id, ethnicity_concept_id)
           VALUES (?, 0, 1980, 0, 0)""",
        [(1,), (2,), (3,)],
    )
    connection.executemany(
        """INSERT INTO omop.condition_occurrence
           (condition_occurrence_id, person_id, condition_concept_id,
            condition_start_date, condition_source_value)
           VALUES (?, ?, ?, ?, ?)""",
        [
            (1, 1, 99, "2024-01-01", "other"),
            (2, 2, 99, "2024-02-01", "other"),
        ],
    )
    # concept_id=0 still participates through the exact system#code source value.
    connection.execute(
        """INSERT INTO omop.drug_exposure
           (drug_exposure_id, person_id, drug_concept_id,
            drug_exposure_start_date, drug_source_value)
           VALUES (1, 2, 0, '2024-02-10',
                   'http://www.nlm.nih.gov/research/umls/rxnorm#rx')"""
    )

    result = resolve_plan_cohorts(connection, _plan())

    rows = {row.person_id: row for row in result.rows}
    assert rows[1].included is True
    assert rows[1].index_date.isoformat() == "2024-01-01"
    assert rows[2].included is False
    assert rows[2].exclusion_reason == "exclusion:excluded-drug"
    assert rows[3].exclusion_reason == "missing_index_date"
    connection.close()
