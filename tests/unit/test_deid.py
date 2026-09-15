from __future__ import annotations

import json
from datetime import date

import pytest

from pheno_rwe.deid.dateshift import (
    apply_date_shift,
    date_registry_errors,
    patient_shift_days,
)
from pheno_rwe.errors import StalePlanError
from pheno_rwe.hashing import hash_file
from pheno_rwe.manifest import read_manifest
from pheno_rwe.omop.ddl import create_schema
from pheno_rwe.omop.loader import OmopLoader, connect_database
from pheno_rwe.plan import analysis_data_hash, load_plan, plan_hash, write_plan
from pheno_rwe.steps.deid import deid_freshness, deidentify, rebuild_deidentified_database
from pheno_rwe.steps.resolve_cohort import resolve_cohort
from pheno_rwe.workspace import create_study


def _identified_database(path) -> None:
    connection = connect_database(path)
    create_schema(connection)
    connection.execute(
        """INSERT INTO omop.person
           (person_id, gender_concept_id, year_of_birth, month_of_birth,
            day_of_birth, birth_datetime, race_concept_id, ethnicity_concept_id,
            person_source_value)
           VALUES (1, 0, 1980, 5, 12, '1980-05-12', 0, 0, 'source-patient')"""
    )
    connection.execute(
        """INSERT INTO omop.observation_period VALUES
           (1, 1, '2024-01-01', '2024-12-31', 32817)"""
    )
    connection.execute(
        """INSERT INTO omop.condition_occurrence
           (condition_occurrence_id, person_id, condition_concept_id,
            condition_start_date, condition_end_date)
           VALUES (1, 1, 99, '2024-02-01', '2024-02-11')"""
    )
    connection.execute("INSERT INTO study.cohort VALUES ('cases', 1, '2024-02-01', true, NULL)")
    connection.execute(
        """INSERT INTO meta.ingest_patient VALUES
           ('source-patient', 1, 1000000, 'h', 'r', 'success', NULL, current_timestamp)"""
    )
    connection.close()


def _study_plan() -> dict:
    return {
        "study": {"name": "Freshness", "question": "Is analysis data current?"},
        "code_sets": [
            {
                "name": "index",
                "domain": "condition",
                "codings": [
                    {
                        "system": "snomed",
                        "code": "1",
                        "concept_id": 99,
                        "mapping_status": "MAPPED",
                    }
                ],
            }
        ],
        "cohorts": [{"name": "cases"}],
        "index_date_rule": {"strategy": "first_occurrence", "code_set": "index"},
        "deid": {"tier": "baseline"},
        "analyses": [{"id": "table", "kind": "table_one", "params": {"cohort": "cases"}}],
    }


def _workspace_after_cohort_resolution(tmp_path):
    workspace = create_study(tmp_path / "study", "Freshness")
    plan = _study_plan()
    write_plan(workspace.plan_path, plan)
    _identified_database(workspace.identified_db)
    resolve_cohort(workspace)
    return workspace, plan


def test_shift_is_deterministic_bounded_and_preserves_intervals() -> None:
    import duckdb

    connection = duckdb.connect()
    create_schema(connection)
    connection.execute(
        """INSERT INTO omop.person
           (person_id, gender_concept_id, year_of_birth, race_concept_id, ethnicity_concept_id)
           VALUES (1, 0, 1980, 0, 0)"""
    )
    connection.execute(
        """INSERT INTO omop.condition_occurrence
           (condition_occurrence_id, person_id, condition_start_date, condition_end_date)
           VALUES (1, 1, '2024-01-01', '2024-01-11')"""
    )
    assert date_registry_errors(connection) == ()
    expected = patient_shift_days(1, b"salt", max_days=30)
    assert -30 <= expected <= 30
    assert expected == patient_shift_days(1, b"salt", max_days=30)

    apply_date_shift(connection, b"salt", max_days=30)
    interval = connection.execute(
        "SELECT condition_start_date, condition_end_date FROM omop.condition_occurrence"
    ).fetchone()
    assert interval is not None
    start, end = interval
    assert (end - start).days == 10
    assert (start - date(2024, 1, 1)).days == expected
    connection.close()


def test_rebuild_drops_identifiers_and_shifts_index_coherently(tmp_path) -> None:
    source = tmp_path / "identified.duckdb"
    output = tmp_path / "deid.duckdb"
    _identified_database(source)

    result = rebuild_deidentified_database(
        source,
        output,
        config={"tier": "date_shift", "date_shift_max_days": 30},
        salt=b"stable-salt",
    )

    assert not any(result.identifier_audit.values())
    import duckdb

    connection = duckdb.connect(str(output), read_only=True)
    person = connection.execute(
        """SELECT year_of_birth, month_of_birth, day_of_birth,
                  birth_datetime, person_source_value
           FROM omop.person"""
    ).fetchone()
    assert person == (None, None, None, None, None)
    condition_row = connection.execute(
        "SELECT condition_start_date FROM omop.condition_occurrence"
    ).fetchone()
    index_row = connection.execute("SELECT index_date FROM study.cohort").fetchone()
    assert condition_row is not None and index_row is not None
    condition_date = condition_row[0]
    index_date = index_row[0]
    assert condition_date == index_date
    connection.close()


def test_resolve_cohort_records_provenance_consumed_by_deid(tmp_path) -> None:
    workspace, _ = _workspace_after_cohort_resolution(tmp_path)
    loaded = load_plan(workspace.plan_path)
    resolve_entry = read_manifest(workspace.manifest_path)[-1]

    assert resolve_entry["step"] == "resolve-cohort"
    assert resolve_entry["params"]["plan_hash"] == plan_hash(loaded)
    assert resolve_entry["params"]["analysis_data_sha256"] == analysis_data_hash(loaded)
    assert resolve_entry["params"]["identified_database_sha256"] == hash_file(
        workspace.identified_db
    )
    assert resolve_entry["outputs"]["omop/omop.duckdb"] == hash_file(workspace.identified_db)

    result = deidentify(workspace, salt=b"provenance-test-salt")

    assert result.status == "success"
    report = json.loads(
        (workspace.root / "reports" / "deid_report.json").read_text(encoding="utf-8")
    )
    assert report["analysis_data_sha256"] == analysis_data_hash(loaded)
    assert report["cohort_resolution_entry_hash"] == resolve_entry["entry_hash"]
    assert report["cohort_resolution_plan_sha256"] == plan_hash(loaded)
    assert deid_freshness(workspace) == (True, "current")


def test_deidentify_refuses_materialized_plan_amendment_after_resolve(tmp_path) -> None:
    workspace, plan = _workspace_after_cohort_resolution(tmp_path)
    plan["code_sets"][0]["codings"][0]["code"] = "2"
    write_plan(workspace.plan_path, plan)

    with pytest.raises(StalePlanError, match="changed after cohort resolution"):
        deidentify(workspace, salt=b"provenance-test-salt")

    assert not workspace.deidentified_db.exists()
    assert not (workspace.root / "omop" / ".deid_salt").exists()


def test_deidentify_refuses_materialization_after_cohort_resolution(tmp_path) -> None:
    workspace, _ = _workspace_after_cohort_resolution(tmp_path)
    with OmopLoader(workspace.identified_db) as loader:
        loader.materialize_patient(
            "source-patient",
            {
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
                            "condition_concept_id": 99,
                            "condition_start_date": "2024-03-01",
                        }
                    ],
                },
                "mappings": [],
                "dropped": [],
            },
            bundle_hash="rematerialized",
        )

    with pytest.raises(RuntimeError, match="database changed after cohort resolution"):
        deidentify(workspace, salt=b"provenance-test-salt")

    assert not workspace.deidentified_db.exists()


def test_deidentify_refuses_config_override_that_differs_from_plan(tmp_path) -> None:
    workspace, _ = _workspace_after_cohort_resolution(tmp_path)

    with pytest.raises(ValueError, match="config override differs from plan.json"):
        deidentify(
            workspace,
            config={"tier": "date_shift", "date_shift_max_days": 30},
            salt=b"provenance-test-salt",
        )


def test_deid_freshness_invalidates_materialized_plan_amendments(tmp_path) -> None:
    workspace, plan = _workspace_after_cohort_resolution(tmp_path)
    deidentify(workspace, salt=b"provenance-test-salt")

    assert deid_freshness(workspace) == (True, "current")
    plan["analyses"][0]["params"]["variables"] = ["age_bucket"]
    write_plan(workspace.plan_path, plan)
    assert deid_freshness(workspace) == (True, "current")

    plan["code_sets"][0]["codings"][0]["code"] = "2"
    write_plan(workspace.plan_path, plan)

    fresh, reason = deid_freshness(workspace)
    assert fresh is False
    assert "code sets, cohorts, or index-date" in reason
