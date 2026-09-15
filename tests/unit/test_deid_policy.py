from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from pheno_rwe.deid.policy import (
    DEIDENTIFIED_COLUMN_POLICY,
    NORMALIZED_OPERATIONAL_TIME,
    POLICY_VERSION,
    ColumnAction,
    keyed_value,
    policy_summary,
    require_deidentified_policy,
    schema_policy_errors,
)
from pheno_rwe.hashing import hash_file, hash_json
from pheno_rwe.manifest import record_step
from pheno_rwe.omop.ddl import create_schema
from pheno_rwe.omop.loader import connect_database
from pheno_rwe.plan import analysis_data_hash, load_plan, plan_hash, write_plan
from pheno_rwe.steps.deid import deidentify
from pheno_rwe.steps.export import export_study
from pheno_rwe.steps.review_codes import review_code_set
from pheno_rwe.steps.trace import trace_study
from pheno_rwe.steps.validate import validate_study
from pheno_rwe.workspace import create_study

SALT = b"synthetic-policy-test-salt"
RAW_PATIENT = "CANARY_PATIENT_RAW_123"
RAW_RESOURCE = "CANARY_RESOURCE_RAW_456"
RAW_DOCUMENT_REFERENCE = "CANARY_DOCUMENT_REFERENCE_RAW_789"


def _plan() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "study": {"name": "Privacy canary", "question": "Does the privacy policy hold?"},
        "code_sets": [
            {
                "name": "condition",
                "domain": "condition",
                "codings": [
                    {
                        "system": "https://example.test/codes",
                        "code": "safe",
                        "display": "Synthetic condition",
                        "concept_id": 123,
                        "mapping_status": "MAPPED",
                    }
                ],
            }
        ],
        "cohorts": [{"name": "All"}],
        "index_date_rule": {"strategy": "fixed_date", "fixed_date": "2024-01-01"},
        "deid": {"tier": "baseline"},
        "analyses": [
            {
                "id": "baseline",
                "kind": "table_one",
                "params": {"cohort": "All", "variables": ["age_bucket"]},
            }
        ],
        "plots": [],
    }


def _safe_string(column: str) -> str:
    return {
        "status": "success",
        "origin": "structured",
        "mapping_status": "MAPPED",
        "resource_type": "Condition",
        "omop_table": "condition_occurrence",
        "column_name": "condition_start_date",
        "cohort_name": "All",
        "code_set_name": "condition",
        "domain": "condition",
        "source_system": "https://example.test/codes",
        "source_code": "safe",
        "source_value": "https://example.test/codes#safe",
        "measurement_source_value": "https://example.test/codes#safe",
        "condition_source_value": "https://example.test/codes#safe",
        "drug_source_value": "https://example.test/codes#safe",
        "procedure_source_value": "https://example.test/codes#safe",
        "observation_source_value": "https://example.test/codes#safe",
        "visit_source_value": "https://example.test/codes#safe",
        "cause_source_value": "https://example.test/codes#safe",
        "unit_source_value": "mg/dL",
        "gender_source_value": "F",
        "race_source_value": "2106-3",
        "ethnicity_source_value": "2186-2",
        "age_bucket": "50-54",
        "stage": "initial",
        "detail": "safe aggregate criterion",
        "exclusion_reason": "age",
        "display": "Safe vocabulary concept",
        "vocab_version": "synthetic-v1",
    }.get(column, f"safe_{column}")


def _raw_hmac_value(namespace: str) -> str:
    if namespace == "source-patient-id":
        return RAW_PATIENT
    if namespace == "fhir-resource-id":
        return RAW_RESOURCE
    if namespace == "document-reference-id":
        return RAW_DOCUMENT_REFERENCE
    return hashlib.sha256(f"IDENTIFIED_CONTENT::{namespace}".encode()).hexdigest()


def _value_for(
    table: str,
    column: str,
    data_type: str,
    raw_values: dict[tuple[str, str], Any],
    canaries: set[str],
) -> Any:
    rule = DEIDENTIFIED_COLUMN_POLICY[table].columns[column]
    if rule.action == ColumnAction.HMAC:
        value = _raw_hmac_value(str(rule.namespace))
        raw_values[(table, column)] = value
        canaries.add(value)
        return value
    if data_type == "VARCHAR":
        if rule.action in {ColumnAction.NULL, ColumnAction.DROP_WITH_TABLE}:
            value = f"CANARY_UNSAFE_FREE_TEXT::{table}.{column}::PATIENT_RAW_123"
            canaries.add(value)
            return value
        return _safe_string(column)
    if data_type in {"BIGINT", "INTEGER"}:
        if table == "omop.person" and column == "year_of_birth":
            return 1970
        return 1
    if data_type == "DOUBLE":
        return 1.5
    if data_type == "BOOLEAN":
        return True
    if data_type == "DATE":
        return date(2024, 1, 1)
    if data_type.startswith("TIMESTAMP"):
        return datetime(2024, 1, 1, 12, 30, 45)
    raise AssertionError(f"unhandled synthetic type {table}.{column}: {data_type}")


def _populate_every_policy_column(connection: Any) -> tuple[dict[tuple[str, str], Any], set[str]]:
    rows = connection.execute(
        """SELECT table_schema, table_name, column_name, data_type
           FROM information_schema.columns
           WHERE table_catalog=current_database()
             AND table_schema NOT IN ('information_schema', 'pg_catalog')"""
    ).fetchall()
    types = {
        (f"{schema}.{table}", str(column)): str(data_type)
        for schema, table, column, data_type in rows
    }
    raw_values: dict[tuple[str, str], Any] = {}
    canaries = {RAW_PATIENT, RAW_RESOURCE, RAW_DOCUMENT_REFERENCE}
    for table, table_rule in DEIDENTIFIED_COLUMN_POLICY.items():
        schema, table_name = table.split(".", 1)
        columns = list(table_rule.columns)
        values = [
            _value_for(table, column, types[(table, column)], raw_values, canaries)
            for column in columns
        ]
        quoted_columns = ", ".join(f'"{column}"' for column in columns)
        placeholders = ", ".join("?" for _ in columns)
        connection.execute(
            f'INSERT INTO "{schema}"."{table_name}" ({quoted_columns}) VALUES ({placeholders})',
            values,
        )
    return raw_values, canaries


def _assert_all_policy_values(
    connection: Any,
    raw_values: dict[tuple[str, str], Any],
) -> int:
    checked = 0
    for table, table_rule in DEIDENTIFIED_COLUMN_POLICY.items():
        schema, table_name = table.split(".", 1)
        columns = list(table_rule.columns)
        if table_rule.delete_rows:
            assert connection.execute(
                f'SELECT COUNT(*) FROM "{schema}"."{table_name}"'
            ).fetchone() == (0,)
            checked += len(columns)
            continue
        quoted_columns = ", ".join(f'"{column}"' for column in columns)
        rows = connection.execute(
            f'SELECT {quoted_columns} FROM "{schema}"."{table_name}"'
        ).fetchall()
        assert rows, table
        for index, column in enumerate(columns):
            rule = table_rule.columns[column]
            values = [row[index] for row in rows]
            if rule.action == ColumnAction.NULL:
                assert all(value is None for value in values), f"{table}.{column}"
            elif rule.action == ColumnAction.HMAC:
                expected = keyed_value(
                    raw_values[(table, column)],
                    SALT,
                    namespace=str(rule.namespace),
                    prefix=str(rule.prefix),
                )
                assert all(value == expected for value in values), f"{table}.{column}"
            elif rule.action == ColumnAction.CONSTANT:
                assert all(value == NORMALIZED_OPERATIONAL_TIME for value in values)
            else:
                # KEEP preserves nullable schema semantics; the physical canary
                # scan below verifies that rewritten rows do not retain PHI.
                assert rule.action == ColumnAction.KEEP
            checked += 1
    return checked


def _workspace_with_canaries(tmp_path: Path):
    workspace = create_study(tmp_path / "study", "Privacy canary")
    write_plan(workspace.plan_path, _plan())
    artifact = {
        "schema_version": "1.0",
        "name": "condition",
        "description": "synthetic condition",
        "domain": "condition",
        "codings": [
            {
                "system": "https://example.test/codes",
                "code": "safe",
                "display": "Synthetic condition",
                "source_value": "https://example.test/codes#safe",
                "concept_id": 123,
                "mapping_status": "MAPPED",
                "accepted": True,
            }
        ],
        "mapping_status_counts": {"MAPPED": 1},
        "approval": {
            "status": "pending",
            "reviewed_by": None,
            "reviewed_at": None,
            "notes": None,
            "review_hash": None,
        },
    }
    artifact_path = workspace.root / "codesets" / "condition.codeset.json"
    artifact_path.write_text(json.dumps(artifact, sort_keys=True) + "\n", encoding="utf-8")
    review_code_set(
        workspace,
        name="condition",
        decision="approved",
        reviewed_by="Synthetic researcher",
        reviewed_at="2026-08-12T12:00:00Z",
    )
    connection = connect_database(workspace.identified_db)
    try:
        create_schema(connection)
        raw_values, canaries = _populate_every_policy_column(connection)
    finally:
        connection.close()
    # This exhaustive policy fixture deliberately pre-populates every cohort
    # column, so invoking the real resolver would replace its synthetic canary
    # row. Record the exact production provenance contract for that prepared
    # database instead.
    plan = load_plan(workspace.plan_path)
    database_hash = hash_file(workspace.identified_db)
    data_hash = analysis_data_hash(plan)
    current_plan_hash = plan_hash(plan)
    record_step(
        workspace.manifest_path,
        step="resolve-cohort",
        status="success",
        started_at=datetime.now(UTC),
        params={
            "cohorts": [cohort.name for cohort in plan.cohorts],
            "plan_hash": current_plan_hash,
            "analysis_data_sha256": data_hash,
            "identified_database_sha256": database_hash,
        },
        input_signature=hash_json(
            {"plan_hash": current_plan_hash, "analysis_data_sha256": data_hash}
        ),
        inputs={workspace.relative(workspace.plan_path): hash_file(workspace.plan_path)},
        outputs={workspace.relative(workspace.identified_db): database_hash},
    )
    deidentify(workspace, salt=SALT)
    validation = validate_study(workspace, include_data=False)
    assert validation.status == "valid_static"
    return workspace, raw_values, canaries


def test_policy_is_fail_closed_for_unknown_tables_and_columns(tmp_path: Path) -> None:
    connection = connect_database(tmp_path / "policy.duckdb")
    try:
        create_schema(connection)
        assert schema_policy_errors(connection) == ()
        connection.execute("ALTER TABLE omop.person ADD COLUMN unsafe_notes VARCHAR")
        assert any("unsafe_notes" in error for error in schema_policy_errors(connection))
        with pytest.raises(RuntimeError, match="unsafe_notes"):
            require_deidentified_policy(connection)
    finally:
        connection.close()


def test_canaries_are_scrubbed_from_every_deidentified_column_and_physical_file(
    tmp_path: Path,
) -> None:
    workspace, raw_values, canaries = _workspace_with_canaries(tmp_path)
    connection = connect_database(workspace.deidentified_db, read_only=True)
    try:
        require_deidentified_policy(connection)
        checked = _assert_all_policy_values(connection, raw_values)
    finally:
        connection.close()
    assert checked == sum(
        len(table_rule.columns) for table_rule in DEIDENTIFIED_COLUMN_POLICY.values()
    )
    database_bytes = workspace.deidentified_db.read_bytes()
    for canary in canaries:
        assert canary.encode() not in database_bytes


def test_exported_parquet_and_database_follow_every_column_rule(tmp_path: Path) -> None:
    workspace, raw_values, canaries = _workspace_with_canaries(tmp_path)
    shared = tmp_path / "shared"
    export_study(workspace, output=shared)

    bundle = json.loads((shared / "bundle.json").read_text(encoding="utf-8"))
    assert bundle["database_policy"] == policy_summary()
    assert bundle["database_policy"]["version"] == POLICY_VERSION
    assert bundle["privacy"] == {
        "identified_paths_included": False,
        "identified_artifact_hashes_included": False,
        "identified_content_hashes_included": False,
        "study_keyed_content_fingerprints_included": True,
        "patient_or_document_item_ledgers_included": False,
        "source_queries_included": False,
        "unsafe_free_text_in_database_included": False,
        "original_invalid_temporal_values_included": False,
    }
    expected_parquet = {
        f"parquet/{table.replace('.', '__')}.parquet" for table in DEIDENTIFIED_COLUMN_POLICY
    }
    assert {path for path in bundle["files"] if path.startswith("parquet/")} == expected_parquet
    assert "deid.duckdb" in bundle["files"]

    import duckdb

    connection = duckdb.connect()
    try:
        checked = 0
        for table, table_rule in DEIDENTIFIED_COLUMN_POLICY.items():
            parquet = shared / "parquet" / f"{table.replace('.', '__')}.parquet"
            cursor = connection.execute("SELECT * FROM read_parquet(?)", [str(parquet)])
            assert [description[0] for description in cursor.description] == list(
                table_rule.columns
            )
            rows = cursor.fetchall()
            if table_rule.delete_rows:
                assert rows == []
                checked += len(table_rule.columns)
                continue
            assert rows
            for index, column in enumerate(table_rule.columns):
                rule = table_rule.columns[column]
                values = [row[index] for row in rows]
                if rule.action == ColumnAction.NULL:
                    assert all(value is None for value in values), f"{table}.{column}"
                elif rule.action == ColumnAction.HMAC:
                    expected = keyed_value(
                        raw_values[(table, column)],
                        SALT,
                        namespace=str(rule.namespace),
                        prefix=str(rule.prefix),
                    )
                    assert all(value == expected for value in values), f"{table}.{column}"
                elif rule.action == ColumnAction.CONSTANT:
                    assert all(value == NORMALIZED_OPERATIONAL_TIME for value in values)
                else:
                    assert rule.action == ColumnAction.KEEP
                checked += 1
    finally:
        connection.close()
    assert checked == sum(
        len(table_rule.columns) for table_rule in DEIDENTIFIED_COLUMN_POLICY.values()
    )

    for artifact in shared.rglob("*"):
        if not artifact.is_file():
            continue
        content = artifact.read_bytes()
        for canary in canaries:
            assert canary.encode() not in content, artifact
    verification = trace_study(shared, verify=True).verification
    assert verification is not None and verification.valid
