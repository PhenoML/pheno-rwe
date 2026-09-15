"""Fail-closed column policy for de-identified and shareable databases.

Every application table and column must appear here. Export refuses databases
whose schema is newer or different from this registry, so a newly added field
cannot silently cross the sharing boundary.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from pheno_rwe.omop.ddl import TABLE_SPECS, quote_identifier

POLICY_VERSION = "shareable-columns-v1"
NORMALIZED_OPERATIONAL_TIME = datetime(1970, 1, 1)


class ColumnAction(StrEnum):
    KEEP = "keep"
    NULL = "null"
    HMAC = "hmac"
    CONSTANT = "constant"
    DROP_WITH_TABLE = "drop_with_table"


@dataclass(frozen=True, slots=True)
class ColumnRule:
    action: ColumnAction
    namespace: str | None = None
    prefix: str | None = None
    constant: Any = None


@dataclass(frozen=True, slots=True)
class TableRule:
    columns: dict[str, ColumnRule]
    delete_rows: bool = False


@dataclass(frozen=True, slots=True)
class PolicyApplication:
    hmac_values: int
    nulled_values: int
    deleted_rows: int
    normalized_values: int


@dataclass(frozen=True, slots=True)
class PolicyAudit:
    valid: bool
    errors: tuple[str, ...]
    violations: dict[str, int]


def _rules(
    *,
    keep: tuple[str, ...] = (),
    null: tuple[str, ...] = (),
    hmac_rules: dict[str, tuple[str, str]] | None = None,
    constants: dict[str, Any] | None = None,
    drop_with_table: tuple[str, ...] = (),
) -> dict[str, ColumnRule]:
    result: dict[str, ColumnRule] = {}

    def add(name: str, rule: ColumnRule) -> None:
        if name in result:
            raise AssertionError(f"duplicate de-identification policy for column {name}")
        result[name] = rule

    for name in keep:
        add(name, ColumnRule(ColumnAction.KEEP))
    for name in null:
        add(name, ColumnRule(ColumnAction.NULL))
    for name, (namespace, prefix) in (hmac_rules or {}).items():
        add(name, ColumnRule(ColumnAction.HMAC, namespace=namespace, prefix=prefix))
    for name, value in (constants or {}).items():
        add(name, ColumnRule(ColumnAction.CONSTANT, constant=value))
    for name in drop_with_table:
        add(name, ColumnRule(ColumnAction.DROP_WITH_TABLE))
    return result


def _drop_table(*columns: str) -> TableRule:
    return TableRule(_rules(drop_with_table=tuple(columns)), delete_rows=True)


# OMOP source-value fields retained below are controlled coding/unit fields used
# for concept_id=0 fallback and analysis labeling. Narrative/value text is not.
DEIDENTIFIED_COLUMN_POLICY: dict[str, TableRule] = {
    "omop.location": _drop_table(
        "location_id",
        "address_1",
        "address_2",
        "city",
        "state",
        "zip",
        "county",
        "location_source_value",
        "country_concept_id",
        "country_source_value",
        "latitude",
        "longitude",
    ),
    "omop.care_site": _drop_table(
        "care_site_id",
        "care_site_name",
        "place_of_service_concept_id",
        "location_id",
        "care_site_source_value",
        "place_of_service_source_value",
    ),
    "omop.provider": _drop_table(
        "provider_id",
        "provider_name",
        "npi",
        "dea",
        "specialty_concept_id",
        "care_site_id",
        "year_of_birth",
        "gender_concept_id",
        "provider_source_value",
        "specialty_source_value",
        "specialty_source_concept_id",
        "gender_source_value",
        "gender_source_concept_id",
    ),
    "omop.person": TableRule(
        _rules(
            keep=(
                "person_id",
                "gender_concept_id",
                "race_concept_id",
                "ethnicity_concept_id",
                "gender_source_value",
                "race_source_value",
                "ethnicity_source_value",
            ),
            null=(
                "year_of_birth",
                "month_of_birth",
                "day_of_birth",
                "birth_datetime",
                "location_id",
                "person_source_value",
            ),
        )
    ),
    "omop.death": TableRule(
        _rules(
            keep=(
                "person_id",
                "death_date",
                "death_datetime",
                "death_type_concept_id",
                "cause_concept_id",
                "cause_source_value",
                "cause_source_concept_id",
            )
        )
    ),
    "omop.observation_period": TableRule(
        _rules(
            keep=(
                "observation_period_id",
                "person_id",
                "observation_period_start_date",
                "observation_period_end_date",
                "period_type_concept_id",
            )
        )
    ),
    "omop.visit_occurrence": TableRule(
        _rules(
            keep=(
                "visit_occurrence_id",
                "person_id",
                "visit_concept_id",
                "visit_start_date",
                "visit_start_datetime",
                "visit_end_date",
                "visit_end_datetime",
                "visit_type_concept_id",
                "visit_source_value",
            ),
            null=("provider_id", "care_site_id"),
        )
    ),
    "omop.condition_occurrence": TableRule(
        _rules(
            keep=(
                "condition_occurrence_id",
                "person_id",
                "condition_concept_id",
                "condition_start_date",
                "condition_start_datetime",
                "condition_end_date",
                "condition_type_concept_id",
                "visit_occurrence_id",
                "condition_source_value",
                "condition_source_concept_id",
                "condition_status_source_value",
            ),
            null=("provider_id",),
        )
    ),
    "omop.drug_exposure": TableRule(
        _rules(
            keep=(
                "drug_exposure_id",
                "person_id",
                "drug_concept_id",
                "drug_exposure_start_date",
                "drug_exposure_start_datetime",
                "drug_exposure_end_date",
                "drug_type_concept_id",
                "visit_occurrence_id",
                "drug_source_value",
                "drug_source_concept_id",
            ),
            null=("stop_reason", "sig", "provider_id"),
        )
    ),
    "omop.procedure_occurrence": TableRule(
        _rules(
            keep=(
                "procedure_occurrence_id",
                "person_id",
                "procedure_concept_id",
                "procedure_date",
                "procedure_datetime",
                "procedure_type_concept_id",
                "visit_occurrence_id",
                "procedure_source_value",
                "procedure_source_concept_id",
            ),
            null=("provider_id",),
        )
    ),
    "omop.measurement": TableRule(
        _rules(
            keep=(
                "measurement_id",
                "person_id",
                "measurement_concept_id",
                "measurement_date",
                "measurement_datetime",
                "measurement_type_concept_id",
                "value_as_number",
                "operator_concept_id",
                "value_as_concept_id",
                "unit_concept_id",
                "range_low",
                "range_high",
                "visit_occurrence_id",
                "measurement_source_value",
                "measurement_source_concept_id",
                "unit_source_value",
            ),
            null=("provider_id", "value_source_value"),
        )
    ),
    "omop.observation": TableRule(
        _rules(
            keep=(
                "observation_id",
                "person_id",
                "observation_concept_id",
                "observation_date",
                "observation_datetime",
                "observation_type_concept_id",
                "value_as_number",
                "value_as_concept_id",
                "unit_concept_id",
                "visit_occurrence_id",
                "observation_source_value",
                "observation_source_concept_id",
                "unit_source_value",
            ),
            null=("value_as_string", "provider_id", "value_source_value"),
        )
    ),
    "meta.ingest_patient": TableRule(
        _rules(
            keep=("person_id", "id_offset", "status"),
            null=("error",),
            hmac_rules={
                "source_patient_id": ("source-patient-id", "tok_"),
                "bundle_hash": ("input-bundle-content", "hmac_"),
                "response_hash": ("fhir2omop-response-content", "hmac_"),
            },
            constants={"updated_at": NORMALIZED_OPERATIONAL_TIME},
        )
    ),
    "meta.mapping": TableRule(
        _rules(
            keep=(
                "resource_type",
                "omop_table",
                "omop_id",
                "source_system",
                "source_code",
                "target_vocabulary",
                "target_code",
                "mapping_status",
                "vocab_version",
            ),
            null=("source_name", "target_name", "note"),
            hmac_rules={
                "source_patient_id": ("source-patient-id", "tok_"),
                "resource_id": ("fhir-resource-id", "tok_"),
            },
        )
    ),
    "meta.coverage": TableRule(
        _rules(
            keep=(
                "codes_already_standard",
                "codes_normalized",
                "codes_unmapped",
                "off_vocab_rate",
                "vocab_version",
            ),
            hmac_rules={"source_patient_id": ("source-patient-id", "tok_")},
        )
    ),
    "meta.row_provenance": TableRule(
        _rules(
            keep=("omop_table", "omop_id", "origin", "resource_type"),
            null=("source_pages",),
            hmac_rules={
                "source_patient_id": ("source-patient-id", "tok_"),
                "resource_id": ("fhir-resource-id", "tok_"),
                "doc_ref_id": ("document-reference-id", "tok_"),
                "document_hash": ("document-content", "hmac_"),
            },
        )
    ),
    "meta.dropped": TableRule(
        _rules(
            keep=("resource_type",),
            null=("reason",),
            hmac_rules={
                "source_patient_id": ("source-patient-id", "tok_"),
                "resource_id": ("fhir-resource-id", "tok_"),
            },
        )
    ),
    "meta.invalid_date": TableRule(
        _rules(
            keep=("omop_table", "omop_id", "column_name"),
            null=("original_value",),
            hmac_rules={"source_patient_id": ("source-patient-id", "tok_")},
        )
    ),
    "study.code_set": TableRule(
        _rules(
            keep=(
                "code_set_name",
                "domain",
                "concept_id",
                "source_system",
                "source_code",
                "source_value",
                "display",
                "mapping_status",
                "accepted",
            )
        )
    ),
    "study.cohort": TableRule(
        _rules(keep=("cohort_name", "person_id", "index_date", "included", "exclusion_reason"))
    ),
    "study.attrition": TableRule(
        _rules(keep=("cohort_name", "stage_order", "stage", "remaining", "removed", "detail"))
    ),
    "study.person_demographic": TableRule(
        _rules(
            keep=(
                "person_id",
                "age_bucket",
                "age_bucket_start",
                "age_bucket_end",
                "age_capped",
                "reference_date",
            )
        )
    ),
}


def _qualified(schema: str, table: str) -> str:
    return f"{quote_identifier(schema)}.{quote_identifier(table)}"


def policy_definition_errors() -> tuple[str, ...]:
    errors: list[str] = []
    expected_omop = {f"omop.{name}": set(spec.column_names) for name, spec in TABLE_SPECS.items()}
    actual_omop = {
        name: set(rule.columns)
        for name, rule in DEIDENTIFIED_COLUMN_POLICY.items()
        if name.startswith("omop.")
    }
    for table in sorted(expected_omop.keys() - actual_omop.keys()):
        errors.append(f"{table}: missing table policy")
    for table in sorted(actual_omop.keys() - expected_omop.keys()):
        errors.append(f"{table}: policy has no OMOP table")
    for table in sorted(expected_omop.keys() & actual_omop.keys()):
        for column in sorted(expected_omop[table] - actual_omop[table]):
            errors.append(f"{table}.{column}: missing column policy")
        for column in sorted(actual_omop[table] - expected_omop[table]):
            errors.append(f"{table}.{column}: policy has no schema column")
    for table, rule in DEIDENTIFIED_COLUMN_POLICY.items():
        if rule.delete_rows:
            invalid = [
                column
                for column, column_rule in rule.columns.items()
                if column_rule.action != ColumnAction.DROP_WITH_TABLE
            ]
            if invalid:
                errors.append(f"{table}: deleted table has retained column rules")
        elif any(
            column_rule.action == ColumnAction.DROP_WITH_TABLE
            for column_rule in rule.columns.values()
        ):
            errors.append(f"{table}: retained table has drop-with-table columns")
    return tuple(errors)


def schema_policy_errors(connection: Any) -> tuple[str, ...]:
    errors = list(policy_definition_errors())
    actual_tables = {
        f"{schema}.{table}"
        for schema, table in connection.execute(
            """SELECT table_schema, table_name
               FROM information_schema.tables
               WHERE table_catalog=current_database() AND table_type='BASE TABLE'
                 AND table_schema NOT IN ('information_schema', 'pg_catalog')"""
        ).fetchall()
    }
    expected_tables = set(DEIDENTIFIED_COLUMN_POLICY)
    for table in sorted(expected_tables - actual_tables):
        errors.append(f"{table}: policy table is missing from database")
    for table in sorted(actual_tables - expected_tables):
        errors.append(f"{table}: database table has no de-identification policy")
    rows = connection.execute(
        """SELECT table_schema, table_name, column_name
           FROM information_schema.columns
           WHERE table_catalog=current_database()
             AND table_schema NOT IN ('information_schema', 'pg_catalog')"""
    ).fetchall()
    actual_columns: dict[str, set[str]] = {}
    for schema, table, column in rows:
        actual_columns.setdefault(f"{schema}.{table}", set()).add(str(column))
    for table in sorted(expected_tables & actual_tables):
        expected = set(DEIDENTIFIED_COLUMN_POLICY[table].columns)
        actual = actual_columns.get(table, set())
        for column in sorted(expected - actual):
            errors.append(f"{table}.{column}: policy column is missing from database")
        for column in sorted(actual - expected):
            errors.append(f"{table}.{column}: database column has no de-identification policy")
    return tuple(dict.fromkeys(errors))


def keyed_value(
    value: Any,
    salt: bytes | str,
    *,
    namespace: str,
    prefix: str,
) -> str | None:
    if value is None:
        return None
    key = salt.encode("utf-8") if isinstance(salt, str) else salt
    if not key:
        raise ValueError("de-identification salt must not be empty")
    message = namespace.encode("utf-8") + b"\0" + str(value).encode("utf-8")
    return prefix + hmac.new(key, message, hashlib.sha256).hexdigest()


def _count(connection: Any, query: str, parameters: list[Any] | None = None) -> int:
    return int(connection.execute(query, parameters or []).fetchone()[0])


def apply_deidentified_policy(connection: Any, salt: bytes | str) -> PolicyApplication:
    errors = schema_policy_errors(connection)
    if errors:
        raise RuntimeError("Unsafe de-identification schema: " + "; ".join(errors))
    hmac_values = 0
    nulled_values = 0
    deleted_rows = 0
    normalized_values = 0
    for table, table_rule in DEIDENTIFIED_COLUMN_POLICY.items():
        schema, table_name = table.split(".", 1)
        qualified = _qualified(schema, table_name)
        if table_rule.delete_rows:
            deleted_rows += _count(connection, f"SELECT COUNT(*) FROM {qualified}")
            connection.execute(f"DELETE FROM {qualified}")
            continue
        for column, rule in table_rule.columns.items():
            quoted_column = quote_identifier(column)
            if rule.action == ColumnAction.KEEP:
                continue
            if rule.action == ColumnAction.NULL:
                nulled_values += _count(
                    connection,
                    f"SELECT COUNT(*) FROM {qualified} WHERE {quoted_column} IS NOT NULL",
                )
                connection.execute(f"UPDATE {qualified} SET {quoted_column}=NULL")
                continue
            if rule.action == ColumnAction.CONSTANT:
                normalized_values += _count(
                    connection,
                    f"SELECT COUNT(*) FROM {qualified} WHERE {quoted_column} IS DISTINCT FROM ?",
                    [rule.constant],
                )
                connection.execute(
                    f"UPDATE {qualified} SET {quoted_column}=?",
                    [rule.constant],
                )
                continue
            if rule.action != ColumnAction.HMAC or not rule.namespace or not rule.prefix:
                raise AssertionError(f"Invalid policy action for {table}.{column}")
            values = [
                str(row[0])
                for row in connection.execute(
                    f"SELECT DISTINCT {quoted_column} FROM {qualified} "
                    f"WHERE {quoted_column} IS NOT NULL"
                ).fetchall()
            ]
            hmac_values += len(values)
            if not values:
                continue
            mapping = [
                (
                    value,
                    keyed_value(
                        value,
                        salt,
                        namespace=rule.namespace,
                        prefix=rule.prefix,
                    ),
                )
                for value in values
            ]
            connection.execute(
                "CREATE OR REPLACE TEMP TABLE _pheno_rwe_privacy_map "
                "(original VARCHAR, replacement VARCHAR)"
            )
            connection.executemany("INSERT INTO _pheno_rwe_privacy_map VALUES (?, ?)", mapping)
            connection.execute(
                f"""UPDATE {qualified} AS target
                    SET {quoted_column}=mapping.replacement
                    FROM _pheno_rwe_privacy_map AS mapping
                    WHERE target.{quoted_column}=mapping.original"""
            )
            connection.execute("DROP TABLE _pheno_rwe_privacy_map")
    return PolicyApplication(hmac_values, nulled_values, deleted_rows, normalized_values)


def audit_deidentified_policy(connection: Any) -> PolicyAudit:
    errors = schema_policy_errors(connection)
    violations: dict[str, int] = {}
    if errors:
        return PolicyAudit(False, errors, violations)
    for table, table_rule in DEIDENTIFIED_COLUMN_POLICY.items():
        schema, table_name = table.split(".", 1)
        qualified = _qualified(schema, table_name)
        if table_rule.delete_rows:
            count = _count(connection, f"SELECT COUNT(*) FROM {qualified}")
            if count:
                violations[f"{table}.*"] = count
            continue
        for column, rule in table_rule.columns.items():
            quoted_column = quote_identifier(column)
            count = 0
            if rule.action == ColumnAction.NULL:
                count = _count(
                    connection,
                    f"SELECT COUNT(*) FROM {qualified} WHERE {quoted_column} IS NOT NULL",
                )
            elif rule.action == ColumnAction.CONSTANT:
                count = _count(
                    connection,
                    f"SELECT COUNT(*) FROM {qualified} WHERE {quoted_column} IS DISTINCT FROM ?",
                    [rule.constant],
                )
            elif rule.action == ColumnAction.HMAC:
                prefix = re.escape(rule.prefix or "")
                count = _count(
                    connection,
                    f"SELECT COUNT(*) FROM {qualified} WHERE {quoted_column} IS NOT NULL "
                    f"AND NOT regexp_full_match(CAST({quoted_column} AS VARCHAR), ?)",
                    [f"{prefix}[0-9a-f]{{64}}"],
                )
            if count:
                violations[f"{table}.{column}"] = count
    return PolicyAudit(not violations, (), violations)


def require_deidentified_policy(connection: Any) -> None:
    audit = audit_deidentified_policy(connection)
    if audit.valid:
        return
    details = [*audit.errors]
    details.extend(f"{column}={count}" for column, count in sorted(audit.violations.items()))
    raise RuntimeError("De-identified column policy audit failed: " + "; ".join(details))


def policy_summary() -> dict[str, Any]:
    columns = sum(len(rule.columns) for rule in DEIDENTIFIED_COLUMN_POLICY.values())
    hmac_columns = sorted(
        f"{table}.{column}"
        for table, table_rule in DEIDENTIFIED_COLUMN_POLICY.items()
        for column, rule in table_rule.columns.items()
        if rule.action == ColumnAction.HMAC
    )
    null_columns = sum(
        rule.action == ColumnAction.NULL
        for table_rule in DEIDENTIFIED_COLUMN_POLICY.values()
        for rule in table_rule.columns.values()
    )
    return {
        "version": POLICY_VERSION,
        "table_count": len(DEIDENTIFIED_COLUMN_POLICY),
        "column_count": columns,
        "hmac_columns": hmac_columns,
        "null_column_count": int(null_columns),
        "deleted_tables": sorted(
            table for table, rule in DEIDENTIFIED_COLUMN_POLICY.items() if rule.delete_rows
        ),
    }


assert not policy_definition_errors(), "; ".join(policy_definition_errors())
