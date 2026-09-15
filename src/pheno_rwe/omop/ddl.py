"""Explicit DuckDB schema registry for the fhir2omop CDM v5.4-lite payload.

The registries in this module are intentionally boring.  Re-keying and date
shifting are high-risk operations, so neither is allowed to infer columns from
suffixes at runtime.  A new column must be classified here before it can be
loaded.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Column:
    name: str
    duckdb_type: str


@dataclass(frozen=True, slots=True)
class TableSpec:
    name: str
    columns: tuple[Column, ...]
    primary_key: str | None
    foreign_keys: Mapping[str, tuple[str, str]]
    person_column: str | None = None

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def column_types(self) -> dict[str, str]:
        return {column.name: column.duckdb_type for column in self.columns}


def _columns(**values: str) -> tuple[Column, ...]:
    return tuple(Column(name, value) for name, value in values.items())


# These are the twelve tables returned by PhenoML's fhir2omop endpoint.  The
# fields follow its CDM v5.4-lite response contract rather than pretending to
# implement every optional field in the full CDM.
TABLE_SPECS: dict[str, TableSpec] = {
    "location": TableSpec(
        "location",
        _columns(
            location_id="BIGINT",
            address_1="VARCHAR",
            address_2="VARCHAR",
            city="VARCHAR",
            state="VARCHAR",
            zip="VARCHAR",
            county="VARCHAR",
            location_source_value="VARCHAR",
            country_concept_id="BIGINT",
            country_source_value="VARCHAR",
            latitude="DOUBLE",
            longitude="DOUBLE",
        ),
        "location_id",
        {},
    ),
    "care_site": TableSpec(
        "care_site",
        _columns(
            care_site_id="BIGINT",
            care_site_name="VARCHAR",
            place_of_service_concept_id="BIGINT",
            location_id="BIGINT",
            care_site_source_value="VARCHAR",
            place_of_service_source_value="VARCHAR",
        ),
        "care_site_id",
        {"location_id": ("location", "location_id")},
    ),
    "provider": TableSpec(
        "provider",
        _columns(
            provider_id="BIGINT",
            provider_name="VARCHAR",
            npi="VARCHAR",
            dea="VARCHAR",
            specialty_concept_id="BIGINT",
            care_site_id="BIGINT",
            year_of_birth="INTEGER",
            gender_concept_id="BIGINT",
            provider_source_value="VARCHAR",
            specialty_source_value="VARCHAR",
            specialty_source_concept_id="BIGINT",
            gender_source_value="VARCHAR",
            gender_source_concept_id="BIGINT",
        ),
        "provider_id",
        {"care_site_id": ("care_site", "care_site_id")},
    ),
    "person": TableSpec(
        "person",
        _columns(
            person_id="BIGINT",
            gender_concept_id="BIGINT",
            year_of_birth="INTEGER",
            month_of_birth="INTEGER",
            day_of_birth="INTEGER",
            birth_datetime="TIMESTAMP",
            race_concept_id="BIGINT",
            ethnicity_concept_id="BIGINT",
            location_id="BIGINT",
            person_source_value="VARCHAR",
            gender_source_value="VARCHAR",
            race_source_value="VARCHAR",
            ethnicity_source_value="VARCHAR",
        ),
        "person_id",
        {"location_id": ("location", "location_id")},
        "person_id",
    ),
    "death": TableSpec(
        "death",
        _columns(
            person_id="BIGINT",
            death_date="DATE",
            death_datetime="TIMESTAMP",
            death_type_concept_id="BIGINT",
            cause_concept_id="BIGINT",
            cause_source_value="VARCHAR",
            cause_source_concept_id="BIGINT",
        ),
        "person_id",
        {"person_id": ("person", "person_id")},
        "person_id",
    ),
    "observation_period": TableSpec(
        "observation_period",
        _columns(
            observation_period_id="BIGINT",
            person_id="BIGINT",
            observation_period_start_date="DATE",
            observation_period_end_date="DATE",
            period_type_concept_id="BIGINT",
        ),
        "observation_period_id",
        {"person_id": ("person", "person_id")},
        "person_id",
    ),
    "visit_occurrence": TableSpec(
        "visit_occurrence",
        _columns(
            visit_occurrence_id="BIGINT",
            person_id="BIGINT",
            visit_concept_id="BIGINT",
            visit_start_date="DATE",
            visit_start_datetime="TIMESTAMP",
            visit_end_date="DATE",
            visit_end_datetime="TIMESTAMP",
            visit_type_concept_id="BIGINT",
            provider_id="BIGINT",
            care_site_id="BIGINT",
            visit_source_value="VARCHAR",
        ),
        "visit_occurrence_id",
        {
            "person_id": ("person", "person_id"),
            "provider_id": ("provider", "provider_id"),
            "care_site_id": ("care_site", "care_site_id"),
        },
        "person_id",
    ),
    "condition_occurrence": TableSpec(
        "condition_occurrence",
        _columns(
            condition_occurrence_id="BIGINT",
            person_id="BIGINT",
            condition_concept_id="BIGINT",
            condition_start_date="DATE",
            condition_start_datetime="TIMESTAMP",
            condition_end_date="DATE",
            condition_type_concept_id="BIGINT",
            visit_occurrence_id="BIGINT",
            provider_id="BIGINT",
            condition_source_value="VARCHAR",
            condition_source_concept_id="BIGINT",
            condition_status_source_value="VARCHAR",
        ),
        "condition_occurrence_id",
        {
            "person_id": ("person", "person_id"),
            "visit_occurrence_id": ("visit_occurrence", "visit_occurrence_id"),
            "provider_id": ("provider", "provider_id"),
        },
        "person_id",
    ),
    "drug_exposure": TableSpec(
        "drug_exposure",
        _columns(
            drug_exposure_id="BIGINT",
            person_id="BIGINT",
            drug_concept_id="BIGINT",
            drug_exposure_start_date="DATE",
            drug_exposure_start_datetime="TIMESTAMP",
            drug_exposure_end_date="DATE",
            drug_type_concept_id="BIGINT",
            stop_reason="VARCHAR",
            sig="VARCHAR",
            visit_occurrence_id="BIGINT",
            provider_id="BIGINT",
            drug_source_value="VARCHAR",
            drug_source_concept_id="BIGINT",
        ),
        "drug_exposure_id",
        {
            "person_id": ("person", "person_id"),
            "visit_occurrence_id": ("visit_occurrence", "visit_occurrence_id"),
            "provider_id": ("provider", "provider_id"),
        },
        "person_id",
    ),
    "procedure_occurrence": TableSpec(
        "procedure_occurrence",
        _columns(
            procedure_occurrence_id="BIGINT",
            person_id="BIGINT",
            procedure_concept_id="BIGINT",
            procedure_date="DATE",
            procedure_datetime="TIMESTAMP",
            procedure_type_concept_id="BIGINT",
            visit_occurrence_id="BIGINT",
            provider_id="BIGINT",
            procedure_source_value="VARCHAR",
            procedure_source_concept_id="BIGINT",
        ),
        "procedure_occurrence_id",
        {
            "person_id": ("person", "person_id"),
            "visit_occurrence_id": ("visit_occurrence", "visit_occurrence_id"),
            "provider_id": ("provider", "provider_id"),
        },
        "person_id",
    ),
    "measurement": TableSpec(
        "measurement",
        _columns(
            measurement_id="BIGINT",
            person_id="BIGINT",
            measurement_concept_id="BIGINT",
            measurement_date="DATE",
            measurement_datetime="TIMESTAMP",
            measurement_type_concept_id="BIGINT",
            value_as_number="DOUBLE",
            operator_concept_id="BIGINT",
            value_as_concept_id="BIGINT",
            unit_concept_id="BIGINT",
            range_low="DOUBLE",
            range_high="DOUBLE",
            visit_occurrence_id="BIGINT",
            provider_id="BIGINT",
            measurement_source_value="VARCHAR",
            measurement_source_concept_id="BIGINT",
            unit_source_value="VARCHAR",
            value_source_value="VARCHAR",
        ),
        "measurement_id",
        {
            "person_id": ("person", "person_id"),
            "visit_occurrence_id": ("visit_occurrence", "visit_occurrence_id"),
            "provider_id": ("provider", "provider_id"),
        },
        "person_id",
    ),
    "observation": TableSpec(
        "observation",
        _columns(
            observation_id="BIGINT",
            person_id="BIGINT",
            observation_concept_id="BIGINT",
            observation_date="DATE",
            observation_datetime="TIMESTAMP",
            observation_type_concept_id="BIGINT",
            value_as_number="DOUBLE",
            value_as_string="VARCHAR",
            value_as_concept_id="BIGINT",
            unit_concept_id="BIGINT",
            visit_occurrence_id="BIGINT",
            provider_id="BIGINT",
            observation_source_value="VARCHAR",
            observation_source_concept_id="BIGINT",
            unit_source_value="VARCHAR",
            value_source_value="VARCHAR",
        ),
        "observation_id",
        {
            "person_id": ("person", "person_id"),
            "visit_occurrence_id": ("visit_occurrence", "visit_occurrence_id"),
            "provider_id": ("provider", "provider_id"),
        },
        "person_id",
    ),
}

TABLE_ORDER: tuple[str, ...] = tuple(TABLE_SPECS)
PRIMARY_KEY_REGISTRY: dict[str, str] = {
    name: spec.primary_key for name, spec in TABLE_SPECS.items() if spec.primary_key
}
FOREIGN_KEY_REGISTRY: dict[str, dict[str, tuple[str, str]]] = {
    name: dict(spec.foreign_keys) for name, spec in TABLE_SPECS.items()
}
ID_COLUMN_REGISTRY: dict[str, tuple[str, ...]] = {
    name: tuple(
        dict.fromkeys(([spec.primary_key] if spec.primary_key else []) + list(spec.foreign_keys))
    )
    for name, spec in TABLE_SPECS.items()
}
PERSON_COLUMN_REGISTRY: dict[str, str] = {
    name: spec.person_column for name, spec in TABLE_SPECS.items() if spec.person_column
}

# Explicit, typed temporal registry consumed by date-shift.  It includes every
# DATE/TIMESTAMP in the OMOP-lite DDL even when baseline de-id nulls the field.
DATE_COLUMN_REGISTRY: dict[str, dict[str, str]] = {
    name: {
        column.name: column.duckdb_type
        for column in spec.columns
        if column.duckdb_type in {"DATE", "TIMESTAMP"}
    }
    for name, spec in TABLE_SPECS.items()
}
DATE_COLUMN_REGISTRY["study.cohort"] = {"index_date": "DATE"}
DATE_COLUMN_REGISTRY["study.person_demographic"] = {"reference_date": "DATE"}


META_DDL: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS meta.ingest_patient (
        source_patient_id VARCHAR PRIMARY KEY,
        person_id BIGINT,
        id_offset BIGINT NOT NULL UNIQUE,
        bundle_hash VARCHAR,
        response_hash VARCHAR,
        status VARCHAR NOT NULL,
        error VARCHAR,
        updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp
    )""",
    """CREATE TABLE IF NOT EXISTS meta.mapping (
        source_patient_id VARCHAR,
        resource_type VARCHAR,
        resource_id VARCHAR,
        omop_table VARCHAR,
        omop_id BIGINT,
        source_system VARCHAR,
        source_code VARCHAR,
        source_name VARCHAR,
        target_vocabulary VARCHAR,
        target_code VARCHAR,
        target_name VARCHAR,
        mapping_status VARCHAR,
        note VARCHAR,
        vocab_version VARCHAR
    )""",
    """CREATE TABLE IF NOT EXISTS meta.coverage (
        source_patient_id VARCHAR PRIMARY KEY,
        codes_already_standard BIGINT,
        codes_normalized BIGINT,
        codes_unmapped BIGINT,
        off_vocab_rate DOUBLE,
        vocab_version VARCHAR
    )""",
    """CREATE TABLE IF NOT EXISTS meta.row_provenance (
        source_patient_id VARCHAR,
        omop_table VARCHAR,
        omop_id BIGINT,
        origin VARCHAR NOT NULL,
        resource_type VARCHAR,
        resource_id VARCHAR,
        doc_ref_id VARCHAR,
        document_hash VARCHAR,
        source_pages VARCHAR
    )""",
    """CREATE TABLE IF NOT EXISTS meta.dropped (
        source_patient_id VARCHAR,
        resource_type VARCHAR,
        resource_id VARCHAR,
        reason VARCHAR
    )""",
    """CREATE TABLE IF NOT EXISTS meta.invalid_date (
        source_patient_id VARCHAR,
        omop_table VARCHAR,
        omop_id BIGINT,
        column_name VARCHAR,
        original_value VARCHAR
    )""",
)

STUDY_DDL: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS study.code_set (
        code_set_name VARCHAR,
        domain VARCHAR,
        concept_id BIGINT,
        source_system VARCHAR,
        source_code VARCHAR,
        source_value VARCHAR,
        display VARCHAR,
        mapping_status VARCHAR,
        accepted BOOLEAN DEFAULT TRUE
    )""",
    """CREATE TABLE IF NOT EXISTS study.cohort (
        cohort_name VARCHAR,
        person_id BIGINT,
        index_date DATE,
        included BOOLEAN NOT NULL,
        exclusion_reason VARCHAR
    )""",
    """CREATE TABLE IF NOT EXISTS study.attrition (
        cohort_name VARCHAR,
        stage_order INTEGER,
        stage VARCHAR,
        remaining BIGINT,
        removed BIGINT,
        detail VARCHAR
    )""",
    """CREATE TABLE IF NOT EXISTS study.person_demographic (
        person_id BIGINT PRIMARY KEY,
        age_bucket VARCHAR,
        age_bucket_start INTEGER,
        age_bucket_end INTEGER,
        age_capped BOOLEAN,
        reference_date DATE
    )""",
)


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def table_ddl(spec: TableSpec) -> str:
    definitions = ",\n        ".join(
        f"{quote_identifier(column.name)} {column.duckdb_type}" for column in spec.columns
    )
    return (
        f"CREATE TABLE IF NOT EXISTS omop.{quote_identifier(spec.name)} (\n"
        f"        {definitions}\n    )"
    )


def all_ddl() -> tuple[str, ...]:
    return (
        "CREATE SCHEMA IF NOT EXISTS omop",
        "CREATE SCHEMA IF NOT EXISTS meta",
        "CREATE SCHEMA IF NOT EXISTS study",
        *(table_ddl(spec) for spec in TABLE_SPECS.values()),
        *META_DDL,
        *STUDY_DDL,
    )


def create_schema(connection: Any) -> None:
    for statement in all_ddl():
        connection.execute(statement)


def registry_errors() -> tuple[str, ...]:
    """Return internal registry inconsistencies (empty means safe to operate)."""
    errors: list[str] = []
    for table, spec in TABLE_SPECS.items():
        columns = set(spec.column_names)
        for column in ID_COLUMN_REGISTRY.get(table, ()):
            if column not in columns:
                errors.append(f"{table}.{column}: ID column is absent from DDL")
        expected_dates = {
            column.name for column in spec.columns if column.duckdb_type in {"DATE", "TIMESTAMP"}
        }
        registered_dates = set(DATE_COLUMN_REGISTRY.get(table, {}))
        for column in sorted(expected_dates - registered_dates):
            errors.append(f"{table}.{column}: temporal column is not registered")
        for column, (target_table, target_column) in spec.foreign_keys.items():
            target = TABLE_SPECS.get(target_table)
            if target is None or target_column not in target.column_names:
                errors.append(f"{table}.{column}: invalid FK target {target_table}.{target_column}")
    return tuple(errors)


def iter_omop_columns() -> Iterable[tuple[str, Column]]:
    for table, spec in TABLE_SPECS.items():
        for column in spec.columns:
            yield table, column


assert not registry_errors(), "; ".join(registry_errors())
