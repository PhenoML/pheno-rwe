"""Consistent HMAC-derived per-person date shifting."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Any

from pheno_rwe.omop.ddl import DATE_COLUMN_REGISTRY, PERSON_COLUMN_REGISTRY, quote_identifier


@dataclass(frozen=True, slots=True)
class DateShiftResult:
    patient_count: int
    max_days: int
    shifts: dict[int, int]
    columns_shifted: tuple[str, ...]


def _salt_bytes(salt: bytes | str) -> bytes:
    result = salt.encode("utf-8") if isinstance(salt, str) else salt
    if not result:
        raise ValueError("date-shift salt must not be empty")
    return result


def patient_shift_days(person_id: int | str, salt: bytes | str, max_days: int = 180) -> int:
    """Map one stable person ID uniformly into ``[-max_days, +max_days]``."""
    if max_days < 0:
        raise ValueError("max_days cannot be negative")
    digest = hmac.new(_salt_bytes(salt), str(person_id).encode("utf-8"), hashlib.sha256).digest()
    width = 2 * max_days + 1
    return int.from_bytes(digest, "big") % width - max_days


# Short alias used by some callers.
shift_days = patient_shift_days


def date_registry_errors(connection: Any | None = None) -> tuple[str, ...]:
    errors: list[str] = []
    for table, columns in DATE_COLUMN_REGISTRY.items():
        if not columns:
            continue
        if not table.startswith("study.") and table not in PERSON_COLUMN_REGISTRY:
            errors.append(f"{table}: temporal table lacks a person ownership column")
    if connection is None:
        return tuple(errors)

    actual: set[tuple[str, str]] = set()
    rows = connection.execute(
        """SELECT table_schema, table_name, column_name, data_type
           FROM information_schema.columns
           WHERE table_schema IN ('omop', 'study')"""
    ).fetchall()
    for schema, table, column, data_type in rows:
        if str(data_type).upper() in {"DATE", "TIMESTAMP", "TIMESTAMP WITH TIME ZONE"}:
            qualified = f"{schema}.{table}" if schema != "omop" else str(table)
            actual.add((qualified, str(column)))
    registered = {
        (table, column) for table, columns in DATE_COLUMN_REGISTRY.items() for column in columns
    }
    for table, column in sorted(actual - registered):
        # Operational timestamps in the metadata ledger are intentionally not
        # clinical and therefore do not belong in the date-shift registry.
        if not table.startswith("meta."):
            errors.append(f"{table}.{column}: database temporal column is unregistered")
    for table, column in sorted(registered - actual):
        errors.append(f"{table}.{column}: registered temporal column is absent from database")
    return tuple(errors)


def apply_date_shift(
    connection: Any,
    salt: bytes | str,
    *,
    max_days: int = 180,
    validate_registry: bool = True,
) -> DateShiftResult:
    if max_days < 0:
        raise ValueError("max_days cannot be negative")
    if validate_registry:
        errors = date_registry_errors(connection)
        if errors:
            raise ValueError("Unsafe date-column registry: " + "; ".join(errors))
    person_ids = [
        int(row[0])
        for row in connection.execute(
            "SELECT person_id FROM omop.person ORDER BY person_id"
        ).fetchall()
    ]
    shifts = {person_id: patient_shift_days(person_id, salt, max_days) for person_id in person_ids}
    connection.execute("DROP TABLE IF EXISTS _pheno_rwe_date_shift")
    connection.execute(
        "CREATE TEMP TABLE _pheno_rwe_date_shift (person_id BIGINT, shift_days INTEGER)"
    )
    if shifts:
        connection.executemany(
            "INSERT INTO _pheno_rwe_date_shift VALUES (?, ?)", list(shifts.items())
        )

    shifted: list[str] = []
    for table, columns in DATE_COLUMN_REGISTRY.items():
        if not columns:
            continue
        schema, table_name = table.split(".", 1) if "." in table else ("omop", table)
        person_column = "person_id" if schema == "study" else PERSON_COLUMN_REGISTRY[table_name]
        qualified = f"{quote_identifier(schema)}.{quote_identifier(table_name)}"
        for column, data_type in columns.items():
            quoted_column = quote_identifier(column)
            expression = f"target.{quoted_column} + shifts.shift_days * INTERVAL '1 day'"
            if data_type == "DATE":
                expression = f"CAST({expression} AS DATE)"
            connection.execute(
                f"""UPDATE {qualified} AS target
                    SET {quoted_column} = {expression}
                    FROM _pheno_rwe_date_shift AS shifts
                    WHERE target.{quote_identifier(person_column)} = shifts.person_id
                      AND target.{quoted_column} IS NOT NULL"""
            )
            shifted.append(f"{table}.{column}")
    connection.execute("DROP TABLE _pheno_rwe_date_shift")
    return DateShiftResult(len(person_ids), max_days, shifts, tuple(shifted))


apply_date_shifts = apply_date_shift
