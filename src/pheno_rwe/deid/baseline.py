"""Always-on baseline de-identification for an OMOP CDM-lite database."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from pheno_rwe.deid.policy import apply_deidentified_policy, keyed_value


@dataclass(frozen=True, slots=True)
class BaselineResult:
    patient_count: int
    tokenized_values: int
    age_bucket_years: int
    age_cap: int
    location_rows_removed: int


def _salt_bytes(salt: bytes | str) -> bytes:
    result = salt.encode("utf-8") if isinstance(salt, str) else salt
    if not result:
        raise ValueError("de-identification salt must not be empty")
    return result


def tokenize(value: Any, salt: bytes | str, *, prefix: str = "tok_") -> str | None:
    return keyed_value(
        value,
        _salt_bytes(salt),
        namespace="identifier",
        prefix=prefix,
    )


# British spelling retained as a harmless convenience.
tokenise = tokenize


def age_bucket(
    year_of_birth: int | None,
    reference_date: date,
    *,
    width: int = 5,
    cap: int = 90,
) -> tuple[str | None, int | None, int | None, bool]:
    if year_of_birth is None:
        return None, None, None, False
    if width < 1:
        raise ValueError("age bucket width must be at least one year")
    if cap < 1:
        raise ValueError("age cap must be positive")
    age = max(0, reference_date.year - int(year_of_birth))
    if age >= cap:
        return f"{cap}+", cap, None, True
    lower = age // width * width
    upper = min(lower + width - 1, cap - 1)
    return f"{lower}-{upper}", lower, upper, False


def _reference_dates(connection: Any, default: date) -> dict[int, date]:
    values: dict[int, date] = {}
    for person_id, index_date in connection.execute(
        """SELECT person_id, MIN(index_date)
           FROM study.cohort WHERE included AND index_date IS NOT NULL
           GROUP BY person_id"""
    ).fetchall():
        values[int(person_id)] = index_date
    for person_id, start_date in connection.execute(
        """SELECT person_id, MIN(observation_period_start_date)
           FROM omop.observation_period
           WHERE observation_period_start_date IS NOT NULL GROUP BY person_id"""
    ).fetchall():
        values.setdefault(int(person_id), start_date)
    return values


def apply_baseline(
    connection: Any,
    salt: bytes | str,
    *,
    age_bucket_years: int = 5,
    age_cap: int = 90,
    reference_date: date | None = None,
) -> BaselineResult:
    """Mutate a copied identified database into its baseline de-id form."""
    if age_bucket_years < 1:
        raise ValueError("age_bucket_years must be at least 1")
    if age_cap < 1:
        raise ValueError("age_cap must be positive")
    salt_value = _salt_bytes(salt)
    default_reference = reference_date or date(2000, 1, 1)
    references = _reference_dates(connection, default_reference)

    people = connection.execute(
        "SELECT person_id, year_of_birth FROM omop.person ORDER BY person_id"
    ).fetchall()
    connection.execute("DELETE FROM study.person_demographic")
    demographics = []
    for person_id, year_of_birth in people:
        anchor = references.get(int(person_id), default_reference)
        label, lower, upper, capped = age_bucket(
            year_of_birth, anchor, width=age_bucket_years, cap=age_cap
        )
        demographics.append((person_id, label, lower, upper, capped, anchor))
    if demographics:
        connection.executemany(
            "INSERT INTO study.person_demographic VALUES (?, ?, ?, ?, ?, ?)", demographics
        )

    location_rows = int(connection.execute("SELECT COUNT(*) FROM omop.location").fetchone()[0])
    application = apply_deidentified_policy(connection, salt_value)
    return BaselineResult(
        patient_count=len(people),
        tokenized_values=application.hmac_values,
        age_bucket_years=age_bucket_years,
        age_cap=age_cap,
        location_rows_removed=location_rows,
    )
