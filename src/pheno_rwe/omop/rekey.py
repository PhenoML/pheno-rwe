"""Stable block-based re-keying for per-patient fhir2omop responses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from pheno_rwe.omop.ddl import ID_COLUMN_REGISTRY, TABLE_SPECS

BLOCK_SIZE = 1_000_000


class RekeyError(ValueError):
    """The response cannot be safely placed in its allocated ID block."""


def _as_dict(value: Any, *, by_alias: bool = True) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=by_alias, exclude_none=False)
    if hasattr(value, "dict"):
        return value.dict(by_alias=by_alias, exclude_none=False)
    raise TypeError(f"Expected a mapping or Pydantic model, got {type(value).__name__}")


def block_offset(block_number: int, *, block_size: int = BLOCK_SIZE) -> int:
    if isinstance(block_number, bool) or not isinstance(block_number, int):
        raise TypeError("block_number must be an integer")
    if block_number < 1:
        raise ValueError("block_number must be at least 1")
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    return block_number * block_size


def validate_offset(offset: int, *, block_size: int = BLOCK_SIZE) -> None:
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise TypeError("offset must be an integer")
    if offset < block_size or offset % block_size:
        raise RekeyError(f"offset must be a positive {block_size:,}-aligned block")


def rekey_id(
    value: Any,
    offset: int,
    *,
    block_size: int = BLOCK_SIZE,
    column: str = "id",
) -> Any:
    """Translate a local row/FK ID while preserving NULL.

    Zero is translated too: OMOP's special zero semantics apply to concept IDs,
    and concept IDs are deliberately absent from ``ID_COLUMN_REGISTRY``.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise RekeyError(f"{column} must be an integer or null, got {value!r}")
    if value < 0:
        raise RekeyError(f"{column} cannot be negative: {value}")
    if value >= block_size:
        raise RekeyError(f"{column} local ID {value} exceeds the {block_size:,}-row patient block")
    return offset + value


def rekey_row(
    table: str,
    row: Mapping[str, Any] | Any,
    offset: int,
    *,
    block_size: int = BLOCK_SIZE,
) -> dict[str, Any]:
    validate_offset(offset, block_size=block_size)
    if table not in TABLE_SPECS:
        raise RekeyError(f"No explicit re-key registry for OMOP table {table!r}")
    result = _as_dict(row)
    for column in ID_COLUMN_REGISTRY[table]:
        if column in result:
            result[column] = rekey_id(
                result[column], offset, block_size=block_size, column=f"{table}.{column}"
            )
    return result


def rekey_tables(
    tables: Mapping[str, Sequence[Mapping[str, Any] | Any]] | Any,
    offset: int,
    *,
    block_size: int = BLOCK_SIZE,
) -> dict[str, list[dict[str, Any]]]:
    validate_offset(offset, block_size=block_size)
    table_values = _as_dict(tables)
    unknown = sorted(set(table_values) - set(TABLE_SPECS))
    if unknown:
        raise RekeyError("No explicit re-key registry for table(s): " + ", ".join(unknown))
    return {
        table: [rekey_row(table, row, offset, block_size=block_size) for row in (rows or [])]
        for table, rows in table_values.items()
    }


def rekey_mapping(
    mapping: Mapping[str, Any] | Any,
    offset: int,
    *,
    block_size: int = BLOCK_SIZE,
) -> dict[str, Any]:
    result = _as_dict(mapping)
    table = result.get("omop_table")
    if result.get("omop_id") is not None:
        if table not in TABLE_SPECS:
            raise RekeyError(f"Mapping points to unregistered OMOP table {table!r}")
        result["omop_id"] = rekey_id(
            result["omop_id"],
            offset,
            block_size=block_size,
            column=f"mapping[{table}].omop_id",
        )
    return result


def rekey_response(
    response: Mapping[str, Any] | Any,
    offset: int,
    *,
    block_size: int = BLOCK_SIZE,
) -> dict[str, Any]:
    """Return a plain JSON-compatible response with every row/FK re-keyed."""
    result = deepcopy(_as_dict(response))
    result["tables"] = rekey_tables(result.get("tables") or {}, offset, block_size=block_size)
    result["mappings"] = [
        rekey_mapping(mapping, offset, block_size=block_size)
        for mapping in (result.get("mappings") or [])
    ]
    return result


def allocate_patient_offset(
    connection: Any,
    source_patient_id: str,
    *,
    block_size: int = BLOCK_SIZE,
) -> int:
    """Return an existing allocation or persist the next free patient block.

    This function must be called inside the caller's write transaction.  The
    allocation is recorded before any clinical rows, so rematerialization can
    delete and replace a patient while retaining stable global IDs.
    """
    if not source_patient_id:
        raise ValueError("source_patient_id cannot be empty")
    existing = connection.execute(
        "SELECT id_offset FROM meta.ingest_patient WHERE source_patient_id = ?",
        [source_patient_id],
    ).fetchone()
    if existing:
        return int(existing[0])
    maximum = connection.execute(
        "SELECT COALESCE(MAX(id_offset), 0) FROM meta.ingest_patient"
    ).fetchone()[0]
    offset = block_offset(int(maximum) // block_size + 1, block_size=block_size)
    connection.execute(
        """INSERT INTO meta.ingest_patient
           (source_patient_id, id_offset, status)
           VALUES (?, ?, 'allocated')""",
        [source_patient_id, offset],
    )
    return offset


# Compatibility-friendly short name for callers/tests.
allocate_offset = allocate_patient_offset
