"""Transactional DuckDB loader for re-keyed fhir2omop responses."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from pheno_rwe.hashing import hash_json
from pheno_rwe.omop.ddl import TABLE_ORDER, TABLE_SPECS, create_schema, quote_identifier
from pheno_rwe.omop.rekey import (
    BLOCK_SIZE,
    allocate_patient_offset,
    rekey_response,
)


def _duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - exercised by lean-install users
        raise RuntimeError("DuckDB is required; install pheno-rwe's core dependencies.") from exc
    return duckdb


def connect_database(path: str | Path, *, read_only: bool = False) -> Any:
    """Open a file under a neutral catalog alias.

    The production file is named ``omop.duckdb`` and also contains an ``omop``
    schema.  DuckDB otherwise considers ``omop.person`` ambiguous (catalog vs
    schema).  Attaching under a neutral alias preserves the useful two-part
    schema names everywhere in the engine.
    """
    value = Path(path)
    value.parent.mkdir(parents=True, exist_ok=True)
    escaped = str(value).replace("'", "''")
    connection = _duckdb().connect()
    option = " (READ_ONLY)" if read_only else ""
    connection.execute(f"ATTACH '{escaped}' AS pheno_rwe{option}")
    connection.execute("USE pheno_rwe")
    return connection


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, date, datetime)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=False)
    if hasattr(value, "dict"):
        return value.dict(by_alias=True, exclude_none=False)
    return value


@dataclass(frozen=True, slots=True)
class PatientLoadResult:
    source_patient_id: str
    person_id: int
    id_offset: int
    row_counts: dict[str, int]
    mapping_count: int
    dropped_count: int
    invalid_date_count: int
    bundle_hash: str
    response_hash: str


@dataclass(frozen=True, slots=True)
class PatientLedger:
    source_patient_id: str
    person_id: int | None
    id_offset: int
    bundle_hash: str | None
    response_hash: str | None
    status: str
    error: str | None


class OmopLoader(AbstractContextManager["OmopLoader"]):
    """Own or wrap a DuckDB connection and load one patient per transaction."""

    def __init__(self, database: str | Path | Any, *, read_only: bool = False) -> None:
        self.connection: Any
        if hasattr(database, "execute"):
            self.connection = database
            self._owns_connection = False
            self.database_path: Path | None = None
        else:
            path = Path(database)
            self.connection = connect_database(path, read_only=read_only)
            self._owns_connection = True
            self.database_path = path
        if not read_only:
            create_schema(self.connection)

    def __exit__(self, *exc_info: object) -> None:
        if self._owns_connection:
            self.connection.close()

    def close(self) -> None:
        if self._owns_connection:
            self.connection.close()
            self._owns_connection = False

    def ledger(self, source_patient_id: str) -> PatientLedger | None:
        row = self.connection.execute(
            """SELECT source_patient_id, person_id, id_offset, bundle_hash,
                      response_hash, status, error
               FROM meta.ingest_patient WHERE source_patient_id = ?""",
            [source_patient_id],
        ).fetchone()
        return PatientLedger(*row) if row else None

    def is_current(self, source_patient_id: str, bundle_hash: str) -> bool:
        ledger = self.ledger(source_patient_id)
        return bool(ledger and ledger.status == "success" and ledger.bundle_hash == bundle_hash)

    def mark_failed(self, source_patient_id: str, error: str) -> None:
        existing = self.ledger(source_patient_id)
        self.connection.execute("BEGIN TRANSACTION")
        try:
            offset = (
                existing.id_offset
                if existing
                else allocate_patient_offset(self.connection, source_patient_id)
            )
            self.connection.execute(
                """UPDATE meta.ingest_patient
                   SET status = 'failed', error = ?, updated_at = current_timestamp
                   WHERE source_patient_id = ?""",
                [error, source_patient_id],
            )
            if offset < BLOCK_SIZE:  # defensive; allocation validates this in normal use
                raise RuntimeError("invalid patient offset")
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def materialize_patient(
        self,
        source_patient_id: str,
        response: Mapping[str, Any] | Any,
        *,
        bundle_hash: str,
        bundle: Mapping[str, Any] | None = None,
        provenance: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    ) -> PatientLoadResult:
        plain_response = _plain(response)
        response_hash = hash_json(plain_response)
        self.connection.execute("BEGIN TRANSACTION")
        try:
            offset = allocate_patient_offset(self.connection, source_patient_id)
            rekeyed = rekey_response(plain_response, offset)
            person_rows = (rekeyed.get("tables") or {}).get("person") or []
            if len(person_rows) != 1 or person_rows[0].get("person_id") is None:
                raise ValueError("Each patient materialization must produce exactly one person row")
            person_id = int(person_rows[0]["person_id"])
            self._delete_patient(source_patient_id, offset, person_id)

            invalid_dates: list[tuple[str, int | None, str, str]] = []
            row_counts: dict[str, int] = {}
            for table in TABLE_ORDER:
                rows = (rekeyed.get("tables") or {}).get(table) or []
                prepared = [
                    self._prepare_row(
                        table,
                        row,
                        source_patient_id=source_patient_id,
                        invalid_dates=invalid_dates,
                    )
                    for row in rows
                ]
                self._insert_rows(f"omop.{table}", TABLE_SPECS[table].column_names, prepared)
                row_counts[table] = len(prepared)

            mappings = list(rekeyed.get("mappings") or [])
            vocab_version = rekeyed.get("vocab_version")
            self._insert_mappings(source_patient_id, mappings, vocab_version)
            self._insert_dropped(source_patient_id, rekeyed.get("dropped") or [])
            self._insert_coverage(source_patient_id, rekeyed.get("summary"), vocab_version)
            self._insert_invalid_dates(source_patient_id, invalid_dates)
            self._insert_provenance(
                source_patient_id,
                mappings,
                bundle=bundle,
                supplied=provenance,
            )
            self.connection.execute(
                """UPDATE meta.ingest_patient
                   SET person_id = ?, bundle_hash = ?, response_hash = ?,
                       status = 'success', error = NULL, updated_at = current_timestamp
                   WHERE source_patient_id = ?""",
                [person_id, bundle_hash, response_hash, source_patient_id],
            )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

        return PatientLoadResult(
            source_patient_id=source_patient_id,
            person_id=person_id,
            id_offset=offset,
            row_counts=row_counts,
            mapping_count=len(mappings),
            dropped_count=len(rekeyed.get("dropped") or []),
            invalid_date_count=len(invalid_dates),
            bundle_hash=bundle_hash,
            response_hash=response_hash,
        )

    def _delete_patient(self, source_patient_id: str, offset: int, person_id: int) -> None:
        for table in reversed(TABLE_ORDER):
            spec = TABLE_SPECS[table]
            if spec.person_column:
                self.connection.execute(
                    f"DELETE FROM omop.{quote_identifier(table)} "
                    f"WHERE {quote_identifier(spec.person_column)} = ?",
                    [person_id],
                )
            elif spec.primary_key:
                self.connection.execute(
                    f"DELETE FROM omop.{quote_identifier(table)} "
                    f"WHERE {quote_identifier(spec.primary_key)} >= ? "
                    f"AND {quote_identifier(spec.primary_key)} < ?",
                    [offset, offset + BLOCK_SIZE],
                )
        for table in ("mapping", "coverage", "row_provenance", "dropped", "invalid_date"):
            self.connection.execute(
                f"DELETE FROM meta.{quote_identifier(table)} WHERE source_patient_id = ?",
                [source_patient_id],
            )

    def _prepare_row(
        self,
        table: str,
        row: Mapping[str, Any],
        *,
        source_patient_id: str,
        invalid_dates: list[tuple[str, int | None, str, str]],
    ) -> dict[str, Any]:
        spec = TABLE_SPECS[table]
        aliases = {"address1": "address_1", "address2": "address_2"}
        values = {aliases.get(key, key): value for key, value in dict(row).items()}
        row_id = values.get(spec.primary_key) if spec.primary_key else None
        result: dict[str, Any] = {}
        for column in spec.columns:
            value = values.get(column.name)
            if value is not None and column.duckdb_type in {"DATE", "TIMESTAMP"}:
                original = str(value)
                try:
                    value = _coerce_temporal(value, column.duckdb_type)
                except (TypeError, ValueError, OverflowError):
                    invalid_dates.append((table, row_id, column.name, original))
                    value = None
            result[column.name] = value
        return result

    def _insert_rows(
        self,
        qualified_table: str,
        columns: Sequence[str],
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        if not rows:
            return
        schema, table = qualified_table.split(".", 1)
        quoted_columns = ", ".join(quote_identifier(column) for column in columns)
        placeholders = ", ".join("?" for _ in columns)
        sql = (
            f"INSERT INTO {quote_identifier(schema)}.{quote_identifier(table)} "
            f"({quoted_columns}) VALUES ({placeholders})"
        )
        self.connection.executemany(sql, [[row.get(column) for column in columns] for row in rows])

    def _insert_mappings(
        self, source_patient_id: str, mappings: Sequence[Mapping[str, Any]], vocab: Any
    ) -> None:
        columns = (
            "source_patient_id",
            "resource_type",
            "resource_id",
            "omop_table",
            "omop_id",
            "source_system",
            "source_code",
            "source_name",
            "target_vocabulary",
            "target_code",
            "target_name",
            "mapping_status",
            "note",
            "vocab_version",
        )
        rows = [
            {
                **{column: mapping.get(column) for column in columns},
                "source_patient_id": source_patient_id,
                "vocab_version": vocab,
            }
            for mapping in mappings
        ]
        self._insert_rows("meta.mapping", columns, rows)

    def _insert_dropped(self, source_patient_id: str, dropped: Sequence[Mapping[str, Any]]) -> None:
        columns = ("source_patient_id", "resource_type", "resource_id", "reason")
        self._insert_rows(
            "meta.dropped",
            columns,
            [
                {
                    "source_patient_id": source_patient_id,
                    "resource_type": item.get("resource_type"),
                    "resource_id": item.get("resource_id"),
                    "reason": item.get("reason"),
                }
                for item in dropped
            ],
        )

    def _insert_coverage(
        self, source_patient_id: str, summary: Mapping[str, Any] | Any, vocab: Any
    ) -> None:
        value = _plain(summary) or {}
        columns = (
            "source_patient_id",
            "codes_already_standard",
            "codes_normalized",
            "codes_unmapped",
            "off_vocab_rate",
            "vocab_version",
        )
        self._insert_rows(
            "meta.coverage",
            columns,
            [
                {
                    "source_patient_id": source_patient_id,
                    **{column: value.get(column) for column in columns[1:-1]},
                    "vocab_version": vocab,
                }
            ],
        )

    def _insert_invalid_dates(
        self,
        source_patient_id: str,
        values: Sequence[tuple[str, int | None, str, str]],
    ) -> None:
        columns = (
            "source_patient_id",
            "omop_table",
            "omop_id",
            "column_name",
            "original_value",
        )
        self._insert_rows(
            "meta.invalid_date",
            columns,
            [
                {
                    "source_patient_id": source_patient_id,
                    "omop_table": table,
                    "omop_id": row_id,
                    "column_name": column,
                    "original_value": original,
                }
                for table, row_id, column, original in values
            ],
        )

    def _insert_provenance(
        self,
        source_patient_id: str,
        mappings: Sequence[Mapping[str, Any]],
        *,
        bundle: Mapping[str, Any] | None,
        supplied: Mapping[tuple[str, str], Mapping[str, Any]] | None,
    ) -> None:
        resource_info = resource_provenance(bundle or {})
        if supplied:
            resource_info.update({key: dict(value) for key, value in supplied.items()})
        unique: dict[tuple[str, int], dict[str, Any]] = {}
        for mapping in mappings:
            table = mapping.get("omop_table")
            row_id = mapping.get("omop_id")
            if not table or row_id is None:
                continue
            resource_type = str(mapping.get("resource_type") or "")
            resource_id = str(mapping.get("resource_id") or "")
            info = resource_info.get((resource_type, resource_id), {})
            candidate = {
                "source_patient_id": source_patient_id,
                "omop_table": table,
                "omop_id": row_id,
                "origin": info.get("origin", "structured"),
                "resource_type": resource_type or None,
                "resource_id": resource_id or None,
                "doc_ref_id": info.get("doc_ref_id"),
                "document_hash": info.get("document_hash"),
                "source_pages": json.dumps(info.get("source_pages"), sort_keys=True)
                if info.get("source_pages") is not None
                else None,
            }
            key = (str(table), int(row_id))
            if key not in unique or candidate["origin"] == "enriched":
                unique[key] = candidate
        columns = (
            "source_patient_id",
            "omop_table",
            "omop_id",
            "origin",
            "resource_type",
            "resource_id",
            "doc_ref_id",
            "document_hash",
            "source_pages",
        )
        self._insert_rows("meta.row_provenance", columns, list(unique.values()))


def _coerce_temporal(value: Any, duckdb_type: str) -> date | datetime:
    if duckdb_type == "DATE":
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if not isinstance(value, str):
            raise TypeError("not a date string")
        return date.fromisoformat(value)
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if not isinstance(value, str):
        raise TypeError("not a datetime string")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def resource_provenance(bundle: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Extract pheno-rwe enrichment tags without depending on one tag spelling."""
    result: dict[tuple[str, str], dict[str, Any]] = {}
    entries = bundle.get("entry") or []
    for entry in entries:
        resource = entry.get("resource", entry) if isinstance(entry, Mapping) else {}
        if not isinstance(resource, Mapping):
            continue
        resource_type = str(resource.get("resourceType") or "")
        resource_id = str(resource.get("id") or "")
        if not resource_type or not resource_id:
            continue
        info: dict[str, Any] = {"origin": "structured"}
        custom = resource.get("_pheno_rwe_provenance")
        if isinstance(custom, Mapping):
            info.update(custom)
            info["origin"] = custom.get("origin", "enriched")
        meta = resource.get("meta") or {}
        for tag in meta.get("tag") or []:
            if not isinstance(tag, Mapping):
                continue
            system = str(tag.get("system") or "").lower()
            code = str(tag.get("code") or "")
            lowered = code.lower()
            if system.endswith("/origin") and lowered == "enriched":
                info["origin"] = "enriched"
            elif "document-hash" in system:
                info["document_hash"] = code
                info["origin"] = "enriched"
            elif "document-reference" in system:
                info["doc_ref_id"] = code
                info["origin"] = "enriched"
            elif "source-pages" in system:
                try:
                    info["source_pages"] = json.loads(code)
                except json.JSONDecodeError:
                    info["source_pages"] = code
                info["origin"] = "enriched"
            if ("pheno-rwe" in system or "provenance" in system) and lowered in {
                "enriched",
                "generated",
                "lang2fhir",
            }:
                info["origin"] = "enriched"
            for prefix, key in (
                ("doc-ref:", "doc_ref_id"),
                ("doc_ref_id:", "doc_ref_id"),
                ("document-hash:", "document_hash"),
                ("document_hash:", "document_hash"),
                ("source-pages:", "source_pages"),
                ("source_pages:", "source_pages"),
            ):
                if lowered.startswith(prefix):
                    raw = code[len(prefix) :]
                    if key == "source_pages":
                        try:
                            raw = json.loads(raw)
                        except json.JSONDecodeError:
                            raw = raw
                    info[key] = raw
                    info["origin"] = "enriched"
        result[(resource_type, resource_id)] = info
    return result


def create_database(path: str | Path) -> Path:
    path = Path(path)
    with OmopLoader(path):
        pass
    return path
