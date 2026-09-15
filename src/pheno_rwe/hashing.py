"""Canonical hashing for plans, files, directories, and logical DuckDB tables."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_paths(paths: Iterable[str | Path], *, root: str | Path | None = None) -> str:
    root_path = Path(root).resolve() if root else None
    entries: list[dict[str, str]] = []
    for raw in sorted((Path(p).resolve() for p in paths), key=str):
        label = (
            str(raw.relative_to(root_path))
            if root_path and raw.is_relative_to(root_path)
            else str(raw)
        )
        entries.append({"path": label, "sha256": hash_file(raw)})
    return hash_json(entries)


def hash_directory(path: str | Path) -> str:
    directory = Path(path).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    files = [item for item in directory.rglob("*") if item.is_file()]
    return hash_paths(files, root=directory)


def logical_table_hash(connection: Any, table: str) -> str:
    """Hash a table independently of DuckDB's physical file representation."""
    safe_table = table.replace('"', '""')
    columns = [
        row[1] for row in connection.execute(f'PRAGMA table_info("{safe_table}")').fetchall()
    ]
    if not columns:
        return hash_json([])
    quoted = ", ".join(f'"{column.replace(chr(34), chr(34) * 2)}"' for column in columns)
    rows = connection.execute(f'SELECT {quoted} FROM "{safe_table}" ORDER BY ALL').fetchall()
    return hash_json({"columns": columns, "rows": rows})
