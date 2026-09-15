"""Study discovery, creation, metadata, and single-writer locking."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pheno_rwe.errors import InvalidStudy, StudyLocked, StudyNotFound
from pheno_rwe.hashing import canonical_json

STUDY_DIRS = (
    "raw/patients",
    "enriched/patients",
    "enriched/documents",
    "codesets",
    "omop",
    "reports",
    "results",
    "plots",
    "exports",
)


@dataclass(frozen=True, slots=True)
class StudyWorkspace:
    root: Path

    @property
    def metadata_path(self) -> Path:
        return self.root / "study.json"

    @property
    def plan_path(self) -> Path:
        return self.root / "plan.json"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.jsonl"

    @property
    def identified_db(self) -> Path:
        return self.root / "omop" / "omop.duckdb"

    @property
    def deidentified_db(self) -> Path:
        return self.root / "omop" / "deid.duckdb"

    def relative(self, path: str | Path) -> str:
        return str(Path(path).resolve().relative_to(self.root.resolve()))

    def read_metadata(self) -> dict[str, Any]:
        try:
            return json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InvalidStudy(f"Invalid study metadata at {self.metadata_path}: {exc}") from exc

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        self.metadata_path.write_text(canonical_json(metadata) + "\n", encoding="utf-8")


def create_study(path: str | Path, name: str, *, force: bool = False) -> StudyWorkspace:
    root = Path(path).expanduser().resolve()
    existing_metadata = root / "study.json"
    if existing_metadata.exists() and not force:
        raise InvalidStudy(f"A study already exists at {root}")
    root.mkdir(parents=True, exist_ok=True)
    for directory in STUDY_DIRS:
        (root / directory).mkdir(parents=True, exist_ok=True)
    workspace = StudyWorkspace(root)
    now = datetime.now(UTC).isoformat()
    if existing_metadata.exists():
        # --force may refresh metadata/directories but must never sever existing
        # artifacts from their stable study identity.
        metadata = workspace.read_metadata()
        metadata.update({"name": name, "updated_at": now})
    else:
        metadata = {
            "schema_version": "1.0",
            "study_id": str(uuid.uuid4()),
            "name": name,
            "created_at": now,
            "updated_at": now,
        }
    workspace.write_metadata(metadata)
    workspace.manifest_path.touch(exist_ok=True)
    return workspace


def find_study(path: str | Path | None = None) -> StudyWorkspace:
    candidate = Path(path or Path.cwd()).expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for current in (candidate, *candidate.parents):
        if (current / "study.json").is_file():
            return StudyWorkspace(current)
    raise StudyNotFound(f"No study.json found at or above {candidate}; run 'pheno-rwe init'.")


def study_secret(workspace: StudyWorkspace) -> bytes:
    """Load or atomically create the secret shared by audit tokens/date shifting."""
    path = workspace.root / "omop" / ".deid_salt"
    if path.exists():
        value = path.read_bytes()
        if not value:
            raise InvalidStudy(f"Study secret is empty: {path}")
        return value
    value = secrets.token_bytes(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return study_secret(workspace)
    try:
        os.write(descriptor, value)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return value


def audit_token(
    workspace: StudyWorkspace,
    value: str,
    *,
    prefix: str = "t",
    length: int = 24,
) -> str:
    digest = hmac.new(
        study_secret(workspace),
        value.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{prefix}-{digest[:length]}"


def patient_token(workspace: StudyWorkspace, source_patient_id: str) -> str:
    return audit_token(workspace, source_patient_id, prefix="p")


@contextmanager
def study_lock(workspace: StudyWorkspace) -> Iterator[None]:
    """Take an advisory non-blocking writer lock for concurrent CLI work."""
    import fcntl

    lock_path = workspace.root / ".pheno-rwe.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StudyLocked(f"Study is already being modified: {workspace.root}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
