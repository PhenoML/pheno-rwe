"""Append-only, hash-chained study manifest."""

from __future__ import annotations

import json
import os
import platform
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pheno_rwe import __version__
from pheno_rwe.hashing import hash_directory, hash_file, hash_json

GENESIS_HASH = "0" * 64

_MANIFEST_STATUSES = {
    "success",
    "partial",
    "failed",
    "skipped",
    "refused",
    "cancelled",
}
_SHAREABLE_STEPS = {
    "init",
    "ingest",
    "pull",
    "enrich",
    "resolve-codes",
    "materialize",
    "resolve-cohort",
    "deid",
    "validate",
    "plan_amended",
    "plan_validated",
    "analyze",
    "analysis_rerun",
    "plot",
    "export",
}
_SHAREABLE_PARAM_KEYS: dict[str, tuple[str, ...]] = {
    "ingest": ("force",),
    "pull": ("approved",),
    "enrich": ("detection_effort", "validation_method"),
    "resolve-codes": ("name", "domain"),
    "materialize": ("force",),
    "resolve-cohort": ("cohorts",),
    "deid": (
        "tier",
        "age_bucket_years",
        "age_cap",
        "date_shift_max_days",
        "k",
    ),
    "validate": ("plan_hash", "phases"),
    "plan_amended": ("old_hash", "new_hash"),
    "plan_validated": ("plan_hash",),
    "analyze": ("analysis_id", "kind", "plan_hash"),
    "analysis_rerun": ("analysis_id", "kind", "plan_hash"),
    "plot": ("plot_id", "kind"),
    "export": ("include_database",),
}
_SHAREABLE_API_ENDPOINTS = {
    "/cohort",
    "/construe",
    "/fhir/search",
    "/fhir2omop/create",
    "/lang2fhir/document",
    "/tools/cohort",
}
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+ -]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(slots=True)
class ManifestEntry:
    step: str
    status: str
    started_at: str
    finished_at: str
    params: dict[str, Any] = field(default_factory=dict)
    input_signature: str | None = None
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    items: list[dict[str, Any]] = field(default_factory=list)
    api_calls: list[dict[str, Any]] = field(default_factory=list)
    versions: dict[str, str] = field(default_factory=dict)
    prev_hash: str = GENESIS_HASH
    entry_hash: str = ""

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("entry_hash", None)
        return value

    def finalize(self, previous_hash: str) -> ManifestEntry:
        self.prev_hash = previous_hash
        self.entry_hash = hash_json(self.payload())
        return self


@dataclass(frozen=True, slots=True)
class ManifestVerification:
    valid: bool
    entries: int
    errors: tuple[str, ...] = ()


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid manifest JSON at line {line_number}: {exc}") from exc
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid manifest JSON at line {line_number}: expected an object")
        entries.append(entry)
    return entries


def append_manifest(path: str | Path, entry: ManifestEntry) -> ManifestEntry:
    import fcntl

    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # Lock the file around both reading the chain head and appending the new
    # entry. O_APPEND alone prevents interleaved bytes but cannot prevent two
    # writers from selecting the same prev_hash.
    with manifest_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            previous = [json.loads(line) for line in handle.read().splitlines() if line.strip()]
            previous_hash = previous[-1]["entry_hash"] if previous else GENESIS_HASH
            entry.versions = entry.versions or runtime_versions()
            entry.finalize(previous_hash)
            serialized = json.dumps(asdict(entry), sort_keys=True, separators=(",", ":")) + "\n"
            handle.seek(0, os.SEEK_END)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return entry


def record_step(
    path: str | Path,
    *,
    step: str,
    status: str,
    started_at: datetime,
    params: dict[str, Any] | None = None,
    input_signature: str | None = None,
    inputs: dict[str, str] | None = None,
    outputs: dict[str, str] | None = None,
    items: Iterable[dict[str, Any]] = (),
    api_calls: Iterable[dict[str, Any]] = (),
) -> ManifestEntry:
    return append_manifest(
        path,
        ManifestEntry(
            step=step,
            status=status,
            started_at=started_at.astimezone(UTC).isoformat(),
            finished_at=datetime.now(UTC).isoformat(),
            params=params or {},
            input_signature=input_signature,
            inputs=inputs or {},
            outputs=outputs or {},
            items=list(items),
            api_calls=list(api_calls),
        ),
    )


def verify_manifest(
    path: str | Path,
    *,
    verify_inputs: bool = False,
    verify_outputs: bool = False,
    root: str | Path | None = None,
    restrict_to_root: bool = False,
) -> ManifestVerification:
    errors: list[str] = []
    expected_previous = GENESIS_HASH
    try:
        entries = read_manifest(path)
    except ValueError as exc:
        return ManifestVerification(False, 0, (str(exc),))
    for index, entry in enumerate(entries, 1):
        if entry.get("prev_hash") != expected_previous:
            errors.append(f"entry {index}: prev_hash does not match")
        claimed = entry.get("entry_hash", "")
        payload = dict(entry)
        payload.pop("entry_hash", None)
        computed = hash_json(payload)
        if claimed != computed:
            errors.append(f"entry {index}: entry_hash does not match payload")
        expected_previous = claimed
    if verify_inputs or verify_outputs:
        base = Path(root).resolve() if root else Path(path).resolve().parent
        # Artifacts are mutable across steps (plan amendments and in-place cohort
        # materialization are expected), so verify the latest declaration for a
        # resolved path rather than comparing today's file with every historical
        # digest. Inputs are visited before outputs so an in-place step's output
        # becomes the final declaration for that entry.
        latest_artifacts: dict[Path, tuple[str, str, int, str]] = {}
        for index, entry in enumerate(entries, 1):
            declarations: list[tuple[str, Any]] = []
            if verify_inputs:
                declarations.append(("input", entry.get("inputs")))
            if verify_outputs:
                declarations.append(("output", entry.get("outputs")))
            for kind, values in declarations:
                if not isinstance(values, Mapping):
                    errors.append(f"entry {index}: {kind}s must be an object")
                    continue
                for raw_label, raw_digest in values.items():
                    label = str(raw_label)
                    try:
                        target = Path(label)
                        if not target.is_absolute():
                            target = base / target
                        target = target.resolve()
                    except (OSError, ValueError):
                        errors.append(f"entry {index}: {kind} path is invalid")
                        continue
                    if restrict_to_root and not target.is_relative_to(base):
                        errors.append(f"entry {index}: {kind} path escapes the verification root")
                        continue
                    latest_artifacts[target] = (
                        label,
                        str(raw_digest),
                        index,
                        kind,
                    )
        for target, (label, expected, index, kind) in sorted(
            latest_artifacts.items(), key=lambda item: str(item[0])
        ):
            if not target.exists():
                errors.append(f"entry {index}: {kind} is missing: {label}")
                continue
            try:
                actual = hash_directory(target) if target.is_dir() else hash_file(target)
            except OSError as exc:
                errors.append(f"entry {index}: {kind} could not be hashed: {label}: {exc}")
                continue
            if actual != expected:
                errors.append(f"entry {index}: {kind} hash does not match: {label}")
    return ManifestVerification(not errors, len(entries), tuple(errors))


def _shareable_value(key: str, value: Any) -> Any:
    """Return only tightly constrained values from source-manifest parameters."""
    if key in {
        "force",
        "approved",
        "include_database",
    }:
        return value if isinstance(value, bool) else None
    if key in {
        "age_bucket_years",
        "age_cap",
        "date_shift_max_days",
        "k",
    }:
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    if key in {"plan_hash", "old_hash", "new_hash"}:
        return value if isinstance(value, str) and _SHA256.fullmatch(value) else None
    if key in {
        "analysis_id",
        "domain",
        "kind",
        "name",
        "plot_id",
        "tier",
    }:
        return value if isinstance(value, str) and _SAFE_IDENTIFIER.fullmatch(value) else None
    if key in {"detection_effort", "validation_method"}:
        return value if isinstance(value, str) and _SAFE_IDENTIFIER.fullmatch(value) else None
    if key in {"cohorts", "phases"} and isinstance(value, list):
        return [
            item for item in value if isinstance(item, str) and _SAFE_IDENTIFIER.fullmatch(item)
        ]
    return None


def _shareable_params(step: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in _SHAREABLE_PARAM_KEYS.get(step, ()):
        if key not in value:
            continue
        safe_value = _shareable_value(key, value[key])
        if safe_value is not None:
            result[key] = safe_value
    return result


def _shareable_item_summary(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        return []
    status_counts: dict[str, int] = {}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        status = item.get("status")
        if isinstance(status, str) and status in _MANIFEST_STATUSES:
            status_counts[status] = status_counts.get(status, 0) + 1
    summary: dict[str, Any] = {"item_count": len(value)}
    if status_counts:
        summary["status_counts"] = dict(sorted(status_counts.items()))
    return [summary]


def _shareable_api_calls(value: Any) -> list[dict[str, Any]]:
    """Keep aggregate call accounting, never request arguments or dynamic paths."""
    if not isinstance(value, list):
        return []
    totals: dict[str, dict[str, Any]] = {}
    for call in value:
        if not isinstance(call, Mapping):
            continue
        endpoint = call.get("endpoint")
        if not isinstance(endpoint, str) or endpoint not in _SHAREABLE_API_ENDPOINTS:
            continue
        aggregate = totals.setdefault(str(endpoint), {"endpoint": endpoint, "count": 0})
        count = call.get("count")
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            aggregate["count"] += count
        milliseconds = call.get("ms")
        if isinstance(milliseconds, (int, float)) and not isinstance(milliseconds, bool):
            aggregate["ms"] = round(float(aggregate.get("ms", 0.0)) + float(milliseconds), 3)
        credits = call.get("credits")
        if isinstance(credits, (int, float)) and not isinstance(credits, bool):
            aggregate["credits"] = float(aggregate.get("credits", 0.0)) + float(credits)
    return [totals[key] for key in sorted(totals)]


def _shareable_timestamp(value: Any, fallback: datetime) -> str:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            pass
        else:
            if parsed.tzinfo is not None:
                return parsed.astimezone(UTC).isoformat()
    return fallback.astimezone(UTC).isoformat()


def build_shareable_manifest(
    source_entries: Iterable[Mapping[str, Any]],
    *,
    payload_files: Mapping[str, str],
    bundle_sha256: str,
    started_at: datetime,
    finished_at: datetime,
    include_database: bool,
) -> list[dict[str, Any]]:
    """Rebuild a raw-free manifest for a de-identified export.

    Source hashes and chains are intentionally not carried over: even opaque
    fingerprints can link a shared bundle back to identified files. The rebuilt
    chain retains safe execution metadata, and its final entry declares every
    included payload file so a recipient can verify the bundle independently.
    """
    entries: list[dict[str, Any]] = []
    previous_hash = GENESIS_HASH
    source_count = 0
    retained_count = 0
    for source in source_entries:
        source_count += 1
        step = source.get("step")
        if not isinstance(step, str) or step not in _SHAREABLE_STEPS:
            continue
        status = source.get("status")
        safe_status = (
            status if isinstance(status, str) and status in _MANIFEST_STATUSES else "failed"
        )
        versions = source.get("versions")
        safe_versions = {
            key: str(versions[key])
            for key in ("pheno_rwe", "python", "platform")
            if isinstance(versions, Mapping)
            and key in versions
            and isinstance(versions[key], (str, int, float))
            and _SAFE_VERSION.fullmatch(str(versions[key]))
        }
        entry = ManifestEntry(
            step=step,
            status=str(safe_status),
            started_at=_shareable_timestamp(source.get("started_at"), started_at),
            finished_at=_shareable_timestamp(source.get("finished_at"), started_at),
            params=_shareable_params(step, source.get("params")),
            input_signature=None,
            inputs={},
            outputs={},
            items=_shareable_item_summary(source.get("items")),
            api_calls=_shareable_api_calls(source.get("api_calls")),
            versions=safe_versions,
        ).finalize(previous_hash)
        serialized = asdict(entry)
        entries.append(serialized)
        previous_hash = entry.entry_hash
        retained_count += 1

    export_entry = ManifestEntry(
        step="export",
        status="success",
        started_at=started_at.astimezone(UTC).isoformat(),
        finished_at=finished_at.astimezone(UTC).isoformat(),
        params={
            "bundle_schema_version": "2.0",
            "include_database": include_database,
            "identified_inputs_omitted": True,
            "manifest_profile": "shareable-v1",
        },
        input_signature=hash_json(dict(sorted(payload_files.items()))),
        inputs=dict(sorted(payload_files.items())),
        outputs={"bundle.json": bundle_sha256},
        items=[
            {
                "source_entry_count": source_count,
                "retained_entry_count": retained_count,
            }
        ],
        versions=runtime_versions(),
    ).finalize(previous_hash)
    entries.append(asdict(export_entry))
    return entries


def write_manifest(path: str | Path, entries: Iterable[Mapping[str, Any]]) -> None:
    """Write a complete manifest deterministically (used for staged exports)."""
    serialized = "".join(
        json.dumps(dict(entry), sort_keys=True, separators=(",", ":")) + "\n" for entry in entries
    )
    Path(path).write_text(serialized, encoding="utf-8")


def find_success(path: str | Path, step: str, input_signature: str) -> dict[str, Any] | None:
    for entry in reversed(read_manifest(path)):
        if (
            entry.get("step") == step
            and entry.get("status") in {"success", "skipped"}
            and entry.get("input_signature") == input_signature
        ):
            return entry
    return None


def runtime_versions() -> dict[str, str]:
    return {
        "pheno_rwe": __version__,
        "python": platform.python_version(),
        "platform": sys.platform,
    }
