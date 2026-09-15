"""Plan serialization, canonicalization, and hashing helpers."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from pheno_rwe.hashing import canonical_json, hash_json

from .schema import CodeSet, Plan

PlanInput = Plan | Mapping[str, Any]
CodeSetInput = CodeSet | Mapping[str, Any]


def parse_plan(value: PlanInput) -> Plan:
    """Validate and normalize a model or mapping as a :class:`Plan`."""

    if isinstance(value, Plan):
        return value
    return Plan.model_validate(value)


def load_plan(path: str | Path) -> Plan:
    """Load a UTF-8 JSON plan and validate it against the current schema."""

    plan_path = Path(path)
    with plan_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return Plan.model_validate(payload)


def normalized_plan_dict(plan: PlanInput) -> dict[str, Any]:
    """Return the complete semantic representation used for plan hashing.

    Defaults and explicit nulls are included.  Consequently two inputs that mean
    the same thing hash identically even if one omitted default-valued fields.
    """

    return parse_plan(plan).model_dump(mode="json", by_alias=True, exclude_none=False)


def canonical_plan_json(plan: PlanInput) -> str:
    """Serialize a normalized plan as canonical, whitespace-free JSON."""

    return canonical_json(normalized_plan_dict(plan))


def plan_hash(plan: PlanInput) -> str:
    """Return the SHA-256 hash used for pre-registration and stale-plan checks."""

    return hash_json(normalized_plan_dict(plan))


def code_set_review_hash(code_set: CodeSetInput) -> str:
    """Bind a researcher decision to one exact normalized coding set.

    The code-set name, prose description, and approval metadata are deliberately
    excluded.  The scientific domain and every field that affects an individual
    coding (including its mapping status and accepted state) are included.  The
    order is normalized so reformatting a set cannot invalidate a review.
    """

    if isinstance(code_set, CodeSet):
        normalized = code_set
    else:
        raw_codings = code_set.get("codings")
        codings = (
            [
                {
                    key: coding.get(key)
                    for key in (
                        "system",
                        "code",
                        "display",
                        "concept_id",
                        "mapping_status",
                        "accepted",
                    )
                    if key in coding
                }
                for coding in raw_codings
                if isinstance(coding, Mapping)
            ]
            if isinstance(raw_codings, list)
            else raw_codings
        )
        normalized = CodeSet.model_validate(
            {
                "name": str(code_set.get("name") or "reviewed-code-set"),
                "domain": code_set.get("domain"),
                "codings": codings,
            }
        )
    codings = [coding.model_dump(mode="json", exclude_none=False) for coding in normalized.codings]
    codings.sort(key=canonical_json)
    return hash_json({"domain": normalized.domain, "codings": codings})


def analysis_data_hash(plan: PlanInput) -> str:
    """Hash plan fields materialized into the analysis database.

    Analysis-only amendments need not rebuild de-identification, while changes
    to code sets, cohorts, or index-date derivation must not be analyzed against
    stale ``study`` tables copied into ``deid.duckdb``.
    """

    normalized = parse_plan(plan)
    return hash_json(
        {
            # Approval changes the canonical preregistration hash and static
            # gate, but it does not alter rows materialized into deid.duckdb.
            "code_sets": [
                item.model_dump(mode="json", exclude={"approval"}) for item in normalized.code_sets
            ],
            "cohorts": [item.model_dump(mode="json") for item in normalized.cohorts],
            "index_date_rule": normalized.index_date_rule.model_dump(mode="json"),
        }
    )


def write_plan(path: str | Path, plan: PlanInput) -> Path:
    """Validate and atomically write a human-readable plan JSON file."""

    plan_path = Path(path)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = parse_plan(plan)
    serialized = json.dumps(
        normalized.model_dump(mode="json", by_alias=True, exclude_none=True),
        ensure_ascii=False,
        indent=2,
    )
    serialized += "\n"

    # NamedTemporaryFile in the destination directory makes os.replace atomic on
    # normal local filesystems and avoids exposing a half-written registration.
    temp_name: str | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=plan_path.parent,
            prefix=f".{plan_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, plan_path)
    finally:
        if temp_name is not None:
            temp_path = Path(temp_name)
            if temp_path.exists():
                temp_path.unlink()
    return plan_path


def plan_json_schema() -> dict[str, Any]:
    """Return the JSON Schema consumed by the CLI, skill, and plan-builder UI."""

    return Plan.model_json_schema()


# Short alias that reads naturally in HTTP handlers.
json_schema = plan_json_schema


def is_plan_current(plan: PlanInput, validated_hash: str | None) -> bool:
    """Whether ``plan`` matches the last successfully validated plan hash."""

    return bool(validated_hash) and plan_hash(plan) == validated_hash
