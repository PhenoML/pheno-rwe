"""Atomically curate one resolved coding in its artifact and declarative plan."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pheno_rwe.hashing import canonical_json, hash_bytes, hash_file, hash_json
from pheno_rwe.manifest import record_step
from pheno_rwe.plan import Plan, load_plan
from pheno_rwe.steps.common import StepResult, safe_identifier
from pheno_rwe.steps.review_codes import (
    _artifact_codings,
    _coding_fingerprint,
    _plan_text,
    _stage_text,
)
from pheno_rwe.workspace import StudyWorkspace, find_study, study_lock


def curate_code_set_coding(
    study: StudyWorkspace | str | Path,
    *,
    name: str,
    system: str,
    code: str,
    accepted: bool,
    curated_by: str,
    curated_at: str | datetime,
    expected_artifact_hash: str,
) -> StepResult:
    """Set one coding's accepted state with optimistic concurrency control."""

    workspace = study if isinstance(study, StudyWorkspace) else find_study(study)
    started = datetime.now(UTC)
    reviewer = curated_by.strip()
    if not reviewer:
        raise ValueError("curated_by must be a non-empty researcher name")
    when = datetime.fromisoformat(curated_at) if isinstance(curated_at, str) else curated_at
    if when.utcoffset() is None:
        raise ValueError("curated_at must include a UTC offset")
    when_text = when.isoformat()
    target = workspace.root / "codesets" / f"{safe_identifier(name)}.codeset.json"
    if not target.is_file() or not workspace.plan_path.is_file():
        raise FileNotFoundError("Resolved code-set artifact and plan.json are both required")

    with study_lock(workspace):
        current_hash = hash_file(target)
        if current_hash != expected_artifact_hash:
            raise ValueError("Code set changed since it was displayed; refresh before curating")
        previous_artifact = target.read_text(encoding="utf-8")
        previous_plan = workspace.plan_path.read_text(encoding="utf-8")
        artifact = json.loads(previous_artifact)
        if not isinstance(artifact, dict) or artifact.get("name") != name:
            raise ValueError(f"Resolved code-set artifact does not declare name '{name}'")
        artifact_codings = _artifact_codings(artifact)
        plan = load_plan(workspace.plan_path)
        matches = [value for value in plan.code_sets if value.name == name]
        if len(matches) != 1:
            raise ValueError(f"plan.json must contain exactly one code set named '{name}'")
        plan_code_set = matches[0]
        plan_codings = [value.model_dump(mode="json") for value in plan_code_set.codings]
        if _coding_fingerprint(artifact_codings) != _coding_fingerprint(plan_codings):
            raise ValueError(
                "Resolved artifact and plan codings differ; reconcile them before curation"
            )

        artifact_matches = [
            value
            for value in artifact_codings
            if value.get("system") == system and value.get("code") == code
        ]
        if len(artifact_matches) != 1:
            raise ValueError("The requested coding is missing or not unique")
        current_accepted = bool(artifact_matches[0].get("accepted", True))
        accepted_count = sum(bool(value.get("accepted", True)) for value in artifact_codings)
        if current_accepted and not accepted and accepted_count <= 1:
            raise ValueError(
                "A code set must retain at least one accepted coding; reject it instead"
            )

        changed = current_accepted != accepted
        if changed:
            for value in artifact.get("codings", []):
                if value.get("system") == system and value.get("code") == code:
                    value["accepted"] = accepted
            artifact["approval"] = {
                "status": "pending",
                "reviewed_by": None,
                "reviewed_at": None,
                "notes": None,
                "review_hash": None,
            }
            history = artifact.setdefault("curation", [])
            if not isinstance(history, list):
                raise ValueError("Resolved code-set curation history must be an array")
            history.append(
                {
                    "system": system,
                    "code": code,
                    "accepted": accepted,
                    "curated_by": reviewer,
                    "curated_at": when_text,
                }
            )

            plan_payload = plan.model_dump(mode="json", by_alias=True, exclude_none=False)
            for code_set in plan_payload["code_sets"]:
                if code_set["name"] != name:
                    continue
                for coding in code_set["codings"]:
                    if coding["system"] == system and coding["code"] == code:
                        coding["accepted"] = accepted
                code_set["approval"] = {
                    "status": "pending",
                    "reviewed_by": None,
                    "reviewed_at": None,
                    "notes": None,
                    "review_hash": None,
                }
            updated_plan = Plan.model_validate(plan_payload)
            artifact_text = canonical_json(artifact) + "\n"
            plan_text = _plan_text(updated_plan)
            staged_artifact = _stage_text(target, artifact_text)
            staged_plan = _stage_text(workspace.plan_path, plan_text)
            try:
                os.replace(staged_artifact, target)
                try:
                    os.replace(staged_plan, workspace.plan_path)
                except Exception:
                    rollback = _stage_text(target, previous_artifact)
                    os.replace(rollback, target)
                    raise
            finally:
                for staged in (staged_artifact, staged_plan):
                    if staged.exists():
                        staged.unlink()

        status = "success" if changed else "skipped"
        inputs = {
            workspace.relative(target): hash_bytes(previous_artifact.encode("utf-8")),
            workspace.relative(workspace.plan_path): hash_bytes(previous_plan.encode("utf-8")),
        }
        outputs = {
            workspace.relative(target): hash_file(target),
            workspace.relative(workspace.plan_path): hash_file(workspace.plan_path),
        }
        item: dict[str, Any] = {
            "code_set": name,
            "system": system,
            "code": code,
            "accepted": accepted,
            "curated_by": reviewer,
            "curated_at": when_text,
        }
        record_step(
            workspace.manifest_path,
            step="curate-codes",
            status=status,
            started_at=started,
            params=item,
            input_signature=hash_json(
                {
                    "artifact_hash": current_hash,
                    "decision": item,
                }
            ),
            inputs=inputs,
            outputs=outputs,
            items=[item],
        )
    return StepResult(
        "curate-codes",
        status,
        f"{'Updated' if changed else 'Kept'} {system}#{code}; code-set approval is pending.",
        outputs,
        [item],
    )
