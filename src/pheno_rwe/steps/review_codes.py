"""Record an auditable researcher decision for an already resolved code set."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from pheno_rwe.hashing import canonical_json, hash_bytes, hash_file, hash_json
from pheno_rwe.manifest import record_step
from pheno_rwe.plan import CodeSetApproval, Plan, code_set_review_hash, load_plan
from pheno_rwe.steps.common import StepResult, safe_identifier
from pheno_rwe.workspace import StudyWorkspace, find_study, study_lock

_MAPPING_STATUSES = ("ALREADY_STANDARD", "MAPPED", "UNCHECKED", "UNMAPPED")


def _artifact_codings(payload: dict[str, Any]) -> list[dict[str, Any]]:
    values = payload.get("codings")
    if not isinstance(values, list) or not values:
        raise ValueError("Resolved code-set artifact must contain at least one coding.")
    codings: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("Resolved code-set codings must be JSON objects.")
        codings.append(
            {
                "system": value.get("system"),
                "code": value.get("code"),
                "display": value.get("display"),
                "concept_id": value.get("concept_id"),
                "mapping_status": value.get("mapping_status"),
                "accepted": value.get("accepted", True),
            }
        )
    return codings


def _coding_fingerprint(codings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("system", "code", "display", "concept_id", "mapping_status", "accepted")
    normalized = [{key: coding.get(key) for key in keys} for coding in codings]
    return sorted(normalized, key=lambda item: (str(item["system"]), str(item["code"])))


def _mapping_counts(codings: list[dict[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in _MAPPING_STATUSES}
    for coding in codings:
        status = coding.get("mapping_status")
        if status not in counts:
            raise ValueError(
                "Every resolved coding requires one of these mapping statuses: "
                + ", ".join(_MAPPING_STATUSES)
            )
        counts[str(status)] += 1
    return counts


def _stage_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)


def _plan_text(plan: Plan) -> str:
    return (
        json.dumps(
            plan.model_dump(mode="json", by_alias=True, exclude_none=True),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


def review_code_set(
    study: StudyWorkspace | str | Path,
    *,
    name: str,
    decision: str,
    reviewed_by: str,
    reviewed_at: str | datetime,
    notes: str | None = None,
) -> StepResult:
    """Approve or reject a resolved code set that exactly matches ``plan.json``.

    This command intentionally cannot create a plan code set or replace its
    codings.  The researcher first reviews/edits the resolved artifact and plan;
    only an exact coding/domain match can receive the recorded decision.
    """

    workspace = study if isinstance(study, StudyWorkspace) else find_study(study)
    started = datetime.now(UTC)
    normalized_decision = decision.strip().lower()
    if normalized_decision not in {"approved", "rejected"}:
        raise ValueError("decision must be 'approved' or 'rejected'")
    target = workspace.root / "codesets" / f"{safe_identifier(name)}.codeset.json"
    if not target.is_file():
        raise FileNotFoundError(
            f"Resolved code-set artifact does not exist for '{name}'; run resolve-codes first."
        )
    if not workspace.plan_path.is_file():
        raise FileNotFoundError(
            "plan.json must already contain the reviewed code set before review-codes can run."
        )

    with study_lock(workspace):
        try:
            artifact = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid resolved code-set artifact at {target}: {exc}") from exc
        if not isinstance(artifact, dict) or artifact.get("name") != name:
            raise ValueError(f"Resolved code-set artifact does not declare name '{name}'.")
        plan = load_plan(workspace.plan_path)
        matches = [code_set for code_set in plan.code_sets if code_set.name == name]
        if len(matches) != 1:
            raise ValueError(
                f"plan.json must contain exactly one existing code set named '{name}'."
            )
        plan_code_set = matches[0]
        artifact_domain = str(artifact.get("domain") or "").strip().lower()
        if artifact_domain != plan_code_set.domain:
            raise ValueError(
                f"Resolved artifact domain '{artifact_domain}' does not match plan domain "
                f"'{plan_code_set.domain}' for code set '{name}'."
            )
        artifact_codings = _artifact_codings(artifact)
        plan_codings = [coding.model_dump(mode="json") for coding in plan_code_set.codings]
        if _coding_fingerprint(artifact_codings) != _coding_fingerprint(plan_codings):
            raise ValueError(
                f"Resolved artifact codings do not exactly match plan code set '{name}'; "
                "reconcile system, code, concept_id, mapping_status, accepted state, and display "
                "before recording a decision."
            )
        counts = _mapping_counts(artifact_codings)
        review_hash = code_set_review_hash(plan_code_set)
        approval = CodeSetApproval.model_validate(
            {
                "status": normalized_decision,
                "reviewed_by": reviewed_by,
                "reviewed_at": reviewed_at,
                "notes": notes,
                "review_hash": review_hash,
            }
        )

        plan_payload = plan.model_dump(mode="json", by_alias=True, exclude_none=False)
        for code_set in plan_payload["code_sets"]:
            if code_set["name"] == name:
                code_set["approval"] = approval.model_dump(mode="json", exclude_none=False)
        updated_plan = Plan.model_validate(plan_payload)
        artifact["mapping_status_counts"] = counts
        artifact["approval"] = approval.model_dump(mode="json", exclude_none=False)

        previous_artifact = target.read_text(encoding="utf-8")
        previous_plan = workspace.plan_path.read_text(encoding="utf-8")
        artifact_text = canonical_json(artifact) + "\n"
        plan_text = _plan_text(updated_plan)
        unchanged = artifact_text == previous_artifact and plan_text == previous_plan
        if not unchanged:
            staged_artifact = _stage_text(target, artifact_text)
            staged_plan = _stage_text(workspace.plan_path, plan_text)
            try:
                # Fail closed: publish the artifact first.  If the plan replace
                # fails, plan approval remains unchanged and validation refuses.
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

        status = "skipped" if unchanged else "success"
        reviewed_at_text = approval.model_dump(mode="json")["reviewed_at"]
        item = {
            "code_set": name,
            "decision": normalized_decision,
            "reviewed_by": approval.reviewed_by,
            "reviewed_at": reviewed_at_text,
            "mapping_status_counts": counts,
            "review_hash": review_hash,
        }
        inputs = {
            workspace.relative(target): hash_bytes(previous_artifact.encode("utf-8")),
            workspace.relative(workspace.plan_path): hash_bytes(previous_plan.encode("utf-8")),
        }
        outputs = {
            workspace.relative(target): hash_file(target),
            workspace.relative(workspace.plan_path): hash_file(workspace.plan_path),
        }
        record_step(
            workspace.manifest_path,
            step="review-codes",
            status=status,
            started_at=started,
            params={
                "name": name,
                "decision": normalized_decision,
                "reviewed_by": approval.reviewed_by,
                "reviewed_at": reviewed_at_text,
                "review_hash": review_hash,
            },
            input_signature=hash_json(
                {
                    "artifact": json.loads(previous_artifact),
                    "plan": json.loads(previous_plan),
                    "approval": approval.model_dump(mode="json"),
                }
            ),
            inputs=inputs,
            outputs=outputs,
            items=[item],
        )

    counts_text = ", ".join(f"{status}={counts[status]}" for status in _MAPPING_STATUSES)
    return StepResult(
        step="review-codes",
        status=status,
        message=(
            f"Recorded '{normalized_decision}' for code set '{name}'. "
            f"Mapping statuses: {counts_text}."
        ),
        outputs=outputs,
        items=[item],
    )


review_codes = review_code_set
