"""Validate preregistration, re-run DATA gates, and dispatch analyses read-only."""

from __future__ import annotations

import inspect
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pheno_rwe.analyses.features import derive_analysis_seed
from pheno_rwe.errors import GuardrailRefusal, StalePlanError
from pheno_rwe.guardrails import validate_analysis_data
from pheno_rwe.hashing import hash_file, hash_json
from pheno_rwe.manifest import record_step
from pheno_rwe.plan import load_plan, plan_hash
from pheno_rwe.runtime import CancellationToken, ProgressEvent, ProgressSink, null_progress
from pheno_rwe.steps.common import StepResult
from pheno_rwe.steps.validate import aggregate_summaries, last_validated_plan_hash
from pheno_rwe.workspace import StudyWorkspace, find_study, study_lock


def derived_seed(seed: int, analysis_id: str) -> int:
    return derive_analysis_seed(seed, analysis_id)


def _declared_artifacts(result_path: Path) -> set[Path]:
    """Return safe, existing artifacts declared by a result envelope."""

    artifacts = {result_path}
    if not result_path.is_file():
        return artifacts
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return artifacts
    output_dir = result_path.parent.resolve()
    for item in payload.get("output_tables", []):
        relative = item.get("path") if isinstance(item, dict) else None
        if not isinstance(relative, str) or not relative:
            continue
        candidate = result_path.parent / relative
        try:
            if candidate.resolve().parent == output_dir:
                artifacts.add(candidate)
        except OSError:
            continue
    return artifacts


def _write_result(result: Any, output_dir: Path) -> tuple[Path, ...]:
    """Atomically refresh the envelope and remove only previously declared stale tables."""

    result_path = output_dir / "result.json"
    previous = _declared_artifacts(result_path)
    writer = getattr(result, "write", None)
    if not callable(writer):
        raise RuntimeError("registered analysis did not return a writable AnalysisResult")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}.result."
    ) as temporary:
        staging = Path(temporary)
        staged_result = staging / "result.json"
        written_value = writer(staging)
        if not isinstance(written_value, (str, Path)):
            raise RuntimeError("registered analysis returned an invalid result path")
        written = Path(written_value)
        if written.resolve() != staged_result.resolve() or not staged_result.is_file():
            raise RuntimeError("registered analysis did not write the expected result.json")
        staged = _declared_artifacts(staged_result)
        missing = sorted(path.name for path in staged if not path.is_file())
        if missing:
            raise RuntimeError(
                "analysis result declares missing output table(s): " + ", ".join(missing)
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        current = {output_dir / path.name for path in staged}
        for artifact in sorted(staged - {staged_result}):
            os.replace(artifact, output_dir / artifact.name)
        # Publish the envelope last so it never references a table that has not
        # yet reached its final location.
        os.replace(staged_result, result_path)
    for stale in sorted(previous - current):
        if stale != result_path and (stale.is_file() or stale.is_symlink()):
            stale.unlink()
    return tuple(sorted(current))


def analyze_study(
    study: StudyWorkspace | str | Path,
    *,
    analysis_id: str | None = None,
    all: bool = False,
    progress: ProgressSink = null_progress,
    cancellation: CancellationToken | None = None,
    force: bool = False,
) -> StepResult:
    workspace = study if isinstance(study, StudyWorkspace) else find_study(study)
    from pheno_rwe.steps.deid import deid_freshness

    token = cancellation or CancellationToken()
    items: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}
    failures = 0
    started = datetime.now(UTC)
    selected: list[Any] = []

    with study_lock(workspace):
        if not workspace.deidentified_db.exists():
            raise FileNotFoundError("deid.duckdb is required and must be current before analysis.")
        fresh, reason = deid_freshness(workspace)
        if not fresh:
            raise StalePlanError(f"deid.duckdb is stale: {reason}; run deid and validate again.")
        plan = load_plan(workspace.plan_path)
        current_hash = plan_hash(plan)
        validated_hash = last_validated_plan_hash(workspace)
        if validated_hash != current_hash:
            raise StalePlanError(
                "plan.json does not match the last successfully validated plan; run validate first."
            )
        selected = [item for item in plan.analyses if item.enabled]
        if analysis_id:
            selected = [item for item in selected if item.id == analysis_id]
        elif not all and len(selected) > 1:
            raise ValueError("Choose --id or --all when the plan has multiple analyses.")
        if not selected:
            raise ValueError("No enabled matching analyses were found in plan.json.")

        summaries = aggregate_summaries(workspace.deidentified_db, plan)
        database_hash = hash_file(workspace.deidentified_db)
        for index, spec in enumerate(selected, 1):
            token.checkpoint()
            progress(
                ProgressEvent(
                    "analyze", "running", f"Running {spec.id}", index, len(selected), spec.id
                )
            )
            summary = summaries.get(spec.id)
            if summary is None:
                raise GuardrailRefusal(
                    f"DATA guardrails could not derive aggregate inputs for analysis '{spec.id}'."
                )
            gate = validate_analysis_data(plan, spec, summary)
            if gate.refused:
                messages = [outcome.message for outcome in gate.outcomes if outcome.refused]
                record_step(
                    workspace.manifest_path,
                    step="analyze",
                    status="refused",
                    started_at=started,
                    params={"analysis_id": spec.id, "plan_hash": current_hash},
                    items=[outcome.model_dump(mode="json") for outcome in gate.outcomes],
                )
                raise GuardrailRefusal("; ".join(messages), list(gate.outcomes))
            output_dir = workspace.root / "results" / spec.id
            result_path = output_dir / "result.json"
            step_name = "analysis_rerun" if result_path.exists() else "analyze"
            try:
                result = _run_analysis(
                    workspace,
                    plan,
                    spec,
                    summary,
                    output_dir,
                    derived_seed(plan.seed, spec.id),
                    tuple(outcome.model_dump(mode="json") for outcome in gate.outcomes),
                )
                artifact_paths = _write_result(result, output_dir)
                artifact_outputs = {
                    workspace.relative(path): hash_file(path) for path in artifact_paths
                }
                outputs.update(artifact_outputs)
                relative = workspace.relative(result_path)
                items.append(
                    {
                        "analysis_id": spec.id,
                        "kind": spec.kind,
                        "status": "success",
                        "guardrails": [
                            outcome.model_dump(mode="json") for outcome in gate.outcomes
                        ],
                        "power_note": gate.power_notes.get(spec.id),
                        "result": relative,
                        "outputs": sorted(artifact_outputs),
                    }
                )
                record_step(
                    workspace.manifest_path,
                    step=step_name,
                    status="success",
                    started_at=started,
                    params={"analysis_id": spec.id, "kind": spec.kind, "plan_hash": current_hash},
                    input_signature=hash_json(
                        {
                            "plan_hash": current_hash,
                            "analysis": spec.model_dump(mode="json"),
                            "database": database_hash,
                        }
                    ),
                    inputs={"omop/deid.duckdb": database_hash},
                    outputs=artifact_outputs,
                    items=[items[-1]],
                )
            except GuardrailRefusal:
                raise
            except Exception as exc:
                failures += 1
                items.append(
                    {
                        "analysis_id": spec.id,
                        "kind": spec.kind,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                record_step(
                    workspace.manifest_path,
                    step="analyze",
                    status="failed",
                    started_at=started,
                    params={"analysis_id": spec.id, "kind": spec.kind, "plan_hash": current_hash},
                    items=[items[-1]],
                )
    status = (
        "partial"
        if failures and failures < len(selected)
        else ("failed" if failures else "success")
    )
    return StepResult(
        "analyze",
        status,
        f"Completed {len(selected) - failures} analyses; {failures} failed.",
        outputs,
        items,
    )


def _run_analysis(
    workspace: StudyWorkspace,
    plan: Any,
    spec: Any,
    summary: Any,
    output_dir: Path,
    seed: int,
    guardrail_outcomes: tuple[dict[str, Any], ...] = (),
) -> Any:
    """Adapt to the analysis package's stable base protocol without importing models into CLI."""
    from pheno_rwe.analyses import AnalysisContext
    from pheno_rwe.analyses.prepare import prepare_context_spec
    from pheno_rwe.analyses.provenance import provenance_from_database
    from pheno_rwe.analyses.registry import run as run_analysis

    prepared_spec = prepare_context_spec(plan, spec, workspace.deidentified_db)
    data = prepared_spec["params"].get("data")
    person_ids = None
    if hasattr(data, "columns") and "person_id" in data.columns:
        person_ids = [int(value) for value in data["person_id"].dropna().unique()]
    provenance = provenance_from_database(str(workspace.deidentified_db), person_ids)
    kwargs = {
        "database": workspace.deidentified_db,
        "database_path": workspace.deidentified_db,
        "db_path": workspace.deidentified_db,
        "analysis_id": spec.id,
        "spec": prepared_spec,
        "params": prepared_spec["params"],
        "output_dir": output_dir,
        "seed": seed,
        "data_summary": summary,
        "guardrails": [],
        "guardrail_outcomes": guardrail_outcomes,
        "provenance": provenance,
    }
    signature = inspect.signature(AnalysisContext)
    context = AnalysisContext(
        **{key: value for key, value in kwargs.items() if key in signature.parameters}
    )
    return run_analysis(context)
