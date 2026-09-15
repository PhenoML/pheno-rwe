"""Render registered plots only from current, manifest-declared tidy CSV artifacts."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pheno_rwe.errors import StalePlanError
from pheno_rwe.hashing import hash_file, hash_json
from pheno_rwe.manifest import read_manifest, record_step
from pheno_rwe.plan import load_plan, plan_hash
from pheno_rwe.steps.common import StepResult, safe_identifier
from pheno_rwe.workspace import StudyWorkspace, find_study, study_lock

_ANALYSIS_PLOT_KINDS = {
    "forest",
    "incidence_bar",
    "km_curve",
    "love_plot",
    "pathway_bars",
    "trajectory",
    "umap_scatter",
}
_AGGREGATE_PLOT_SOURCES = {
    "attrition": ("reports/cohort_attrition.csv", "resolve-cohort"),
    "mapping_coverage": ("reports/mapping_coverage.csv", "materialize"),
}


def plot_study(
    study: StudyWorkspace | str | Path,
    *,
    plot_id: str | None = None,
    all: bool = False,
    from_csv: str | Path | None = None,
    kind: str | None = None,
    force: bool = False,
) -> StepResult:
    workspace = study if isinstance(study, StudyWorkspace) else find_study(study)
    if from_csv is not None:
        return _plot_replay(
            workspace,
            from_csv=from_csv,
            plot_id=plot_id,
            kind=kind,
            force=force,
        )
    if kind is not None:
        raise ValueError("--kind is only valid with --from-csv")
    return _plot_registered(workspace, plot_id=plot_id, all=all)


def _plot_registered(
    workspace: StudyWorkspace,
    *,
    plot_id: str | None,
    all: bool,
) -> StepResult:
    from pheno_rwe.steps.deid import deid_freshness

    started = datetime.now(UTC)
    items: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}
    failures = 0
    with study_lock(workspace):
        # The plan, validation head, source declarations, and output manifest
        # records all live under one study-lock snapshot.
        plan = load_plan(workspace.plan_path)
        current_plan_hash = plan_hash(plan)
        manifest = read_manifest(workspace.manifest_path)
        if _last_successful_validation_hash(manifest) != current_plan_hash:
            raise StalePlanError(
                "plan.json does not match the last successfully validated plan; "
                "validate the current plan before plotting."
            )
        fresh, reason = deid_freshness(workspace)
        if not fresh:
            raise StalePlanError(f"deid.duckdb is stale: {reason}; rebuild and validate again.")
        deid_hash = hash_file(workspace.deidentified_db)
        specs = [item for item in plan.plots if item.enabled]
        if plot_id:
            specs = [item for item in specs if item.id == plot_id]
        elif not all and len(specs) > 1:
            raise ValueError("Choose --id or --all when the plan has multiple plots.")
        if not specs:
            raise ValueError("No enabled matching plots were found in plan.json.")

        for spec in specs:
            params = spec.params.model_dump(mode="json")
            source, source_hash, producer = _registered_source(
                workspace,
                kind=spec.kind,
                params=params,
                manifest=manifest,
                current_plan_hash=current_plan_hash,
                deid_hash=deid_hash,
            )
            source_label = workspace.relative(source)
            output_dir = workspace.root / "plots" / spec.id
            try:
                artifacts = _render(
                    spec.kind,
                    source,
                    output_dir,
                    params,
                    provenance=None,
                )
                artifact_paths = _artifact_paths(artifacts, output_dir)
                artifact_outputs = {
                    workspace.relative(path): hash_file(path)
                    for path in artifact_paths
                    if path.exists()
                }
                outputs.update(artifact_outputs)
                item = {
                    "plot_id": spec.id,
                    "kind": spec.kind,
                    "status": "success",
                    "source_csv": source_label,
                    "source_sha256": source_hash,
                    "plan_sha256": current_plan_hash,
                    "deid_sha256": deid_hash,
                    "producer_entry_hash": producer.get("entry_hash"),
                }
                items.append(item)
                record_step(
                    workspace.manifest_path,
                    step="plot",
                    status="success",
                    started_at=started,
                    params={
                        "plot_id": spec.id,
                        "kind": spec.kind,
                        "plan_hash": current_plan_hash,
                        "deid_sha256": deid_hash,
                        "source_sha256": source_hash,
                    },
                    input_signature=hash_json(
                        {"kind": spec.kind, "params": params, "csv": source_hash}
                    ),
                    inputs={source_label: source_hash},
                    outputs=artifact_outputs,
                    items=[item],
                )
            except StalePlanError:
                raise
            except Exception as exc:
                failures += 1
                item = {
                    "plot_id": spec.id,
                    "kind": spec.kind,
                    "status": "failed",
                    "error": str(exc),
                    "source_csv": source_label,
                    "source_sha256": source_hash,
                    "plan_sha256": current_plan_hash,
                    "deid_sha256": deid_hash,
                }
                items.append(item)
                record_step(
                    workspace.manifest_path,
                    step="plot",
                    status="failed",
                    started_at=started,
                    params={
                        "plot_id": spec.id,
                        "kind": spec.kind,
                        "plan_hash": current_plan_hash,
                        "deid_sha256": deid_hash,
                        "source_sha256": source_hash,
                    },
                    inputs={source_label: source_hash},
                    items=[item],
                )
    status = (
        "partial" if failures and failures < len(specs) else ("failed" if failures else "success")
    )
    return StepResult(
        "plot",
        status,
        f"Rendered {len(specs) - failures} registered plots; {failures} failed.",
        outputs,
        items,
    )


def _plot_replay(
    workspace: StudyWorkspace,
    *,
    from_csv: str | Path,
    plot_id: str | None,
    kind: str | None,
    force: bool,
) -> StepResult:
    if not kind:
        raise ValueError("--kind is required with --from-csv")
    source = Path(from_csv).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".csv":
        raise FileNotFoundError("--from-csv must identify an existing CSV file")
    identifier = safe_identifier(plot_id or source.stem)
    if plot_id is not None and identifier != plot_id:
        raise ValueError("Replay --id must contain only letters, numbers, '.', '-', or '_'")
    source_hash = hash_file(source)
    started = datetime.now(UTC)
    output_dir = workspace.root / "replays" / "plots" / identifier
    provenance = {
        "registration": "unregistered_replay",
        "shareable": False,
        "source_sha256": source_hash,
    }
    with study_lock(workspace):
        artifacts = _render(kind, source, output_dir, {}, provenance=provenance)
        artifact_paths = _artifact_paths(artifacts, output_dir)
        artifact_outputs = {
            workspace.relative(path): hash_file(path) for path in artifact_paths if path.exists()
        }
        source_label = (
            workspace.relative(source) if source.is_relative_to(workspace.root) else str(source)
        )
        item = {
            "plot_id": identifier,
            "kind": kind,
            "status": "success",
            "registration": "unregistered_replay",
            "shareable": False,
            "source_sha256": source_hash,
        }
        record_step(
            workspace.manifest_path,
            step="plot_replay",
            status="success",
            started_at=started,
            params={
                "plot_id": identifier,
                "kind": kind,
                "force": force,
                "shareable": False,
                "source_sha256": source_hash,
            },
            input_signature=hash_json(
                {"kind": kind, "source_sha256": source_hash, "registered": False}
            ),
            inputs={source_label: source_hash},
            outputs=artifact_outputs,
            items=[item],
        )
    return StepResult(
        "plot_replay",
        "success",
        "Rendered one unregistered, non-shareable CSV replay under replays/plots.",
        artifact_outputs,
        [item],
        ["Replay plots are excluded from the registered plots directory and shareable export."],
    )


def _last_successful_validation_hash(manifest: list[dict[str, Any]]) -> str | None:
    for entry in reversed(manifest):
        if entry.get("status") != "success":
            continue
        raw_params = entry.get("params")
        params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
        if entry.get("step") == "plan_amended":
            value = params.get("new_hash")
            return str(value) if value else None
        if entry.get("step") == "plan_validated":
            value = params.get("plan_hash")
            return str(value) if value else None
    return None


def _registered_source(
    workspace: StudyWorkspace,
    *,
    kind: str,
    params: dict[str, Any],
    manifest: list[dict[str, Any]],
    current_plan_hash: str,
    deid_hash: str,
) -> tuple[Path, str, dict[str, Any]]:
    if kind in _ANALYSIS_PLOT_KINDS:
        return _analysis_source(
            workspace,
            params=params,
            manifest=manifest,
            current_plan_hash=current_plan_hash,
            deid_hash=deid_hash,
        )
    if kind in _AGGREGATE_PLOT_SOURCES:
        relative, producer_step = _AGGREGATE_PLOT_SOURCES[kind]
        expected = (workspace.root / relative).resolve()
        explicit = params.get("source_csv")
        if explicit and _resolve_source(workspace, explicit) != expected:
            raise ValueError(
                f"Registered {kind} plots must use the current aggregate report at {relative}."
            )
        if not expected.is_file():
            raise StalePlanError(
                f"Current aggregate source {relative} is missing; rerun {producer_step}."
            )
        source_hash = hash_file(expected)
        producer = _find_declaring_entry(
            workspace,
            manifest,
            source=expected,
            source_hash=source_hash,
            steps={producer_step},
        )
        if producer is None:
            raise StalePlanError(
                f"Current aggregate source {relative} is not declared by a successful "
                f"{producer_step} step; rerun that step."
            )
        return expected, source_hash, producer
    raise ValueError(f"Plot kind '{kind}' has no registered source provenance contract.")


def _analysis_source(
    workspace: StudyWorkspace,
    *,
    params: dict[str, Any],
    manifest: list[dict[str, Any]],
    current_plan_hash: str,
    deid_hash: str,
) -> tuple[Path, str, dict[str, Any]]:
    analysis_id = str(params.get("analysis_id") or "")
    if not analysis_id:
        raise ValueError("Registered analysis plots require params.analysis_id.")
    result_root = (workspace.root / "results" / analysis_id).resolve()
    explicit = params.get("source_csv")
    if explicit:
        candidates = [_resolve_source(workspace, explicit)]
    else:
        preferred = result_root / "data.csv"
        candidates = [preferred, *sorted(result_root.glob("*.csv"))]
        candidates = list(dict.fromkeys(candidates))
    for source in candidates:
        if not source.is_file() or source.suffix.lower() != ".csv" or source.is_symlink():
            continue
        resolved = source.resolve()
        if not resolved.is_relative_to(result_root):
            if explicit:
                raise ValueError(
                    "A registered analysis plot source must be inside its "
                    "analysis result directory."
                )
            continue
        source_hash = hash_file(resolved)
        producer = _find_declaring_entry(
            workspace,
            manifest,
            source=resolved,
            source_hash=source_hash,
            steps={"analyze", "analysis_rerun"},
            analysis_id=analysis_id,
            plan_hash_value=current_plan_hash,
            deid_hash=deid_hash,
        )
        if producer is not None:
            return resolved, source_hash, producer
    if explicit and not candidates[0].resolve().is_relative_to(result_root):
        raise ValueError(
            "A registered analysis plot source must be inside its analysis result directory."
        )
    raise StalePlanError(
        f"No current manifest-declared CSV exists for analysis '{analysis_id}'; rerun analysis."
    )


def _find_declaring_entry(
    workspace: StudyWorkspace,
    manifest: list[dict[str, Any]],
    *,
    source: Path,
    source_hash: str,
    steps: set[str],
    analysis_id: str | None = None,
    plan_hash_value: str | None = None,
    deid_hash: str | None = None,
) -> dict[str, Any] | None:
    for entry in reversed(manifest):
        if entry.get("status") != "success" or entry.get("step") not in steps:
            continue
        raw_params = entry.get("params")
        params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
        if analysis_id is not None and params.get("analysis_id") != analysis_id:
            continue
        if plan_hash_value is not None and params.get("plan_hash") != plan_hash_value:
            continue
        if not _declares_path(
            workspace,
            entry.get("outputs"),
            target=source,
            digest=source_hash,
        ):
            continue
        if deid_hash is not None and not _declares_path(
            workspace,
            entry.get("inputs"),
            target=workspace.deidentified_db.resolve(),
            digest=deid_hash,
        ):
            continue
        return entry
    return None


def _declares_path(
    workspace: StudyWorkspace,
    declarations: Any,
    *,
    target: Path,
    digest: str,
) -> bool:
    if not isinstance(declarations, dict):
        return False
    resolved_target = target.resolve()
    for label, value in declarations.items():
        try:
            declared = Path(str(label))
            declared = declared if declared.is_absolute() else workspace.root / declared
            if declared.resolve() == resolved_target and value == digest:
                return True
        except (OSError, ValueError):
            continue
    return False


def _resolve_source(workspace: StudyWorkspace, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else workspace.root / path).resolve()


def _render(
    kind: str,
    source: Path,
    output_dir: Path,
    params: dict[str, Any],
    *,
    provenance: dict[str, Any] | None,
) -> Any:
    from pheno_rwe.plots import render_plot

    signature = inspect.signature(render_plot)
    kwargs = {
        "kind": kind,
        "data": source,
        "csv_path": source,
        "output_dir": output_dir,
        "spec": params,
        "params": params,
        "provenance": provenance,
    }
    return render_plot(
        **{key: value for key, value in kwargs.items() if key in signature.parameters}
    )


def _artifact_paths(value: Any, output_dir: Path) -> list[Path]:
    paths = [output_dir / name for name in ("plot.png", "plot.svg", "data.csv", "spec.json")]
    if value:
        for item in vars(value).values() if hasattr(value, "__dict__") else []:
            if isinstance(item, (str, Path)):
                path = Path(item)
                if path not in paths:
                    paths.append(path)
    return paths
