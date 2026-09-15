from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any

from pheno_rwe.deid.policy import (
    DEIDENTIFIED_COLUMN_POLICY,
    policy_summary,
    require_deidentified_policy,
)
from pheno_rwe.hashing import canonical_json, hash_directory, hash_file, hash_json
from pheno_rwe.manifest import (
    build_shareable_manifest,
    read_manifest,
    record_step,
    verify_manifest,
    write_manifest,
)
from pheno_rwe.omop.loader import connect_database
from pheno_rwe.plan import Plan, load_plan, plan_hash
from pheno_rwe.steps.common import StepResult, safe_identifier
from pheno_rwe.steps.validate import last_validated_plan_hash
from pheno_rwe.workspace import StudyWorkspace, find_study, study_lock

_REPORT_PRODUCERS: dict[str, tuple[str, frozenset[str]]] = {
    "reports/cohort_attrition.csv": ("resolve-cohort", frozenset({"success"})),
    "reports/cohort_attrition.json": ("resolve-cohort", frozenset({"success"})),
    "reports/mapping_coverage.json": (
        "materialize",
        frozenset({"success", "partial"}),
    ),
    "reports/mapping_coverage.md": (
        "materialize",
        frozenset({"success", "partial"}),
    ),
    "reports/validation_report.json": ("validate", frozenset({"success"})),
}
_UNPRODUCED_REPORTS = frozenset({"reports/mapping_coverage.csv"})
_SAFE_PLOT_FILES = frozenset({"data.csv", "plot.png", "plot.svg", "spec.json"})
_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "analysis_id",
        "kind",
        "status",
        "n",
        "estimates",
        "provenance",
        "assumptions_checked",
        "guardrail_outcomes",
        "power_note",
        "seed",
        "package_versions",
        "output_tables",
        "metadata",
    }
)
_RESULT_TABLE_KEYS = frozenset({"name", "path", "format", "rows", "columns"})


@dataclass(frozen=True, slots=True)
class _OutputDeclaration:
    entry_index: int
    entry: dict[str, Any]
    digest: str


def _integer_fields(value: Any, keys: tuple[str, ...]) -> dict[str, int | None]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int | None] = {}
    for key in keys:
        item = value.get(key)
        if item is None:
            result[key] = None
        elif isinstance(item, int) and not isinstance(item, bool):
            result[key] = item
    return result


def _entry_params(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    params = entry.get("params")
    return params if isinstance(params, Mapping) else {}


def _latest_output_declaration(
    entries: list[dict[str, Any]], relative: str
) -> _OutputDeclaration | None:
    """Return the latest manifest OUTPUT declaration for one workspace path.

    Inputs deliberately do not authorize export. A result/report may be declared
    later as a plot input, but its bytes still have to match the producing output.
    """

    for index in range(len(entries) - 1, -1, -1):
        entry = entries[index]
        outputs = entry.get("outputs")
        if not isinstance(outputs, Mapping) or relative not in outputs:
            continue
        digest = outputs[relative]
        if not isinstance(digest, str):
            raise RuntimeError(f"Manifest output digest is invalid for {relative}")
        return _OutputDeclaration(index, entry, digest)
    return None


def _require_regular_artifact(workspace: StudyWorkspace, relative: str) -> Path:
    source = workspace.root / relative
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"Refusing to export non-regular artifact: {relative}")
    if not source.resolve().is_relative_to(workspace.root.resolve()):
        raise RuntimeError(f"Refusing to export artifact outside the study: {relative}")
    return source


def _verify_output_declaration(
    workspace: StudyWorkspace,
    entries: list[dict[str, Any]],
    relative: str,
    *,
    producer_step: str,
    statuses: frozenset[str],
    entry_index: int | None = None,
) -> tuple[Path, _OutputDeclaration]:
    declaration = _latest_output_declaration(entries, relative)
    if declaration is None:
        raise RuntimeError(f"Refusing to export artifact without a producer output: {relative}")
    entry = declaration.entry
    if declaration.entry_index != entry_index and entry_index is not None:
        raise RuntimeError(f"Refusing to export artifact superseded after its run: {relative}")
    if entry.get("step") != producer_step or entry.get("status") not in statuses:
        raise RuntimeError(
            f"Refusing to export artifact whose latest producer is not {producer_step}: {relative}"
        )
    source = _require_regular_artifact(workspace, relative)
    if hash_file(source) != declaration.digest:
        raise RuntimeError(f"Refusing to export artifact with a producer hash mismatch: {relative}")
    return source, declaration


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return payload


def _safe_table_name(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError("Analysis result has an invalid output table name")
    if any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in value
    ):
        raise RuntimeError(f"Analysis result has an unsafe output table name: {value!r}")
    return value


def _validate_csv_description(path: Path, description: Mapping[str, Any]) -> None:
    columns = description.get("columns")
    rows = description.get("rows")
    if (
        not isinstance(columns, list)
        or not all(isinstance(column, str) for column in columns)
        or len(columns) != len(set(columns))
    ):
        raise RuntimeError(f"Analysis result declares invalid CSV columns: {path.name}")
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
        raise RuntimeError(f"Analysis result declares an invalid CSV row count: {path.name}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            actual_rows = sum(1 for _ in reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise RuntimeError(f"Could not validate analysis CSV {path.name}: {exc}") from exc
    if header != columns:
        raise RuntimeError(f"Analysis CSV columns do not match result.json: {path.name}")
    if actual_rows != rows:
        raise RuntimeError(f"Analysis CSV row count does not match result.json: {path.name}")


def _latest_analysis_attempt(
    entries: list[dict[str, Any]], analysis_id: str
) -> tuple[int, dict[str, Any]] | None:
    for index in range(len(entries) - 1, -1, -1):
        entry = entries[index]
        if entry.get("step") not in {"analyze", "analysis_rerun"}:
            continue
        if _entry_params(entry).get("analysis_id") == analysis_id:
            return index, entry
    return None


def _select_results(
    workspace: StudyWorkspace,
    plan: Plan,
    current_plan_hash: str,
    entries: list[dict[str, Any]],
) -> tuple[list[Path], dict[str, frozenset[Path]]]:
    """Authorize only current-plan result envelopes and their declared tables."""

    candidates = {
        path
        for path in (workspace.root / "results").glob("*/*")
        if path.suffix.lower() in {".csv", ".json"}
    }
    authorized: set[Path] = set()
    by_analysis: dict[str, frozenset[Path]] = {}
    for spec in plan.analyses:
        if not spec.enabled:
            continue
        output_dir = workspace.root / "results" / spec.id
        local_candidates = {path for path in candidates if path.parent == output_dir}
        if not local_candidates:
            continue
        result_relative = f"results/{spec.id}/result.json"
        result_path = _require_regular_artifact(workspace, result_relative)
        payload = _read_json_object(result_path, label=result_relative)
        if set(payload) != _RESULT_KEYS:
            raise RuntimeError(f"Analysis result envelope has unexpected fields: {result_relative}")
        if (
            payload.get("schema_version") != "1.0"
            or payload.get("analysis_id") != spec.id
            or payload.get("kind") != spec.kind
            or payload.get("status") != "success"
        ):
            raise RuntimeError(f"Analysis result envelope does not match plan: {result_relative}")
        if not isinstance(payload.get("n"), dict):
            raise RuntimeError(f"Analysis result has invalid n metadata: {result_relative}")
        for key in ("estimates", "assumptions_checked", "guardrail_outcomes"):
            if not isinstance(payload.get(key), list):
                raise RuntimeError(f"Analysis result has invalid {key}: {result_relative}")
        for key in ("provenance", "package_versions", "metadata"):
            if not isinstance(payload.get(key), dict):
                raise RuntimeError(f"Analysis result has invalid {key}: {result_relative}")
        if not isinstance(payload.get("power_note"), str):
            raise RuntimeError(f"Analysis result has invalid power_note: {result_relative}")
        seed = payload.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise RuntimeError(f"Analysis result has invalid seed: {result_relative}")

        table_values = payload.get("output_tables")
        if not isinstance(table_values, list):
            raise RuntimeError(f"Analysis result has invalid output_tables: {result_relative}")
        expected_relatives = {result_relative}
        table_paths: set[Path] = set()
        seen_names: set[str] = set()
        for value in table_values:
            if not isinstance(value, Mapping) or set(value) != _RESULT_TABLE_KEYS:
                raise RuntimeError(
                    f"Analysis result has an invalid output table description: {result_relative}"
                )
            name = _safe_table_name(value.get("name"))
            if name in seen_names:
                raise RuntimeError(f"Analysis result repeats output table {name!r}")
            seen_names.add(name)
            filename = f"{name}.csv"
            if value.get("path") != filename or value.get("format") != "text/csv":
                raise RuntimeError(f"Analysis result has an unsafe table path: {result_relative}")
            relative = f"results/{spec.id}/{filename}"
            table_path = _require_regular_artifact(workspace, relative)
            _validate_csv_description(table_path, value)
            expected_relatives.add(relative)
            table_paths.add(table_path)

        attempt = _latest_analysis_attempt(entries, spec.id)
        if attempt is None:
            raise RuntimeError(f"Refusing to export result without an analysis run: {spec.id}")
        entry_index, entry = attempt
        params = _entry_params(entry)
        if (
            entry.get("status") != "success"
            or params.get("kind") != spec.kind
            or params.get("plan_hash") != current_plan_hash
        ):
            raise RuntimeError(f"Refusing to export stale or unsuccessful result: {spec.id}")
        outputs = entry.get("outputs")
        if not isinstance(outputs, Mapping) or set(outputs) != expected_relatives:
            raise RuntimeError(f"Analysis manifest outputs do not match result.json: {spec.id}")
        for relative in sorted(expected_relatives):
            _verify_output_declaration(
                workspace,
                entries,
                relative,
                producer_step=str(entry.get("step")),
                statuses=frozenset({"success"}),
                entry_index=entry_index,
            )
        result_files = {workspace.root / relative for relative in expected_relatives}
        authorized.update(result_files)
        by_analysis[spec.id] = frozenset(table_paths)

    unexpected = sorted(str(path.relative_to(workspace.root)) for path in candidates - authorized)
    if unexpected:
        raise RuntimeError(
            "Refusing to export untracked or non-plan result artifact(s): " + ", ".join(unexpected)
        )
    return sorted(authorized), by_analysis


def _select_reports(
    workspace: StudyWorkspace,
    entries: list[dict[str, Any]],
    *,
    current_plan_hash: str,
) -> list[Path]:
    selected: list[Path] = []
    for relative, (producer_step, statuses) in sorted(_REPORT_PRODUCERS.items()):
        source = workspace.root / relative
        if not source.exists():
            continue
        source, declaration = _verify_output_declaration(
            workspace,
            entries,
            relative,
            producer_step=producer_step,
            statuses=statuses,
        )
        if (
            producer_step == "validate"
            and _entry_params(declaration.entry).get("plan_hash") != current_plan_hash
        ):
            raise RuntimeError("Refusing to export a validation report for a different plan")
        selected.append(source)
    unexpected = [
        relative for relative in sorted(_UNPRODUCED_REPORTS) if (workspace.root / relative).exists()
    ]
    if unexpected:
        raise RuntimeError(
            "Refusing to export report artifact(s) without an engine producer: "
            + ", ".join(unexpected)
        )
    return selected


def _copy_regular_file(source: Path, destination: Path, *, workspace_root: Path) -> None:
    """Copy one in-workspace regular file without following symlink escapes."""
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"Refusing to export non-regular artifact: {source.name}")
    if not source.resolve().is_relative_to(workspace_root.resolve()):
        raise RuntimeError(f"Refusing to export artifact outside the study: {source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _approved_explicit_plot_source(spec: Any, params: Mapping[str, Any]) -> str | None:
    source_csv = params.get("source_csv")
    if not source_csv:
        return None
    value = str(source_csv)
    candidate = Path(value)
    if (
        candidate.is_absolute()
        or PureWindowsPath(value).is_absolute()
        or ".." in candidate.parts
        or value.startswith("~")
    ):
        raise RuntimeError(
            "Refusing to export plan.json with a non-study-relative plot source_csv; "
            "use a study-relative artifact path and revalidate the plan."
        )
    relative = candidate.as_posix()
    analysis_id = params.get("analysis_id")
    if isinstance(analysis_id, str):
        expected_parent = Path("results") / analysis_id
        if candidate.parent != expected_parent or candidate.suffix.lower() != ".csv":
            raise RuntimeError(
                f"Plot '{spec.id}' source_csv must be a CSV output of analysis '{analysis_id}'"
            )
        return relative
    expected_report = {
        "attrition": "reports/cohort_attrition.csv",
        "mapping_coverage": "reports/mapping_coverage.csv",
    }.get(spec.kind)
    if expected_report is None or relative != expected_report:
        raise RuntimeError(
            f"Plot '{spec.id}' source_csv is not an approved generated report artifact"
        )
    return relative


def _validate_plan_plot_sources(plan: Plan) -> None:
    for spec in plan.plots:
        params = spec.params.model_dump(mode="json")
        _approved_explicit_plot_source(spec, params)


def _latest_plot_entry(
    entries: list[dict[str, Any]], plot_id: str
) -> tuple[int, dict[str, Any]] | None:
    for index in range(len(entries) - 1, -1, -1):
        entry = entries[index]
        if entry.get("step") != "plot":
            continue
        if _entry_params(entry).get("plot_id") == plot_id:
            return index, entry
    return None


def _manifest_plot_input(workspace: StudyWorkspace, entry: Mapping[str, Any]) -> tuple[Path, str]:
    inputs = entry.get("inputs")
    if not isinstance(inputs, Mapping) or len(inputs) != 1:
        raise RuntimeError("Plot manifest must declare exactly one source CSV")
    label, digest = next(iter(inputs.items()))
    if not isinstance(label, str) or not isinstance(digest, str):
        raise RuntimeError("Plot manifest source declaration is invalid")
    raw = Path(label)
    candidate = raw if raw.is_absolute() else workspace.root / raw
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise RuntimeError("Plot manifest source path is invalid") from exc
    if not resolved.is_relative_to(workspace.root.resolve()):
        raise RuntimeError("Refusing to export a plot generated from outside the study")
    return resolved, digest


def _expected_plot_source(
    workspace: StudyWorkspace,
    spec: Any,
    params: Mapping[str, Any],
    result_tables: Mapping[str, frozenset[Path]],
    report_paths: frozenset[Path],
) -> Path:
    explicit = _approved_explicit_plot_source(spec, params)
    analysis_id = params.get("analysis_id")
    if explicit is not None:
        source = _require_regular_artifact(workspace, explicit)
    elif isinstance(analysis_id, str):
        output_dir = workspace.root / "results" / analysis_id
        preferred = output_dir / "data.csv"
        source = (
            preferred if preferred.exists() else next(iter(sorted(output_dir.glob("*.csv"))), None)
        )
        if source is None:
            raise RuntimeError(f"Plot '{spec.id}' has no current analysis CSV source")
        relative = str(source.relative_to(workspace.root))
        source = _require_regular_artifact(workspace, relative)
    else:
        raise RuntimeError(f"Plot '{spec.id}' does not declare a shareable CSV source")

    if isinstance(analysis_id, str):
        if source not in result_tables.get(analysis_id, frozenset()):
            raise RuntimeError(f"Plot '{spec.id}' source is not a current declared analysis table")
    elif source not in report_paths:
        raise RuntimeError(f"Plot '{spec.id}' source is not a current generated report")
    return source


def _validate_plot_spec(path: Path, *, kind: str, params: Mapping[str, Any], data: Path) -> None:
    payload = _read_json_object(path, label=str(path.name))
    if set(payload) != {
        "schema_version",
        "kind",
        "params",
        "columns",
        "renderer",
        "dpi",
        "style",
    }:
        raise RuntimeError(f"Plot spec has unexpected fields: {path}")
    render_params = dict(params)
    render_params.pop("source_csv", None)
    render_params.pop("analysis_id", None)
    if (
        payload.get("schema_version") != "1.0"
        or payload.get("kind") != kind
        or payload.get("params") != render_params
        or payload.get("renderer") != "matplotlib"
        or payload.get("dpi") != 300
        or payload.get("style") != "pheno_rwe.mplstyle"
    ):
        raise RuntimeError(f"Plot spec does not match the current plan: {path}")
    columns = payload.get("columns")
    if not isinstance(columns, list) or not all(isinstance(value, str) for value in columns):
        raise RuntimeError(f"Plot spec has invalid columns: {path}")
    try:
        with data.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), [])
    except (OSError, UnicodeError, csv.Error) as exc:
        raise RuntimeError(f"Could not validate plot data: {exc}") from exc
    if header != columns:
        raise RuntimeError(f"Plot data columns do not match spec.json: {path.parent}")


def _select_plots(
    workspace: StudyWorkspace,
    plan: Plan,
    entries: list[dict[str, Any]],
    *,
    result_tables: Mapping[str, frozenset[Path]],
    report_paths: frozenset[Path],
) -> list[Path]:
    candidates = {
        path for path in (workspace.root / "plots").glob("*/*") if path.name in _SAFE_PLOT_FILES
    }
    authorized: set[Path] = set()
    for spec in plan.plots:
        if not spec.enabled:
            continue
        output_dir = workspace.root / "plots" / spec.id
        local_candidates = {path for path in candidates if path.parent == output_dir}
        if not local_candidates:
            continue
        expected_relatives = {f"plots/{spec.id}/{name}" for name in _SAFE_PLOT_FILES}
        expected_paths = {workspace.root / relative for relative in expected_relatives}
        missing = sorted(
            str(path.relative_to(workspace.root)) for path in expected_paths if not path.is_file()
        )
        if missing:
            raise RuntimeError(
                f"Plot '{spec.id}' is incomplete; missing artifact(s): " + ", ".join(missing)
            )
        params = spec.params.model_dump(mode="json")
        source = _expected_plot_source(
            workspace,
            spec,
            params,
            result_tables,
            report_paths,
        )
        source_hash = hash_file(source)
        attempt = _latest_plot_entry(entries, spec.id)
        if attempt is None:
            raise RuntimeError(f"Refusing to export plot without a registered run: {spec.id}")
        entry_index, entry = attempt
        entry_params = _entry_params(entry)
        if (
            entry.get("status") != "success"
            or entry_params.get("kind") != spec.kind
            or entry.get("input_signature")
            != hash_json({"kind": spec.kind, "params": params, "csv": source_hash})
        ):
            raise RuntimeError(f"Refusing to export stale or ad-hoc plot: {spec.id}")
        manifest_source, manifest_source_hash = _manifest_plot_input(workspace, entry)
        if manifest_source != source.resolve() or manifest_source_hash != source_hash:
            raise RuntimeError(f"Plot '{spec.id}' source lineage does not match the current plan")
        outputs = entry.get("outputs")
        if not isinstance(outputs, Mapping) or set(outputs) != expected_relatives:
            raise RuntimeError(f"Plot manifest outputs are incomplete or unexpected: {spec.id}")
        for relative in sorted(expected_relatives):
            _verify_output_declaration(
                workspace,
                entries,
                relative,
                producer_step="plot",
                statuses=frozenset({"success"}),
                entry_index=entry_index,
            )
        _validate_plot_spec(
            output_dir / "spec.json",
            kind=spec.kind,
            params=params,
            data=output_dir / "data.csv",
        )
        authorized.update(expected_paths)

    unexpected = sorted(str(path.relative_to(workspace.root)) for path in candidates - authorized)
    if unexpected:
        raise RuntimeError(
            "Refusing to export untracked or non-plan plot artifact(s): " + ", ".join(unexpected)
        )
    return sorted(authorized)


def _copy_selected_artifacts(
    workspace: StudyWorkspace,
    staging: Path,
    *,
    plan: Plan,
    current_plan_hash: str,
    entries: list[dict[str, Any]],
) -> None:
    reports = _select_reports(workspace, entries, current_plan_hash=current_plan_hash)
    results, result_tables = _select_results(workspace, plan, current_plan_hash, entries)
    plots = _select_plots(
        workspace,
        plan,
        entries,
        result_tables=result_tables,
        report_paths=frozenset(reports),
    )
    # Resolved code-set artifacts are intentionally omitted. The validated plan
    # already contains the canonical codings and approval state without copying
    # resolver prompts or arbitrary stale files from codesets/.
    for source in sorted([*reports, *results, *plots]):
        relative = source.relative_to(workspace.root)
        _copy_regular_file(
            source,
            staging / relative,
            workspace_root=workspace.root,
        )


def _write_shareable_deid_report(
    workspace: StudyWorkspace,
    staging: Path,
    *,
    entries: list[dict[str, Any]],
) -> None:
    source, _ = _verify_output_declaration(
        workspace,
        entries,
        "reports/deid_report.json",
        producer_step="deid",
        statuses=frozenset({"success"}),
    )
    report = _read_json_object(source, label="de-identification report")
    safe_report: dict[str, Any] = {
        "tier": report.get("tier")
        if report.get("tier") in {"baseline", "date_shift"}
        else "unknown",
        "baseline": _integer_fields(
            report.get("baseline"),
            (
                "patient_count",
                "tokenized_values",
                "age_bucket_years",
                "age_cap",
                "location_rows_removed",
            ),
        ),
        "k_anonymity": _integer_fields(
            report.get("k_anonymity"),
            ("k", "minimum_cell_size", "groups_below_k", "patient_count"),
        ),
        "identifier_audit": _integer_fields(
            report.get("identifier_audit"),
            (
                "location_rows",
                "person_source_values",
                "precise_birth_fields",
                "provider_identifiers",
                "care_site_identifiers",
                "raw_ledger_ids",
                "column_policy_violations",
            ),
        ),
    }
    date_shift = report.get("date_shift")
    if isinstance(date_shift, dict):
        safe_date_shift: dict[str, Any] = _integer_fields(
            date_shift,
            ("patient_count", "max_days"),
        )
        columns = date_shift.get("columns_shifted")
        if isinstance(columns, list):
            safe_date_shift["columns_shifted"] = [
                value
                for value in columns
                if isinstance(value, str)
                and value
                and all(character.isalnum() or character in "_." for character in value)
            ]
        safe_date_shift["shift_values_exported"] = False
        safe_report["date_shift"] = safe_date_shift
    else:
        safe_report["date_shift"] = None
    safe_report["warnings"] = []
    safe_report["column_policy"] = policy_summary()
    safe_report["shareable_provenance"] = {
        "identified_source_fingerprint_included": False,
        "deidentification_secret_fingerprint_included": False,
    }
    destination = staging / "reports" / "deid_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(safe_report) + "\n", encoding="utf-8")


def _validate_shareable_plan(path: Path) -> None:
    """Reject plot source paths that would disclose a source machine location."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read plan.json for export: {exc}") from exc
    plots = payload.get("plots", []) if isinstance(payload, dict) else []
    for plot in plots if isinstance(plots, list) else []:
        params = plot.get("params", {}) if isinstance(plot, dict) else {}
        source_csv = params.get("source_csv") if isinstance(params, dict) else None
        if not source_csv:
            continue
        value = str(source_csv)
        candidate = Path(value)
        if (
            candidate.is_absolute()
            or PureWindowsPath(value).is_absolute()
            or ".." in candidate.parts
            or value.startswith("~")
        ):
            raise RuntimeError(
                "Refusing to export plan.json with a non-study-relative plot source_csv; "
                "use a study-relative artifact path and revalidate the plan."
            )


def _publish_staged_bundle(staging: Path, target: Path, *, study_id: str) -> None:
    """Publish atomically, replacing only a recognized bundle for this study."""
    if not target.exists():
        os.replace(staging, target)
        return
    if target.is_symlink() or not target.is_dir():
        raise FileExistsError(f"Export target is not a directory: {target}")
    contents = list(target.iterdir())
    if contents:
        metadata_path = target / "bundle.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FileExistsError(
                f"Refusing to replace unrecognized non-empty export target: {target}"
            ) from exc
        if (
            not isinstance(metadata, dict)
            or metadata.get("study_id") != study_id
            or metadata.get("identified_data_included") is not False
        ):
            raise FileExistsError(
                f"Refusing to replace an export target belonging to another study: {target}"
            )
    backup = target.with_name(f".{target.name}.previous-{uuid.uuid4().hex}")
    os.replace(target, backup)
    try:
        os.replace(staging, target)
    except Exception:
        os.replace(backup, target)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def _payload_inventory(staging: Path) -> dict[str, str]:
    return {
        str(path.relative_to(staging)): hash_file(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path.name not in {"bundle.json", "manifest.jsonl"}
    }


def export_study(
    study: StudyWorkspace | str | Path,
    *,
    output: str | Path | None = None,
    include_database: bool = True,
) -> StepResult:
    workspace = study if isinstance(study, StudyWorkspace) else find_study(study)
    from pheno_rwe.steps.deid import deid_freshness

    started = datetime.now(UTC)
    outputs: dict[str, str] = {}
    staging: Path | None = None
    target: Path | None = None
    study_id = ""
    try:
        with study_lock(workspace):
            if not workspace.deidentified_db.exists():
                raise FileNotFoundError("deid.duckdb is required; run the deid step first.")
            plan = load_plan(workspace.plan_path)
            current_plan_hash = plan_hash(plan)
            _validate_shareable_plan(workspace.plan_path)
            _validate_plan_plot_sources(plan)
            fresh, reason = deid_freshness(workspace)
            if not fresh:
                raise RuntimeError(f"Refusing to export a stale de-identified database: {reason}.")
            source_entries = read_manifest(workspace.manifest_path)
            source_verification = verify_manifest(workspace.manifest_path)
            if not source_verification.valid:
                detail = "; ".join(source_verification.errors)
                raise RuntimeError(f"Refusing to export an invalid source manifest: {detail}")
            if last_validated_plan_hash(workspace) != current_plan_hash:
                raise RuntimeError(
                    "Refusing to export a plan that does not match the latest successful "
                    "validation; run validate first."
                )

            metadata = workspace.read_metadata()
            study_id = str(metadata["study_id"])
            target = Path(
                output or workspace.root / "exports" / safe_identifier(study_id)
            ).resolve()
            if target == workspace.root.resolve():
                raise ValueError("The export target cannot be the study root")
            protected_roots = [
                workspace.root / name
                for name in (
                    "raw",
                    "enriched",
                    "omop",
                    "reports",
                    "results",
                    "plots",
                    "codesets",
                )
            ]
            if any(target.is_relative_to(path.resolve()) for path in protected_roots):
                raise ValueError("The export target cannot be inside a source artifact directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(
                tempfile.mkdtemp(
                    prefix=f".{safe_identifier(target.name)}.staging-",
                    dir=target.parent,
                )
            )
            deid_digest = hash_file(workspace.deidentified_db)
            parquet_dir = staging / "parquet"
            parquet_dir.mkdir()
            connection = connect_database(workspace.deidentified_db, read_only=True)
            try:
                require_deidentified_policy(connection)
                seen_filenames: set[str] = set()
                for table, table_rule in DEIDENTIFIED_COLUMN_POLICY.items():
                    filename = safe_identifier(table.replace(".", "__")) + ".parquet"
                    if filename in seen_filenames:
                        raise RuntimeError(f"Parquet filename collision for table {table!r}")
                    seen_filenames.add(filename)
                    destination = parquet_dir / filename
                    quoted_table = ".".join(
                        f'"{part.replace(chr(34), chr(34) * 2)}"' for part in table.split(".")
                    )
                    quoted_columns = ", ".join(
                        f'"{column.replace(chr(34), chr(34) * 2)}"' for column in table_rule.columns
                    )
                    connection.execute(
                        f"COPY (SELECT {quoted_columns} FROM {quoted_table}) "
                        "TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
                        [str(destination)],
                    )
            finally:
                connection.close()

            for relative in ("study.json", "plan.json"):
                source = workspace.root / relative
                if source.exists():
                    _copy_regular_file(
                        source,
                        staging / relative,
                        workspace_root=workspace.root,
                    )
            _copy_selected_artifacts(
                workspace,
                staging,
                plan=plan,
                current_plan_hash=current_plan_hash,
                entries=source_entries,
            )
            _write_shareable_deid_report(workspace, staging, entries=source_entries)
            if include_database:
                _copy_regular_file(
                    workspace.deidentified_db,
                    staging / "deid.duckdb",
                    workspace_root=workspace.root,
                )

            payload_files = _payload_inventory(staging)
            finished = datetime.now(UTC)
            bundle_manifest: dict[str, Any] = {
                "schema_version": "2.0",
                "study_id": study_id,
                "created_at": finished.isoformat(),
                "identified_data_included": False,
                "manifest": {
                    "path": "manifest.jsonl",
                    "profile": "shareable-v1",
                    "source_chain_fingerprint_included": False,
                },
                "privacy": {
                    "identified_paths_included": False,
                    "identified_artifact_hashes_included": False,
                    "identified_content_hashes_included": False,
                    "study_keyed_content_fingerprints_included": True,
                    "patient_or_document_item_ledgers_included": False,
                    "source_queries_included": False,
                    "unsafe_free_text_in_database_included": False,
                    "original_invalid_temporal_values_included": False,
                },
                "database_policy": policy_summary(),
                "files": payload_files,
            }
            bundle_path = staging / "bundle.json"
            bundle_path.write_text(
                canonical_json(bundle_manifest) + "\n",
                encoding="utf-8",
            )
            shareable_entries = build_shareable_manifest(
                source_entries,
                payload_files=payload_files,
                bundle_sha256=hash_file(bundle_path),
                started_at=started,
                finished_at=finished,
                include_database=include_database,
            )
            write_manifest(staging / "manifest.jsonl", shareable_entries)
            outputs = {
                str(path.relative_to(staging)): hash_file(path)
                for path in sorted(staging.rglob("*"))
                if path.is_file()
            }
            _publish_staged_bundle(staging, target, study_id=study_id)
            record_step(
                workspace.manifest_path,
                step="export",
                status="success",
                started_at=started,
                params={"output": str(target), "include_database": include_database},
                input_signature=hash_json({"deid": deid_digest}),
                inputs={workspace.relative(workspace.deidentified_db): deid_digest},
                outputs={str(target): hash_directory(target)},
            )
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
    if target is None:
        raise RuntimeError("Export did not resolve an output target")
    return StepResult(
        "export",
        "success",
        f"Exported de-identified study bundle to {target}.",
        outputs,
    )
