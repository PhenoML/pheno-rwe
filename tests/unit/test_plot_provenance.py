from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from pheno_rwe.cli import app
from pheno_rwe.errors import StalePlanError
from pheno_rwe.hashing import canonical_json, hash_file, hash_json
from pheno_rwe.manifest import read_manifest, record_step
from pheno_rwe.plan import Plan, analysis_data_hash, plan_hash, write_plan
from pheno_rwe.steps.plot import plot_study
from pheno_rwe.workspace import StudyWorkspace, create_study


def _plan(source_csv: str | None = "results/baseline/forest.csv") -> Plan:
    params: dict[str, object] = {"analysis_id": "baseline"}
    if source_csv is not None:
        params["source_csv"] = source_csv
    return Plan.model_validate(
        {
            "study": {"name": "Plot provenance", "question": "Synthetic question"},
            "code_sets": [
                {
                    "name": "demo",
                    "domain": "condition",
                    "codings": [
                        {
                            "system": "https://example.test",
                            "code": "demo",
                            "concept_id": 1,
                            "mapping_status": "MAPPED",
                        }
                    ],
                }
            ],
            "cohorts": [{"name": "all"}],
            "index_date_rule": {"strategy": "fixed_date", "fixed_date": "2024-01-01"},
            "analyses": [
                {
                    "id": "baseline",
                    "kind": "table_one",
                    "params": {"cohort": "all", "variables": ["age_bucket"]},
                }
            ],
            "plots": [{"id": "forest-1", "kind": "forest", "params": params}],
        }
    )


def _attrition_plan() -> Plan:
    payload = _plan().model_dump(mode="json")
    payload["plots"] = [
        {
            "id": "attrition-1",
            "kind": "attrition",
            "params": {"cohort": "all", "source_csv": "reports/cohort_attrition.csv"},
        }
    ]
    return Plan.model_validate(payload)


def _forest_csv(path: Path, *, estimate: float = 1.2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"label": ["Synthetic"], "estimate": [estimate], "lower": [0.9], "upper": [1.6]}
    ).to_csv(path, index=False)


def _current_study(tmp_path: Path, *, plan: Plan | None = None) -> tuple[StudyWorkspace, Path]:
    workspace = create_study(tmp_path / "study", "Plot provenance")
    current_plan = plan or _plan()
    write_plan(workspace.plan_path, current_plan)
    workspace.identified_db.write_bytes(b"synthetic identified database")
    workspace.deidentified_db.write_bytes(b"synthetic deidentified database")
    current_hash = plan_hash(current_plan)
    current_analysis_data_hash = analysis_data_hash(current_plan)
    identified_hash = hash_file(workspace.identified_db)
    cohort_entry = record_step(
        workspace.manifest_path,
        step="resolve-cohort",
        status="success",
        started_at=datetime.now(UTC),
        params={
            "plan_hash": current_hash,
            "analysis_data_sha256": current_analysis_data_hash,
            "identified_database_sha256": identified_hash,
        },
        inputs={"plan.json": hash_file(workspace.plan_path)},
        outputs={"omop/omop.duckdb": identified_hash},
    )
    report = {
        "source_database_sha256": identified_hash,
        "deid_config_sha256": hash_json(current_plan.deid.model_dump(mode="json")),
        "analysis_data_sha256": current_analysis_data_hash,
        "cohort_resolution_entry_hash": cohort_entry.entry_hash,
        "cohort_resolution_plan_sha256": current_hash,
    }
    (workspace.root / "reports" / "deid_report.json").write_text(
        canonical_json(report) + "\n", encoding="utf-8"
    )
    record_step(
        workspace.manifest_path,
        step="plan_validated",
        status="success",
        started_at=datetime.now(UTC),
        params={"plan_hash": current_hash},
        inputs={"plan.json": hash_file(workspace.plan_path)},
    )
    source_value = current_plan.plots[0].params.source_csv
    source = (
        Path(source_value).resolve()
        if source_value and Path(source_value).is_absolute()
        else workspace.root / str(source_value or "results/baseline/forest.csv")
    )
    _forest_csv(source)
    record_step(
        workspace.manifest_path,
        step="analyze",
        status="success",
        started_at=datetime.now(UTC),
        params={"analysis_id": "baseline", "kind": "table_one", "plan_hash": current_hash},
        inputs={"omop/deid.duckdb": hash_file(workspace.deidentified_db)},
        outputs={
            workspace.relative(source) if source.is_relative_to(workspace.root) else str(source): (
                hash_file(source)
            )
        },
    )
    return workspace, source


def test_registered_plot_refuses_unvalidated_plan_edit_with_exit_four(tmp_path) -> None:
    workspace, _ = _current_study(tmp_path)
    payload = json.loads(workspace.plan_path.read_text(encoding="utf-8"))
    payload["study"]["question"] = "Edited after validation"
    write_plan(workspace.plan_path, payload)

    with pytest.raises(StalePlanError, match="last successfully validated plan"):
        plot_study(workspace, plot_id="forest-1")

    result = CliRunner().invoke(
        app,
        ["--study", str(workspace.root), "--json", "plot", "--id", "forest-1"],
    )
    assert result.exit_code == 4
    assert json.loads(result.stdout)["exit_code"] == 4


def test_registered_plot_refuses_stale_deid_and_changed_result(tmp_path) -> None:
    stale_workspace, _ = _current_study(tmp_path / "deid")
    stale_workspace.identified_db.write_bytes(b"identified database changed")
    with pytest.raises(StalePlanError, match="deid.duckdb is stale"):
        plot_study(stale_workspace, plot_id="forest-1")

    result_workspace, source = _current_study(tmp_path / "result")
    _forest_csv(source, estimate=2.4)
    with pytest.raises(StalePlanError, match="rerun analysis"):
        plot_study(result_workspace, plot_id="forest-1")


def test_external_csv_is_replay_only_and_never_enters_registered_plots(tmp_path) -> None:
    external = tmp_path / "outside" / "external.csv"
    _forest_csv(external)
    workspace, _ = _current_study(tmp_path / "registered", plan=_plan(str(external.resolve())))

    with pytest.raises(ValueError, match="inside its analysis result directory"):
        plot_study(workspace, plot_id="forest-1")

    replay = plot_study(
        workspace,
        plot_id="external-review",
        from_csv=external,
        kind="forest",
    )
    assert replay.step == "plot_replay"
    assert replay.status == "success"
    assert not (workspace.root / "plots" / "external-review").exists()
    replay_dir = workspace.root / "replays" / "plots" / "external-review"
    spec = json.loads((replay_dir / "spec.json").read_text(encoding="utf-8"))
    assert spec["provenance"]["registration"] == "unregistered_replay"
    assert spec["provenance"]["shareable"] is False
    entry = read_manifest(workspace.manifest_path)[-1]
    assert entry["step"] == "plot_replay"
    assert entry["items"][0]["shareable"] is False


def test_valid_registered_plot_records_current_plan_deid_source_and_producer(tmp_path) -> None:
    workspace, source = _current_study(tmp_path)
    result = plot_study(workspace, plot_id="forest-1")
    assert result.status == "success"
    output_dir = workspace.root / "plots" / "forest-1"
    assert (output_dir / "spec.json").is_file()
    entry = read_manifest(workspace.manifest_path)[-1]
    assert entry["step"] == "plot"
    assert entry["status"] == "success"
    assert entry["params"]["plan_hash"] == plan_hash(_plan())
    assert entry["params"]["deid_sha256"] == hash_file(workspace.deidentified_db)
    assert entry["params"]["source_sha256"] == hash_file(source)
    assert entry["inputs"]["results/baseline/forest.csv"] == hash_file(source)


def test_valid_aggregate_plot_requires_expected_current_report_producer(tmp_path) -> None:
    plan = _attrition_plan()
    workspace, source = _current_study(tmp_path, plan=plan)
    pd.DataFrame({"stage": ["Eligible", "Included"], "n": [12, 10], "excluded": [0, 2]}).to_csv(
        source, index=False
    )
    identified_hash = hash_file(workspace.identified_db)
    cohort_entry = record_step(
        workspace.manifest_path,
        step="resolve-cohort",
        status="success",
        started_at=datetime.now(UTC),
        params={
            "plan_hash": plan_hash(plan),
            "analysis_data_sha256": analysis_data_hash(plan),
            "identified_database_sha256": identified_hash,
        },
        inputs={"plan.json": hash_file(workspace.plan_path)},
        outputs={
            "omop/omop.duckdb": identified_hash,
            "reports/cohort_attrition.csv": hash_file(source),
        },
    )
    report_path = workspace.root / "reports" / "deid_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["cohort_resolution_entry_hash"] = cohort_entry.entry_hash
    report["cohort_resolution_plan_sha256"] = plan_hash(plan)
    report_path.write_text(canonical_json(report) + "\n", encoding="utf-8")

    result = plot_study(workspace, plot_id="attrition-1")
    assert result.status == "success"
    entry = read_manifest(workspace.manifest_path)[-1]
    assert entry["step"] == "plot"
    assert entry["items"][0]["source_csv"] == "reports/cohort_attrition.csv"
    assert entry["items"][0]["producer_entry_hash"] == cohort_entry.entry_hash
