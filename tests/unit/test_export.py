from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from pheno_rwe.hashing import canonical_json, hash_file, hash_json
from pheno_rwe.manifest import read_manifest, record_step
from pheno_rwe.omop.ddl import create_schema
from pheno_rwe.omop.loader import connect_database
from pheno_rwe.plan import load_plan, plan_hash
from pheno_rwe.steps.deid import deidentify
from pheno_rwe.steps.export import export_study
from pheno_rwe.steps.plot import plot_study
from pheno_rwe.steps.resolve_cohort import resolve_cohort
from pheno_rwe.steps.review_codes import review_code_set
from pheno_rwe.steps.trace import trace_study
from pheno_rwe.steps.validate import validate_study
from pheno_rwe.workspace import create_study


def _plan(*, source_csv: str | None = None) -> dict:
    plots = []
    if source_csv is not None:
        plots = [
            {
                "id": "attrition",
                "kind": "attrition",
                "params": {"cohort": "All", "source_csv": source_csv},
            }
        ]
    return {
        "schema_version": "1.0",
        "study": {"name": "Shareable study", "question": "A safe aggregate question"},
        "code_sets": [
            {
                "name": "condition",
                "domain": "condition",
                "codings": [
                    {
                        "system": "https://example.test",
                        "code": "safe",
                        "concept_id": 123,
                        "mapping_status": "MAPPED",
                    }
                ],
            }
        ],
        "cohorts": [{"name": "All"}],
        "index_date_rule": {"strategy": "fixed_date", "fixed_date": "2024-01-01"},
        "deid": {"tier": "baseline"},
        "analyses": [
            {
                "id": "baseline",
                "kind": "table_one",
                "params": {"cohort": "All", "variables": []},
            }
        ],
        "plots": plots,
    }


def _empty_identified_database(path) -> None:
    connection = connect_database(path)
    create_schema(connection)
    connection.execute(
        """INSERT INTO omop.person
           (person_id, gender_concept_id, year_of_birth, race_concept_id,
            ethnicity_concept_id, person_source_value)
           VALUES (1, 0, 1970, 0, 0, 'identified-source-patient')"""
    )
    connection.close()


def _review_codes(workspace) -> None:
    artifact = {
        "schema_version": "1.0",
        "name": "condition",
        "description": "synthetic condition",
        "domain": "condition",
        "codings": [
            {
                "system": "https://example.test",
                "code": "safe",
                "display": None,
                "source_value": "https://example.test#safe",
                "concept_id": 123,
                "mapping_status": "MAPPED",
                "accepted": True,
            }
        ],
        "mapping_status_counts": {"MAPPED": 1},
        "approval": {
            "status": "pending",
            "reviewed_by": None,
            "reviewed_at": None,
            "notes": None,
            "review_hash": None,
        },
    }
    target = workspace.root / "codesets" / "condition.codeset.json"
    target.write_text(canonical_json(artifact) + "\n", encoding="utf-8")
    review_code_set(
        workspace,
        name="condition",
        decision="approved",
        reviewed_by="Synthetic researcher",
        reviewed_at="2026-08-12T12:00:00Z",
    )


def _workspace_ready_for_export(
    tmp_path,
    *,
    source_csv: str | None = None,
    plan_payload: dict | None = None,
):
    workspace = create_study(tmp_path / "study", "Export")
    workspace.plan_path.write_text(
        canonical_json(plan_payload or _plan(source_csv=source_csv)) + "\n",
        encoding="utf-8",
    )
    _review_codes(workspace)
    _empty_identified_database(workspace.identified_db)
    resolve_cohort(workspace)
    deidentify(workspace, salt=b"export-test-salt")
    validation = validate_study(workspace, include_data=False)
    assert validation.status == "valid_static"
    return workspace


def _analysis_plot_plan() -> dict:
    payload = _plan()
    payload["plots"] = [
        {
            "id": "forest",
            "kind": "forest",
            "params": {
                "analysis_id": "baseline",
                "source_csv": "results/baseline/data.csv",
            },
        }
    ]
    return payload


def _record_result(workspace, *, value: str = "1", step: str = "analyze"):
    result_dir = workspace.root / "results" / "baseline"
    result_dir.mkdir(exist_ok=True)
    data = result_dir / "data.csv"
    data.write_text(
        f"label,estimate,ci_low,ci_high\nSynthetic,{value},0.5,1.5\n",
        encoding="utf-8",
    )
    result_payload = {
        "schema_version": "1.0",
        "analysis_id": "baseline",
        "kind": "table_one",
        "status": "success",
        "n": {"total": 1},
        "estimates": [],
        "provenance": {"available": True},
        "assumptions_checked": [],
        "guardrail_outcomes": [],
        "power_note": "Synthetic test result.",
        "seed": 2025,
        "package_versions": {},
        "output_tables": [
            {
                "name": "data",
                "path": "data.csv",
                "format": "text/csv",
                "rows": 1,
                "columns": ["label", "estimate", "ci_low", "ci_high"],
            }
        ],
        "metadata": {},
    }
    result_json = result_dir / "result.json"
    result_json.write_text(canonical_json(result_payload) + "\n", encoding="utf-8")
    outputs = {
        workspace.relative(data): hash_file(data),
        workspace.relative(result_json): hash_file(result_json),
    }
    record_step(
        workspace.manifest_path,
        step=step,
        status="success",
        started_at=datetime.now(UTC),
        params={
            "analysis_id": "baseline",
            "kind": "table_one",
            "plan_hash": plan_hash(load_plan(workspace.plan_path)),
        },
        outputs=outputs,
    )
    return result_json, data


def _record_plot(workspace):
    source = workspace.root / "results" / "baseline" / "data.csv"
    plot_dir = workspace.root / "plots" / "forest"
    plot_dir.mkdir(exist_ok=True)
    (plot_dir / "data.csv").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    (plot_dir / "plot.png").write_bytes(b"synthetic png")
    (plot_dir / "plot.svg").write_text("<svg><text>Synthetic</text></svg>\n", encoding="utf-8")
    plan = load_plan(workspace.plan_path)
    spec = next(item for item in plan.plots if item.id == "forest")
    params = spec.params.model_dump(mode="json")
    plot_spec = {
        "schema_version": "1.0",
        "kind": "forest",
        "params": {},
        "columns": ["label", "estimate", "ci_low", "ci_high"],
        "renderer": "matplotlib",
        "dpi": 300,
        "style": "pheno_rwe.mplstyle",
    }
    (plot_dir / "spec.json").write_text(canonical_json(plot_spec) + "\n", encoding="utf-8")
    outputs = {
        workspace.relative(plot_dir / name): hash_file(plot_dir / name)
        for name in ("data.csv", "plot.png", "plot.svg", "spec.json")
    }
    source_hash = hash_file(source)
    record_step(
        workspace.manifest_path,
        step="plot",
        status="success",
        started_at=datetime.now(UTC),
        params={"plot_id": "forest", "kind": "forest"},
        input_signature=hash_json({"kind": "forest", "params": params, "csv": source_hash}),
        inputs={str(source): source_hash},
        outputs=outputs,
    )
    return plot_dir


def test_export_rebuilds_raw_free_manifest_and_remains_verifiable(tmp_path) -> None:
    workspace = _workspace_ready_for_export(tmp_path)
    identified_source = tmp_path / "identified" / "patient-raw-123.ndjson"
    identified_source.parent.mkdir()
    identified_source.write_text('{"id":"patient-raw-123"}\n', encoding="utf-8")
    raw_output = workspace.root / "raw" / "patients" / "patient-raw-123.bundle.json"
    raw_output.write_text('{"document":"document-raw-456"}\n', encoding="utf-8")
    record_step(
        workspace.manifest_path,
        step="pull",
        status="success",
        started_at=datetime.now(UTC),
        params={
            "text": "secret FHIR query",
            "provider_id": "private-provider",
            "approved": True,
        },
        input_signature="identified-input-signature",
        inputs={str(identified_source): hash_file(identified_source)},
        outputs={workspace.relative(raw_output): hash_file(raw_output)},
        items=[
            {
                "patient_token": "patient-raw-123",
                "document_token": "document-raw-456",
                "status": "success",
            }
        ],
    )
    (workspace.root / "reports" / "ingest_orphans.json").write_text(
        '{"orphans":[{"resource_id":"raw-orphan-789"}]}\n',
        encoding="utf-8",
    )
    result_dir = workspace.root / "results" / "baseline"
    result_dir.mkdir()
    (result_dir / "debug.txt").write_text("private debug output", encoding="utf-8")

    local_deid_report = json.loads(
        (workspace.root / "reports" / "deid_report.json").read_text(encoding="utf-8")
    )
    identified_database_hash = local_deid_report["source_database_sha256"]
    source_chain_hash = read_manifest(workspace.manifest_path)[-1]["entry_hash"]
    shared = tmp_path / "shared"

    result = export_study(workspace, output=shared)

    assert result.status == "success"
    assert not (shared / "reports" / "ingest_orphans.json").exists()
    assert not (shared / "results" / "baseline" / "debug.txt").exists()
    bundle = json.loads((shared / "bundle.json").read_text(encoding="utf-8"))
    assert bundle["schema_version"] == "2.0"
    assert bundle["manifest"]["profile"] == "shareable-v1"
    assert bundle["identified_data_included"] is False
    assert "manifest.jsonl" not in bundle["files"]

    exported_report = json.loads(
        (shared / "reports" / "deid_report.json").read_text(encoding="utf-8")
    )
    assert "source_database_sha256" not in exported_report
    assert "salt_sha256" not in exported_report
    assert "deid_config_sha256" not in exported_report

    manifest_text = (shared / "manifest.jsonl").read_text(encoding="utf-8")
    for secret in (
        str(identified_source),
        "patient-raw-123",
        "document-raw-456",
        "raw-orphan-789",
        "secret FHIR query",
        "private-provider",
        "identified-input-signature",
        identified_database_hash,
        source_chain_hash,
    ):
        assert secret not in manifest_text
    exported_entries = read_manifest(shared / "manifest.jsonl")
    assert all(not entry["inputs"] and not entry["outputs"] for entry in exported_entries[:-1])
    assert exported_entries[-1]["params"]["manifest_profile"] == "shareable-v1"
    assert exported_entries[-1]["inputs"] == bundle["files"]
    assert exported_entries[-1]["outputs"] == {"bundle.json": hash_file(shared / "bundle.json")}

    verification = trace_study(shared, verify=True).verification
    assert verification is not None and verification.valid
    (shared / "plan.json").write_text("{}\n", encoding="utf-8")
    verification = trace_study(shared, verify=True).verification
    assert verification is not None and not verification.valid
    assert any("input hash does not match: plan.json" in error for error in verification.errors)


def test_export_refuses_source_machine_paths_in_plan(tmp_path) -> None:
    workspace = _workspace_ready_for_export(
        tmp_path,
        source_csv="/Users/researcher/identified/patient-level.csv",
    )

    with pytest.raises(RuntimeError, match="non-study-relative plot source_csv"):
        export_study(workspace, output=tmp_path / "shared")
    assert not (tmp_path / "shared").exists()


def test_export_requires_the_current_plan_to_be_validated(tmp_path) -> None:
    workspace = _workspace_ready_for_export(tmp_path)
    payload = json.loads(workspace.plan_path.read_text(encoding="utf-8"))
    payload["study"]["question"] = "An amended but not validated question"
    workspace.plan_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="latest successful validation"):
        export_study(workspace, output=tmp_path / "shared")
    assert not (tmp_path / "shared").exists()


@pytest.mark.parametrize("suffix", ["csv", "json"])
def test_export_refuses_untracked_result_artifacts(tmp_path, suffix: str) -> None:
    workspace = _workspace_ready_for_export(tmp_path)
    rogue = workspace.root / "results" / "rogue"
    rogue.mkdir()
    (rogue / f"patient-level.{suffix}").write_text("PATIENT-CANARY\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="untracked or non-plan result"):
        export_study(workspace, output=tmp_path / "shared")
    assert not (tmp_path / "shared").exists()


def test_later_plot_input_does_not_invalidate_a_produced_result(tmp_path) -> None:
    workspace = _workspace_ready_for_export(tmp_path)
    _, data = _record_result(workspace)
    record_step(
        workspace.manifest_path,
        step="plot",
        status="success",
        started_at=datetime.now(UTC),
        params={"plot_id": "not-persisted", "kind": "forest"},
        inputs={str(data): hash_file(data)},
    )

    shared = tmp_path / "shared"
    export_study(workspace, output=shared)

    assert (shared / "results" / "baseline" / "data.csv").read_text(
        encoding="utf-8"
    ) == data.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["result.json", "data.csv"])
def test_export_refuses_tampered_analysis_outputs(tmp_path, name: str) -> None:
    workspace = _workspace_ready_for_export(tmp_path)
    result_json, data = _record_result(workspace)
    target = result_json if name == "result.json" else data
    target.write_text(target.read_text(encoding="utf-8") + " \n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Analysis CSV row count|producer hash mismatch"):
        export_study(workspace, output=tmp_path / "shared")
    assert not (tmp_path / "shared").exists()


def test_export_omits_all_code_set_artifacts(tmp_path) -> None:
    workspace = _workspace_ready_for_export(tmp_path)
    (workspace.root / "codesets" / "rogue.codeset.json").write_text(
        '{"patient":"CODESET-PATIENT-CANARY"}\n',
        encoding="utf-8",
    )
    shared = tmp_path / "shared"

    export_study(workspace, output=shared)

    bundle = json.loads((shared / "bundle.json").read_text(encoding="utf-8"))
    assert not any(relative.startswith("codesets/") for relative in bundle["files"])
    assert not (shared / "codesets").exists()


@pytest.mark.parametrize("relative", ["reports/deid_report.json", "reports/validation_report.json"])
def test_export_refuses_tampered_source_reports(tmp_path, relative: str) -> None:
    workspace = _workspace_ready_for_export(tmp_path)
    report = workspace.root / relative
    report.write_text(report.read_text(encoding="utf-8") + " \n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="producer hash mismatch"):
        export_study(workspace, output=tmp_path / "shared")
    assert not (tmp_path / "shared").exists()


def test_export_refuses_unproduced_mapping_coverage_csv(tmp_path) -> None:
    workspace = _workspace_ready_for_export(tmp_path)
    (workspace.root / "reports" / "mapping_coverage.csv").write_text(
        "domain,mapping_status,count\ncondition,MAPPED,1\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="without an engine producer"):
        export_study(workspace, output=tmp_path / "shared")
    assert not (tmp_path / "shared").exists()


@pytest.mark.parametrize("name", ["data.csv", "plot.png", "plot.svg", "spec.json"])
def test_export_refuses_tampered_plot_artifacts(tmp_path, name: str) -> None:
    workspace = _workspace_ready_for_export(tmp_path, plan_payload=_analysis_plot_plan())
    _record_result(workspace)
    plot_dir = _record_plot(workspace)
    artifact = plot_dir / name
    artifact.write_bytes(artifact.read_bytes() + b"PLOT-PATIENT-CANARY")

    with pytest.raises(RuntimeError, match="producer hash mismatch"):
        export_study(workspace, output=tmp_path / "shared")
    assert not (tmp_path / "shared").exists()


def test_export_refuses_plot_stale_after_analysis_rerun(tmp_path) -> None:
    workspace = _workspace_ready_for_export(tmp_path, plan_payload=_analysis_plot_plan())
    _record_result(workspace, value="1")
    _record_plot(workspace)
    _record_result(workspace, value="2", step="analysis_rerun")

    with pytest.raises(RuntimeError, match="stale or ad-hoc plot|source lineage"):
        export_study(workspace, output=tmp_path / "shared")
    assert not (tmp_path / "shared").exists()


def test_export_excludes_external_from_csv_replay(tmp_path) -> None:
    workspace = _workspace_ready_for_export(tmp_path)
    external = tmp_path / "identified-patient.csv"
    external.write_text("stage,n\nPATIENT-CANARY,1\n", encoding="utf-8")
    rendered = plot_study(
        workspace,
        plot_id="external-replay",
        from_csv=external,
        kind="attrition",
    )
    assert rendered.status == "success"

    shared = tmp_path / "shared"
    export_study(workspace, output=shared)

    bundle = json.loads((shared / "bundle.json").read_text(encoding="utf-8"))
    assert not any(path.startswith("replays/") for path in bundle["files"])
    assert not (shared / "replays").exists()
    assert b"PATIENT-CANARY" not in b"".join(
        path.read_bytes() for path in shared.rglob("*") if path.is_file()
    )


def test_export_refuses_study_relative_raw_plot_source(tmp_path) -> None:
    workspace = _workspace_ready_for_export(tmp_path, source_csv="raw/patient-level.csv")

    with pytest.raises(RuntimeError, match="not an approved generated report artifact"):
        export_study(workspace, output=tmp_path / "shared")
    assert not (tmp_path / "shared").exists()
