from __future__ import annotations

import json
from typing import Any

from typer.testing import CliRunner

from pheno_rwe.cli import app
from pheno_rwe.hashing import canonical_json
from pheno_rwe.manifest import read_manifest
from pheno_rwe.plan import load_plan, write_plan
from pheno_rwe.steps.common import StepResult
from pheno_rwe.steps.pull import PullPreview
from pheno_rwe.steps.resolve_codes import resolve_code_set
from pheno_rwe.workspace import create_study


def test_global_options_work_before_or_after_subcommand(tmp_path) -> None:
    workspace = create_study(tmp_path / "study", "CLI options")
    runner = CliRunner()

    before = runner.invoke(
        app,
        ["--study", str(workspace.root), "--json", "status"],
    )
    after = runner.invoke(
        app,
        ["status", "--study", str(workspace.root), "--json"],
    )

    assert before.exit_code == 0, before.output
    assert after.exit_code == 0, after.output
    assert json.loads(before.output) == json.loads(after.output)


def test_global_value_option_supports_equals_form_after_subcommand(tmp_path) -> None:
    workspace = create_study(tmp_path / "study", "CLI equals option")
    result = CliRunner().invoke(
        app,
        ["status", f"--study={workspace.root}", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["study"]["study_id"] == workspace.read_metadata()["study_id"]


def _stub_pull_dependencies(monkeypatch, *, preview: PullPreview) -> list[dict[str, Any]]:
    executions: list[dict[str, Any]] = []
    transport = object()
    monkeypatch.setattr("pheno_rwe.client.build_transport", lambda *args, **kwargs: transport)
    monkeypatch.setattr(
        "pheno_rwe.steps.pull.preview_live_cohort",
        lambda received, text, provider, *, queries_only=False: preview,
    )

    def execute(study, received, **kwargs):
        assert received is transport
        executions.append(kwargs)
        return StepResult(
            step="pull",
            status="success",
            message="Fetched one synthetic bundle.",
        )

    monkeypatch.setattr("pheno_rwe.steps.pull.pull_live_cohort", execute)
    return executions


def test_pull_json_yes_executes_after_preview(tmp_path, monkeypatch) -> None:
    workspace = create_study(tmp_path / "study", "JSON pull")
    preview = PullPreview(
        patient_ids=["synthetic-patient"],
        queries=[{"resourceType": "Condition", "code": "synthetic"}],
    )
    executions = _stub_pull_dependencies(monkeypatch, preview=preview)

    result = CliRunner().invoke(
        app,
        [
            "pull",
            "--study",
            str(workspace.root),
            "--json",
            "--cohort",
            "synthetic cohort",
            "--provider",
            "synthetic-provider",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "success"
    assert len(executions) == 1
    assert executions[0]["approved"] is True
    assert executions[0]["preview"] is preview


def test_pull_queries_only_is_the_explicit_preview_path(tmp_path, monkeypatch) -> None:
    workspace = create_study(tmp_path / "study", "Query preview")
    preview = PullPreview(
        patient_ids=["must-not-be-returned"],
        queries=[{"resourceType": "Condition", "code": "synthetic"}],
    )
    executions = _stub_pull_dependencies(monkeypatch, preview=preview)

    result = CliRunner().invoke(
        app,
        [
            "pull",
            "--study",
            str(workspace.root),
            "--json",
            "--cohort",
            "synthetic cohort",
            "--provider",
            "synthetic-provider",
            "--queries-only",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "queries": preview.queries,
        "patient_count": 1,
        "patient_ids": [],
    }
    assert executions == []


def test_pull_json_without_approval_fails_instead_of_returning_preview(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = create_study(tmp_path / "study", "Unapproved JSON pull")
    preview = PullPreview(patient_ids=["synthetic-patient"], queries=[])
    executions = _stub_pull_dependencies(monkeypatch, preview=preview)

    result = CliRunner().invoke(
        app,
        [
            "pull",
            "--study",
            str(workspace.root),
            "--json",
            "--cohort",
            "synthetic cohort",
            "--provider",
            "synthetic-provider",
        ],
    )

    assert result.exit_code == 1
    assert "requires --yes" in json.loads(result.output)["message"]
    assert executions == []


def test_review_codes_records_decision_counts_and_controls_static_exit_3(
    tmp_path,
    fake_client,
) -> None:
    workspace = create_study(tmp_path / "study", "Code review")
    resolve_code_set(
        workspace,
        fake_client,
        name="diabetes",
        text="type 2 diabetes",
        domain="condition",
    )
    artifact_path = workspace.root / "codesets" / "diabetes.codeset.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    coding = artifact["codings"][0]
    write_plan(
        workspace.plan_path,
        {
            "study": {"name": "Code review", "question": "Who is in the cohort?"},
            "code_sets": [
                {
                    "name": "diabetes",
                    "domain": "condition",
                    "codings": [
                        {
                            "system": coding["system"],
                            "code": coding["code"],
                            "display": coding["display"],
                            "concept_id": coding["concept_id"],
                            "mapping_status": coding["mapping_status"],
                            "accepted": coding["accepted"],
                        }
                    ],
                }
            ],
            "cohorts": [{"name": "all"}],
            "index_date_rule": {"strategy": "fixed_date", "fixed_date": "2024-01-01"},
            "analyses": [{"id": "baseline", "kind": "table_one", "params": {"cohort": "all"}}],
        },
    )
    runner = CliRunner()

    pending = runner.invoke(
        app,
        ["validate", "--static-only", "--study", str(workspace.root), "--json"],
    )
    assert pending.exit_code == 3, pending.output
    assert any(
        item["rule_id"] == "CODESET-APPROVAL"
        for item in json.loads(pending.output)["report"]["outcomes"]
    )

    rejected = runner.invoke(
        app,
        [
            "review-codes",
            "--name",
            "diabetes",
            "--decision",
            "rejected",
            "--reviewed-by",
            "Synthetic researcher",
            "--reviewed-at",
            "2026-08-12T12:00:00Z",
            "--study",
            str(workspace.root),
            "--json",
        ],
    )
    assert rejected.exit_code == 0, rejected.output
    rejected_payload = json.loads(rejected.output)
    assert rejected_payload["items"][0]["mapping_status_counts"] == {
        "ALREADY_STANDARD": 0,
        "MAPPED": 1,
        "UNCHECKED": 0,
        "UNMAPPED": 0,
    }
    refused = runner.invoke(
        app,
        ["validate", "--static-only", "--study", str(workspace.root), "--json"],
    )
    assert refused.exit_code == 3, refused.output

    approved = runner.invoke(
        app,
        [
            "review-codes",
            "--name",
            "diabetes",
            "--decision",
            "approved",
            "--reviewed-by",
            "Synthetic researcher",
            "--reviewed-at",
            "2026-08-12T12:05:00Z",
            "--study",
            str(workspace.root),
            "--json",
        ],
    )
    assert approved.exit_code == 0, approved.output
    assert load_plan(workspace.plan_path).code_sets[0].approval.status == "approved"
    assert load_plan(workspace.plan_path).code_sets[0].approval.review_hash
    reviewed_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert reviewed_artifact["approval"]["status"] == "approved"
    assert reviewed_artifact["approval"]["review_hash"]
    passed = runner.invoke(
        app,
        ["validate", "--static-only", "--study", str(workspace.root), "--json"],
    )
    assert passed.exit_code == 0, passed.output
    assert [
        entry["params"]["decision"]
        for entry in read_manifest(workspace.manifest_path)
        if entry["step"] == "review-codes"
    ] == ["rejected", "approved"]

    reviewed_artifact["codings"][0]["display"] = "Edited without a new review"
    artifact_path.write_text(canonical_json(reviewed_artifact) + "\n", encoding="utf-8")
    stale_evidence = runner.invoke(
        app,
        ["validate", "--static-only", "--study", str(workspace.root), "--json"],
    )
    assert stale_evidence.exit_code == 3, stale_evidence.output
    refusal = json.loads(stale_evidence.output)["report"]["outcomes"][0]
    assert refusal["details"]["review_evidence_valid"] is False
