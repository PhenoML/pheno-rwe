from __future__ import annotations

import json
from typing import Any

from typer.testing import CliRunner

from pheno_rwe import __version__
from pheno_rwe.cli import app
from pheno_rwe.hashing import canonical_json
from pheno_rwe.manifest import read_manifest
from pheno_rwe.plan import load_plan, write_plan
from pheno_rwe.steps.common import StepResult
from pheno_rwe.steps.pull import PullPreview
from pheno_rwe.steps.resolve_codes import resolve_code_set
from pheno_rwe.workspace import create_study


def test_version_flag_prints_version_standalone() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == __version__


def test_version_flag_short_circuits_before_subcommand() -> None:
    result = CliRunner().invoke(app, ["status", "--version"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == __version__


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


class DiscoveryCliTransport:
    """Offline transport backing the discovery commands and explicit-set pulls."""

    def resolve_codings(self, text: str, domain: str | None = None) -> dict[str, Any]:
        del text, domain
        return {
            "codings": [{"system": "http://snomed.info/sct", "code": "44054006", "display": "T2DM"}]
        }

    def crosswalk(self, system: str, code: str, targets: Any) -> dict[str, Any]:
        del system, code
        return {
            "targets": [
                {"system": target, "matches": [{"code": "rx", "display": "mapped", "cui": "C0"}]}
                for target in targets
            ]
        }

    def fhir_search(self, provider: str, fhir_path: str, **params: Any) -> dict[str, Any]:
        del provider
        if fhir_path.endswith("/$everything"):
            patient_id = fhir_path.split("/")[1]
            return {
                "resourceType": "Bundle",
                "type": "searchset",
                "entry": [
                    {
                        "resource": {
                            "resourceType": "Patient",
                            "id": patient_id,
                            "text": {"status": "generated", "div": "private narrative"},
                        }
                    }
                ],
            }
        if params.get("_summary") == "count":
            return {"resourceType": "Bundle", "total": 5}
        return {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "MedicationRequest",
                        "id": "m1",
                        "subject": {"reference": "Patient/P1"},
                    }
                },
                {
                    "resource": {
                        "resourceType": "MedicationRequest",
                        "id": "m2",
                        "subject": {"reference": "Patient/P2"},
                    }
                },
            ],
        }


def _discovery_runner(monkeypatch, transport: DiscoveryCliTransport) -> CliRunner:
    monkeypatch.setattr("pheno_rwe.client.build_transport", lambda *args, **kwargs: transport)
    return CliRunner()


def test_extract_codes_command_prints_codings(tmp_path, monkeypatch) -> None:
    workspace = create_study(tmp_path / "study", "Extract")
    runner = _discovery_runner(monkeypatch, DiscoveryCliTransport())

    result = runner.invoke(
        app,
        ["extract-codes", "GLP-1", "--domain", "drug", "--study", str(workspace.root), "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["step"] == "extract-codes"
    assert payload["items"] == [
        {"system": "http://snomed.info/sct", "code": "44054006", "display": "T2DM"}
    ]


def test_crosswalk_command_prints_target_codings(tmp_path, monkeypatch) -> None:
    workspace = create_study(tmp_path / "study", "Crosswalk")
    runner = _discovery_runner(monkeypatch, DiscoveryCliTransport())

    result = runner.invoke(
        app,
        [
            "crosswalk",
            "--system",
            "http://snomed.info/sct",
            "--code",
            "44054006",
            "--to",
            "http://www.nlm.nih.gov/research/umls/rxnorm",
            "--to",
            "http://hl7.org/fhir/sid/ndc",
            "--study",
            str(workspace.root),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert [coding["system"] for coding in json.loads(result.output)["items"]] == [
        "http://www.nlm.nih.gov/research/umls/rxnorm",
        "http://hl7.org/fhir/sid/ndc",
    ]


def test_fhir_search_count_and_patient_extraction(tmp_path, monkeypatch) -> None:
    workspace = create_study(tmp_path / "study", "Search")
    runner = _discovery_runner(monkeypatch, DiscoveryCliTransport())

    count = runner.invoke(
        app,
        [
            "fhir-search",
            "MedicationRequest",
            "--param",
            "code=x",
            "--count",
            "--provider",
            "prov",
            "--study",
            str(workspace.root),
            "--json",
        ],
    )
    assert count.exit_code == 0, count.output
    assert json.loads(count.output)["items"] == [{"total": 5}]

    patients = runner.invoke(
        app,
        [
            "fhir-search",
            "MedicationRequest",
            "--param",
            "code=x",
            "--patients",
            "--provider",
            "prov",
            "--study",
            str(workspace.root),
            "--json",
        ],
    )
    assert patients.exit_code == 0, patients.output
    assert json.loads(patients.output)["items"] == [{"id": "P1"}, {"id": "P2"}]


def test_fhir_search_rejects_malformed_param(tmp_path, monkeypatch) -> None:
    workspace = create_study(tmp_path / "study", "Bad param")
    runner = _discovery_runner(monkeypatch, DiscoveryCliTransport())

    result = runner.invoke(
        app,
        [
            "fhir-search",
            "Patient",
            "--param",
            "gendermale",
            "--provider",
            "prov",
            "--study",
            str(workspace.root),
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert "expected key=value" in json.loads(result.output)["message"]


def test_pull_patients_fetches_the_listed_patients(tmp_path, monkeypatch) -> None:
    workspace = create_study(tmp_path / "study", "Explicit pull")
    ids_file = tmp_path / "cohort.txt"
    ids_file.write_text("explicit-1 explicit-2\nexplicit-1\n", encoding="utf-8")
    runner = _discovery_runner(monkeypatch, DiscoveryCliTransport())

    result = runner.invoke(
        app,
        [
            "pull",
            "--patients",
            str(ids_file),
            "--provider",
            "prov",
            "--yes",
            "--study",
            str(workspace.root),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "success"
    bundles = sorted((workspace.root / "raw" / "patients").glob("*.bundle.json"))
    # Whitespace/newline delimited, de-duplicated to two distinct patients.
    assert len(bundles) == 2
    for bundle_path in bundles:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        assert bundle["resourceType"] == "Bundle"
        patient = bundle["entry"][0]["resource"]
        assert patient["resourceType"] == "Patient"
        assert "text" not in patient  # narrative stripped exactly like the cohort path


def test_pull_requires_exactly_one_of_cohort_or_patients(tmp_path, monkeypatch) -> None:
    workspace = create_study(tmp_path / "study", "XOR")
    runner = _discovery_runner(monkeypatch, DiscoveryCliTransport())

    neither = runner.invoke(
        app,
        ["pull", "--provider", "prov", "--study", str(workspace.root), "--json"],
    )
    assert neither.exit_code == 1
    assert "exactly one" in json.loads(neither.output)["message"]

    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("p1\n", encoding="utf-8")
    both = runner.invoke(
        app,
        [
            "pull",
            "--cohort",
            "diabetics",
            "--patients",
            str(ids_file),
            "--provider",
            "prov",
            "--study",
            str(workspace.root),
            "--json",
        ],
    )
    assert both.exit_code == 1
    assert "exactly one" in json.loads(both.output)["message"]


def test_pull_queries_only_is_rejected_with_patients(tmp_path, monkeypatch) -> None:
    workspace = create_study(tmp_path / "study", "Queries-only guard")
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("p1\n", encoding="utf-8")
    runner = _discovery_runner(monkeypatch, DiscoveryCliTransport())

    result = runner.invoke(
        app,
        [
            "pull",
            "--patients",
            str(ids_file),
            "--queries-only",
            "--provider",
            "prov",
            "--study",
            str(workspace.root),
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert "only valid with --cohort" in json.loads(result.output)["message"]
