from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from pheno_rwe.hashing import hash_file
from pheno_rwe.plan import Plan, write_plan
from pheno_rwe.steps.curate_codes import curate_code_set_coding
from pheno_rwe.workspace import create_study


def _plan() -> Plan:
    return Plan.model_validate(
        {
            "study": {"name": "Curation", "question": "Synthetic question"},
            "code_sets": [
                {
                    "name": "demo",
                    "domain": "condition",
                    "codings": [
                        {
                            "system": "https://example.test",
                            "code": "one",
                            "display": "One",
                            "concept_id": 1,
                            "mapping_status": "MAPPED",
                        },
                        {
                            "system": "https://example.test",
                            "code": "two",
                            "display": "Two",
                            "concept_id": 2,
                            "mapping_status": "UNCHECKED",
                        },
                    ],
                    "approval": {
                        "status": "approved",
                        "reviewed_by": "Earlier Researcher",
                        "reviewed_at": "2026-08-11T12:00:00+00:00",
                    },
                }
            ],
            "cohorts": [{"name": "all"}],
            "index_date_rule": {"strategy": "fixed_date", "fixed_date": "2024-01-01"},
            "analyses": [{"id": "baseline", "kind": "table_one", "params": {"cohort": "all"}}],
        }
    )


def _artifact(plan: Plan) -> dict[str, object]:
    code_set = plan.code_sets[0]
    return {
        "schema_version": "1.0",
        "name": code_set.name,
        "domain": code_set.domain,
        "codings": [coding.model_dump(mode="json") for coding in code_set.codings],
        "mapping_status_counts": {"MAPPED": 1, "UNCHECKED": 1},
        "approval": code_set.approval.model_dump(mode="json"),
    }


def test_curate_coding_atomically_updates_artifact_and_plan_and_resets_approval(tmp_path) -> None:
    workspace = create_study(tmp_path / "study", "Curation")
    plan = _plan()
    write_plan(workspace.plan_path, plan)
    target = workspace.root / "codesets" / "demo.codeset.json"
    target.write_text(json.dumps(_artifact(plan), sort_keys=True) + "\n", encoding="utf-8")
    result = curate_code_set_coding(
        workspace,
        name="demo",
        system="https://example.test",
        code="two",
        accepted=False,
        curated_by="Researcher",
        curated_at=datetime(2026, 8, 12, 12, tzinfo=UTC),
        expected_artifact_hash=hash_file(target),
    )
    assert result.status == "success"
    artifact = json.loads(target.read_text(encoding="utf-8"))
    updated_plan = json.loads(workspace.plan_path.read_text(encoding="utf-8"))
    assert artifact["codings"][1]["accepted"] is False
    assert updated_plan["code_sets"][0]["codings"][1]["accepted"] is False
    assert artifact["approval"]["status"] == "pending"
    assert updated_plan["code_sets"][0]["approval"]["status"] == "pending"
    assert artifact["curation"][0]["curated_by"] == "Researcher"
    entry = json.loads(workspace.manifest_path.read_text(encoding="utf-8").splitlines()[-1])
    assert entry["step"] == "curate-codes"


def test_curate_coding_refuses_stale_ui_and_last_accepted_removal(tmp_path) -> None:
    workspace = create_study(tmp_path / "study", "Curation")
    plan = _plan()
    write_plan(workspace.plan_path, plan)
    target = workspace.root / "codesets" / "demo.codeset.json"
    target.write_text(json.dumps(_artifact(plan), sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed since it was displayed"):
        curate_code_set_coding(
            workspace,
            name="demo",
            system="https://example.test",
            code="two",
            accepted=False,
            curated_by="Researcher",
            curated_at="2026-08-12T12:00:00+00:00",
            expected_artifact_hash="0" * 64,
        )
    curate_code_set_coding(
        workspace,
        name="demo",
        system="https://example.test",
        code="two",
        accepted=False,
        curated_by="Researcher",
        curated_at="2026-08-12T12:00:00+00:00",
        expected_artifact_hash=hash_file(target),
    )
    with pytest.raises(ValueError, match="retain at least one"):
        curate_code_set_coding(
            workspace,
            name="demo",
            system="https://example.test",
            code="one",
            accepted=False,
            curated_by="Researcher",
            curated_at="2026-08-12T12:01:00+00:00",
            expected_artifact_hash=hash_file(target),
        )
