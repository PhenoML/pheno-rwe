from __future__ import annotations

import base64
import json

import pytest

from pheno_rwe.manifest import read_manifest
from pheno_rwe.plan import write_plan
from pheno_rwe.steps.enrich import enrich_documents
from pheno_rwe.steps.pull import preview_live_cohort, pull_live_cohort
from pheno_rwe.steps.resolve_codes import resolve_code_set
from pheno_rwe.steps.review_codes import review_code_set
from pheno_rwe.workspace import create_study


def test_live_pull_requires_approval_and_strips_narrative(tmp_path, fake_client):
    workspace = create_study(tmp_path / "study", "Pull")
    preview = preview_live_cohort(fake_client, "diabetes", "provider")
    refused = pull_live_cohort(
        workspace,
        fake_client,
        text="diabetes",
        provider_id="provider",
        preview=preview,
    )
    assert refused.status == "preview"
    assert not list((workspace.root / "raw" / "patients").glob("*.json"))

    result = pull_live_cohort(
        workspace,
        fake_client,
        text="diabetes",
        provider_id="provider",
        preview=preview,
        approved=True,
    )
    assert result.status == "success"
    bundle_path = next((workspace.root / "raw" / "patients").glob("*.json"))
    assert "patient-1" not in bundle_path.name
    bundle = json.loads(bundle_path.read_text())
    assert "text" not in bundle["entry"][0]["resource"]
    assert "patient-1" not in json.dumps(read_manifest(workspace.manifest_path))


def test_enrich_caches_and_tags_resources(tmp_path, fake_client):
    workspace = create_study(tmp_path / "study", "Enrich")
    content = base64.b64encode(b"clinical note").decode()
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p1"}},
            {
                "resource": {
                    "resourceType": "DocumentReference",
                    "id": "doc1",
                    "content": [{"attachment": {"contentType": "text/plain", "data": content}}],
                }
            },
        ],
    }
    source = workspace.root / "raw" / "patients" / "p1.bundle.json"
    source.write_text(json.dumps(bundle), encoding="utf-8")
    result = enrich_documents(workspace, fake_client)
    assert result.status == "success"
    enriched = json.loads((workspace.root / "enriched" / "patients" / source.name).read_text())
    added = enriched["entry"][-1]["resource"]
    assert added["id"].startswith("enr-")
    assert any(tag["code"] == "enriched" for tag in added["meta"]["tag"])
    assert len(list((workspace.root / "enriched" / "documents").glob("*.json"))) == 1


def test_enrich_accepts_single_resource_document_response(
    tmp_path,
    fake_client,
    monkeypatch,
):
    workspace = create_study(tmp_path / "study", "Single resource")
    content = base64.b64encode(b"synthetic clinical note").decode()
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "synthetic-patient"}},
            {
                "resource": {
                    "resourceType": "DocumentReference",
                    "id": "synthetic-document",
                    "content": [
                        {
                            "attachment": {
                                "contentType": "text/plain",
                                "data": content,
                            }
                        }
                    ],
                }
            },
        ],
    }
    source = workspace.root / "raw" / "patients" / "synthetic.bundle.json"
    source.write_text(json.dumps(bundle), encoding="utf-8")
    monkeypatch.setattr(
        fake_client,
        "document",
        lambda content, mime_type, **options: {
            "resourceType": "Condition",
            "id": "temporary-single",
            "code": {"text": "Synthetic condition"},
        },
    )

    result = enrich_documents(workspace, fake_client)

    assert result.status == "success"
    enriched = json.loads((workspace.root / "enriched" / "patients" / source.name).read_text())
    assert enriched["entry"][-1]["resource"]["resourceType"] == "Condition"
    assert enriched["entry"][-1]["resource"]["id"].startswith("enr-")


def test_resolve_codes_round_trips_through_fhir2omop(tmp_path, fake_client):
    workspace = create_study(tmp_path / "study", "Codes")
    result = resolve_code_set(
        workspace,
        fake_client,
        name="diabetes",
        text="type 2 diabetes",
        domain="condition",
    )
    assert result.items[0]["concept_id"] == 201826
    assert result.items[0]["mapping_status"] == "MAPPED"
    assert result.warnings


def test_review_codes_refuses_plan_artifact_coding_mismatch_without_mutation(
    tmp_path,
    fake_client,
) -> None:
    workspace = create_study(tmp_path / "study", "Codes")
    resolve_code_set(
        workspace,
        fake_client,
        name="diabetes",
        text="type 2 diabetes",
        domain="condition",
    )
    write_plan(
        workspace.plan_path,
        {
            "study": {"name": "Codes", "question": "A question"},
            "code_sets": [
                {
                    "name": "diabetes",
                    "domain": "condition",
                    "codings": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": "different-code",
                            "concept_id": 201826,
                            "mapping_status": "MAPPED",
                        }
                    ],
                }
            ],
            "cohorts": [{"name": "all"}],
            "index_date_rule": {"strategy": "fixed_date", "fixed_date": "2024-01-01"},
            "analyses": [{"id": "baseline", "kind": "table_one", "params": {"cohort": "all"}}],
        },
    )
    artifact_path = workspace.root / "codesets" / "diabetes.codeset.json"
    before_artifact = artifact_path.read_bytes()
    before_plan = workspace.plan_path.read_bytes()

    with pytest.raises(ValueError, match="do not exactly match"):
        review_code_set(
            workspace,
            name="diabetes",
            decision="approved",
            reviewed_by="Synthetic researcher",
            reviewed_at="2026-08-12T12:00:00Z",
        )

    assert artifact_path.read_bytes() == before_artifact
    assert workspace.plan_path.read_bytes() == before_plan
