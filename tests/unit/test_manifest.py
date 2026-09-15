from __future__ import annotations

import json
from datetime import UTC, datetime

from pheno_rwe.hashing import hash_file
from pheno_rwe.manifest import (
    build_shareable_manifest,
    read_manifest,
    record_step,
    verify_manifest,
    write_manifest,
)


def test_manifest_chain_detects_tampering(tmp_path):
    path = tmp_path / "manifest.jsonl"
    for step in ("ingest", "materialize"):
        record_step(path, step=step, status="success", started_at=datetime.now(UTC))
    verification = verify_manifest(path)
    assert verification.valid
    assert verification.entries == 2

    entries = read_manifest(path)
    entries[0]["step"] = "tampered"
    path.write_text("\n".join(json.dumps(item) for item in entries) + "\n", encoding="utf-8")
    verification = verify_manifest(path)
    assert not verification.valid
    assert any("entry_hash" in error for error in verification.errors)


def test_manifest_chain_links_entries(tmp_path):
    path = tmp_path / "manifest.jsonl"
    first = record_step(path, step="init", status="success", started_at=datetime.now(UTC))
    second = record_step(path, step="ingest", status="partial", started_at=datetime.now(UTC))
    assert second.prev_hash == first.entry_hash


def test_manifest_verifies_latest_declared_artifact(tmp_path):
    path = tmp_path / "manifest.jsonl"
    artifact = tmp_path / "result.json"
    artifact.write_text('{"value":1}\n', encoding="utf-8")
    record_step(
        path,
        step="analyze",
        status="success",
        started_at=datetime.now(UTC),
        outputs={"result.json": hash_file(artifact)},
    )
    assert verify_manifest(path, verify_outputs=True, root=tmp_path).valid
    artifact.write_text('{"value":2}\n', encoding="utf-8")
    verification = verify_manifest(path, verify_outputs=True, root=tmp_path)
    assert not verification.valid
    assert "output hash does not match" in verification.errors[0]


def test_manifest_verifies_inputs_and_uses_latest_path_declaration(tmp_path):
    path = tmp_path / "manifest.jsonl"
    artifact = tmp_path / "plan.json"
    artifact.write_text('{"version":1}\n', encoding="utf-8")
    record_step(
        path,
        step="init",
        status="success",
        started_at=datetime.now(UTC),
        outputs={"plan.json": hash_file(artifact)},
    )
    artifact.write_text('{"version":2}\n', encoding="utf-8")
    record_step(
        path,
        step="validate",
        status="success",
        started_at=datetime.now(UTC),
        inputs={str(artifact): hash_file(artifact)},
    )

    assert verify_manifest(
        path,
        verify_inputs=True,
        verify_outputs=True,
        root=tmp_path,
    ).valid
    artifact.write_text('{"version":3}\n', encoding="utf-8")
    verification = verify_manifest(path, verify_inputs=True, root=tmp_path)
    assert not verification.valid
    assert "input hash does not match" in verification.errors[0]


def test_shareable_manifest_rebuild_strips_identified_lineage_and_verifies(tmp_path):
    payload = tmp_path / "plan.json"
    payload.write_text('{"study":"safe"}\n', encoding="utf-8")
    bundle = tmp_path / "bundle.json"
    bundle.write_text('{"identified_data_included":false}\n', encoding="utf-8")
    now = datetime.now(UTC)
    secret_path = "/identified/source/patient-raw-123.ndjson"
    source = [
        {
            "step": "pull",
            "status": "success",
            "started_at": now.isoformat(),
            "finished_at": now.isoformat(),
            "params": {
                "text": "secret FHIR query",
                "provider_id": "private-provider",
                "approved": True,
            },
            "input_signature": "identified-fingerprint",
            "inputs": {secret_path: "a" * 64},
            "outputs": {"raw/patients/patient-raw-123.json": "b" * 64},
            "items": [
                {
                    "patient_token": "patient-raw-123",
                    "document_token": "document-raw-456",
                    "status": "success",
                }
            ],
            "api_calls": [
                {
                    "endpoint": "/fhir/search",
                    "count": 1,
                    "ms": 4.0,
                    "request": "Patient/patient-raw-123/$everything",
                }
            ],
            "versions": {"pheno_rwe": "0.1.0", "python": "3.13.0"},
        }
    ]
    entries = build_shareable_manifest(
        source,
        payload_files={"plan.json": hash_file(payload)},
        bundle_sha256=hash_file(bundle),
        started_at=now,
        finished_at=now,
        include_database=False,
    )
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, entries)

    serialized = manifest.read_text(encoding="utf-8")
    for secret in (
        secret_path,
        "patient-raw-123",
        "document-raw-456",
        "secret FHIR query",
        "private-provider",
        "identified-fingerprint",
    ):
        assert secret not in serialized
    assert '"endpoint":"/fhir/search"' in serialized
    assert '"manifest_profile":"shareable-v1"' in serialized
    assert verify_manifest(
        manifest,
        verify_inputs=True,
        verify_outputs=True,
        root=tmp_path,
        restrict_to_root=True,
    ).valid


def test_restricted_manifest_verification_never_reads_outside_bundle(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = bundle / "manifest.jsonl"
    record_step(
        manifest,
        step="export",
        status="success",
        started_at=datetime.now(UTC),
        inputs={str(outside): hash_file(outside)},
    )

    verification = verify_manifest(
        manifest,
        verify_inputs=True,
        root=bundle,
        restrict_to_root=True,
    )
    assert not verification.valid
    assert verification.errors == ("entry 1: input path escapes the verification root",)
