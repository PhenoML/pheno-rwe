from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from pheno_rwe.cli import app
from pheno_rwe.hashing import canonical_json


class SyntheticPhenoTransport:
    """Deterministic transport exercising the live-service seams without egress."""

    def resolve_codings(self, text: str, domain: str | None = None) -> dict[str, Any]:
        del text, domain
        return {
            "codings": [
                {
                    "system": "https://example.test/codes",
                    "code": "demo",
                    "display": "Synthetic demo concept",
                }
            ]
        }

    def fhir2omop(self, bundle: dict[str, Any]) -> dict[str, Any]:
        resources = [
            entry.get("resource", {})
            for entry in bundle.get("entry", [])
            if isinstance(entry, dict)
        ]
        patient = next(
            resource for resource in resources if resource.get("resourceType") == "Patient"
        )
        patient_id = str(patient["id"])
        try:
            ordinal = int(patient_id.rsplit("-", 1)[-1])
        except ValueError:
            ordinal = 1
        sex = "F" if ordinal % 2 else "M"
        conditions: list[dict[str, Any]] = []
        mappings: list[dict[str, Any]] = []
        for resource in resources:
            if resource.get("resourceType") != "Condition":
                continue
            coding = next(iter(resource.get("code", {}).get("coding", [])), None)
            if not coding:
                continue
            row_id = len(conditions) + 1
            conditions.append(
                {
                    "condition_occurrence_id": row_id,
                    "person_id": 1,
                    "condition_concept_id": 123,
                    "condition_start_date": "2024-02-01",
                    "condition_source_value": f"{coding.get('system', '')}#{coding['code']}",
                }
            )
            mappings.append(
                {
                    "resource_type": "Condition",
                    "resource_id": resource.get("id"),
                    "omop_table": "condition_occurrence",
                    "omop_id": row_id,
                    "source_coding": coding,
                    "source_system": coding.get("system"),
                    "source_code": coding.get("code"),
                    "mapping_status": "MAPPED",
                    "concept_id": 123,
                }
            )
        return {
            "tables": {
                "person": [
                    {
                        "person_id": 1,
                        "gender_concept_id": 8532 if sex == "F" else 8507,
                        "year_of_birth": 1970 + ordinal,
                        "race_concept_id": 0,
                        "ethnicity_concept_id": 0,
                        "person_source_value": patient_id,
                        "gender_source_value": sex,
                    }
                ],
                "observation_period": [
                    {
                        "observation_period_id": 1,
                        "person_id": 1,
                        "observation_period_start_date": "2023-01-01",
                        "observation_period_end_date": "2025-01-01",
                        "period_type_concept_id": 0,
                    }
                ],
                "condition_occurrence": conditions,
            },
            "mappings": mappings,
            "summary": {
                "codes_already_standard": 0,
                "codes_normalized": len(mappings),
                "codes_unmapped": 0,
                "off_vocab_rate": 0.0,
            },
            "dropped": [],
            "vocab_version": "synthetic-v1",
        }

    def document(self, content: bytes, mime_type: str, **options: Any) -> dict[str, Any]:
        del content, mime_type, options
        return {
            "resourceType": "Bundle",
            "type": "transaction",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": "generated-note-finding",
                        "code": {"text": "Synthetic note finding"},
                    }
                }
            ],
        }

    def fhir_search(self, provider: str, fhir_path: str, **params: Any) -> dict[str, Any]:
        raise AssertionError(
            f"inline synthetic attachments should not call FHIR search: "
            f"{provider=} {fhir_path=} {params=}"
        )


def _write_synthetic_ndjson(path: Path, patient_count: int = 6) -> None:
    resources: list[dict[str, Any]] = []
    for ordinal in range(1, patient_count + 1):
        patient_id = f"synthetic-{ordinal}"
        resources.append({"resourceType": "Patient", "id": patient_id})
        if ordinal == 1:
            resources.append(
                {
                    "resourceType": "DocumentReference",
                    "id": "synthetic-document",
                    "subject": {"reference": f"Patient/{patient_id}"},
                    "content": [
                        {
                            "attachment": {
                                "contentType": "text/plain",
                                "data": base64.b64encode(b"synthetic clinical note").decode(),
                            }
                        }
                    ],
                }
            )
    path.write_text(
        "".join(json.dumps(resource, sort_keys=True) + "\n" for resource in resources),
        encoding="utf-8",
    )


def _plan() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "study": {
            "name": "Synthetic smoke study",
            "question": "What are the baseline characteristics of the synthetic cohort?",
        },
        "code_sets": [
            {
                "name": "demo",
                "domain": "condition",
                "codings": [
                    {
                        "system": "https://example.test/codes",
                        "code": "demo",
                        "display": "Synthetic demo concept",
                        "concept_id": 123,
                        "mapping_status": "MAPPED",
                    }
                ],
                "approval": {
                    "status": "pending",
                },
            }
        ],
        "cohorts": [
            {"name": "All"},
            {"name": "ArmF", "demographics": {"sex": ["F"]}},
            {"name": "ArmM", "demographics": {"sex": ["M"]}},
        ],
        "index_date_rule": {"strategy": "fixed_date", "fixed_date": "2024-01-01"},
        "deid": {"tier": "baseline"},
        "seed": 2025,
        "analyses": [
            {
                "id": "baseline",
                "kind": "table_one",
                "params": {
                    "cohort": "All",
                    "stratify_by": "sex",
                    "variables": ["age_bucket", "sex"],
                },
            }
        ],
        "plots": [
            {
                "id": "attrition",
                "kind": "attrition",
                "params": {
                    "cohort": "All",
                    "source_csv": "reports/cohort_attrition.csv",
                },
            }
        ],
    }


def test_synthetic_cli_pipeline_refusal_and_staleness(tmp_path, monkeypatch) -> None:
    transport = SyntheticPhenoTransport()
    monkeypatch.setattr(
        "pheno_rwe.client.build_transport",
        lambda *args, **kwargs: transport,
    )
    runner = CliRunner()
    study = tmp_path / "study"
    source = tmp_path / "synthetic.ndjson"
    shared = tmp_path / "shared"
    _write_synthetic_ndjson(source)

    def invoke(*arguments: str, expected: int = 0):
        result = runner.invoke(app, list(arguments))
        assert result.exit_code == expected, result.output
        return result

    # Global flags intentionally follow the command to cover the documented CLI form.
    invoke("init", str(study), "--name", "Synthetic smoke", "--json")
    invoke("ingest", str(source), "--study", str(study), "--json")
    invoke("enrich", "--study", str(study), "--json")
    invoke(
        "resolve-codes",
        "--name",
        "demo",
        "--text",
        "synthetic demo concept",
        "--domain",
        "condition",
        "--study",
        str(study),
        "--json",
    )
    invoke("materialize", "--study", str(study), "--json")

    original_plan = _plan()
    (study / "plan.json").write_text(canonical_json(original_plan) + "\n", encoding="utf-8")
    invoke(
        "review-codes",
        "--name",
        "demo",
        "--decision",
        "approved",
        "--reviewed-by",
        "Synthetic test researcher",
        "--reviewed-at",
        "2026-08-12T12:00:00Z",
        "--study",
        str(study),
        "--json",
    )
    original_plan = json.loads((study / "plan.json").read_text(encoding="utf-8"))
    invoke("resolve-cohort", "--study", str(study), "--json")
    invoke("deid", "--study", str(study), "--json")
    invoke("validate", "--study", str(study), "--json")
    invoke("analyze", "--all", "--study", str(study), "--json")
    invoke("plot", "--all", "--study", str(study), "--json")
    invoke(
        "export",
        "--output",
        str(shared),
        "--study",
        str(study),
        "--json",
    )
    invoke("trace", "--verify", "--study", str(study), "--json")

    assert (study / "results" / "baseline" / "result.json").is_file()
    assert (study / "plots" / "attrition" / "plot.svg").is_file()
    assert (shared / "bundle.json").is_file()

    stale_plan = copy.deepcopy(original_plan)
    stale_plan["study"]["question"] = "A changed, unvalidated question"
    (study / "plan.json").write_text(canonical_json(stale_plan) + "\n", encoding="utf-8")
    stale = invoke(
        "analyze",
        "--all",
        "--study",
        str(study),
        "--json",
        expected=4,
    )
    assert json.loads(stale.output)["exit_code"] == 4

    small_plan = copy.deepcopy(original_plan)
    small_plan["analyses"] = [
        {
            "id": "small-comparison",
            "kind": "cohort_compare",
            "params": {
                "exposed_cohort": "ArmF",
                "comparator_cohort": "ArmM",
                "outcomes": [
                    {
                        "code_set": "demo",
                        "risk_window": {
                            "start_day": 0,
                            "end_day": 90,
                            "washout_days": 0,
                        },
                    }
                ],
            },
        }
    ]
    small_plan["plots"] = []
    (study / "plan.json").write_text(canonical_json(small_plan) + "\n", encoding="utf-8")
    refused = invoke("validate", "--study", str(study), "--json", expected=3)
    payload = json.loads(refused.output)
    assert payload["status"] == "refused"
    assert any(outcome["rule_id"] == "GLOBAL-MIN-N" for outcome in payload["report"]["outcomes"])
