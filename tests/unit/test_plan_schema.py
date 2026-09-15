from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from pheno_rwe.plan import (
    Plan,
    SurvivalAnalysis,
    analysis_data_hash,
    canonical_plan_json,
    load_plan,
    plan_hash,
    plan_json_schema,
    write_plan,
)


def valid_plan_dict() -> dict:
    return {
        "schema_version": "1.0",
        "study": {"name": "Example", "question": "Does treatment alter outcome?"},
        "code_sets": [
            {
                "name": "exposure",
                "domain": "drug",
                "codings": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "1"}],
                "approval": {
                    "status": "approved",
                    "reviewed_by": "Example researcher",
                    "reviewed_at": "2026-08-12T12:00:00Z",
                },
            },
            {
                "name": "outcome",
                "domain": "condition",
                "codings": [{"system": "http://snomed.info/sct", "code": "2"}],
                "approval": {
                    "status": "approved",
                    "reviewed_by": "Example researcher",
                    "reviewed_at": "2026-08-12T12:00:00Z",
                },
            },
        ],
        "cohorts": [
            {
                "name": "exposed",
                "inclusion": [{"code_set": "exposure", "min_count": 1}],
            },
            {"name": "comparator"},
        ],
        "index_date_rule": {"strategy": "first_occurrence", "code_set": "exposure"},
        "analyses": [
            {
                "id": "survival-1",
                "kind": "survival",
                "params": {
                    "exposed_cohort": "exposed",
                    "comparator_cohort": "comparator",
                    "outcome": "outcome",
                    "censor_rule": {
                        "strategy": "end_of_observation",
                        "description": "End of continuous observation",
                    },
                },
            }
        ],
        "plots": [
            {
                "id": "km-1",
                "kind": "km_curve",
                "params": {"analysis_id": "survival-1"},
            }
        ],
    }


def test_analysis_and_plot_are_discriminated_models() -> None:
    plan = Plan.model_validate(valid_plan_dict())

    assert isinstance(plan.analyses[0], SurvivalAnalysis)
    assert plan.plots[0].kind == "km_curve"
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        Plan.model_validate(
            valid_plan_dict()
            | {"analyses": [{"id": "bad", "kind": "made_up", "params": {"cohort": "exposed"}}]}
        )


def test_schema_rejects_invalid_index_rules_and_dangling_references() -> None:
    payload = valid_plan_dict()
    payload["index_date_rule"] = {"strategy": "fixed_date"}
    with pytest.raises(ValidationError, match="fixed_date is required"):
        Plan.model_validate(payload)

    payload = valid_plan_dict()
    payload["cohorts"][0]["inclusion"][0]["code_set"] = "unknown"
    with pytest.raises(ValidationError, match="unknown code-set reference"):
        Plan.model_validate(payload)

    payload = valid_plan_dict()
    payload["plots"][0]["params"]["analysis_id"] = "unknown"
    with pytest.raises(ValidationError, match="unknown analysis reference"):
        Plan.model_validate(payload)


def test_schema_rejects_duplicates_and_unknown_fields() -> None:
    payload = valid_plan_dict()
    payload["cohorts"].append({"name": "exposed"})
    with pytest.raises(ValidationError, match="duplicate cohort name"):
        Plan.model_validate(payload)

    payload = valid_plan_dict()
    payload["analyses"][0]["params"]["cox_penality"] = 1
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Plan.model_validate(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["analyses"][0]["params"].update({"exposed_cohort": "missing"}),
            "unknown cohort 'missing'",
        ),
        (
            lambda payload: payload["analyses"][0]["params"].update({"outcome": "missing"}),
            "unknown code set 'missing'",
        ),
        (
            lambda payload: payload["analyses"][0]["params"].update(
                {"comparator_cohort": "exposed"}
            ),
            "exposed and comparator cohorts must be distinct",
        ),
    ],
)
def test_schema_rejects_dangling_analysis_references(mutate, message: str) -> None:
    payload = valid_plan_dict()
    mutate(payload)

    with pytest.raises(ValidationError, match=message):
        Plan.model_validate(payload)


def test_schema_rejects_non_executable_analysis_choices() -> None:
    payload = valid_plan_dict()
    payload["analyses"] = [
        {
            "id": "compare",
            "kind": "cohort_compare",
            "params": {
                "exposed_cohort": "exposed",
                "comparator_cohort": "comparator",
                "outcomes": [
                    {
                        "code_set": "outcome",
                        "risk_window": {
                            "start_day": 0,
                            "end_day": 90,
                            "washout_days": 0,
                        },
                    }
                ],
                "adjusted": True,
            },
        }
    ]
    payload["plots"] = []
    with pytest.raises(ValidationError, match="requires a non-empty adjustment_set"):
        Plan.model_validate(payload)

    payload = valid_plan_dict()
    payload["analyses"] = [
        {
            "id": "rates",
            "kind": "incidence_rate",
            "params": {"cohorts": ["exposed"], "outcome": "outcome"},
        }
    ]
    payload["plots"] = []
    with pytest.raises(ValidationError, match="at least 2 items"):
        Plan.model_validate(payload)


def test_plan_hash_is_canonical_and_semantic() -> None:
    payload = valid_plan_dict()
    reordered = json.loads(json.dumps(payload, sort_keys=True))
    model = Plan.model_validate(payload)

    assert plan_hash(payload) == plan_hash(reordered) == plan_hash(model)
    assert len(plan_hash(model)) == 64
    assert canonical_plan_json(payload) == canonical_plan_json(reordered)

    changed = valid_plan_dict()
    changed["seed"] = 7
    assert plan_hash(changed) != plan_hash(payload)


def test_binary_risk_contract_and_code_set_approval_are_hashed() -> None:
    payload = valid_plan_dict()
    payload["analyses"] = [
        {
            "id": "compare",
            "kind": "cohort_compare",
            "params": {
                "exposed_cohort": "exposed",
                "comparator_cohort": "comparator",
                "outcomes": [
                    {
                        "code_set": "outcome",
                        "risk_window": {
                            "start_day": 0,
                            "end_day": 90,
                            "washout_days": 30,
                        },
                    }
                ],
            },
        }
    ]
    payload["plots"] = []
    model = Plan.model_validate(payload)
    canonical = json.loads(canonical_plan_json(model))

    assert canonical["analyses"][0]["params"]["outcomes"][0]["risk_window"] == {
        "end_day": 90,
        "observation_censor": "end_of_observation",
        "start_day": 0,
        "washout_days": 30,
    }
    changed_window = json.loads(json.dumps(payload))
    changed_window["analyses"][0]["params"]["outcomes"][0]["risk_window"]["end_day"] = 91
    assert plan_hash(changed_window) != plan_hash(payload)

    changed_approval = json.loads(json.dumps(payload))
    changed_approval["code_sets"][0]["approval"]["reviewed_by"] = "Second researcher"
    assert plan_hash(changed_approval) != plan_hash(payload)
    assert analysis_data_hash(changed_approval) == analysis_data_hash(payload)


def test_binary_risk_contract_refuses_temporally_ambiguous_legacy_outcomes() -> None:
    payload = valid_plan_dict()
    payload["analyses"] = [
        {
            "id": "compare",
            "kind": "cohort_compare",
            "params": {
                "exposed_cohort": "exposed",
                "comparator_cohort": "comparator",
                "outcomes": ["outcome"],
            },
        }
    ]
    payload["plots"] = []

    with pytest.raises(ValidationError, match="each outcome must declare code_set and risk_window"):
        Plan.model_validate(payload)


def test_reviewed_code_set_requires_reviewer_and_timestamp() -> None:
    payload = valid_plan_dict()
    payload["code_sets"][0]["approval"] = {"status": "approved"}

    with pytest.raises(ValidationError, match="reviewed_by and reviewed_at are required"):
        Plan.model_validate(payload)

    payload = valid_plan_dict()
    payload["code_sets"][0]["approval"]["reviewed_at"] = "2026-08-12T12:00:00"
    with pytest.raises(ValidationError, match="reviewed_at must include a UTC offset"):
        Plan.model_validate(payload)


def test_plan_round_trip_and_json_schema(tmp_path) -> None:
    plan_path = tmp_path / "study" / "plan.json"
    original = Plan.model_validate(valid_plan_dict())

    assert write_plan(plan_path, original) == plan_path
    loaded = load_plan(plan_path)

    assert loaded == original
    assert plan_hash(loaded) == plan_hash(original)
    schema = plan_json_schema()
    assert schema["title"] == "Plan"
    assert "discriminator" in json.dumps(schema)
