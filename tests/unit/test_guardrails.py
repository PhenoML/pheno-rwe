from __future__ import annotations

import copy

import pytest

from pheno_rwe.errors import GuardrailRefusal
from pheno_rwe.guardrails import (
    DataSummary,
    ExitCode,
    RuleOutcome,
    Severity,
    catalog,
    exit_code_for,
    minimum_detectable_effect_note,
    raise_for_refusal,
    validate_analysis_data,
    validate_plan_static,
)
from pheno_rwe.plan import Plan, code_set_review_hash


def risk_outcome(code_set: str = "outcome") -> dict:
    return {
        "code_set": code_set,
        "risk_window": {"start_day": 0, "end_day": 90, "washout_days": 0},
    }


def plan_for(analysis: dict, *, override: bool = False) -> Plan:
    payload = {
        "study": {"name": "Rules", "question": "Can this analysis be estimated?"},
        "code_sets": [
            {
                "name": "index",
                "domain": "drug",
                "codings": [{"system": "rxnorm", "code": "1", "mapping_status": "MAPPED"}],
            },
            {
                "name": "outcome",
                "domain": "condition",
                "codings": [{"system": "snomed", "code": "2", "mapping_status": "MAPPED"}],
            },
            {
                "name": "lab",
                "domain": "measurement",
                "codings": [{"system": "loinc", "code": "3", "mapping_status": "MAPPED"}],
            },
        ],
        "cohorts": [{"name": "exposed"}, {"name": "comparator"}],
        "index_date_rule": {"strategy": "first_occurrence", "code_set": "index"},
        "analyses": [analysis],
    }
    for code_set in payload["code_sets"]:
        code_set["approval"] = {
            "status": "approved",
            "reviewed_by": "Test researcher",
            "reviewed_at": "2026-08-12T12:00:00Z",
            "review_hash": code_set_review_hash(code_set),
        }
    if override:
        payload["provenance_override"] = {
            "justification": "Chart review supports use of the unresolved source codes."
        }
    return Plan.model_validate(payload)


def outcome_ids(report) -> set[str]:
    return {outcome.rule_id for outcome in report.outcomes}


def test_static_rules_refuse_missing_survival_censor_rule() -> None:
    plan = plan_for(
        {
            "id": "s1",
            "kind": "survival",
            "params": {
                "exposed_cohort": "exposed",
                "comparator_cohort": "comparator",
                "outcome": "outcome",
            },
        }
    )

    report = validate_plan_static(plan)

    assert report.refused
    assert report.exit_code is ExitCode.REFUSED
    assert outcome_ids(report) == {catalog.SURV_CENSOR}


def test_static_rules_require_causal_covariate_rationales() -> None:
    plan = plan_for(
        {
            "id": "c1",
            "kind": "causal_effect",
            "params": {
                "exposed_cohort": "exposed",
                "comparator_cohort": "comparator",
                "outcome": risk_outcome(),
                "adjustment_set": [{"name": "age", "rationale": ""}],
            },
        }
    )

    report = validate_plan_static(plan)

    assert outcome_ids(report) == {catalog.CAUSAL_ADJUSTMENT}
    assert report.outcomes[0].details["covariates"] == ["age"]


def test_static_rules_refuse_unapproved_code_sets_with_mapping_counts() -> None:
    approved = plan_for(
        {
            "id": "table",
            "kind": "table_one",
            "params": {"cohort": "exposed"},
        }
    ).model_dump(mode="json")
    approved["code_sets"][1]["approval"] = {"status": "pending"}

    report = validate_plan_static(Plan.model_validate(approved))
    outcomes = [
        outcome for outcome in report.outcomes if outcome.rule_id == catalog.CODESET_APPROVAL
    ]

    assert len(outcomes) == 1
    assert outcomes[0].analysis_id is None
    assert outcomes[0].details == {
        "code_set": "outcome",
        "approval_status": "pending",
        "mapping_status_counts": {"MAPPED": 1},
        "mapping_statuses_missing": 0,
        "review_hash_matches": False,
    }


def test_static_rules_refuse_approval_retained_after_coding_edit() -> None:
    plan = plan_for(
        {
            "id": "table",
            "kind": "table_one",
            "params": {"cohort": "exposed"},
        }
    ).model_dump(mode="json")
    plan["code_sets"][0]["codings"][0]["display"] = "Edited after review"

    report = validate_plan_static(Plan.model_validate(plan))
    outcome = next(
        value
        for value in report.outcomes
        if value.rule_id == catalog.CODESET_APPROVAL and value.details["code_set"] == "index"
    )

    assert outcome.severity is Severity.REFUSE
    assert outcome.details["review_hash_matches"] is False


def test_multiplicity_is_automatic_and_cannot_be_disabled() -> None:
    analysis = {
        "id": "cmp",
        "kind": "cohort_compare",
        "params": {
            "exposed_cohort": "exposed",
            "comparator_cohort": "comparator",
            "outcomes": [risk_outcome(), risk_outcome("index")],
        },
    }
    automatic = validate_plan_static(plan_for(analysis))
    assert automatic.outcomes[0].severity is Severity.INFO

    disabled_analysis = copy.deepcopy(analysis)
    disabled_analysis["params"]["multiplicity"] = "none"
    disabled = validate_plan_static(plan_for(disabled_analysis))
    assert disabled.outcomes[0].severity is Severity.REFUSE


def test_comparative_n_and_arm_thresholds_have_exact_boundaries() -> None:
    plan = plan_for(
        {
            "id": "cmp",
            "kind": "cohort_compare",
            "params": {
                "exposed_cohort": "exposed",
                "comparator_cohort": "comparator",
                "outcomes": [risk_outcome()],
            },
        }
    )
    below = validate_analysis_data(
        plan,
        "cmp",
        {"analysis_id": "cmp", "total_n": 9, "arm_n": {"a": 4, "b": 5}},
    )
    boundary = validate_analysis_data(
        plan,
        "cmp",
        {"analysis_id": "cmp", "total_n": 10, "arm_n": {"a": 5, "b": 5}},
    )

    assert outcome_ids(below) == {catalog.GLOBAL_MIN_N, catalog.COMP_MIN_ARM}
    assert not boundary.refused


@pytest.mark.parametrize(
    ("events", "expected"),
    [(19, Severity.REFUSE), (20, Severity.WARN), (39, Severity.WARN), (40, None)],
)
def test_epv_logistic_thresholds(events: int, expected: Severity | None) -> None:
    plan = plan_for(
        {
            "id": "cmp",
            "kind": "cohort_compare",
            "params": {
                "exposed_cohort": "exposed",
                "comparator_cohort": "comparator",
                "outcomes": [risk_outcome()],
                "adjusted": True,
                "adjustment_set": [{"name": "age", "rationale": "Pre-specified confounder."}],
            },
        }
    )
    report = validate_analysis_data(
        plan,
        "cmp",
        {
            "analysis_id": "cmp",
            "total_n": 100,
            "arm_n": {"exposed": 50, "comparator": 50},
            "events_total": events,
            "model_parameters": 4,
        },
    )
    epv = [outcome for outcome in report.outcomes if outcome.rule_id == catalog.EPV_LOGISTIC]

    if expected is None:
        assert epv == []
    else:
        assert epv[0].severity is expected


@pytest.mark.parametrize(
    ("n", "severity"),
    [(29, Severity.REFUSE), (30, Severity.WARN), (99, Severity.WARN), (100, None)],
)
def test_umap_minimum_n(n: int, severity: Severity | None) -> None:
    plan = plan_for(
        {
            "id": "sig",
            "kind": "patient_signature",
            "params": {"cohort": "exposed"},
        }
    )
    report = validate_analysis_data(plan, "sig", {"analysis_id": "sig", "n": n})
    outcomes = [outcome for outcome in report.outcomes if outcome.rule_id == catalog.UMAP_MIN_N]

    assert (outcomes[0].severity if outcomes else None) is severity
    if n < 30:
        assert outcomes[0].details["downgrade"] == "pca"


def test_unchecked_fraction_override_is_hashed_and_auditable() -> None:
    analysis = {
        "id": "t1",
        "kind": "table_one",
        "params": {"cohort": "exposed"},
    }
    refused = validate_analysis_data(
        plan_for(analysis),
        "t1",
        {"analysis_id": "t1", "n": 100, "unchecked_fraction": 0.51},
    )
    overridden = validate_analysis_data(
        plan_for(analysis, override=True),
        "t1",
        {"analysis_id": "t1", "n": 100, "unchecked_fraction": 0.51},
    )

    assert refused.refused
    assert overridden.outcomes[0].severity is Severity.WARN
    assert overridden.outcomes[0].details["override_applied"] is True
    assert "Chart review" in overridden.outcomes[0].details["override_justification"]


def test_trajectory_unit_and_mixed_model_rules() -> None:
    plan = plan_for(
        {
            "id": "traj",
            "kind": "trajectory",
            "params": {
                "cohort": "exposed",
                "measurement": "lab",
                "mixed_model": True,
            },
        }
    )
    report = validate_analysis_data(
        plan,
        "traj",
        {
            "analysis_id": "traj",
            "n": 29,
            "measurement_units": ["mg/dL", "mmol/L"],
            "median_measurements_per_patient": 2,
        },
    )

    assert outcome_ids(report) == {catalog.MEAS_UNITS, catalog.TRAJECTORY_MIXED_MIN}


def test_causal_minimums_and_static_adjustment_pass() -> None:
    plan = plan_for(
        {
            "id": "causal",
            "kind": "causal_effect",
            "params": {
                "exposed_cohort": "exposed",
                "comparator_cohort": "comparator",
                "outcome": risk_outcome(),
                "adjustment_set": [
                    {"name": "age", "rationale": "Common cause of treatment and outcome."}
                ],
            },
        }
    )
    assert not validate_plan_static(plan).refused

    report = validate_analysis_data(
        plan,
        "causal",
        {
            "analysis_id": "causal",
            "n": 49,
            "arm_n": {"exposed": 19, "comparator": 30},
            "events_total": 9,
        },
    )
    assert catalog.CAUSAL_MIN in outcome_ids(report)


def test_refusal_and_exit_semantics_remain_library_level() -> None:
    outcome = RuleOutcome(
        rule_id="TEST",
        phase="data",
        severity="refuse",
        message="Refused for test",
    )
    assert exit_code_for([outcome]) is ExitCode.REFUSED
    assert exit_code_for(stale_plan=True) is ExitCode.STALE_PLAN
    assert exit_code_for(partial=True) is ExitCode.PARTIAL

    plan = plan_for(
        {
            "id": "s1",
            "kind": "survival",
            "params": {
                "exposed_cohort": "exposed",
                "comparator_cohort": "comparator",
                "outcome": "outcome",
            },
        }
    )
    with pytest.raises(GuardrailRefusal):
        raise_for_refusal(validate_plan_static(plan))


def test_power_note_is_design_stage_mde_not_post_hoc_power() -> None:
    note = minimum_detectable_effect_note(
        DataSummary(analysis_id="a", arm_n={"exposed": 50, "comparator": 50})
    )

    assert note is not None
    assert "minimum detectable standardized effect" in note
    assert "not post-hoc power" in note
