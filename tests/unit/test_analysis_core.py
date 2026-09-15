from __future__ import annotations

import pandas as pd
import pytest

from pheno_rwe.analyses import AnalysisContext
from pheno_rwe.analyses.cohort_compare import ANALYSIS as cohort_compare
from pheno_rwe.analyses.features import build_feature_matrix, derive_analysis_seed
from pheno_rwe.analyses.survival import ANALYSIS as survival
from pheno_rwe.analyses.survival import kaplan_meier, restricted_mean_survival
from pheno_rwe.analyses.table_one import ANALYSIS as table_one


def context(tmp_path, kind: str, params: dict, analysis_id: str = "a1") -> AnalysisContext:
    return AnalysisContext(
        db_path=None,
        spec={"id": analysis_id, "kind": kind, "params": params},
        analysis_id=analysis_id,
        seed=771,
        output_dir=tmp_path / analysis_id,
    )


def test_table_one_suppresses_small_cells_and_reports_smd(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "group": ["A"] * 5 + ["B"] * 5,
            "age": [20, 22, 24, 26, None, 30, 32, 34, 36, 38],
            "sex": ["F", "F", "M", "M", "Rare", "F", "F", "M", "M", "M"],
        }
    )
    result = table_one.run(
        context(
            tmp_path,
            "table_one",
            {
                "data": frame,
                "group_column": "group",
                "continuous": ["age"],
                "categorical": ["sex"],
                "small_cell_threshold": 3,
            },
        )
    )
    rare = [
        row
        for row in result.tables["data"]
        if row["variable"] == "sex" and row["level"] == "Rare" and row["group"] == "A"
    ]
    assert rare[0]["display"] == "<3"
    assert rare[0]["suppressed"] is True
    assert rare[0]["value"] is None
    age_smd = next(row for row in result.estimates if row["variable"] == "age")
    assert age_smd["estimate"] == pytest.approx(-3.8105, abs=0.01)
    missing = result.assumptions_checked[1]["counts"]
    assert missing["age"] == 1


def test_cohort_compare_known_two_by_two_and_bh(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "outcome_1": [1] * 7 + [0] * 3 + [1] * 2 + [0] * 8,
            "outcome_2": [1] * 6 + [0] * 4 + [1] * 3 + [0] * 7,
        }
    )
    result = cohort_compare.run(
        context(
            tmp_path,
            "cohort_compare",
            {
                "data": frame,
                "group_column": "group",
                "exposed_value": "A",
                "comparator_value": "B",
                "outcomes": [
                    {"column": "outcome_1", "type": "binary"},
                    {"column": "outcome_2", "type": "binary"},
                ],
                "permutations": 99,
                "bootstrap_iterations": 100,
            },
        )
    )
    odds = next(
        row
        for row in result.estimates
        if row["outcome"] == "outcome_1" and row["estimand"] == "odds_ratio"
    )
    difference = next(
        row
        for row in result.estimates
        if row["outcome"] == "outcome_1" and row["estimand"] == "risk_difference"
    )
    assert odds["estimate"] == pytest.approx(8.1538354145)
    assert odds["p_value"] == pytest.approx(0.0697785187)
    assert difference["estimate"] == pytest.approx(0.5)
    assert difference["lower"] < difference["estimate"] < difference["upper"]
    assert all(row["q_value"] >= row["p_value"] for row in result.estimates)


def test_native_kaplan_meier_and_rmst_are_known() -> None:
    curve = kaplan_meier([1, 2, 2, 3], [1, 1, 0, 1])
    assert [(row["time"], row["at_risk"], row["events"]) for row in curve] == [
        (0.0, 4, 0),
        (1.0, 4, 1),
        (2.0, 3, 1),
        (3.0, 1, 1),
    ]
    assert curve[1]["survival"] == pytest.approx(0.75)
    assert curve[2]["survival"] == pytest.approx(0.5)
    assert restricted_mean_survival(curve, 3) == pytest.approx(2.25)


def test_survival_result_has_rmst_logrank_and_curve_csv(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "group": ["A"] * 6 + ["B"] * 6,
            "duration": [1, 2, 3, 4, 5, 6] * 2,
            "event": [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        }
    )
    result = survival.run(
        context(
            tmp_path,
            "survival",
            {
                "data": frame,
                "censor_rule": "end of observation",
                "rmst_horizon": 6,
                "bootstrap_iterations": 100,
                "cox": False,
            },
        )
    )
    assert [row["estimand"] for row in result.estimates] == [
        "rmst_difference",
        "survival_curve_difference",
    ]
    assert (
        result.estimates[0]["lower"]
        <= result.estimates[0]["estimate"]
        <= result.estimates[0]["upper"]
    )
    assert set(row["group"] for row in result.tables["data"]) == {"A", "B"}
    assert "minimum detectable effect" in result.power_note.lower()


def test_feature_matrix_keeps_unchecked_source_value_and_is_deterministic() -> None:
    cohort = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "index_date": ["2024-01-10"] * 3,
            "age_bucket": ["50-54", "55-59", "60-64"],
        }
    )
    events = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 1],
            "event_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-11"],
            "concept_id": [0, 0, 0, 99],
            "source_value": ["local#x", "local#x", "local#x", "std#99"],
            "domain": ["condition"] * 4,
            "mapping_status": ["UNCHECKED"] * 3 + ["MAPPED"],
        }
    )
    first = build_feature_matrix(
        events,
        cohort,
        analysis_id="sig",
        plan_seed=42,
        prevalence_floor=3,
        demographic_columns=["age_bucket"],
    )
    second = build_feature_matrix(
        events,
        cohort,
        analysis_id="sig",
        plan_seed=42,
        prevalence_floor=3,
        demographic_columns=["age_bucket"],
    )
    assert any("source:local#x" in column for column in first.matrix.columns)
    assert set(first.feature_provenance["mapping_status"]) == {"UNCHECKED"}
    assert first.matrix_hash == second.matrix_hash
    assert first.seed == second.seed == derive_analysis_seed(42, "sig")
