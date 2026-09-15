from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from pheno_rwe.analyses import AnalysisContext
from pheno_rwe.analyses.features import derive_analysis_seed
from pheno_rwe.analyses.prepare import prepare_context_spec
from pheno_rwe.analyses.registry import run as run_analysis
from pheno_rwe.omop.ddl import create_schema


@pytest.fixture
def analysis_database(tmp_path: Path) -> Path:
    path = tmp_path / "deid.duckdb"
    connection = duckdb.connect(str(path))
    create_schema(connection)
    for person_id in range(1, 7):
        connection.execute(
            """INSERT INTO omop.person
               (person_id, gender_concept_id, year_of_birth, race_concept_id,
                ethnicity_concept_id, gender_source_value, race_source_value,
                ethnicity_source_value)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                person_id,
                1 if person_id % 2 else 2,
                1970 + person_id,
                10,
                20,
                "F" if person_id % 2 else "M",
                "race",
                "ethnicity",
            ],
        )
        connection.execute(
            "INSERT INTO study.person_demographic VALUES (?, ?, ?, ?, ?, ?)",
            [person_id, "50-59", 50, 59, False, "2024-01-01"],
        )
        arm = "A" if person_id <= 3 else "B"
        connection.execute(
            "INSERT INTO study.cohort VALUES (?, ?, ?, TRUE, NULL)",
            [arm, person_id, "2024-01-01"],
        )
        connection.execute(
            "INSERT INTO study.cohort VALUES ('All', ?, ?, TRUE, NULL)",
            [person_id, "2024-01-01"],
        )
        connection.execute(
            "INSERT INTO omop.observation_period VALUES (?, ?, ?, ?, ?)",
            [person_id, person_id, "2023-01-01", "2024-12-31", 1],
        )
    code_sets = [
        ("outcome", "condition", 100, "sys#outcome"),
        ("covariate", "condition", 101, "sys#covariate"),
        ("risk", "condition", 103, "sys#risk"),
        ("drugA", "drug", 200, "sys#drugA"),
        ("drugB", "drug", 201, "sys#drugB"),
        ("lab", "measurement", 300, "sys#lab"),
    ]
    for name, domain, concept, source in code_sets:
        connection.execute(
            """INSERT INTO study.code_set
               (code_set_name, domain, concept_id, source_value, mapping_status, accepted)
               VALUES (?, ?, ?, ?, 'MAPPED', TRUE)""",
            [name, domain, concept, source],
        )
    condition_id = 1
    drug_id = 1
    measurement_id = 1
    for person_id in range(1, 7):
        connection.execute(
            """INSERT INTO omop.condition_occurrence
               (condition_occurrence_id, person_id, condition_concept_id,
                condition_start_date, condition_source_value)
               VALUES (?, ?, 101, '2023-12-01', 'sys#covariate')""",
            [condition_id, person_id],
        )
        condition_id += 1
        if person_id % 2:
            connection.execute(
                """INSERT INTO omop.condition_occurrence
                   (condition_occurrence_id, person_id, condition_concept_id,
                    condition_start_date, condition_source_value)
                   VALUES (?, ?, 100, '2024-02-01', 'sys#outcome')""",
                [condition_id, person_id],
            )
            condition_id += 1
        if person_id in {1, 2, 4, 5}:
            connection.execute(
                """INSERT INTO omop.condition_occurrence
                   (condition_occurrence_id, person_id, condition_concept_id,
                    condition_start_date, condition_source_value)
                   VALUES (?, ?, 103, '2023-11-01', 'sys#risk')""",
                [condition_id, person_id],
            )
            condition_id += 1
        drug_concept = 200 if person_id <= 3 else 201
        drug_source = "sys#drugA" if person_id <= 3 else "sys#drugB"
        connection.execute(
            """INSERT INTO omop.drug_exposure
               (drug_exposure_id, person_id, drug_concept_id,
                drug_exposure_start_date, drug_exposure_end_date, drug_source_value)
               VALUES (?, ?, ?, '2024-01-10', '2024-02-10', ?)""",
            [drug_id, person_id, drug_concept, drug_source],
        )
        drug_id += 1
        for day, value in [(0, 10.0 + person_id), (30, 11.0 + person_id), (60, 12.0 + person_id)]:
            connection.execute(
                """INSERT INTO omop.measurement
                   (measurement_id, person_id, measurement_concept_id, measurement_date,
                    value_as_number, measurement_source_value, unit_source_value)
                   VALUES (?, ?, 300, DATE '2024-01-01' + ?, ?, 'sys#lab', 'mg/dL')""",
                [measurement_id, person_id, day, value],
            )
            measurement_id += 1
    connection.close()
    return path


def spec(kind: str, params: dict, identifier: str | None = None) -> dict:
    return {"id": identifier or kind, "kind": kind, "params": params}


def risk_outcome(
    code_set: str = "outcome",
    *,
    start_day: int = 0,
    end_day: int = 90,
    washout_days: int = 0,
) -> dict:
    return {
        "code_set": code_set,
        "risk_window": {
            "start_day": start_day,
            "end_day": end_day,
            "washout_days": washout_days,
        },
    }


def analysis_plan() -> dict:
    return {
        "seed": 99,
        "code_sets": [
            {"name": name, "codings": [{"concept_id": concept, "source_value": source}]}
            for name, concept, source in [
                ("outcome", 100, "sys#outcome"),
                ("covariate", 101, "sys#covariate"),
                ("risk", 103, "sys#risk"),
                ("drugA", 200, "sys#drugA"),
                ("drugB", 201, "sys#drugB"),
                ("lab", 300, "sys#lab"),
            ]
        ],
    }


def analysis_specs() -> list[dict]:
    return [
        spec("table_one", {"cohort": "All", "variables": ["age_bucket", "sex"]}),
        spec(
            "cohort_compare",
            {
                "exposed_cohort": "A",
                "comparator_cohort": "B",
                "outcomes": [risk_outcome()],
                "adjustment_set": [{"name": "risk", "code_set": "risk", "rationale": "confounder"}],
                "adjusted": True,
            },
        ),
        spec(
            "survival",
            {
                "exposed_cohort": "A",
                "comparator_cohort": "B",
                "outcome": "outcome",
                "censor_rule": "end_of_observation",
                "horizon_days": 90,
                "cox": False,
            },
        ),
        spec("incidence_rate", {"cohorts": ["A", "B"], "outcome": "outcome"}),
        spec(
            "treatment_pathways",
            {"cohort": "All", "drug_classes": ["drugA", "drugB"]},
        ),
        spec(
            "trajectory",
            {"cohort": "All", "measurement": "lab", "group_by": "sex"},
        ),
        spec(
            "patient_signature",
            {
                "cohort": "All",
                "feature_domains": ["condition", "drug", "measurement"],
                "min_cluster_size": 2,
            },
        ),
        spec(
            "causal_effect",
            {
                "exposed_cohort": "A",
                "comparator_cohort": "B",
                "outcome": risk_outcome(),
                "adjustment_set": [{"name": "risk", "code_set": "risk", "rationale": "confounder"}],
            },
        ),
    ]


def test_prepare_context_spec_materializes_all_eight_kinds(analysis_database: Path) -> None:
    plan = analysis_plan()
    specifications = analysis_specs()
    prepared = [prepare_context_spec(plan, item, analysis_database) for item in specifications]
    assert [item["kind"] for item in prepared] == [item["kind"] for item in specifications]
    assert all(len(item["params"]["data"]) > 0 for item in prepared)
    comparison = prepared[1]["params"]
    assert comparison["data"]["outcome"].sum() == 3
    assert comparison["adjustment_covariates"] == ["risk"]
    survival = prepared[2]["params"]["data"]
    assert set(survival["event"]) == {0, 1}
    incidence = prepared[3]["params"]
    assert incidence["exposed_value"] == "A"
    assert incidence["comparator_value"] == "B"
    assert incidence["person_time_unit"] == "person_years"
    trajectory = prepared[5]["params"]["data"]
    assert trajectory["time"].tolist()[:3] == [0.0, 30.0, 60.0]
    signature = prepared[6]["params"]
    assert signature["feature_columns"]
    assert signature["matrix_hash"]
    assert signature["feature_seed"] == derive_analysis_seed(99, "patient_signature")
    assert signature["characterization_columns"]
    assert all(column.endswith("|presence") for column in signature["characterization_columns"])


def test_prepared_specs_dispatch_through_all_registered_analyses(
    analysis_database: Path, tmp_path: Path
) -> None:
    pytest.importorskip("umap")
    pytest.importorskip("sklearn")
    pytest.importorskip("statsmodels")

    results = []
    for declared in analysis_specs():
        prepared = prepare_context_spec(analysis_plan(), declared, analysis_database)
        context = AnalysisContext(
            db_path=analysis_database,
            spec=prepared,
            analysis_id=declared["id"],
            seed=derive_analysis_seed(99, declared["id"]),
            output_dir=tmp_path / declared["id"],
        )
        result = run_analysis(context)
        result.write()
        results.append(result)

    assert [result.kind for result in results] == [item["kind"] for item in analysis_specs()]
    assert all((tmp_path / result.analysis_id / "result.json").is_file() for result in results)


def test_patient_signature_uses_omop_events_outside_declared_code_sets(
    analysis_database: Path,
) -> None:
    connection = duckdb.connect(str(analysis_database))
    connection.execute(
        """INSERT INTO omop.condition_occurrence
           (condition_occurrence_id, person_id, condition_concept_id,
            condition_start_date, condition_source_value)
           VALUES (999, 1, 777, '2023-11-01', 'local#uncatalogued'),
                  (1000, 2, 777, '2023-11-01', 'local#uncatalogued'),
                  (1001, 3, 777, '2023-11-01', 'local#uncatalogued')"""
    )
    connection.close()

    prepared = prepare_context_spec(
        {
            "seed": 7,
            "code_sets": [
                {
                    "name": "outcome",
                    "codings": [{"concept_id": 100, "source_value": "sys#outcome"}],
                }
            ],
        },
        spec(
            "patient_signature",
            {"cohort": "All", "feature_domains": ["condition"], "min_cluster_size": 2},
            "signature",
        ),
        analysis_database,
    )

    assert any("concept:777" in column for column in prepared["params"]["feature_columns"])


def test_survival_censor_code_set_and_post_index_event_filter(
    analysis_database: Path,
) -> None:
    connection = duckdb.connect(str(analysis_database))
    connection.execute(
        """INSERT INTO study.code_set
           (code_set_name, domain, concept_id, source_value, mapping_status, accepted)
           VALUES ('competing', 'condition', 102, 'sys#competing', 'MAPPED', TRUE)"""
    )
    connection.execute(
        """INSERT INTO omop.condition_occurrence
           (condition_occurrence_id, person_id, condition_concept_id,
            condition_start_date, condition_source_value)
           VALUES (999, 2, 100, '2023-12-15', 'sys#outcome'),
                  (1000, 2, 102, '2024-01-20', 'sys#competing')"""
    )
    connection.close()

    prepared = prepare_context_spec(
        {},
        spec(
            "survival",
            {
                "exposed_cohort": "A",
                "comparator_cohort": "B",
                "outcome": "outcome",
                "censor_rule": {"strategy": "competing_event", "code_set": "competing"},
                "cox": False,
            },
        ),
        analysis_database,
    )["params"]["data"].set_index("person_id")

    assert prepared.loc[2, "event"] == 0
    assert prepared.loc[2, "duration"] == 19


def test_treatment_pathways_refuses_overlapping_drug_classes(
    analysis_database: Path,
) -> None:
    connection = duckdb.connect(str(analysis_database))
    connection.execute(
        """INSERT INTO study.code_set
           (code_set_name, domain, concept_id, source_value, mapping_status, accepted)
           VALUES ('overlap', 'drug', 200, 'sys#drugA', 'MAPPED', TRUE)"""
    )
    connection.close()

    with pytest.raises(ValueError, match="drug classes overlap"):
        prepare_context_spec(
            {},
            spec(
                "treatment_pathways",
                {"cohort": "All", "drug_classes": ["drugA", "overlap"]},
            ),
            analysis_database,
        )


@pytest.mark.parametrize("kind", ["cohort_compare", "causal_effect"])
def test_binary_risk_window_washout_boundaries_and_observation_censoring(
    analysis_database: Path,
    kind: str,
) -> None:
    connection = duckdb.connect(str(analysis_database))
    connection.execute("DELETE FROM omop.condition_occurrence WHERE condition_concept_id=100")
    connection.execute(
        """UPDATE omop.observation_period
           SET observation_period_end_date='2024-01-21'
           WHERE person_id IN (5, 6)"""
    )
    connection.execute(
        """INSERT INTO omop.condition_occurrence
           (condition_occurrence_id, person_id, condition_concept_id,
            condition_start_date, condition_source_value)
           VALUES (900, 1, 100, '2023-12-01', 'sys#outcome'),
                  (901, 1, 100, '2024-01-11', 'sys#outcome'),
                  (902, 2, 100, '2023-12-02', 'sys#outcome'),
                  (903, 2, 100, '2024-01-11', 'sys#outcome'),
                  (904, 3, 100, '2024-01-31', 'sys#outcome'),
                  (905, 4, 100, '2024-02-01', 'sys#outcome'),
                  (906, 5, 100, '2024-01-21', 'sys#outcome'),
                  (907, 6, 100, '2024-01-22', 'sys#outcome')"""
    )
    connection.close()

    params: dict = {
        "exposed_cohort": "A",
        "comparator_cohort": "B",
    }
    definition = risk_outcome(start_day=10, end_day=30, washout_days=30)
    if kind == "cohort_compare":
        params["outcomes"] = [definition]
    else:
        params["outcome"] = definition
        params["adjustment_set"] = [{"name": "risk", "code_set": "risk", "rationale": "confounder"}]

    prepared = prepare_context_spec(analysis_plan(), spec(kind, params), analysis_database)[
        "params"
    ]["data"].set_index("person_id")["outcome"]

    assert prepared.loc[1] == 1  # pre-washout history ignored; exact risk start included
    assert pd.isna(prepared.loc[2])  # exact washout boundary is prevalent
    assert prepared.loc[3] == 1  # exact risk end included
    assert prepared.loc[4] == 0  # first event after the risk window is not a case
    assert prepared.loc[5] == 1  # event on an early observation end is still observed
    assert pd.isna(prepared.loc[6])  # event after observation end; non-case is censored
