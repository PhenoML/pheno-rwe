"""Translate declarative plan references into flat, read-only analysis datasets.

Plan parameters deliberately name cohorts and code sets rather than SQL columns.
This module is the single deterministic adapter between that scientific plan
contract and the execution-level analysis implementations.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pheno_rwe.analyses.base import AnalysisInputError
from pheno_rwe.analyses.features import build_feature_matrix
from pheno_rwe.steps.resolve_cohort import DOMAIN_TABLES, normalize_domain


def _mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    if isinstance(value, Mapping):
        return dict(value)
    raise AnalysisInputError("plan analysis spec must be a mapping or pydantic model")


def _connect(database: str | Path) -> Any:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - core dependency
        from pheno_rwe.analyses.base import OptionalDependencyError

        raise OptionalDependencyError("duckdb", "analysis dataset preparation") from exc
    path = Path(database)
    if not path.is_file():
        raise AnalysisInputError(f"de-identified database does not exist: {path}")
    return duckdb.connect(str(path), read_only=True)


def _cohort_frame(connection: Any, cohorts: list[str]) -> Any:
    import pandas as pd

    if not cohorts or any(not str(name).strip() for name in cohorts):
        raise AnalysisInputError("at least one named cohort is required")
    placeholders = ", ".join("?" for _ in cohorts)
    frame = connection.execute(
        f"""SELECT c.person_id, c.cohort_name AS \"group\", c.index_date,
                    p.gender_concept_id, p.gender_source_value,
                    p.race_concept_id, p.race_source_value,
                    p.ethnicity_concept_id, p.ethnicity_source_value,
                    p.year_of_birth, d.age_bucket, d.age_bucket_start, d.age_bucket_end
             FROM study.cohort c
             JOIN omop.person p USING (person_id)
             LEFT JOIN study.person_demographic d USING (person_id)
             WHERE c.included AND c.cohort_name IN ({placeholders})
             ORDER BY c.person_id, c.cohort_name""",
        cohorts,
    ).fetchdf()
    found = set(frame["group"].astype(str)) if len(frame) else set()
    missing = [cohort for cohort in cohorts if cohort not in found]
    if missing:
        raise AnalysisInputError(f"included cohort has no patients: {', '.join(missing)}")
    duplicated = frame.loc[frame["person_id"].duplicated(keep=False), ["person_id", "group"]]
    if not duplicated.empty:
        patients = ", ".join(str(value) for value in duplicated["person_id"].unique()[:5])
        raise AnalysisInputError(
            "comparison cohorts overlap; a patient cannot contribute to two arms "
            f"(example person_id values: {patients})"
        )
    frame["index_date"] = pd.to_datetime(frame["index_date"], errors="coerce")
    if frame["index_date"].isna().any():
        raise AnalysisInputError("included cohort rows require an index_date")
    frame["age"] = frame["age_bucket_start"]
    frame["sex"] = frame["gender_source_value"].fillna(frame["gender_concept_id"].astype("string"))
    frame["gender"] = frame["sex"]
    frame["race"] = frame["race_source_value"].fillna(frame["race_concept_id"].astype("string"))
    frame["ethnicity"] = frame["ethnicity_source_value"].fillna(
        frame["ethnicity_concept_id"].astype("string")
    )
    return frame


def _code_set_domain(connection: Any, name: str) -> tuple[str, Any]:
    rows = connection.execute(
        "SELECT DISTINCT domain FROM study.code_set WHERE code_set_name=? AND accepted ORDER BY 1",
        [name],
    ).fetchall()
    if not rows:
        raise AnalysisInputError(f"code set is missing or has no accepted codings: {name}")
    domains = [normalize_domain(str(row[0])) for row in rows]
    if len(set(domains)) != 1:
        raise AnalysisInputError(f"code set '{name}' spans multiple OMOP domains")
    domain = domains[0]
    return domain, DOMAIN_TABLES[domain]


_EVENT_ID_COLUMNS = {
    "condition": "condition_occurrence_id",
    "drug": "drug_exposure_id",
    "procedure": "procedure_occurrence_id",
    "measurement": "measurement_id",
    "observation": "observation_id",
    "visit": "visit_occurrence_id",
}


def _domain_event_frame(
    connection: Any,
    domain: str,
    *,
    where: str = "",
    parameters: list[Any] | None = None,
) -> Any:
    """Load a fixed-registry OMOP event shape without accepting SQL identifiers."""

    domain = normalize_domain(domain)
    table = DOMAIN_TABLES[domain]
    end_column: str | None = None
    value_columns = "NULL::DOUBLE AS value, NULL::VARCHAR AS unit"
    if domain == "drug":
        end_column = "drug_exposure_end_date"
    if domain == "measurement":
        value_columns = "e.value_as_number AS value, e.unit_source_value AS unit"
    id_column = _EVENT_ID_COLUMNS[domain]
    end_expression = f"e.{end_column}" if end_column else f"e.{table.date_column}"
    return connection.execute(
        f"""SELECT e.person_id, e.{id_column} AS event_id,
                    e.{table.date_column} AS event_date,
                    {end_expression} AS end_date,
                    e.{table.concept_column} AS concept_id,
                    e.{table.source_column} AS source_value,
                    {value_columns},
                    ? AS domain,
                    COALESCE((SELECT m.mapping_status FROM meta.mapping m
                              WHERE m.omop_table=? AND m.omop_id=e.{id_column}
                              ORDER BY CASE UPPER(COALESCE(m.mapping_status, ''))
                                  WHEN 'UNMAPPED' THEN 0
                                  WHEN 'UNCHECKED' THEN 1
                                  WHEN 'MAPPED' THEN 2
                                  WHEN 'ALREADY_STANDARD' THEN 3
                                  ELSE 4 END,
                                  m.mapping_status
                              LIMIT 1),
                        CASE WHEN e.{table.concept_column} > 0 THEN 'MAPPED' ELSE 'UNCHECKED' END
                    ) AS mapping_status,
                    COALESCE((SELECT rp.origin FROM meta.row_provenance rp
                              WHERE rp.omop_table=? AND rp.omop_id=e.{id_column}
                              ORDER BY CASE LOWER(COALESCE(rp.origin, ''))
                                  WHEN 'enriched' THEN 0 ELSE 1 END,
                                  rp.origin
                              LIMIT 1), 'structured') AS origin
             FROM omop.{table.table} e
             {where}
             ORDER BY e.person_id, event_date, e.{id_column}""",
        [domain, table.table, table.table, *(parameters or [])],
    ).fetchdf()


def _event_frame(connection: Any, name: str) -> Any:
    domain, table = _code_set_domain(connection, name)
    return _domain_event_frame(
        connection,
        domain,
        where=f"""WHERE EXISTS (SELECT 1 FROM study.code_set cs
                           WHERE cs.code_set_name=? AND cs.accepted
                             AND ((cs.concept_id > 0
                                   AND cs.concept_id=e.{table.concept_column})
                               OR (cs.source_value IS NOT NULL
                                   AND cs.source_value=e.{table.source_column})))""",
        parameters=[name],
    )


def _presence(frame: Any, events: Any, *, post_index: bool = True) -> Any:
    import pandas as pd

    joined = frame[["person_id", "index_date"]].merge(events, on="person_id", how="left")
    joined["event_date"] = pd.to_datetime(joined["event_date"], errors="coerce")
    if post_index:
        matched = joined.loc[joined["event_date"] >= joined["index_date"], "person_id"]
    else:
        relative = (joined["event_date"] - joined["index_date"]).dt.days
        matched = joined.loc[relative.between(-365, -1), "person_id"]
    return frame["person_id"].isin(set(matched)).astype(int)


def _outcome_definition(value: Any) -> tuple[str, dict[str, Any]]:
    """Normalize a plan outcome while refusing temporally ambiguous legacy input."""

    if isinstance(value, str):
        raise AnalysisInputError(
            "binary-risk outcomes must declare code_set and risk_window with start_day, "
            "end_day, and washout_days"
        )
    definition = _mapping(value)
    code_set = str(definition.get("code_set") or "").strip()
    if not code_set:
        raise AnalysisInputError("outcome definition is missing code_set")
    risk_value = definition.get("risk_window")
    if risk_value is None:
        raise AnalysisInputError(f"outcome '{code_set}' is missing risk_window")
    risk_window = _mapping(risk_value)
    required = {"start_day", "end_day", "washout_days"}
    missing = sorted(required - risk_window.keys())
    if missing:
        raise AnalysisInputError(
            f"outcome '{code_set}' risk_window is missing: {', '.join(missing)}"
        )
    try:
        start_day = int(risk_window["start_day"])
        end_day = int(risk_window["end_day"])
        washout_days = int(risk_window["washout_days"])
    except (TypeError, ValueError) as exc:
        raise AnalysisInputError(f"outcome '{code_set}' risk-window days must be integers") from exc
    if start_day < 0 or end_day < start_day or washout_days < 0:
        raise AnalysisInputError(
            f"outcome '{code_set}' requires 0 <= start_day <= end_day and washout_days >= 0"
        )
    observation_censor = risk_window.get("observation_censor", "end_of_observation")
    if observation_censor != "end_of_observation":
        raise AnalysisInputError(
            f"outcome '{code_set}' only supports observation_censor='end_of_observation'"
        )
    return code_set, {
        "start_day": start_day,
        "end_day": end_day,
        "washout_days": washout_days,
        "observation_censor": observation_censor,
    }


def _risk_outcome_presence(frame: Any, events: Any, risk_window: Mapping[str, Any]) -> Any:
    """Materialize fixed-horizon incident outcome status with honest censoring.

    A qualifying event is observed only inside the inclusive risk window and no
    later than the individual's observation end.  An event-free individual is
    a non-case only after completing the full horizon; earlier loss to
    observation remains missing so it cannot be analyzed as a false non-event.
    """

    import pandas as pd

    start_day = int(risk_window["start_day"])
    end_day = int(risk_window["end_day"])
    washout_days = int(risk_window["washout_days"])
    index_date = pd.to_datetime(frame["index_date"], errors="coerce")
    observation_start = pd.to_datetime(frame["observation_start"], errors="coerce")
    observation_end = pd.to_datetime(frame["observation_end"], errors="coerce")
    washout_start = index_date - pd.to_timedelta(washout_days, unit="D")
    risk_start = index_date + pd.to_timedelta(start_day, unit="D")
    risk_end = index_date + pd.to_timedelta(end_day, unit="D")

    joined = frame[["person_id", "index_date", "observation_end"]].merge(
        events[["person_id", "event_date"]], on="person_id", how="left"
    )
    joined["event_date"] = pd.to_datetime(joined["event_date"], errors="coerce")
    joined["washout_start"] = joined["index_date"] - pd.to_timedelta(washout_days, unit="D")
    joined["risk_start"] = joined["index_date"] + pd.to_timedelta(start_day, unit="D")
    joined["risk_end"] = joined["index_date"] + pd.to_timedelta(end_day, unit="D")

    prevalent = joined.loc[
        (joined["event_date"] >= joined["washout_start"])
        & (joined["event_date"] < joined["index_date"]),
        "person_id",
    ]
    observed = joined.loc[
        (joined["event_date"] >= joined["risk_start"])
        & (joined["event_date"] <= joined["risk_end"])
        & (joined["event_date"] <= joined["observation_end"]),
        "person_id",
    ]
    prevalent_ids = set(prevalent)
    observed_ids = set(observed)

    has_required_coverage = (
        observation_start.notna()
        & observation_end.notna()
        & (observation_start <= washout_start)
        & (observation_end >= risk_start)
    )
    incident_eligible = has_required_coverage & ~frame["person_id"].isin(prevalent_ids)
    has_event = incident_eligible & frame["person_id"].isin(observed_ids)
    completed_horizon = incident_eligible & (observation_end >= risk_end)

    result = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    result.loc[completed_horizon] = 0
    result.loc[has_event] = 1
    return result


def _materialize_variable(
    connection: Any,
    frame: Any,
    name: str,
    *,
    code_set: str | None = None,
    post_index: bool = False,
) -> None:
    aliases = {
        "age": "age",
        "age_bucket": "age_bucket",
        "sex": "sex",
        "gender": "gender",
        "race": "race",
        "ethnicity": "ethnicity",
        "gender_concept_id": "gender_concept_id",
        "race_concept_id": "race_concept_id",
        "ethnicity_concept_id": "ethnicity_concept_id",
    }
    source = aliases.get(name) if code_set is None else None
    if source:
        frame[name] = frame[source]
        return
    if code_set is not None and name in aliases:
        raise AnalysisInputError(
            f"covariate '{name}' shadows a built-in demographic; use a distinct name"
        )
    selected_code_set = code_set or name
    events = _event_frame(connection, selected_code_set)
    frame[name] = _presence(frame, events, post_index=post_index)


def _adjustment_columns(
    connection: Any,
    frame: Any,
    adjustment: list[Any],
    *,
    protected: set[str] | None = None,
) -> list[str]:
    columns: list[str] = []
    protected_names = {"person_id", "group", "index_date", *(protected or set())}
    for item in adjustment:
        value = _mapping(item)
        name = str(value.get("name") or value.get("column") or "")
        if not name:
            raise AnalysisInputError("adjustment covariate is missing its name")
        if name in protected_names:
            raise AnalysisInputError(f"adjustment covariate name collides with '{name}'")
        if name in columns:
            raise AnalysisInputError(f"duplicate adjustment covariate name: {name}")
        _materialize_variable(connection, frame, name, code_set=value.get("code_set"))
        columns.append(name)
    return columns


def _table_one(connection: Any, params: dict[str, Any]) -> dict[str, Any]:
    cohort = str(params["cohort"])
    frame = _cohort_frame(connection, [cohort])
    stratify = params.get("stratify_by")
    if stratify:
        _materialize_variable(connection, frame, str(stratify))
        frame["group"] = frame[str(stratify)].fillna("Missing").astype(str)
    else:
        frame["group"] = "Overall"
    variables = [str(value) for value in params.get("variables", [])]
    if not variables:
        variables = ["age_bucket", "sex", "race", "ethnicity"]
    for variable in variables:
        _materialize_variable(connection, frame, variable)
    categorical = [
        variable
        for variable in variables
        if variable in {"age_bucket", "sex", "gender", "race", "ethnicity"}
        or str(frame[variable].dtype) in {"object", "string", "category"}
    ]
    continuous = [variable for variable in variables if variable not in categorical]
    return {
        "data": frame[["person_id", "group", *variables]],
        "group_column": "group",
        "categorical": categorical,
        "continuous": continuous,
        "small_cell_threshold": params.get("small_cell_threshold", 5),
    }


def _comparison_frame(
    connection: Any,
    exposed: str,
    comparator: str,
    outcomes: list[tuple[str, dict[str, Any]]],
    adjustment: list[Any],
) -> tuple[Any, list[str]]:
    frame = _with_observation_window(connection, _cohort_frame(connection, [exposed, comparator]))
    outcome_names = [outcome for outcome, _ in outcomes]
    for outcome, risk_window in outcomes:
        frame[outcome] = _risk_outcome_presence(
            frame, _event_frame(connection, outcome), risk_window
        )
    columns = _adjustment_columns(connection, frame, adjustment, protected=set(outcome_names))
    return frame, columns


def _cohort_compare(connection: Any, params: dict[str, Any]) -> dict[str, Any]:
    exposed = str(params["exposed_cohort"])
    comparator = str(params["comparator_cohort"])
    outcome_definitions = [_outcome_definition(value) for value in params["outcomes"]]
    outcomes = [name for name, _ in outcome_definitions]
    adjustment = list(params.get("adjustment_set", []))
    frame, covariates = _comparison_frame(
        connection, exposed, comparator, outcome_definitions, adjustment
    )
    execution_adjustment = [
        {
            "column": covariates[index],
            "rationale": str(_mapping(item).get("rationale") or "pre-specified covariate"),
        }
        for index, item in enumerate(adjustment)
    ]
    return {
        "data": frame[["person_id", "group", *outcomes, *covariates]],
        "group_column": "group",
        "exposed_value": exposed,
        "comparator_value": comparator,
        "outcomes": [{"column": outcome, "type": "binary"} for outcome in outcomes],
        "binary_test": params.get("exact_test", "fisher"),
        "permutations": params.get("permutations", 10_000),
        "multiplicity": params.get("multiplicity", "auto_bh"),
        "adjusted_estimator": params.get("adjusted_estimator", "firth_logistic"),
        "adjustment_set": execution_adjustment,
        "adjustment_covariates": covariates if params.get("adjusted") or adjustment else [],
    }


def _with_observation_window(connection: Any, frame: Any) -> Any:
    """Attach continuous observed bounds from periods that contain index."""

    import pandas as pd

    periods = connection.execute(
        """SELECT person_id, observation_period_start_date AS observation_start,
                  observation_period_end_date AS observation_end
           FROM omop.observation_period"""
    ).fetchdf()
    periods["observation_start"] = pd.to_datetime(periods["observation_start"], errors="coerce")
    periods["observation_end"] = pd.to_datetime(periods["observation_end"], errors="coerce")
    joined = frame[["person_id", "index_date"]].merge(periods, on="person_id", how="left")
    eligible = joined.loc[
        (joined["observation_start"] <= joined["index_date"])
        & (joined["observation_end"] >= joined["index_date"])
    ]
    windows = eligible.groupby("person_id", sort=True).agg(
        observation_start=("observation_start", "min"),
        observation_end=("observation_end", "max"),
    )
    return frame.merge(windows, on="person_id", how="left")


def _with_observation_end(connection: Any, frame: Any) -> Any:
    """Attach the containing observation end for analyses with legacy contracts."""

    return _with_observation_window(connection, frame).drop(columns=["observation_start"])


def _censor_limit(censor_rule: Any) -> int | None:
    if hasattr(censor_rule, "model_dump"):
        censor_rule = censor_rule.model_dump(mode="python")
    if isinstance(censor_rule, Mapping):
        value = censor_rule.get("max_followup_days")
        return int(value) if value is not None else None
    return None


def _first_post_index_date(frame: Any, events: Any, name: str) -> Any:
    import pandas as pd

    event_dates = events[["person_id", "event_date"]].copy()
    event_dates["event_date"] = pd.to_datetime(event_dates["event_date"], errors="coerce")
    joined = frame[["person_id", "index_date"]].merge(event_dates, on="person_id", how="left")
    joined = joined.loc[
        joined["event_date"].isna() | (joined["event_date"] >= joined["index_date"])
    ]
    first = joined.groupby("person_id", sort=True)["event_date"].min().rename(name)
    return frame.merge(first, on="person_id", how="left")


def _survival(connection: Any, params: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    exposed = str(params["exposed_cohort"])
    comparator = str(params["comparator_cohort"])
    outcome_name = str(params["outcome"])
    frame = _cohort_frame(connection, [exposed, comparator])
    frame = _first_post_index_date(frame, _event_frame(connection, outcome_name), "outcome_date")
    frame = _with_observation_end(connection, frame)
    frame["observation_end"] = pd.to_datetime(frame["observation_end"], errors="coerce")
    censor_rule = params.get("censor_rule")
    rule = (
        _mapping(censor_rule)
        if censor_rule is not None and not isinstance(censor_rule, str)
        else {}
    )
    maximum = _censor_limit(censor_rule)
    if maximum is not None:
        administrative = frame["index_date"] + pd.to_timedelta(maximum, unit="D")
        frame["observation_end"] = frame["observation_end"].where(
            frame["observation_end"].notna() & (frame["observation_end"] < administrative),
            administrative,
        )
    if rule.get("strategy") in {"outcome", "competing_event"}:
        censor_code_set = str(rule.get("code_set") or "")
        if not censor_code_set:
            raise AnalysisInputError(
                f"censor strategy '{rule.get('strategy')}' requires a code_set"
            )
        frame = _first_post_index_date(
            frame,
            _event_frame(connection, censor_code_set),
            "event_censor_date",
        )
        frame["observation_end"] = frame["observation_end"].where(
            frame["event_censor_date"].isna()
            | (
                frame["observation_end"].notna()
                & (frame["observation_end"] <= frame["event_censor_date"])
            ),
            frame["event_censor_date"],
        )
    if frame["observation_end"].isna().any():
        raise AnalysisInputError(
            "survival censoring requires observation_period end dates or max_followup_days"
        )
    if (frame["observation_end"] < frame["index_date"]).any():
        raise AnalysisInputError("observation period ends before cohort index date")
    frame["event"] = (
        frame["outcome_date"].notna() & (frame["outcome_date"] <= frame["observation_end"])
    ).astype(int)
    end = frame["outcome_date"].where(frame["event"].eq(1), frame["observation_end"])
    frame["duration"] = (end - frame["index_date"]).dt.total_seconds() / 86_400
    rule_text = (
        str(censor_rule)
        if isinstance(censor_rule, str)
        else str(rule.get("description") or rule.get("strategy"))
    )
    adjustment = list(params.get("adjustment_set", []))
    covariates = _adjustment_columns(
        connection,
        frame,
        adjustment,
        protected={"duration", "event", "outcome_date", "observation_end"},
    )
    return {
        "data": frame[["person_id", "group", "duration", "event", *covariates]],
        "group_column": "group",
        "exposed_value": exposed,
        "comparator_value": comparator,
        "censor_rule": rule_text,
        "rmst_horizon": params.get("horizon_days"),
        "cox": params.get("cox", True),
        "cox_covariates": covariates,
        "cox_penalizer": params.get("ridge_penalizer"),
    }


def _incidence(connection: Any, params: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    cohorts = [str(value) for value in params["cohorts"]]
    frame = _with_observation_end(connection, _cohort_frame(connection, cohorts))
    frame["observation_end"] = pd.to_datetime(frame["observation_end"], errors="coerce")
    if (
        frame["observation_end"].isna().any()
        or (frame["observation_end"] < frame["index_date"]).any()
    ):
        raise AnalysisInputError("incidence rates require valid post-index observation periods")
    outcome = _event_frame(connection, str(params["outcome"]))
    outcome["event_date"] = pd.to_datetime(outcome["event_date"], errors="coerce")
    joined = frame[["person_id", "index_date", "observation_end"]].merge(
        outcome[["person_id", "event_date"]], on="person_id", how="left"
    )
    valid = joined["event_date"].between(
        joined["index_date"], joined["observation_end"], inclusive="both"
    )
    counts = joined.loc[valid].groupby("person_id").size()
    frame["events"] = frame["person_id"].map(counts).fillna(0).astype(int)
    days = (frame["observation_end"] - frame["index_date"]).dt.total_seconds() / 86_400
    if (days <= 0).any():
        raise AnalysisInputError("incidence person-time must be positive")
    time_scale = params.get("time_scale", "person_years")
    frame["person_time"] = days if time_scale == "days" else days / 365.25
    return {
        "data": frame[["person_id", "group", "events", "person_time"]],
        "group_column": "group",
        "exposed_value": cohorts[0],
        "comparator_value": cohorts[1],
        "event_count_column": "events",
        "person_time_column": "person_time",
        "rate_scale": 1_000.0,
        "person_time_unit": time_scale,
    }


def _pathways(connection: Any, params: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    cohort = str(params["cohort"])
    members = _cohort_frame(connection, [cohort])[["person_id", "index_date"]]
    rows: list[Any] = []
    event_classes: dict[tuple[int, int], str] = {}
    for drug_class in params["drug_classes"]:
        events = _event_frame(connection, str(drug_class))
        if normalize_domain(str(_code_set_domain(connection, str(drug_class))[0])) != "drug":
            raise AnalysisInputError(
                f"treatment pathway class is not a drug code set: {drug_class}"
            )
        for person_id, event_id in events[["person_id", "event_id"]].itertuples(
            index=False, name=None
        ):
            key = (int(person_id), int(event_id))
            previous = event_classes.setdefault(key, str(drug_class))
            if previous != str(drug_class):
                raise AnalysisInputError(
                    "treatment pathway drug classes overlap for an OMOP exposure: "
                    f"{previous}, {drug_class}"
                )
        events["drug_class"] = str(drug_class)
        rows.append(events)
    exposures = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    exposures["event_date"] = pd.to_datetime(exposures["event_date"], errors="coerce")
    exposures["end_date"] = pd.to_datetime(exposures["end_date"], errors="coerce")
    exposures = exposures.merge(members, on="person_id", how="inner")
    exposures = exposures.loc[exposures["event_date"] >= exposures["index_date"]]
    return {
        "data": exposures.rename(columns={"event_date": "start_date"})[
            ["person_id", "drug_class", "start_date", "end_date"]
        ],
        "drug_class_column": "drug_class",
        "start_column": "start_date",
        "end_column": "end_date",
        "drug_classes": [str(value) for value in params["drug_classes"]],
        "permissible_gap_days": params.get("permissible_gap_days", 30),
        "max_lines": params.get("max_lines", 3),
    }


def _trajectory(connection: Any, params: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    cohort = str(params["cohort"])
    members = _cohort_frame(connection, [cohort])
    events = _event_frame(connection, str(params["measurement"]))
    if _code_set_domain(connection, str(params["measurement"]))[0] != "measurement":
        raise AnalysisInputError("trajectory measurement must reference a measurement code set")
    frame = events.merge(members, on="person_id", how="inner")
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    frame["time"] = (frame["event_date"] - frame["index_date"]).dt.total_seconds() / 86_400
    group_by = params.get("group_by")
    group_column = None
    if group_by:
        if str(group_by) in {"person_id", "time", "value", "unit", "event_date"}:
            raise AnalysisInputError(
                f"trajectory group_by collides with required column '{group_by}'"
            )
        _materialize_variable(connection, frame, str(group_by))
        group_column = str(group_by)
    columns = ["person_id", "time", "value", "unit", *([group_column] if group_column else [])]
    frame = frame.dropna(subset=["person_id", "time", "value"])
    if frame.empty:
        raise AnalysisInputError("trajectory found no usable measurements in the cohort")
    return {
        "data": frame[columns],
        "patient_column": "person_id",
        "time_column": "time",
        "value_column": "value",
        "unit_column": "unit",
        "group_column": group_column,
        "mixed_model": params.get("mixed_model", False),
        "unit_harmonization": params.get("unit_harmonization", {}),
    }


def _all_feature_events(connection: Any, domains: list[str]) -> Any:
    import pandas as pd

    requested = {normalize_domain(value) for value in domains}
    frames: list[Any] = []
    for domain in DOMAIN_TABLES:
        if requested and domain not in requested:
            continue
        frames.append(_domain_event_frame(connection, domain))
    if not frames:
        return pd.DataFrame(
            columns=pd.Index(
                [
                    "person_id",
                    "event_id",
                    "event_date",
                    "concept_id",
                    "source_value",
                    "domain",
                    "mapping_status",
                    "origin",
                ]
            )
        )
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["domain", "event_id"], keep="first"
    )


def _patient_signature(
    connection: Any, plan: Any, spec_id: str, params: dict[str, Any]
) -> dict[str, Any]:
    cohort = str(params["cohort"])
    members = _cohort_frame(connection, [cohort])
    events = _all_feature_events(connection, list(params.get("feature_domains", [])))
    plan_value = _mapping(plan)
    code_sets = {item["name"]: item for item in plan_value.get("code_sets", [])}
    baseline = params.get("baseline_window", {"start_day": -365, "end_day": -1})
    post = params.get("post_window", {"start_day": 0, "end_day": 90})
    matrix = build_feature_matrix(
        events,
        members,
        analysis_id=spec_id,
        plan_seed=plan_value.get("seed", 2025),
        windows={"baseline": baseline, "post": post},
        code_sets=code_sets,
        demographic_columns=["age_bucket", "sex", "race", "ethnicity"],
    )
    frame = matrix.matrix.reset_index()
    characterization = matrix.presence_matrix.reset_index()
    return {
        "data": frame,
        "patient_column": "person_id",
        "feature_columns": [str(column) for column in matrix.matrix.columns],
        "characterization_data": characterization,
        "characterization_columns": [str(column) for column in matrix.presence_matrix.columns],
        "feature_provenance": matrix.feature_provenance.to_dict(orient="records"),
        "dropped_features": matrix.dropped_features,
        "matrix_hash": matrix.matrix_hash,
        "feature_seed": matrix.seed,
        "n_neighbors": params.get("n_neighbors"),
        "min_cluster_size": params.get("min_cluster_size", 5),
    }


def _causal(connection: Any, params: dict[str, Any]) -> dict[str, Any]:
    exposed = str(params["exposed_cohort"])
    comparator = str(params["comparator_cohort"])
    outcome_definition = _outcome_definition(params["outcome"])
    outcome = outcome_definition[0]
    adjustment = list(params.get("adjustment_set", []))
    frame, covariates = _comparison_frame(
        connection, exposed, comparator, [outcome_definition], adjustment
    )
    execution_adjustment = [
        {
            "name": str(_mapping(item).get("name")),
            "rationale": str(_mapping(item).get("rationale", "")),
            "code_set": _mapping(item).get("code_set"),
        }
        for item in adjustment
    ]
    return {
        "data": frame[["person_id", "group", outcome, *covariates]],
        "treatment_column": "group",
        "outcome_column": outcome,
        "outcome_type": "binary",
        "exposed_value": exposed,
        "comparator_value": comparator,
        "adjustment_set": execution_adjustment,
        "method": params.get("method", "iptw"),
        "estimand": params.get("estimand", "ate"),
        "weight_trim_quantiles": params.get("weight_trim_quantiles", (0.01, 0.99)),
        "matching_caliper": params.get("matching_caliper"),
    }


def prepare_context_spec(plan: Any, spec: Any, database: str | Path) -> dict[str, Any]:
    """Return an execution spec with one in-memory dataframe and explicit columns.

    The database is always opened read-only and all SQL identifiers come from a
    fixed OMOP registry.  This object is intended to be passed directly as
    ``AnalysisContext.spec``.
    """

    value = _mapping(spec)
    kind = str(value.get("kind", ""))
    spec_id = str(value.get("id") or value.get("analysis_id") or "")
    if not kind:
        raise AnalysisInputError("analysis spec must include a kind")
    if not spec_id:
        raise AnalysisInputError("analysis spec must include a non-empty id")
    params = _mapping(value.get("params", {}))
    builders = {
        "table_one": lambda connection: _table_one(connection, params),
        "cohort_compare": lambda connection: _cohort_compare(connection, params),
        "survival": lambda connection: _survival(connection, params),
        "incidence_rate": lambda connection: _incidence(connection, params),
        "treatment_pathways": lambda connection: _pathways(connection, params),
        "trajectory": lambda connection: _trajectory(connection, params),
        "patient_signature": lambda connection: _patient_signature(
            connection, plan, spec_id, params
        ),
        "causal_effect": lambda connection: _causal(connection, params),
    }
    try:
        builder = builders[kind]
    except KeyError as exc:
        raise AnalysisInputError(
            f"no dataset preparer is registered for analysis kind '{kind}'"
        ) from exc
    connection = _connect(database)
    try:
        execution_params = builder(connection)
        # Keep the declarative adapter and registered execution protocol from
        # drifting independently.  This validates column/parameter shape while
        # the database is still open, before any model is fit or output written.
        from pheno_rwe.analyses.registry import validate_params

        validate_params(kind, execution_params)
    finally:
        connection.close()
    return {"id": spec_id, "kind": kind, "params": execution_params}


prepare_analysis_spec = prepare_context_spec
