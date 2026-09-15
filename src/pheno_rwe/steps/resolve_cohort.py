"""Deterministic code-set matching, index dates, and cohort attrition."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from pheno_rwe.hashing import canonical_json, hash_file, hash_json
from pheno_rwe.manifest import record_step
from pheno_rwe.omop.ddl import TABLE_SPECS, create_schema, quote_identifier
from pheno_rwe.omop.loader import connect_database
from pheno_rwe.plan.io import analysis_data_hash, load_plan, parse_plan, plan_hash
from pheno_rwe.plan.schema import CodeSet, CohortDef, ConceptCriterion, IndexDateRule, Plan
from pheno_rwe.steps.common import StepResult
from pheno_rwe.workspace import StudyWorkspace, find_study, study_lock


@dataclass(frozen=True, slots=True)
class Event:
    person_id: int
    event_date: date | None


@dataclass(frozen=True, slots=True)
class CohortRow:
    cohort_name: str
    person_id: int
    index_date: date | None
    included: bool
    exclusion_reason: str | None


@dataclass(frozen=True, slots=True)
class AttritionRow:
    cohort_name: str
    stage_order: int
    stage: str
    remaining: int
    removed: int
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class CohortResolution:
    rows: tuple[CohortRow, ...]
    attrition: tuple[AttritionRow, ...]

    @property
    def included_count(self) -> int:
        return sum(row.included for row in self.rows)


@dataclass(frozen=True, slots=True)
class DomainTable:
    table: str
    concept_column: str
    source_column: str
    date_column: str


DOMAIN_TABLES: dict[str, DomainTable] = {
    "condition": DomainTable(
        "condition_occurrence",
        "condition_concept_id",
        "condition_source_value",
        "condition_start_date",
    ),
    "drug": DomainTable(
        "drug_exposure", "drug_concept_id", "drug_source_value", "drug_exposure_start_date"
    ),
    "procedure": DomainTable(
        "procedure_occurrence", "procedure_concept_id", "procedure_source_value", "procedure_date"
    ),
    "measurement": DomainTable(
        "measurement", "measurement_concept_id", "measurement_source_value", "measurement_date"
    ),
    "observation": DomainTable(
        "observation", "observation_concept_id", "observation_source_value", "observation_date"
    ),
    "visit": DomainTable(
        "visit_occurrence", "visit_concept_id", "visit_source_value", "visit_start_date"
    ),
}

DOMAIN_ALIASES = {
    "condition occurrence": "condition",
    "condition_occurrence": "condition",
    "drug exposure": "drug",
    "drug_exposure": "drug",
    "procedure occurrence": "procedure",
    "procedure_occurrence": "procedure",
    "visit occurrence": "visit",
    "visit_occurrence": "visit",
}


def normalize_domain(value: str) -> str:
    normalized = value.strip().lower()
    normalized = DOMAIN_ALIASES.get(normalized, normalized)
    if normalized not in DOMAIN_TABLES:
        raise ValueError(
            f"Unsupported code-set domain {value!r}; expected one of {', '.join(DOMAIN_TABLES)}"
        )
    return normalized


def sync_code_sets(connection: Any, code_sets: Iterable[CodeSet]) -> None:
    connection.execute("DELETE FROM study.code_set")
    rows: list[list[Any]] = []
    for code_set in code_sets:
        domain = normalize_domain(code_set.domain)
        for coding in code_set.codings:
            rows.append(
                [
                    code_set.name,
                    domain,
                    coding.concept_id,
                    coding.system,
                    coding.code,
                    coding.source_value,
                    coding.display,
                    coding.mapping_status,
                    coding.accepted,
                ]
            )
    if rows:
        connection.executemany(
            "INSERT INTO study.code_set VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
        )


def code_set_events(connection: Any, code_set_name: str) -> list[Event]:
    rows = connection.execute(
        """SELECT domain, concept_id, source_value
           FROM study.code_set WHERE code_set_name = ? AND accepted""",
        [code_set_name],
    ).fetchall()
    if not rows:
        raise ValueError(f"Code set {code_set_name!r} is missing or has no accepted codings")
    grouped: dict[str, dict[str, set[Any]]] = defaultdict(
        lambda: {"concept_ids": set(), "source_values": set()}
    )
    for domain, concept_id, source_value in rows:
        normalized = normalize_domain(str(domain))
        if concept_id not in (None, 0):
            grouped[normalized]["concept_ids"].add(int(concept_id))
        if source_value:
            grouped[normalized]["source_values"].add(str(source_value))

    result: list[Event] = []
    for domain, matches in grouped.items():
        spec = DOMAIN_TABLES[domain]
        clauses: list[str] = []
        params: list[Any] = []
        concept_ids = sorted(matches["concept_ids"])
        source_values = sorted(matches["source_values"])
        if concept_ids:
            placeholders = ", ".join("?" for _ in concept_ids)
            clauses.append(f"{quote_identifier(spec.concept_column)} IN ({placeholders})")
            params.extend(concept_ids)
        if source_values:
            placeholders = ", ".join("?" for _ in source_values)
            clauses.append(f"{quote_identifier(spec.source_column)} IN ({placeholders})")
            params.extend(source_values)
        if not clauses:
            continue
        if spec.table not in TABLE_SPECS:
            raise AssertionError(f"Unregistered event table {spec.table}")
        query = (
            f"SELECT person_id, {quote_identifier(spec.date_column)} "
            f"FROM omop.{quote_identifier(spec.table)} WHERE " + " OR ".join(clauses)
        )
        result.extend(
            Event(int(person_id), event_date)
            for person_id, event_date in connection.execute(query, params).fetchall()
        )
    return result


def derive_index_dates(
    person_ids: Iterable[int],
    rule: IndexDateRule,
    events: Mapping[str, list[Event]],
) -> dict[int, date | None]:
    if rule.strategy == "fixed_date":
        return {person_id: rule.fixed_date for person_id in person_ids}
    assert rule.code_set is not None
    by_person: dict[int, list[date]] = defaultdict(list)
    for event in events[rule.code_set]:
        if event.event_date is not None:
            by_person[event.person_id].append(event.event_date)
    chooser = min if rule.strategy == "first_occurrence" else max
    return {
        person_id: chooser(by_person[person_id]) if by_person[person_id] else None
        for person_id in person_ids
    }


def criterion_matches(
    criterion: ConceptCriterion,
    person_id: int,
    index_date: date | None,
    events: Mapping[str, list[Event]],
) -> bool:
    matches = [event for event in events[criterion.code_set] if event.person_id == person_id]
    if criterion.window is not None:
        if criterion.window.anchor != "index_date":
            raise ValueError(
                f"Unsupported temporal anchor {criterion.window.anchor!r}; "
                "only index_date is available"
            )
        if index_date is None:
            return False
        earliest = (
            index_date - timedelta(days=criterion.window.days_before)
            if criterion.window.days_before is not None
            else None
        )
        latest = (
            index_date + timedelta(days=criterion.window.days_after)
            if criterion.window.days_after is not None
            else None
        )
        matches = [
            event
            for event in matches
            if event.event_date is not None
            and (earliest is None or event.event_date >= earliest)
            and (latest is None or event.event_date <= latest)
        ]
    return len(matches) >= criterion.min_count


def _demographic_reason(
    person: tuple[Any, ...], index_date: date | None, cohort: CohortDef
) -> str | None:
    _, gender, year_of_birth, race, ethnicity = person
    demographics = cohort.demographics
    if demographics.sex and str(gender or "").lower() not in {
        value.lower() for value in demographics.sex
    }:
        return "demographics:sex"
    if demographics.race and str(race or "").lower() not in {
        value.lower() for value in demographics.race
    }:
        return "demographics:race"
    if demographics.ethnicity and str(ethnicity or "").lower() not in {
        value.lower() for value in demographics.ethnicity
    }:
        return "demographics:ethnicity"
    if demographics.min_age is not None or demographics.max_age is not None:
        if index_date is None or year_of_birth is None:
            return "demographics:age_missing"
        age = index_date.year - int(year_of_birth)
        if demographics.min_age is not None and age < demographics.min_age:
            return "demographics:min_age"
        if demographics.max_age is not None and age > demographics.max_age:
            return "demographics:max_age"
    return None


def resolve_one_cohort(
    connection: Any,
    cohort: CohortDef,
    default_index_rule: IndexDateRule,
    events: Mapping[str, list[Event]],
) -> CohortResolution:
    people = connection.execute(
        """SELECT person_id, gender_source_value, year_of_birth,
                  race_source_value, ethnicity_source_value
           FROM omop.person ORDER BY person_id"""
    ).fetchall()
    person_ids = [int(person[0]) for person in people]
    rule = cohort.index_date_rule or default_index_rule
    indexes = derive_index_dates(person_ids, rule, events)
    reasons: dict[int, str | None] = {person_id: None for person_id in person_ids}
    attrition: list[AttritionRow] = [
        AttritionRow(cohort.name, 0, "source population", len(person_ids), 0)
    ]

    def exclude(stage: str, predicate: Any, detail: str | None = None) -> None:
        before = sum(reason is None for reason in reasons.values())
        for person_id in person_ids:
            if reasons[person_id] is None and predicate(person_id):
                reasons[person_id] = stage
        after = sum(reason is None for reason in reasons.values())
        attrition.append(
            AttritionRow(cohort.name, len(attrition), stage, after, before - after, detail)
        )

    if rule.strategy != "fixed_date":
        exclude("missing_index_date", lambda person_id: indexes[person_id] is None)
    for criterion in cohort.inclusion:
        exclude(
            f"inclusion:{criterion.code_set}",
            lambda person_id, criterion=criterion: (
                not criterion_matches(criterion, person_id, indexes[person_id], events)
            ),
            f"required count >= {criterion.min_count}",
        )
    people_by_id = {int(person[0]): person for person in people}
    exclude(
        "demographics",
        lambda person_id: (
            _demographic_reason(people_by_id[person_id], indexes[person_id], cohort) is not None
        ),
    )
    # Replace the generic demographics reason with its precise reason while
    # retaining a single readable attrition stage.
    for person_id, reason in list(reasons.items()):
        if reason == "demographics":
            reasons[person_id] = _demographic_reason(
                people_by_id[person_id], indexes[person_id], cohort
            )
    for criterion in cohort.exclusion:
        exclude(
            f"exclusion:{criterion.code_set}",
            lambda person_id, criterion=criterion: criterion_matches(
                criterion, person_id, indexes[person_id], events
            ),
            f"excluded at count >= {criterion.min_count}",
        )
    rows = tuple(
        CohortRow(
            cohort.name,
            person_id,
            indexes[person_id],
            reasons[person_id] is None,
            reasons[person_id],
        )
        for person_id in person_ids
    )
    return CohortResolution(rows, tuple(attrition))


def resolve_plan_cohorts(connection: Any, plan: Plan | Mapping[str, Any]) -> CohortResolution:
    parsed = parse_plan(plan)
    create_schema(connection)
    sync_code_sets(connection, parsed.code_sets)
    required_names = {
        criterion.code_set
        for cohort in parsed.cohorts
        for criterion in [*cohort.inclusion, *cohort.exclusion]
    }
    if parsed.index_date_rule.code_set:
        required_names.add(parsed.index_date_rule.code_set)
    required_names.update(
        cohort.index_date_rule.code_set
        for cohort in parsed.cohorts
        if cohort.index_date_rule and cohort.index_date_rule.code_set
    )
    events = {name: code_set_events(connection, name) for name in sorted(required_names)}
    all_rows: list[CohortRow] = []
    all_attrition: list[AttritionRow] = []
    for cohort in parsed.cohorts:
        result = resolve_one_cohort(connection, cohort, parsed.index_date_rule, events)
        all_rows.extend(result.rows)
        all_attrition.extend(result.attrition)
    connection.execute("DELETE FROM study.cohort")
    connection.execute("DELETE FROM study.attrition")
    if all_rows:
        connection.executemany(
            "INSERT INTO study.cohort VALUES (?, ?, ?, ?, ?)",
            [
                [row.cohort_name, row.person_id, row.index_date, row.included, row.exclusion_reason]
                for row in all_rows
            ],
        )
    if all_attrition:
        connection.executemany(
            "INSERT INTO study.attrition VALUES (?, ?, ?, ?, ?, ?)",
            [
                [
                    row.cohort_name,
                    row.stage_order,
                    row.stage,
                    row.remaining,
                    row.removed,
                    row.detail,
                ]
                for row in all_attrition
            ],
        )
    return CohortResolution(tuple(all_rows), tuple(all_attrition))


def resolve_cohort(
    study: StudyWorkspace | str | Path,
    *,
    plan: Plan | Mapping[str, Any] | None = None,
) -> StepResult:
    workspace = study if isinstance(study, StudyWorkspace) else find_study(study)
    started = datetime.now(UTC)
    with study_lock(workspace):
        # Load and fingerprint one plan snapshot while holding the study writer
        # lock.  These exact values are the provenance contract consumed by deid.
        parsed = parse_plan(plan) if plan is not None else load_plan(workspace.plan_path)
        resolved_plan_hash = plan_hash(parsed)
        resolved_analysis_data_hash = analysis_data_hash(parsed)
        plan_inputs = (
            {workspace.relative(workspace.plan_path): hash_file(workspace.plan_path)}
            if plan is None and workspace.plan_path.exists()
            else {}
        )
        connection = connect_database(workspace.identified_db)
        try:
            connection.execute("BEGIN TRANSACTION")
            result = resolve_plan_cohorts(connection, parsed)
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        report_payload = {
            "cohorts": [
                {
                    "name": name,
                    "included": sum(row.included for row in result.rows if row.cohort_name == name),
                    "total": sum(1 for row in result.rows if row.cohort_name == name),
                    "attrition": [
                        asdict(row) for row in result.attrition if row.cohort_name == name
                    ],
                }
                for name in sorted({row.cohort_name for row in result.rows})
            ]
        }
        report = workspace.root / "reports" / "cohort_attrition.json"
        report.write_text(canonical_json(report_payload) + "\n", encoding="utf-8")
        tidy_report = workspace.root / "reports" / "cohort_attrition.csv"
        with tidy_report.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "cohort",
                    "stage_order",
                    "stage",
                    "n",
                    "excluded",
                    "detail",
                ],
            )
            writer.writeheader()
            writer.writerows(
                {
                    "cohort": row.cohort_name,
                    "stage_order": row.stage_order,
                    "stage": row.stage,
                    "n": row.remaining,
                    "excluded": row.removed,
                    "detail": row.detail,
                }
                for row in result.attrition
            )
        identified_database_hash = hash_file(workspace.identified_db)
        outputs = {
            workspace.relative(workspace.identified_db): identified_database_hash,
            workspace.relative(report): hash_file(report),
            workspace.relative(tidy_report): hash_file(tidy_report),
        }
        items = [asdict(row) for row in result.attrition]
        record_step(
            workspace.manifest_path,
            step="resolve-cohort",
            status="success",
            started_at=started,
            params={
                "cohorts": [cohort.name for cohort in parsed.cohorts],
                "plan_hash": resolved_plan_hash,
                "analysis_data_sha256": resolved_analysis_data_hash,
                "identified_database_sha256": identified_database_hash,
            },
            input_signature=hash_json(
                {
                    "plan_hash": resolved_plan_hash,
                    "analysis_data_sha256": resolved_analysis_data_hash,
                }
            ),
            inputs=plan_inputs,
            outputs=outputs,
            items=items,
        )
    return StepResult(
        "resolve-cohort",
        "success",
        f"Resolved {len(parsed.cohorts)} cohorts; {result.included_count} included memberships.",
        outputs,
        items,
    )


resolve_cohorts = resolve_cohort
