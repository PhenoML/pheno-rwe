"""Preregister a plan and run STATIC plus available aggregate DATA guardrails."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pheno_rwe.guardrails import (
    DataSummary,
    GuardrailPhase,
    RuleOutcome,
    Severity,
    ValidationReport,
    catalog,
    validate_plan,
)
from pheno_rwe.hashing import canonical_json, hash_file
from pheno_rwe.manifest import read_manifest, record_step, verify_manifest
from pheno_rwe.omop.loader import connect_database
from pheno_rwe.plan import code_set_review_hash, load_plan, normalized_plan_dict, plan_hash
from pheno_rwe.plan.schema import Plan
from pheno_rwe.workspace import StudyWorkspace, find_study, study_lock


@dataclass(slots=True)
class ValidateResult:
    status: str
    report: dict[str, Any]
    output_path: str
    exit_code: int

    @property
    def step(self) -> str:
        return "validate"

    def model_dump(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "status": self.status,
            "report": self.report,
            "output_path": self.output_path,
            "exit_code": self.exit_code,
        }


def last_validated_plan_hash(workspace: StudyWorkspace) -> str | None:
    for entry in reversed(read_manifest(workspace.manifest_path)):
        if entry.get("status") != "success":
            continue
        if entry.get("step") == "plan_amended":
            return entry.get("params", {}).get("new_hash")
        if entry.get("step") == "plan_validated":
            return entry.get("params", {}).get("plan_hash")
    return None


def _plan_diff(before: Any, after: Any, prefix: str = "") -> list[dict[str, Any]]:
    """Compact semantic diff stored with a preregistration amendment."""
    if isinstance(before, dict) and isinstance(after, dict):
        changes: list[dict[str, Any]] = []
        for key in sorted(set(before) | set(after)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in before:
                changes.append({"path": path, "change": "added", "after": after[key]})
            elif key not in after:
                changes.append({"path": path, "change": "removed", "before": before[key]})
            else:
                changes.extend(_plan_diff(before[key], after[key], path))
        return changes
    if isinstance(before, list) and isinstance(after, list):
        if before == after:
            return []
        return [{"path": prefix, "change": "replaced", "before": before, "after": after}]
    if before != after:
        return [{"path": prefix, "change": "changed", "before": before, "after": after}]
    return []


def aggregate_summaries(database: Path, plan: Plan) -> dict[str, DataSummary]:
    """Read only aggregate counts needed by deterministic DATA rules."""
    if not database.exists():
        return {}
    connection = connect_database(database, read_only=True)
    summaries: dict[str, DataSummary] = {}
    try:
        cohort_counts = {
            str(name): int(count)
            for name, count in connection.execute(
                "SELECT cohort_name, COUNT(DISTINCT person_id) FROM study.cohort "
                "WHERE included GROUP BY cohort_name"
            ).fetchall()
        }
        for analysis in plan.analyses:
            if not analysis.enabled:
                continue
            params = analysis.params
            unchecked_fraction = _analysis_unchecked_fraction(connection, plan, analysis)
            arm_n: dict[str, int] = {}
            for field in ("exposed_cohort", "comparator_cohort"):
                value = getattr(params, field, None)
                if value:
                    arm_n[str(value)] = cohort_counts.get(str(value), 0)
            cohort = getattr(params, "cohort", None)
            cohorts = getattr(params, "cohorts", None)
            if cohort:
                total_n = cohort_counts.get(str(cohort), 0)
            elif cohorts:
                total_n = sum(cohort_counts.get(str(name), 0) for name in cohorts)
            elif arm_n:
                total_n = sum(arm_n.values())
            else:
                total_n = len(
                    {
                        row[0]
                        for row in connection.execute(
                            "SELECT DISTINCT person_id FROM study.cohort WHERE included"
                        ).fetchall()
                    }
                )
            outcome = getattr(params, "outcome", None)
            outcomes = getattr(params, "outcomes", None)
            risk_outcomes: list[Any] = []
            if analysis.kind == "cohort_compare":
                risk_outcomes = list(outcomes or [])
            elif analysis.kind == "causal_effect" and outcome is not None:
                risk_outcomes = [outcome]
            outcome_names = (
                [_outcome_code_set(item) for item in risk_outcomes]
                if risk_outcomes
                else ([str(outcome)] if outcome else [str(item) for item in outcomes or []])
            )
            if analysis.kind in {"survival", "causal_effect"} and not outcome_names:
                raise RuntimeError(
                    f"Analysis '{analysis.id}' requires an outcome code set for DATA guardrails."
                )
            events_total: int | None = None
            events_by_arm: dict[str, int] = {}
            if risk_outcomes:
                risk_values = [
                    _risk_outcome_counts(connection, item, arm_n) for item in risk_outcomes
                ]
                if any(value[0] is None for value in risk_values):
                    raise RuntimeError(
                        f"Could not derive outcome event counts for analysis '{analysis.id}'."
                    )
                # Every outcome must independently clear the DATA gate.  The
                # scalar summary contract therefore uses conservative minima.
                events_total = min(int(value[0] or 0) for value in risk_values)
                events_by_arm = {
                    arm: min(value[1].get(arm, 0) for value in risk_values) for arm in arm_n
                }
                arm_n = {arm: min(value[3].get(arm, 0) for value in risk_values) for arm in arm_n}
                total_n = sum(arm_n.values())
                events_by_arm = {
                    arm: min(events_by_arm.get(arm, 0), count) for arm, count in arm_n.items()
                }
                events_total = min(events_total, sum(events_by_arm.values()), total_n)
            elif outcome_names:
                event_values = [
                    _event_counts(connection, outcome_name, arm_n) for outcome_name in outcome_names
                ]
                if any(value[0] is None for value in event_values):
                    raise RuntimeError(
                        f"Could not derive outcome event counts for analysis '{analysis.id}'."
                    )
                # Guard on the sparsest prespecified outcome; otherwise a dense
                # secondary outcome can hide an underpowered primary contrast.
                events_total, events_by_arm = min(
                    event_values,
                    key=lambda value: int(value[0] or 0),
                )
            units: tuple[str, ...] = ()
            median_measurements: float | None = None
            if analysis.kind == "trajectory":
                units, median_measurements = _measurement_summary(
                    connection, str(getattr(params, "measurement", "")), str(cohort or "")
                )
            parameters = None
            adjustment = getattr(params, "adjustment_set", None)
            if adjustment:
                parameters = len(adjustment) + 1
            summaries[analysis.id] = DataSummary(
                analysis_id=analysis.id,
                total_n=total_n,
                arm_n=arm_n,
                events_total=events_total,
                events_by_arm=events_by_arm,
                model_parameters=parameters,
                unchecked_fraction=unchecked_fraction,
                measurement_units=units,
                median_measurements_per_patient=median_measurements,
                outcome_count=len(outcomes) if outcomes else (1 if outcome else None),
            )
    finally:
        connection.close()
    return summaries


def _unchecked_fraction(connection: Any) -> float:
    try:
        row = connection.execute(
            "SELECT COUNT(*) FILTER (WHERE mapping_status = 'UNCHECKED'), COUNT(*) "
            "FROM meta.mapping"
        ).fetchone()
    except Exception as exc:
        raise RuntimeError(
            "Could not derive mapping-provenance counts for DATA guardrails."
        ) from exc
    return (float(row[0]) / float(row[1])) if row and row[1] else 0.0


def _analysis_unchecked_fraction(connection: Any, plan: Plan, analysis: Any) -> float:
    """Scope provenance to every code set/domain that can affect one analysis."""

    from pheno_rwe.steps.resolve_cohort import normalize_domain

    params = analysis.params
    known_code_sets = {code_set.name for code_set in plan.code_sets}
    names: set[str] = set()
    for field in ("outcome", "measurement", "stratify_by", "group_by"):
        value = getattr(params, field, None)
        candidate = _outcome_code_set(value) if field == "outcome" and value else str(value or "")
        if candidate in known_code_sets:
            names.add(candidate)
    for field in ("outcomes", "drug_classes", "variables"):
        names.update(
            _outcome_code_set(item) if field == "outcomes" else str(item)
            for item in (getattr(params, field, None) or [])
            if (_outcome_code_set(item) if field == "outcomes" else str(item)) in known_code_sets
        )
    for covariate in getattr(params, "adjustment_set", None) or []:
        candidate = getattr(covariate, "code_set", None) or getattr(covariate, "name", None)
        if candidate and str(candidate) in known_code_sets:
            names.add(str(candidate))

    cohort_names: set[str] = set()
    for field in ("cohort", "exposed_cohort", "comparator_cohort"):
        value = getattr(params, field, None)
        if value:
            cohort_names.add(str(value))
    cohort_names.update(str(item) for item in (getattr(params, "cohorts", None) or []))
    for cohort in plan.cohorts:
        if cohort.name not in cohort_names:
            continue
        names.update(item.code_set for item in [*cohort.inclusion, *cohort.exclusion])
        index_rule = cohort.index_date_rule or plan.index_date_rule
        if index_rule.code_set:
            names.add(index_rule.code_set)

    tables: set[str] = set()
    for domain in getattr(params, "feature_domains", None) or []:
        normalized = normalize_domain(str(domain))
        mapping = _domain_table(normalized)
        if mapping:
            tables.add(mapping[0])

    clauses: list[str] = []
    parameters: list[Any] = []
    if names:
        placeholders = ",".join("?" for _ in names)
        clauses.append(
            f"""EXISTS (SELECT 1 FROM study.code_set cs
                    WHERE cs.code_set_name IN ({placeholders}) AND cs.accepted
                      AND cs.source_system=m.source_system
                      AND cs.source_code=m.source_code)"""
        )
        parameters.extend(sorted(names))
    if tables:
        placeholders = ",".join("?" for _ in tables)
        clauses.append(f"m.omop_table IN ({placeholders})")
        parameters.extend(sorted(tables))
    if not clauses:
        return 0.0
    try:
        row = connection.execute(
            f"""SELECT COUNT(*) FILTER (WHERE m.mapping_status = 'UNCHECKED'), COUNT(*)
                FROM meta.mapping m
                WHERE {" OR ".join(f"({clause})" for clause in clauses)}""",
            parameters,
        ).fetchone()
    except Exception as exc:
        raise RuntimeError(
            f"Could not derive mapping provenance for analysis '{analysis.id}'."
        ) from exc
    return float(row[0]) / float(row[1]) if row and row[1] else 0.0


def _domain_table(domain: str) -> tuple[str, str, str, str] | None:
    return {
        "condition": (
            "condition_occurrence",
            "condition_concept_id",
            "condition_source_value",
            "condition_start_date",
        ),
        "drug": (
            "drug_exposure",
            "drug_concept_id",
            "drug_source_value",
            "drug_exposure_start_date",
        ),
        "procedure": (
            "procedure_occurrence",
            "procedure_concept_id",
            "procedure_source_value",
            "procedure_date",
        ),
        "measurement": (
            "measurement",
            "measurement_concept_id",
            "measurement_source_value",
            "measurement_date",
        ),
        "observation": (
            "observation",
            "observation_concept_id",
            "observation_source_value",
            "observation_date",
        ),
        "visit": (
            "visit_occurrence",
            "visit_concept_id",
            "visit_source_value",
            "visit_start_date",
        ),
    }.get(domain.lower())


def _outcome_code_set(value: Any) -> str:
    if value is None:
        return ""
    code_set = getattr(value, "code_set", None)
    if code_set is not None:
        return str(code_set)
    if isinstance(value, dict):
        return str(value.get("code_set") or "")
    return str(value)


def _risk_outcome_counts(
    connection: Any,
    outcome: Any,
    arm_n: dict[str, int],
) -> tuple[int | None, dict[str, int], int, dict[str, int]]:
    """Aggregate fixed-horizon cases and analyzable denominators by arm."""

    code_set = _outcome_code_set(outcome)
    risk_window = getattr(outcome, "risk_window", None)
    if risk_window is None and isinstance(outcome, dict):
        risk_window = outcome.get("risk_window")
    if risk_window is not None and hasattr(risk_window, "model_dump"):
        risk_window = risk_window.model_dump(mode="python")
    if not isinstance(risk_window, dict):
        raise RuntimeError(f"Outcome '{code_set}' is missing its risk-window contract.")
    try:
        start_day = int(risk_window["start_day"])
        end_day = int(risk_window["end_day"])
        washout_days = int(risk_window["washout_days"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Outcome '{code_set}' has an invalid risk-window contract.") from exc
    if not code_set:
        return None, {}, 0, {}
    try:
        domain_row = connection.execute(
            "SELECT domain FROM study.code_set WHERE code_set_name = ? LIMIT 1", [code_set]
        ).fetchone()
        if not domain_row or not (mapping := _domain_table(str(domain_row[0]))):
            return None, {}, 0, {}
        table, concept_column, source_column, date_column = mapping
        arms = list(arm_n)
        if not arms:
            return None, {}, 0, {}
        placeholders = ", ".join("?" for _ in arms)
        rows = connection.execute(
            f"""WITH matched_events AS (
                    SELECT e.person_id, e.{date_column} AS event_date
                    FROM omop.{table} e
                    WHERE EXISTS (
                        SELECT 1 FROM study.code_set cs
                        WHERE cs.code_set_name=? AND cs.accepted
                          AND ((cs.concept_id > 0
                                AND cs.concept_id=e.{concept_column})
                            OR (cs.source_value IS NOT NULL
                                AND cs.source_value=e.{source_column})))
                ), cohort_windows AS (
                    SELECT c.person_id, c.cohort_name AS arm, c.index_date,
                           MIN(op.observation_period_start_date) AS observation_start,
                           MAX(op.observation_period_end_date) AS observation_end
                    FROM study.cohort c
                    JOIN omop.observation_period op
                      ON op.person_id=c.person_id
                     AND op.observation_period_start_date <= c.index_date
                     AND op.observation_period_end_date >= c.index_date
                    WHERE c.included AND c.cohort_name IN ({placeholders})
                    GROUP BY c.person_id, c.cohort_name, c.index_date
                ), classified AS (
                    SELECT w.*,
                           EXISTS (
                               SELECT 1 FROM matched_events e
                               WHERE e.person_id=w.person_id
                                 AND e.event_date >= w.index_date - CAST(? AS INTEGER)
                                 AND e.event_date < w.index_date
                           ) AS prevalent,
                           EXISTS (
                               SELECT 1 FROM matched_events e
                               WHERE e.person_id=w.person_id
                                 AND e.event_date >= w.index_date + CAST(? AS INTEGER)
                                 AND e.event_date <= w.index_date + CAST(? AS INTEGER)
                                 AND e.event_date <= w.observation_end
                           ) AS outcome_event
                    FROM cohort_windows w
                ), eligible AS (
                    SELECT *,
                           observation_start <= index_date - CAST(? AS INTEGER)
                           AND observation_end >= index_date + CAST(? AS INTEGER)
                           AND NOT prevalent
                           AND (outcome_event
                                OR observation_end >= index_date + CAST(? AS INTEGER))
                               AS analyzable
                    FROM classified
                )
                SELECT arm,
                       COUNT(DISTINCT person_id) FILTER (WHERE analyzable) AS eligible_n,
                       COUNT(DISTINCT person_id) FILTER (
                           WHERE analyzable AND outcome_event
                       ) AS event_n
                FROM eligible
                GROUP BY arm
                ORDER BY arm""",
            [
                code_set,
                *arms,
                washout_days,
                start_day,
                end_day,
                washout_days,
                start_day,
                end_day,
            ],
        ).fetchall()
        eligible_by_arm = {arm: 0 for arm in arms}
        events_by_arm = {arm: 0 for arm in arms}
        for arm, eligible_n, event_n in rows:
            eligible_by_arm[str(arm)] = int(eligible_n)
            events_by_arm[str(arm)] = int(event_n)
        return (
            sum(events_by_arm.values()),
            events_by_arm,
            sum(eligible_by_arm.values()),
            eligible_by_arm,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not derive risk-window event counts for code set '{code_set}'."
        ) from exc


def _event_counts(
    connection: Any, code_set: str, arm_n: dict[str, int]
) -> tuple[int | None, dict[str, int]]:
    if not code_set:
        return None, {}
    try:
        domain_row = connection.execute(
            "SELECT domain FROM study.code_set WHERE code_set_name = ? LIMIT 1", [code_set]
        ).fetchone()
        if not domain_row or not (mapping := _domain_table(str(domain_row[0]))):
            return None, {}
        table, concept_column, source_column, date_column = mapping
        condition = (
            f"EXISTS (SELECT 1 FROM study.code_set cs WHERE cs.code_set_name = ? "
            f"AND cs.accepted AND ((cs.concept_id > 0 AND cs.concept_id = e.{concept_column}) "
            f"OR (cs.source_value IS NOT NULL AND cs.source_value = e.{source_column})))"
        )
        events_by_arm: dict[str, int] = {}
        for arm in arm_n:
            events_by_arm[arm] = int(
                connection.execute(
                    f"SELECT COUNT(DISTINCT e.person_id) FROM omop.{table} e "
                    "JOIN study.cohort c ON c.person_id=e.person_id "
                    "AND c.cohort_name=? AND c.included "
                    f"WHERE e.{date_column} >= c.index_date AND {condition}",
                    [arm, code_set],
                ).fetchone()[0]
            )
        if events_by_arm:
            arms = sorted(events_by_arm)
            placeholders = ", ".join("?" for _ in arms)
            events_total = int(
                connection.execute(
                    f"SELECT COUNT(DISTINCT e.person_id) FROM omop.{table} e "
                    "JOIN study.cohort c ON c.person_id=e.person_id AND c.included "
                    f"AND c.cohort_name IN ({placeholders}) "
                    f"WHERE e.{date_column} >= c.index_date AND {condition}",
                    [*arms, code_set],
                ).fetchone()[0]
            )
        else:
            events_total = int(
                connection.execute(
                    f"SELECT COUNT(DISTINCT e.person_id) FROM omop.{table} e WHERE {condition}",
                    [code_set],
                ).fetchone()[0]
            )
        return events_total, events_by_arm
    except Exception as exc:
        raise RuntimeError(f"Could not derive event counts for code set '{code_set}'.") from exc


def _measurement_summary(
    connection: Any, code_set: str, cohort: str
) -> tuple[tuple[str, ...], float | None]:
    if not code_set or not cohort:
        return (), None
    try:
        rows = connection.execute(
            """SELECT DISTINCT m.unit_source_value
               FROM omop.measurement m
               JOIN study.cohort c ON c.person_id=m.person_id AND c.cohort_name=? AND c.included
               JOIN study.code_set cs ON cs.code_set_name=? AND cs.accepted
                 AND ((cs.concept_id > 0 AND cs.concept_id=m.measurement_concept_id)
                      OR cs.source_value=m.measurement_source_value)
               WHERE m.unit_source_value IS NOT NULL""",
            [cohort, code_set],
        ).fetchall()
        median = connection.execute(
            """SELECT MEDIAN(n) FROM (
                 SELECT m.person_id, COUNT(*) n
                 FROM omop.measurement m
                 JOIN study.cohort c ON c.person_id=m.person_id AND c.cohort_name=? AND c.included
                 JOIN study.code_set cs ON cs.code_set_name=? AND cs.accepted
                   AND ((cs.concept_id > 0 AND cs.concept_id=m.measurement_concept_id)
                        OR cs.source_value=m.measurement_source_value)
                 GROUP BY m.person_id)""",
            [cohort, code_set],
        ).fetchone()[0]
        return tuple(str(row[0]) for row in rows), float(median) if median is not None else None
    except Exception:
        return (), None


def _code_set_review_evidence(
    workspace: StudyWorkspace,
    plan: Plan,
) -> tuple[RuleOutcome, ...]:
    """Verify that each apparently current approval came from ``review-codes``."""

    verification = verify_manifest(workspace.manifest_path)
    try:
        entries = read_manifest(workspace.manifest_path) if verification.valid else []
    except ValueError:
        entries = []
    outcomes: list[RuleOutcome] = []
    for code_set in plan.code_sets:
        approval = code_set.approval
        expected_hash = code_set_review_hash(code_set)
        # STATIC validation already reports pending, rejected, incomplete, and
        # coding-hash-mismatched approvals. Avoid producing a second refusal for
        # the same defect here.
        if (
            approval.status != "approved"
            or approval.review_hash != expected_hash
            or any(coding.mapping_status is None for coding in code_set.codings)
        ):
            continue

        reason: str | None = None
        artifact_path = workspace.root / "codesets" / f"{code_set.name}.codeset.json"
        artifact_label = workspace.relative(artifact_path)
        artifact: dict[str, Any] | None = None
        if not verification.valid:
            reason = "the study manifest hash chain is invalid"
        elif not artifact_path.is_file():
            reason = "the reviewed code-set artifact is missing"
        else:
            try:
                candidate = json.loads(artifact_path.read_text(encoding="utf-8"))
                artifact = candidate if isinstance(candidate, dict) else None
            except (OSError, json.JSONDecodeError):
                artifact = None
            if artifact is None:
                reason = "the reviewed code-set artifact is invalid"
            elif artifact.get("name") != code_set.name:
                reason = "the reviewed artifact declares a different code-set name"
            else:
                try:
                    artifact_hash = code_set_review_hash(artifact)
                except (TypeError, ValueError):
                    artifact_hash = None
                if artifact_hash != expected_hash:
                    reason = "the reviewed artifact no longer matches the plan codings"
                elif artifact.get("approval") != approval.model_dump(
                    mode="json", exclude_none=False
                ):
                    reason = "the artifact and plan carry different review decisions"

        latest = next(
            (
                entry
                for entry in reversed(entries)
                if entry.get("step") == "review-codes"
                and entry.get("params", {}).get("name") == code_set.name
            ),
            None,
        )
        if reason is None and latest is None:
            reason = "no review-codes manifest evidence exists"
        if reason is None and latest is not None:
            params = latest.get("params", {})
            outputs = latest.get("outputs", {})
            approval_payload = approval.model_dump(mode="json", exclude_none=False)
            evidence_matches = (
                latest.get("status") in {"success", "skipped"}
                and params.get("decision") == "approved"
                and params.get("review_hash") == expected_hash
                and params.get("reviewed_by") == approval_payload["reviewed_by"]
                and params.get("reviewed_at") == approval_payload["reviewed_at"]
                and outputs.get(artifact_label) == hash_file(artifact_path)
            )
            if not evidence_matches:
                reason = "the latest review-codes manifest evidence is stale or inconsistent"

        if reason is not None:
            outcomes.append(
                RuleOutcome(
                    rule_id=catalog.CODESET_APPROVAL,
                    phase=GuardrailPhase.STATIC,
                    severity=Severity.REFUSE,
                    message=(
                        f"Code set '{code_set.name}' lacks current artifact and manifest "
                        "evidence for its approval; run review-codes again."
                    ),
                    details={
                        "code_set": code_set.name,
                        "approval_status": approval.status,
                        "review_hash_matches": True,
                        "review_evidence_valid": False,
                        "reason": reason,
                    },
                )
            )
    return tuple(outcomes)


def validate_study(
    study: StudyWorkspace | str | Path,
    *,
    include_data: bool = True,
) -> ValidateResult:
    workspace = study if isinstance(study, StudyWorkspace) else find_study(study)
    started = datetime.now(UTC)
    with study_lock(workspace):
        plan = load_plan(workspace.plan_path)
        summaries = aggregate_summaries(workspace.deidentified_db, plan) if include_data else {}
        report_model = validate_plan(plan, summaries=summaries or None)
        evidence_outcomes = _code_set_review_evidence(workspace, plan)
        if evidence_outcomes:
            report_model = ValidationReport(
                plan_hash=report_model.plan_hash,
                phases=report_model.phases,
                outcomes=(*report_model.outcomes, *evidence_outcomes),
                power_notes=report_model.power_notes,
            )
        report = report_model.model_dump(mode="json")
        report["status"] = (
            "refused" if report_model.refused else ("valid" if summaries else "valid_static")
        )
        report["data_guardrails_pending"] = include_data and not bool(summaries)
        output = workspace.root / "reports" / "validation_report.json"
        previous_hash = last_validated_plan_hash(workspace)
        current_hash = plan_hash(plan)
        history_dir = workspace.root / "reports" / "plan_history"
        history = history_dir / f"{current_hash}.json"
        previous_plan: dict[str, Any] | None = None
        if previous_hash:
            previous_path = history_dir / f"{previous_hash}.json"
            if previous_path.exists():
                previous_plan = json.loads(previous_path.read_text(encoding="utf-8"))
        history_dir.mkdir(parents=True, exist_ok=True)
        history.write_text(canonical_json(normalized_plan_dict(plan)) + "\n", encoding="utf-8")
        output.write_text(canonical_json(report) + "\n", encoding="utf-8")
        status = "refused" if report_model.refused else "success"
        record_step(
            workspace.manifest_path,
            step="validate",
            status=status,
            started_at=started,
            params={"plan_hash": current_hash, "phases": report.get("phases", [])},
            inputs={"plan.json": hash_file(workspace.plan_path)},
            outputs={
                workspace.relative(output): hash_file(output),
                workspace.relative(history): hash_file(history),
            },
            items=[item for item in report.get("outcomes", [])],
        )
        if not report_model.refused:
            if previous_hash and previous_hash != current_hash:
                record_step(
                    workspace.manifest_path,
                    step="plan_amended",
                    status="success",
                    started_at=started,
                    params={
                        "old_hash": previous_hash,
                        "new_hash": current_hash,
                        "diff": _plan_diff(previous_plan or {}, normalized_plan_dict(plan)),
                    },
                    inputs={"plan.json": hash_file(workspace.plan_path)},
                    outputs={workspace.relative(history): hash_file(history)},
                )
            else:
                record_step(
                    workspace.manifest_path,
                    step="plan_validated",
                    status="success",
                    started_at=started,
                    params={"plan_hash": current_hash},
                    inputs={"plan.json": hash_file(workspace.plan_path)},
                    outputs={workspace.relative(history): hash_file(history)},
                )
    return ValidateResult(
        report["status"], report, workspace.relative(output), int(report_model.exit_code)
    )


validate = validate_study
