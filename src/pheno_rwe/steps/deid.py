"""Physical identified-to-de-identified DuckDB rebuild."""

from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import uuid
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pheno_rwe.deid.baseline import BaselineResult, apply_baseline
from pheno_rwe.deid.dateshift import DateShiftResult, apply_date_shift
from pheno_rwe.deid.policy import (
    DEIDENTIFIED_COLUMN_POLICY,
    audit_deidentified_policy,
    policy_summary,
)
from pheno_rwe.errors import StalePlanError
from pheno_rwe.hashing import canonical_json, hash_file, hash_json
from pheno_rwe.manifest import read_manifest, record_step, verify_manifest
from pheno_rwe.omop.ddl import create_schema
from pheno_rwe.plan.io import analysis_data_hash, load_plan, plan_hash
from pheno_rwe.plan.schema import DeidConfig
from pheno_rwe.runtime import CancellationToken, ProgressEvent, ProgressSink, null_progress
from pheno_rwe.steps.common import StepResult
from pheno_rwe.workspace import StudyWorkspace, find_study, study_lock


@dataclass(frozen=True, slots=True)
class KAnonymityResult:
    k: int
    minimum_cell_size: int | None
    groups_below_k: int
    patient_count: int

    @property
    def warning(self) -> str | None:
        if self.groups_below_k:
            return (
                f"{self.groups_below_k} quasi-identifier group(s) contain fewer than "
                f"k={self.k} patients (minimum cell size {self.minimum_cell_size})."
            )
        return None


@dataclass(frozen=True, slots=True)
class DeidBuildResult:
    output_path: Path
    tier: str
    baseline: BaselineResult
    date_shift: DateShiftResult | None
    k_anonymity: KAnonymityResult
    salt_sha256: str
    identifier_audit: dict[str, int]


@dataclass(frozen=True, slots=True)
class CohortResolutionEvidence:
    entry_hash: str
    plan_hash: str
    analysis_data_sha256: str
    identified_database_sha256: str


def _config(value: DeidConfig | Mapping[str, Any]) -> DeidConfig:
    return value if isinstance(value, DeidConfig) else DeidConfig.model_validate(value)


def _require_current_cohort_resolution(
    workspace: StudyWorkspace,
    *,
    expected_analysis_data_sha256: str,
    expected_identified_database_sha256: str,
) -> CohortResolutionEvidence:
    """Require a verified resolve-cohort record for the exact current inputs."""

    verification = verify_manifest(workspace.manifest_path)
    if not verification.valid:
        detail = "; ".join(verification.errors)
        raise RuntimeError(
            "Cohort-resolution provenance cannot be trusted because manifest verification "
            f"failed: {detail}"
        )
    latest = next(
        (
            entry
            for entry in reversed(read_manifest(workspace.manifest_path))
            if entry.get("step") == "resolve-cohort" and entry.get("status") == "success"
        ),
        None,
    )
    if latest is None:
        raise RuntimeError(
            "No successful resolve-cohort provenance exists; run resolve-cohort before deid."
        )
    params = latest.get("params")
    outputs = latest.get("outputs")
    entry_hash = latest.get("entry_hash")
    database_path = workspace.relative(workspace.identified_db)
    if not isinstance(params, Mapping) or not isinstance(outputs, Mapping):
        raise RuntimeError(
            "The latest resolve-cohort record lacks required provenance; run resolve-cohort again."
        )
    recorded_plan_hash = params.get("plan_hash")
    recorded_analysis_hash = params.get("analysis_data_sha256")
    recorded_database_param = params.get("identified_database_sha256")
    recorded_database_output = outputs.get(database_path)
    if (
        not isinstance(entry_hash, str)
        or not entry_hash
        or not isinstance(recorded_plan_hash, str)
        or not recorded_plan_hash
        or not isinstance(recorded_analysis_hash, str)
        or not recorded_analysis_hash
        or not isinstance(recorded_database_param, str)
        or not recorded_database_param
        or not isinstance(recorded_database_output, str)
        or not recorded_database_output
    ):
        raise RuntimeError(
            "The latest resolve-cohort record predates required provenance; "
            "run resolve-cohort again."
        )
    if recorded_database_param != recorded_database_output:
        raise RuntimeError(
            "The latest resolve-cohort record contains inconsistent database provenance; "
            "run resolve-cohort again."
        )
    if recorded_analysis_hash != expected_analysis_data_sha256:
        raise StalePlanError(
            "The plan code sets, cohorts, or index-date rule changed after cohort resolution; "
            "run resolve-cohort again before deid."
        )
    if recorded_database_output != expected_identified_database_sha256:
        raise RuntimeError(
            "The identified database changed after cohort resolution; "
            "run resolve-cohort again before deid."
        )
    return CohortResolutionEvidence(
        entry_hash=entry_hash,
        plan_hash=recorded_plan_hash,
        analysis_data_sha256=recorded_analysis_hash,
        identified_database_sha256=recorded_database_output,
    )


def k_anonymity_screen(connection: Any, *, k: int = 5) -> KAnonymityResult:
    if k < 2:
        raise ValueError("k must be at least 2")
    sizes = [
        int(row[0])
        for row in connection.execute(
            """SELECT COUNT(*) AS n
               FROM omop.person AS person
               LEFT JOIN study.person_demographic AS demo USING (person_id)
               GROUP BY demo.age_bucket, person.gender_concept_id,
                        person.race_concept_id, person.ethnicity_concept_id"""
        ).fetchall()
    ]
    return KAnonymityResult(
        k=k,
        minimum_cell_size=min(sizes) if sizes else None,
        groups_below_k=sum(size < k for size in sizes),
        patient_count=sum(sizes),
    )


def direct_identifier_audit(connection: Any) -> dict[str, int]:
    checks = {
        "location_rows": "SELECT COUNT(*) FROM omop.location",
        "person_source_values": (
            "SELECT COUNT(*) FROM omop.person WHERE person_source_value IS NOT NULL"
        ),
        "precise_birth_fields": """SELECT COUNT(*) FROM omop.person
            WHERE year_of_birth IS NOT NULL OR month_of_birth IS NOT NULL
               OR day_of_birth IS NOT NULL
               OR birth_datetime IS NOT NULL""",
        "provider_identifiers": """SELECT COUNT(*) FROM omop.provider
            WHERE provider_name IS NOT NULL OR npi IS NOT NULL OR dea IS NOT NULL
               OR provider_source_value IS NOT NULL""",
        "care_site_identifiers": """SELECT COUNT(*) FROM omop.care_site
            WHERE care_site_name IS NOT NULL OR care_site_source_value IS NOT NULL""",
        "raw_ledger_ids": """SELECT COUNT(*) FROM meta.ingest_patient
            WHERE source_patient_id IS NOT NULL AND source_patient_id NOT LIKE 'tok_%'""",
    }
    return {name: int(connection.execute(query).fetchone()[0]) for name, query in checks.items()}


def deid_freshness(workspace: StudyWorkspace) -> tuple[bool, str]:
    """Verify deid.duckdb was built from the current identified DB and plan de-id config."""
    if not workspace.deidentified_db.exists():
        return False, "deid.duckdb is missing"
    report = workspace.root / "reports" / "deid_report.json"
    if not report.exists():
        return False, "deid_report.json is missing"
    try:
        import json

        payload = json.loads(report.read_text(encoding="utf-8"))
        plan = load_plan(workspace.plan_path)
    except Exception as exc:
        return False, f"de-id provenance could not be read: {exc}"
    expected_input = (
        hash_file(workspace.identified_db) if workspace.identified_db.exists() else None
    )
    expected_config = hash_json(plan.deid.model_dump(mode="json"))
    if payload.get("source_database_sha256") != expected_input:
        return False, "the identified database changed after de-identification"
    if payload.get("deid_config_sha256") != expected_config:
        return False, "the plan de-identification config changed"
    expected_analysis_data = analysis_data_hash(plan)
    if payload.get("analysis_data_sha256") != expected_analysis_data:
        return False, "the plan code sets, cohorts, or index-date rule changed"
    if expected_input is None:
        return False, "the identified database is missing"
    try:
        evidence = _require_current_cohort_resolution(
            workspace,
            expected_analysis_data_sha256=expected_analysis_data,
            expected_identified_database_sha256=expected_input,
        )
    except (RuntimeError, StalePlanError) as exc:
        return False, str(exc)
    if payload.get("cohort_resolution_entry_hash") != evidence.entry_hash:
        return False, "cohort resolution was rerun after de-identification"
    if payload.get("cohort_resolution_plan_sha256") != evidence.plan_hash:
        return False, "cohort-resolution plan provenance does not match"
    return True, "current"


def _policy_audit_details(audit: Any) -> str:
    details = [*audit.errors]
    details.extend(f"{column}={count}" for column, count in sorted(audit.violations.items()))
    return "; ".join(details)


def _scrub_working_database(
    path: Path,
    *,
    config: DeidConfig,
    salt: bytes,
    reference_date: date | None,
    k: int,
) -> tuple[BaselineResult, DateShiftResult | None, KAnonymityResult, dict[str, int]]:
    import duckdb

    connection = duckdb.connect(str(path))
    try:
        create_schema(connection)
        connection.execute("BEGIN TRANSACTION")
        baseline = apply_baseline(
            connection,
            salt,
            age_bucket_years=config.age_bucket_years,
            age_cap=config.age_cap,
            reference_date=reference_date,
        )
        date_shift_result = (
            apply_date_shift(connection, salt, max_days=config.date_shift_max_days)
            if config.tier == "date_shift"
            else None
        )
        screen = k_anonymity_screen(connection, k=k)
        audit = direct_identifier_audit(connection)
        policy_audit = audit_deidentified_policy(connection)
        if not policy_audit.valid:
            raise RuntimeError(
                "De-identified column policy audit failed: " + _policy_audit_details(policy_audit)
            )
        audit["column_policy_violations"] = sum(policy_audit.violations.values())
        if any(audit.values()):
            failures = ", ".join(f"{name}={count}" for name, count in audit.items() if count)
            raise RuntimeError(f"Baseline identifier audit failed: {failures}")
        connection.execute("COMMIT")
        connection.execute("CHECKPOINT")
        return baseline, date_shift_result, screen, audit
    except Exception:
        with suppress(Exception):
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def _materialize_clean_database(working: Path, clean: Path) -> None:
    """Copy only the scrubbed logical rows into a fresh physical database."""
    import duckdb

    connection = duckdb.connect(str(clean))
    try:
        create_schema(connection)
        escaped = str(working).replace("'", "''")
        connection.execute(f"ATTACH '{escaped}' AS scrubbed (READ_ONLY)")
        connection.execute("BEGIN TRANSACTION")
        for table, table_rule in DEIDENTIFIED_COLUMN_POLICY.items():
            schema, table_name = table.split(".", 1)
            quoted_schema = f'"{schema.replace(chr(34), chr(34) * 2)}"'
            quoted_table = f'"{table_name.replace(chr(34), chr(34) * 2)}"'
            columns = ", ".join(
                f'"{column.replace(chr(34), chr(34) * 2)}"' for column in table_rule.columns
            )
            connection.execute(
                f"INSERT INTO {quoted_schema}.{quoted_table} ({columns}) "
                f"SELECT {columns} FROM scrubbed.{quoted_schema}.{quoted_table}"
            )
        connection.execute("COMMIT")
        clean_audit = audit_deidentified_policy(connection)
        if not clean_audit.valid:
            raise RuntimeError(
                "Clean de-identified database audit failed: " + _policy_audit_details(clean_audit)
            )
        connection.execute("DETACH scrubbed")
        connection.execute("CHECKPOINT")
    except Exception:
        with suppress(Exception):
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def rebuild_deidentified_database(
    identified_path: str | Path,
    output_path: str | Path,
    *,
    config: DeidConfig | Mapping[str, Any],
    salt: bytes | str,
    reference_date: date | None = None,
    k: int = 5,
) -> DeidBuildResult:
    source = Path(identified_path).resolve()
    output = Path(output_path).resolve()
    if source == output:
        raise ValueError("Identified and de-identified database paths must differ")
    if not source.is_file():
        raise FileNotFoundError(source)
    parsed = _config(config)
    salt_bytes = salt.encode("utf-8") if isinstance(salt, str) else salt
    if not salt_bytes:
        raise ValueError("de-identification salt must not be empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    working = output.with_name(f".{output.name}.{nonce}.working.duckdb")
    clean = output.with_name(f".{output.name}.{nonce}.clean.duckdb")
    try:
        shutil.copy2(source, working)
        baseline, date_shift_result, screen, audit = _scrub_working_database(
            working,
            config=parsed,
            salt=salt_bytes,
            reference_date=reference_date,
            k=k,
        )
        # Never publish the scrubbed-in-place copy: freed DuckDB blocks can
        # retain bytes from the identified source.
        _materialize_clean_database(working, clean)
        os.replace(clean, output)
        return DeidBuildResult(
            output_path=output,
            tier=parsed.tier,
            baseline=baseline,
            date_shift=date_shift_result,
            k_anonymity=screen,
            salt_sha256=hashlib.sha256(salt_bytes).hexdigest(),
            identifier_audit=audit,
        )
    finally:
        for temporary in (
            working,
            Path(f"{working}.wal"),
            clean,
            Path(f"{clean}.wal"),
        ):
            with suppress(FileNotFoundError):
                temporary.unlink()


def _load_or_create_salt(path: Path, supplied: bytes | str | None) -> bytes:
    if supplied is not None:
        value = supplied.encode("utf-8") if isinstance(supplied, str) else supplied
        if not value:
            raise ValueError("de-identification salt must not be empty")
        if path.exists() and path.read_bytes() != value:
            raise ValueError("Supplied salt differs from the study's persisted de-id salt")
        if not path.exists():
            path.write_bytes(value)
            path.chmod(0o600)
        return value
    if path.exists():
        value = path.read_bytes()
        if not value:
            raise ValueError(f"Persisted de-identification salt is empty: {path}")
        return value
    value = secrets.token_bytes(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, value)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return value


def deidentify(
    study: StudyWorkspace | str | Path,
    *,
    config: DeidConfig | Mapping[str, Any] | None = None,
    salt: bytes | str | None = None,
    progress: ProgressSink = null_progress,
    cancellation: CancellationToken | None = None,
    k: int = 5,
) -> StepResult:
    workspace = study if isinstance(study, StudyWorkspace) else find_study(study)
    started = datetime.now(UTC)
    token = cancellation or CancellationToken()
    salt_path = workspace.root / "omop" / ".deid_salt"
    with study_lock(workspace):
        plan_snapshot = load_plan(workspace.plan_path)
        parsed_config = plan_snapshot.deid
        if config is not None:
            supplied_config = _config(config)
            if supplied_config.model_dump(mode="json") != parsed_config.model_dump(mode="json"):
                raise ValueError(
                    "De-identification config override differs from plan.json; "
                    "amend the plan instead."
                )
        if not workspace.identified_db.is_file():
            raise FileNotFoundError(workspace.identified_db)
        identified_database_hash = hash_file(workspace.identified_db)
        current_plan_hash = plan_hash(plan_snapshot)
        current_analysis_data_hash = analysis_data_hash(plan_snapshot)
        cohort_evidence = _require_current_cohort_resolution(
            workspace,
            expected_analysis_data_sha256=current_analysis_data_hash,
            expected_identified_database_sha256=identified_database_hash,
        )
        salt_value = _load_or_create_salt(salt_path, salt)
        token.checkpoint()
        progress(ProgressEvent("deid", "running", "Applying baseline de-identification", 1, 2))
        result = rebuild_deidentified_database(
            workspace.identified_db,
            workspace.deidentified_db,
            config=parsed_config,
            salt=salt_value,
            k=k,
        )
        token.checkpoint()
        progress(ProgressEvent("deid", "running", "Auditing de-identified database", 2, 2))
        report_payload = {
            "tier": result.tier,
            "salt_sha256": result.salt_sha256,
            "source_database_sha256": identified_database_hash,
            "deid_config_sha256": hash_json(parsed_config.model_dump(mode="json")),
            "analysis_data_sha256": current_analysis_data_hash,
            "plan_sha256": current_plan_hash,
            "cohort_resolution_entry_hash": cohort_evidence.entry_hash,
            "cohort_resolution_plan_sha256": cohort_evidence.plan_hash,
            "column_policy": policy_summary(),
            "baseline": asdict(result.baseline),
            "date_shift": (
                {
                    "patient_count": result.date_shift.patient_count,
                    "max_days": result.date_shift.max_days,
                    "columns_shifted": list(result.date_shift.columns_shifted),
                    "shift_values_exported": False,
                }
                if result.date_shift
                else None
            ),
            "k_anonymity": asdict(result.k_anonymity),
            "identifier_audit": result.identifier_audit,
            "warnings": [result.k_anonymity.warning] if result.k_anonymity.warning else [],
        }
        report = workspace.root / "reports" / "deid_report.json"
        report.write_text(canonical_json(report_payload) + "\n", encoding="utf-8")
        outputs = {
            workspace.relative(workspace.deidentified_db): hash_file(workspace.deidentified_db),
            workspace.relative(report): hash_file(report),
        }
        warnings = [result.k_anonymity.warning] if result.k_anonymity.warning else []
        items = [
            {
                "identifier_audit": result.identifier_audit,
                "k_anonymity": asdict(result.k_anonymity),
            }
        ]
        record_step(
            workspace.manifest_path,
            step="deid",
            status="success",
            started_at=started,
            params={
                "tier": parsed_config.tier,
                "age_bucket_years": parsed_config.age_bucket_years,
                "age_cap": parsed_config.age_cap,
                "date_shift_max_days": parsed_config.date_shift_max_days,
                "k": k,
                "salt_sha256": result.salt_sha256,
                "plan_hash": current_plan_hash,
                "analysis_data_sha256": current_analysis_data_hash,
                "cohort_resolution_entry_hash": cohort_evidence.entry_hash,
            },
            input_signature=hash_json(
                {
                    "identified_db": identified_database_hash,
                    "config": parsed_config.model_dump(mode="json"),
                    "salt_sha256": result.salt_sha256,
                    "analysis_data_sha256": current_analysis_data_hash,
                    "cohort_resolution_entry_hash": cohort_evidence.entry_hash,
                }
            ),
            inputs={workspace.relative(workspace.identified_db): identified_database_hash},
            outputs=outputs,
            items=items,
        )
    return StepResult(
        "deid",
        "success",
        f"Rebuilt deid.duckdb with the {parsed_config.tier} tier.",
        outputs,
        items,
        warnings,
    )


deid = deidentify
deidentify_database = deidentify
