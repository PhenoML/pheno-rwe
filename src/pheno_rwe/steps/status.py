from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pheno_rwe.config import Settings
from pheno_rwe.manifest import read_manifest, verify_manifest
from pheno_rwe.workspace import StudyWorkspace, find_study


@dataclass(slots=True)
class StatusResult:
    study: dict[str, Any]
    artifacts: dict[str, bool]
    last_steps: dict[str, str]
    environment: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return {
            "study": self.study,
            "artifacts": self.artifacts,
            "last_steps": self.last_steps,
            "environment": self.environment,
        }


def study_status(study: StudyWorkspace | str | Path, *, check_env: bool = False) -> StatusResult:
    workspace = study if isinstance(study, StudyWorkspace) else find_study(study)
    entries = read_manifest(workspace.manifest_path)
    last_steps: dict[str, str] = {}
    for entry in entries:
        last_steps[str(entry.get("step"))] = str(entry.get("status"))
    study_env = workspace.root / ".env"
    settings = Settings.from_env(study_env)
    environment: dict[str, Any] = {
        "manifest_valid": verify_manifest(workspace.manifest_path).valid,
    }
    if check_env:
        environment.update(
            {
                "phenoml_auth": settings.auth_mode or "missing",
                "fhir_provider": bool(settings.fhir_provider_id),
            }
        )
    return StatusResult(
        workspace.read_metadata(),
        {
            "plan": workspace.plan_path.exists(),
            "identified_db": workspace.identified_db.exists(),
            "deidentified_db": workspace.deidentified_db.exists(),
            "validation_report": (workspace.root / "reports" / "validation_report.json").exists(),
            "results": any((workspace.root / "results").glob("*/result.json")),
            "plots": any((workspace.root / "plots").glob("*/plot.svg")),
        },
        last_steps,
        environment,
    )
