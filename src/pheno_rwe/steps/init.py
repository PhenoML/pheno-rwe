from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pheno_rwe.hashing import hash_file, hash_json
from pheno_rwe.manifest import record_step
from pheno_rwe.steps.common import StepResult
from pheno_rwe.workspace import create_study


def initialize_study(path: str | Path, name: str, *, force: bool = False) -> StepResult:
    started = datetime.now(UTC)
    workspace = create_study(path, name, force=force)
    relative = workspace.relative(workspace.metadata_path)
    record_step(
        workspace.manifest_path,
        step="init",
        status="success",
        started_at=started,
        params={"name": name},
        input_signature=hash_json({"name": name}),
        outputs={relative: hash_file(workspace.metadata_path)},
    )
    return StepResult(
        step="init",
        message=f"Created study '{name}'.",
        outputs={relative: hash_file(workspace.metadata_path)},
    )
