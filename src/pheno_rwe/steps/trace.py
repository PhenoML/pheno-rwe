from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pheno_rwe.manifest import ManifestVerification, read_manifest, verify_manifest
from pheno_rwe.workspace import StudyWorkspace, find_study


def _is_shareable_bundle(root: Path) -> bool:
    path = root / "bundle.json"
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(value, dict):
        return False
    manifest = value.get("manifest")
    return (
        value.get("identified_data_included") is False
        and isinstance(manifest, dict)
        and manifest.get("profile") == "shareable-v1"
    )


@dataclass(slots=True)
class TraceResult:
    entries: list[dict[str, Any]]
    verification: ManifestVerification | None = None
    graph: list[tuple[str, str]] | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "entries": self.entries,
            "verification": (
                {
                    "valid": self.verification.valid,
                    "entries": self.verification.entries,
                    "errors": list(self.verification.errors),
                }
                if self.verification
                else None
            ),
            "graph": self.graph,
        }


def trace_study(
    study: StudyWorkspace | str | Path,
    *,
    verify: bool = False,
    graph: bool = False,
) -> TraceResult:
    workspace = study if isinstance(study, StudyWorkspace) else find_study(study)
    entries = read_manifest(workspace.manifest_path)
    shareable_bundle = _is_shareable_bundle(workspace.root)
    edges = None
    if graph:
        edges = []
        previous: str | None = None
        for index, entry in enumerate(entries):
            node = f"{index + 1}:{entry.get('step')}"
            if previous:
                edges.append((previous, node))
            previous = node
    return TraceResult(
        entries,
        verify_manifest(
            workspace.manifest_path,
            verify_inputs=True,
            verify_outputs=True,
            root=workspace.root,
            restrict_to_root=shareable_bundle,
        )
        if verify
        else None,
        edges,
    )
