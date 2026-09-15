from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class StepResult:
    step: str
    status: str = "success"
    message: str = ""
    outputs: dict[str, str] = field(default_factory=dict)
    items: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        if self.status == "partial":
            return 2
        if self.status == "refused":
            return 3
        if self.status in {"failed", "fatal"}:
            return 1
        return 0

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


def safe_identifier(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in value)
    return safe[:180] or "unknown"


def ensure_output(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
