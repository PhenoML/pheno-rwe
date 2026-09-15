"""Plot protocol and deterministic artifact result."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class PlotError(ValueError):
    pass


class PlotInputError(PlotError):
    pass


@dataclass(frozen=True, slots=True)
class PlotResult:
    kind: str
    output_dir: Path
    png_path: Path
    svg_path: Path
    data_path: Path
    spec_path: Path

    def model_dump(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "output_dir": str(self.output_dir),
            "png": str(self.png_path),
            "svg": str(self.svg_path),
            "data": str(self.data_path),
            "spec": str(self.spec_path),
        }


@runtime_checkable
class PlotRenderer(Protocol):
    @property
    def kind(self) -> str: ...

    @property
    def required_columns(self) -> tuple[str, ...]: ...

    def draw(self, frame: Any, params: dict[str, Any]) -> Any: ...
