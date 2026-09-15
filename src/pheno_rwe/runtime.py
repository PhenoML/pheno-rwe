"""Frontend-neutral progress and cancellation seams."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from threading import Event
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    step: str
    status: str
    message: str
    current: int | None = None
    total: int | None = None
    item_id: str | None = None
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            object.__setattr__(self, "timestamp", datetime.now(UTC).isoformat())

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class ProgressSink(Protocol):
    def __call__(self, event: ProgressEvent) -> None: ...


def null_progress(event: ProgressEvent) -> None:
    del event
    return None


class CancellationToken:
    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    def should_stop(self) -> bool:
        return self._event.is_set()

    def checkpoint(self) -> None:
        if self.should_stop():
            from pheno_rwe.errors import CancelledError

            raise CancelledError("The run was cancelled between items; completed items were kept.")


ShouldStop = Callable[[], bool]
