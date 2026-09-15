"""Normalize generated-SDK and pydantic values into JSON-compatible data."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast


def to_data(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return to_data(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_data(item) for item in value]
    if hasattr(value, "model_dump"):
        return to_data(value.model_dump(mode="json"))
    if hasattr(value, "dict"):
        return to_data(value.dict())
    if is_dataclass(value):
        return to_data(asdict(cast(Any, value)))
    if hasattr(value, "__dict__"):
        return {key: to_data(item) for key, item in vars(value).items() if not key.startswith("_")}
    return str(value)


def first_present(value: Any, *names: str, default: Any = None) -> Any:
    data = to_data(value)
    if not isinstance(data, dict):
        return default
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
    return default
