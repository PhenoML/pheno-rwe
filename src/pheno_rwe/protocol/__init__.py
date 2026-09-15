"""Shared human/agent study protocol resources."""

from importlib.resources import files


def read_protocol(name: str) -> str:
    return files("pheno_rwe.protocol").joinpath(name).read_text(encoding="utf-8")


def system_protocol(exploratory: bool = False) -> str:
    parts = [read_protocol("prohibitions.md")]
    if not exploratory:
        parts.append(read_protocol("interview.md"))
    return "\n\n".join(parts)
