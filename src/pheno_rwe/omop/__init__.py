"""OMOP CDM-lite schema, re-keying, and DuckDB loading utilities."""

from pheno_rwe.omop.ddl import (
    DATE_COLUMN_REGISTRY,
    FOREIGN_KEY_REGISTRY,
    ID_COLUMN_REGISTRY,
    TABLE_SPECS,
)

__all__ = [
    "DATE_COLUMN_REGISTRY",
    "FOREIGN_KEY_REGISTRY",
    "ID_COLUMN_REGISTRY",
    "TABLE_SPECS",
]
