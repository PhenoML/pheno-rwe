"""Analysis-level mapping and enrichment provenance summaries."""

from __future__ import annotations

from typing import Any


def summarize_provenance(
    rows: Any,
    *,
    mapping_column: str = "mapping_status",
    origin_column: str = "origin",
    domain_column: str = "domain",
) -> dict[str, Any]:
    """Summarize fractions overall and by domain from a dataframe-like object."""

    import pandas as pd

    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    total = int(len(frame))

    def fraction(mask: Any) -> float:
        return float(mask.sum() / total) if total else 0.0

    overall = {
        "rows": total,
        "unchecked_fraction": fraction(
            frame[mapping_column].fillna("").astype(str).str.upper().eq("UNCHECKED")
        )
        if mapping_column in frame
        else None,
        "unmapped_fraction": fraction(
            frame[mapping_column].fillna("").astype(str).str.upper().eq("UNMAPPED")
        )
        if mapping_column in frame
        else None,
        "enriched_fraction": fraction(
            frame[origin_column].fillna("").astype(str).str.lower().eq("enriched")
        )
        if origin_column in frame
        else None,
    }
    domains: dict[str, Any] = {}
    if domain_column in frame:
        for domain, group in frame.groupby(domain_column, dropna=False, sort=True):
            domains[str(domain)] = summarize_provenance(
                group.drop(columns=[domain_column]),
                mapping_column=mapping_column,
                origin_column=origin_column,
                domain_column=domain_column,
            )["overall"]
    return {"overall": overall, "by_domain": domains}


def provenance_from_database(db_path: str, person_ids: list[int] | None = None) -> dict[str, Any]:
    """Best-effort summary from the de-identified sidecar tables.

    Schema evolution is tolerated: absence of sidecars yields an explicit
    ``available=False`` payload instead of silently claiming zero provenance.
    """

    try:
        import duckdb
    except ImportError:
        return {"available": False, "reason": "duckdb_not_installed"}
    connection = duckdb.connect(db_path, read_only=True)
    try:
        tables = {
            tuple(row)
            for row in connection.execute(
                "SELECT table_schema, table_name FROM information_schema.tables"
            ).fetchall()
        }
        if ("meta", "mapping") not in tables:
            return {"available": False, "reason": "meta.mapping_missing"}
        where = ""
        parameters: list[Any] = []
        if person_ids is not None:
            unique = sorted(set(int(value) for value in person_ids))
            if not unique:
                return {
                    "available": True,
                    "overall": {
                        "rows": 0,
                        "unchecked_fraction": 0.0,
                        "unmapped_fraction": 0.0,
                        "enriched_fraction": 0.0,
                    },
                    "by_domain": {},
                }
            where = "WHERE i.person_id IN (" + ", ".join("?" for _ in unique) + ")"
            parameters.extend(unique)
        frame = connection.execute(
            f"""SELECT m.mapping_status,
                       COALESCE(rp.origin, 'structured') AS origin,
                       CASE
                         WHEN m.omop_table='condition_occurrence' THEN 'condition'
                         WHEN m.omop_table='drug_exposure' THEN 'drug'
                         WHEN m.omop_table='procedure_occurrence' THEN 'procedure'
                         WHEN m.omop_table='measurement' THEN 'measurement'
                         WHEN m.omop_table='observation' THEN 'observation'
                         WHEN m.omop_table='visit_occurrence' THEN 'visit'
                         ELSE COALESCE(m.omop_table, 'unknown')
                       END AS domain
                FROM meta.mapping m
                LEFT JOIN meta.ingest_patient i USING (source_patient_id)
                LEFT JOIN meta.row_provenance rp
                  ON rp.source_patient_id=m.source_patient_id
                 AND rp.omop_table=m.omop_table AND rp.omop_id=m.omop_id
                {where}""",
            parameters,
        ).fetchdf()
        return {"available": True, **summarize_provenance(frame)}
    finally:
        connection.close()
