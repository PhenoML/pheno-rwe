from __future__ import annotations

import pytest
from pydantic import BaseModel

from pheno_rwe.analyses import AnalysisContext, AnalysisInputError, AnalysisResult
from pheno_rwe.analyses.registry import get, kinds

EXPECTED_KINDS = {
    "table_one",
    "cohort_compare",
    "survival",
    "incidence_rate",
    "treatment_pathways",
    "trajectory",
    "patient_signature",
    "causal_effect",
}


def test_builtin_analysis_registry_is_complete_and_lazy() -> None:
    assert set(kinds()) == EXPECTED_KINDS
    for kind in EXPECTED_KINDS:
        analysis = get(kind)
        assert analysis.kind == kind
        assert issubclass(analysis.Params, BaseModel)
        assert callable(analysis.run)


def test_registry_reports_unknown_kind() -> None:
    with pytest.raises(AnalysisInputError, match="unknown analysis kind"):
        get("not_a_method")


def test_result_write_is_canonical_and_writes_tidy_tables(tmp_path) -> None:
    result = AnalysisResult(
        analysis_id="a1",
        kind="example",
        n={"total": 2, "per_group": {"A": 1, "B": 1}, "excluded": 0},
        tables={"data": [{"group": "A", "value": 1.0}, {"group": "B", "value": 2.0}]},
        output_dir=tmp_path,
    )
    first = result.write()
    first_bytes = first.read_bytes()
    assert (tmp_path / "data.csv").read_text() == "group,value\nA,1.0\nB,2.0\n"
    assert result.write().read_bytes() == first_bytes
    assert result.model_dump()["output_tables"] == [
        {
            "name": "data",
            "path": "data.csv",
            "format": "text/csv",
            "rows": 2,
            "columns": ["group", "value"],
        }
    ]


def test_context_accepts_outer_plan_shape(tmp_path) -> None:
    context = AnalysisContext(
        db_path=None,
        spec={"id": "a1", "kind": "table_one", "params": {"group_column": "arm"}},
        analysis_id="a1",
        seed=42,
        output_dir=tmp_path,
    )
    assert context.kind == "table_one"
    assert context.params == {"group_column": "arm"}
