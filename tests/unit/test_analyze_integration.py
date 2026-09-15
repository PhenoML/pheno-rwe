from __future__ import annotations

import json
from contextlib import contextmanager

import pytest

import pheno_rwe.steps.analyze as analyze_module
from pheno_rwe.analyses import AnalysisResult
from pheno_rwe.steps.analyze import _write_result, derived_seed
from pheno_rwe.workspace import StudyWorkspace


def test_write_result_refreshes_envelope_and_removes_declared_stale_tables(tmp_path) -> None:
    output = tmp_path / "results" / "a1"
    first = AnalysisResult(
        analysis_id="a1",
        kind="test",
        n={"total": 1},
        tables={"old": [{"value": 1}]},
    )
    first_paths = _write_result(first, output)
    unrelated = output / "researcher-note.txt"
    unrelated.write_text("keep", encoding="utf-8")

    second = AnalysisResult(
        analysis_id="a1",
        kind="test",
        n={"total": 2},
        tables={"new": [{"value": 2}]},
    )
    second_paths = _write_result(second, output)

    assert {path.name for path in first_paths} == {"old.csv", "result.json"}
    assert {path.name for path in second_paths} == {"new.csv", "result.json"}
    assert not (output / "old.csv").exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"
    envelope = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert envelope["n"]["total"] == 2


def test_registered_and_feature_builder_seeds_are_identical() -> None:
    # This guards the adapter/analysis integration contract: the feature matrix
    # and UMAP/HDBSCAN execution must derive the same seed from the plan.
    from pheno_rwe.analyses.features import derive_analysis_seed

    assert derived_seed(2025, "signature-1") == derive_analysis_seed(2025, "signature-1")


def test_analysis_checks_freshness_only_after_taking_study_lock(tmp_path, monkeypatch) -> None:
    workspace = StudyWorkspace(tmp_path)
    workspace.deidentified_db.parent.mkdir(parents=True)
    workspace.deidentified_db.touch()
    state = {"locked": False}

    @contextmanager
    def tracked_lock(_workspace):
        state["locked"] = True
        try:
            yield
        finally:
            state["locked"] = False

    class CheckedFreshness(RuntimeError):
        pass

    def check_freshness(_workspace):
        assert state["locked"] is True
        raise CheckedFreshness

    monkeypatch.setattr(analyze_module, "study_lock", tracked_lock)
    monkeypatch.setattr("pheno_rwe.steps.deid.deid_freshness", check_freshness)

    with pytest.raises(CheckedFreshness):
        analyze_module.analyze_study(workspace)
    assert state["locked"] is False
