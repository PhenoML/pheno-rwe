from __future__ import annotations

from pheno_rwe.workspace import create_study, find_study


def test_create_and_find_nearest_study(tmp_path):
    workspace = create_study(tmp_path / "study", "Test study")
    nested = workspace.root / "results" / "a1"
    nested.mkdir(parents=True)
    found = find_study(nested)
    assert found.root == workspace.root
    assert workspace.read_metadata()["study_id"]
    assert workspace.identified_db.parent.is_dir()
