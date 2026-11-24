"""Documentation coverage checks for the Phase 1 data contract."""

from __future__ import annotations

from pathlib import Path

from src.data.schema import RAW_COLUMNS
from src.features.snapshots import FEATURE_COLUMNS, TARGET_COLUMNS


def test_data_dictionary_documents_every_raw_field(project_root: Path):
    text = (project_root / "docs" / "Data_Dictionary.md").read_text(encoding="utf-8")
    missing = [column for column in RAW_COLUMNS if column not in text]
    assert not missing, f"Undocumented raw columns: {missing}"


def test_data_dictionary_documents_every_feature_and_target(project_root: Path):
    text = (project_root / "docs" / "Data_Dictionary.md").read_text(encoding="utf-8")
    missing_features = [column for column in FEATURE_COLUMNS if column not in text]
    missing_targets = [column for column in TARGET_COLUMNS if column not in text]
    assert not missing_features, f"Undocumented features: {missing_features}"
    assert not missing_targets, f"Undocumented targets: {missing_targets}"


def test_readme_contains_synthetic_disclaimer(project_root: Path):
    text = (project_root / "README.md").read_text(encoding="utf-8")
    assert "synthetic" in text.lower()
    assert "not evidence" in text.lower() or "not real" in text.lower()
    assert "Phase 1" in text


def test_agents_file_states_phase_and_safety_rules(project_root: Path):
    text = (project_root / "AGENTS.md").read_text(encoding="utf-8")
    assert "Phase 1" in text
    assert "Synthetic only" in text
    assert "LLM off by default" in text
    assert "Human review is mandatory" in text
