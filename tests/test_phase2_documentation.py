"""Phase 2 documentation coverage and reproducibility-command tests."""

from __future__ import annotations

from pathlib import Path

import pytest

REQUIRED_MODEL_CARD_SECTIONS = (
    "Intended use",
    "Intended users",
    "Prediction target",
    "Model type",
    "Input features",
    "Data source",
    "Synthetic",
    "Training approach",
    "Temporal evaluation",
    "Baseline comparison",
    "Performance",
    "Confidence intervals",
    "Known limitations",
    "Fairness",
    "Explainability",
    "Prohibited uses",
    "Monitoring",
    "should not be used",
)

REQUIRED_EVALUATION_SECTIONS = (
    "Objective",
    "Dataset and temporal split",
    "Revenue-decline label",
    "Transparent baseline",
    "Models evaluated",
    "Validation results",
    "Model selection decision",
    "Locked test results",
    "Precision@K comparison",
    "Confidence intervals",
    "Calibration",
    "Error analysis",
    "Segment analysis",
    "Limitations",
    "Pilot implication",
    "Go / Iterate / Stop",
)


def _read(project_root: Path, relative: str) -> str:
    path = project_root / relative
    assert path.exists(), f"Missing Phase 2 document: {relative}"
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["docs/Model_Card.md"])
def test_model_card_covers_required_sections(project_root: Path, name: str):
    text = _read(project_root, name)
    missing = [section for section in REQUIRED_MODEL_CARD_SECTIONS if section.lower() not in text.lower()]
    assert not missing, f"Model card is missing sections: {missing}"
    assert "synthetic" in text.lower()
    assert "does not establish real-world" in text.lower() or "not real-world" in text.lower()


def test_evaluation_report_has_all_sections(project_root: Path):
    text = _read(project_root, "docs/Evaluation_Results.md")
    missing = [
        section
        for section in REQUIRED_EVALUATION_SECTIONS
        if section.lower() not in text.lower()
    ]
    assert not missing, f"Evaluation report is missing sections: {missing}"
    assert "synthetic" in text.lower()


def test_error_analysis_document_exists(project_root: Path):
    text = _read(project_root, "docs/Error_Analysis.md")
    assert "false positive" in text.lower()
    assert "false negative" in text.lower()
    assert "synthetic" in text.lower()


def test_build_notes_cover_phase_two_decisions(project_root: Path):
    text = _read(project_root, "docs/Phase_2_Build_Notes.md")
    for topic in (
        "class imbalance",
        "preprocessing",
        "model selection",
        "hardcoded",
        "limitations",
        "bootstrap",
    ):
        assert topic.lower() in text.lower(), f"Build notes do not discuss: {topic}"


def test_readme_documents_phase_two_commands(project_root: Path):
    text = _read(project_root, "README.md")
    for command in (
        "python -m src.data.generate_synthetic",
        "python -m src.features.build_snapshots",
        "python -m src.models.train",
        "python -m src.evaluation.evaluate",
        "pytest",
    ):
        assert command in text, f"README is missing command: {command}"
    assert "artifacts/model/revenuepulse_model.joblib" in text
    assert "Phase 2" in text
    assert "not real-world" in text.lower()
