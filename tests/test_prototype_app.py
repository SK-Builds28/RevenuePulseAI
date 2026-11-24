"""Smoke tests for the Streamlit prototype module (no UI automation)."""

from __future__ import annotations

import importlib

import pytest


def test_streamlit_is_installed():
    streamlit = pytest.importorskip("streamlit")
    assert streamlit.__version__


def test_prototype_app_imports_and_exposes_main():
    module = importlib.import_module("prototype.app")
    assert callable(module.main)
    assert module.PAGES == (
        "Executive Dashboard",
        "Review Queue",
        "Account Detail",
        "Human Review",
    )
    assert "synthetic" in module.DISCLAIMER.lower()
    assert "not a statement that the customer will churn" in module.DISCLAIMER


def test_app_renders_dashboard_without_exception(project_root):
    """End-to-end smoke check through Streamlit's own test harness."""

    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from src.utils.config import load_config

    config = load_config(project_root / "config.yaml")
    artifact = config.artifacts_model_dir / "revenuepulse_model.joblib"
    if not (config.full_snapshots_path.exists() and artifact.exists()):
        pytest.skip("Trained artifact or snapshots are not present in this checkout")

    app = AppTest.from_file(
        str(project_root / "prototype" / "app.py"), default_timeout=120
    ).run()
    assert not app.exception, [str(item.value) for item in app.exception]
    labels = [metric.label for metric in app.metric]
    assert "Accounts reviewed" in labels
    assert "High risk" in labels
    assert any("synthetic" in warning.value.lower() for warning in app.warning)


def test_prototype_uses_product_services_not_duplicated_scoring():
    """The app must call the shared product layer rather than score by hand."""

    from pathlib import Path

    text = Path("prototype/app.py").read_text(encoding="utf-8")
    assert "build_queue" in text
    assert "recommend_action" not in text or "account_detail" in text
    assert "predict_proba" not in text, "Scoring belongs in src.product.review"
    assert "joblib.load" not in text, "Model loading belongs in the product layer"
