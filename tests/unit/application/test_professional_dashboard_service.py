"""Tests for the professional dashboard service."""

import json
from pathlib import Path

from src.application.professional_dashboard_service import ProfessionalDashboardService


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_generate_dashboard_without_threshold_comparison(tmp_path):
    monitoring_dir = tmp_path / "monitoring"
    output_file = monitoring_dir / "professional_dashboard.html"

    _write_json(
        monitoring_dir / "train_metrics.json",
        {
            "f1_score": 0.72,
            "recall": 0.81,
            "precision": 0.66,
            "auc": 0.79,
            "risk_threshold": 0.5,
            "threshold_strategy": "fairness_f1",
        },
    )
    _write_json(monitoring_dir / "temporal_cv_report.json", {"folds": []})
    _write_json(
        monitoring_dir / "drift_report.json",
        {"psi_metrics": [], "psi_alerts": [], "strategic_monitoring": {}},
    )

    service = ProfessionalDashboardService(
        monitoring_dir=str(monitoring_dir),
        output_file=str(output_file),
    )
    service.generate_dashboard()

    html = output_file.read_text(encoding="utf-8")
    assert "fairness_f1" in html
    assert "id='threshold-table'" in html
    assert "<script src='https://cdn.plot.ly" not in html
    assert "evidently" not in html.lower()
