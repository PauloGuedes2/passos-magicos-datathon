"""Testes do endpoint de dashboard profissional."""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.monitoring_controller as monitoring_controller
from src.api.monitoring_controller import ControladorMonitoramento


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_dashboard_endpoint_retorna_dashboard_profissional(tmp_path, monkeypatch):
    monitoring_dir = tmp_path / "monitoring"

    _write_json(
        monitoring_dir / "train_metrics.json",
        {
            "f1_score": 0.75,
            "recall": 0.83,
            "precision": 0.69,
            "auc": 0.81,
            "risk_threshold": 0.51,
            "threshold_strategy": "fairness_f1",
            "calibration_curve": {
                "mean_predicted_value": [0.1, 0.4, 0.7],
                "fraction_of_positives": [0.05, 0.38, 0.74],
            },
        },
    )
    _write_json(
        monitoring_dir / "temporal_cv_report.json",
        {
            "folds": [
                {"fold": 1, "f1": 0.73, "recall": 0.9, "precision": 0.62, "auc": 0.78, "threshold": 0.49, "ano_validacao": 2023},
                {"fold": 2, "f1": 0.79, "recall": 0.93, "precision": 0.71, "auc": 0.81, "threshold": 0.51, "ano_validacao": 2024},
            ],
            "interpretacao": {"conclusoes": ["Modelo estavel temporalmente."]},
        },
    )
    _write_json(
        monitoring_dir / "drift_report.json",
        {
            "psi_metrics": [
                {
                    "feature": "IDADE",
                    "psi": 0.08,
                    "reference_mean": 12.9,
                    "current_mean": 12.6,
                    "delta_mean": -0.3,
                    "drift_flag": False,
                }
            ],
            "psi_alerts": [],
            "strategic_monitoring": {
                "reference_high_risk_rate_pct": 55.0,
                "current_high_risk_rate_pct": 47.0,
                "delta_high_risk_pp": -8.0,
                "threshold_pct": 20.0,
                "significant_shift_alert": False,
            },
        },
    )

    monkeypatch.setattr(monitoring_controller.Configuracoes, "MONITORING_DIR", str(monitoring_dir))

    app = FastAPI()
    controller = ControladorMonitoramento()
    app.include_router(controller.roteador, prefix="/api/v1/monitoring")

    client = TestClient(app)
    response = client.get("/api/v1/monitoring/dashboard")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"].lower()
    assert "content-disposition" not in response.headers
    assert "Dashboard de Monitoramento" in response.text
    assert "id='sidebar'" in response.text
    assert "plotly.js" in response.text.lower()
    assert "<script src='https://cdn.plot.ly" not in response.text

    assert "Evidently" not in response.text
    assert "evidently-standard-theme" not in response.text
    assert "evidently_standardized" not in response.text
    assert "master_report.html" not in response.text
    assert "standardized_report.html" not in response.text
