"""Testes de publicação de métricas de modelo no New Relic."""

from unittest.mock import Mock

from src.infrastructure.monitoring import newrelic_model_metrics as nr_metrics


def test_build_event_aplica_defaults():
    evento = nr_metrics._build_event({})

    assert evento["drift_score"] == 0.0
    assert evento["number_of_drifted_features"] == 0
    assert evento["drift_status"] == "unknown"
    assert "window_timestamp" in evento


def test_build_threshold_events_define_ativo():
    train_metrics = {"threshold_strategy": "f1_otimizado"}
    comparativo = {
        "strategies": [
            {"strategy": "f1_otimizado", "threshold": 0.7, "recall": 0.8, "f1_score": 0.75},
            {"strategy": "alta_cobertura", "threshold": 0.4, "recall": 0.95, "f1_score": 0.6},
        ]
    }

    eventos = nr_metrics._build_threshold_events(train_metrics, comparativo, {"service_name": "svc"})

    assert len(eventos) == 2
    assert eventos[0]["is_active"] is True
    assert eventos[1]["is_active"] is False


def test_build_fairness_events_sem_grupo():
    eventos = nr_metrics._build_fairness_events({}, {"service_name": "svc"})
    assert eventos == []


def test_build_feature_importance_events_fallback():
    train_metrics = {
        "feature_importance": {
            "f1": 0.9,
            "f2": 0.1,
        }
    }

    eventos = nr_metrics._build_feature_importance_events(train_metrics, {"service_name": "svc"})

    assert len(eventos) == 2
    assert eventos[0]["feature"] == "f1"
    assert eventos[0]["rank"] == 1


def test_publish_model_metrics_emite_eventos(monkeypatch):
    monkeypatch.setenv("NEW_RELIC_LICENSE_KEY", "license")
    monkeypatch.setenv("NEW_RELIC_APP_NAME", "app")

    agent_mock = Mock()
    application = object()
    agent_mock.application.return_value = application
    monkeypatch.setattr(nr_metrics, "newrelic_agent", agent_mock)

    recorded = []

    def registrar(event_type, payload, application=None):
        recorded.append((event_type, payload, application))

    monkeypatch.setattr(nr_metrics, "_emit_custom_event", lambda event_type, payload, app: registrar(event_type, payload, app))
    monkeypatch.setattr(nr_metrics, "_safe_load_json", lambda path: {})

    nr_metrics.publish_model_metrics(
        summary={"drift_score": 0.2, "service_name": "svc"},
        psi_records=[{"feature": "IDADE", "psi": 0.3, "drift_flag": True}],
    )

    tipos = [item[0] for item in recorded]
    assert "ModelMonitoringSnapshot" in tipos
    assert "ModelMonitoringPsiFeature" in tipos
