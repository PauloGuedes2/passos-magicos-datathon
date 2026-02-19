"""Publica metricas agregadas de monitoramento de modelo no New Relic."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from src.config.settings import Configuracoes
from src.util.logger import logger

try:
    import newrelic.agent as newrelic_agent
except Exception:
    newrelic_agent = None


_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nr-model-metrics")
_EVENT_TYPE = "ModelMonitoringSnapshot"


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _newrelic_enabled() -> bool:
    if newrelic_agent is None:
        return False
    if not os.getenv("NEW_RELIC_LICENSE_KEY"):
        return False
    if not os.getenv("NEW_RELIC_APP_NAME"):
        return False
    return True


def _build_event(summary: dict) -> dict:
    now_iso = datetime.now(timezone.utc).isoformat()
    model_version = str(summary.get("model_version") or Configuracoes.MODEL_VERSION)
    window_timestamp = str(summary.get("window_timestamp") or now_iso)

    return {
        "drift_score": _to_float(summary.get("drift_score"), 0.0),
        "number_of_drifted_features": _to_int(summary.get("number_of_drifted_features"), 0),
        "risk_rate": _to_float(summary.get("risk_rate"), _to_float(summary.get("prediction_mean"), 0.0)),
        "missing_ratio": _to_float(summary.get("missing_ratio"), 0.0),
        "model_version": model_version,
        "window_timestamp": window_timestamp,
        "environment": str(summary.get("environment") or Configuracoes.APP_ENV),
        "service_name": str(summary.get("service_name") or Configuracoes.SERVICE_NAME),
    }


def _emit_custom_event(event_payload: dict) -> None:
    if not _newrelic_enabled():
        return

    try:
        newrelic_agent.record_custom_event(_EVENT_TYPE, event_payload)
        newrelic_agent.add_custom_attributes(
            {
                "model_version": event_payload.get("model_version", "unknown"),
                "environment": event_payload.get("environment", "dev"),
                "service_name": event_payload.get("service_name", "service"),
            }
        )
    except Exception as erro:  # pragma: no cover - protecao operacional
        logger.warning(f"Falha ao enviar metricas de modelo para New Relic: {erro}")


def publish_model_metrics(summary: dict) -> None:
    """Publica evento customizado com metricas agregadas do monitoramento."""
    if not isinstance(summary, dict):
        return

    if not _newrelic_enabled():
        return

    try:
        event_payload = _build_event(summary)
        _EXECUTOR.submit(_emit_custom_event, event_payload)
    except Exception as erro:  # pragma: no cover - protecao operacional
        logger.warning(f"Falha ao agendar envio de metricas de modelo para New Relic: {erro}")
