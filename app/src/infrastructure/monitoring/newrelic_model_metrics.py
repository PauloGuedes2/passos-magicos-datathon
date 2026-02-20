"""Publica metricas agregadas de monitoramento de modelo no New Relic."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from src.config.settings import Configuracoes
from src.util.logger import logger

try:
    import newrelic.agent as newrelic_agent
except Exception:
    newrelic_agent = None


_EVENT_TYPE_SNAPSHOT = "ModelMonitoringSnapshot"
_EVENT_TYPE_PSI_FEATURE = "ModelMonitoringPsiFeature"
_EVENT_TYPE_THRESHOLD = "ModelMonitoringThresholdStrategy"
_EVENT_TYPE_FAIRNESS = "ModelMonitoringFairnessGroup"
_EVENT_TYPE_FEATURE_IMPORTANCE = "ModelMonitoringFeatureImportance"
_EVENT_TYPE_PERFORMANCE = "ModelMonitoringPerformance"
_MAX_FEATURE_IMPORTANCE_EVENTS = 20


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


def _safe_load_json(path: str) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _build_event(summary: dict) -> dict:
    now_iso = datetime.now(timezone.utc).isoformat()
    model_version = str(summary.get("model_version") or Configuracoes.MODEL_VERSION)
    window_timestamp = str(summary.get("window_timestamp") or now_iso)

    return {
        "drift_score": _to_float(summary.get("drift_score"), 0.0),
        "number_of_drifted_features": _to_int(summary.get("number_of_drifted_features"), 0),
        "risk_rate": _to_float(summary.get("risk_rate"), _to_float(summary.get("prediction_mean"), 0.0)),
        "missing_ratio": _to_float(summary.get("missing_ratio"), 0.0),
        "drift_status": str(summary.get("drift_status") or "unknown"),
        "high_risk_change_pct": _to_float(summary.get("high_risk_change_pct"), 0.0),
        "significant_shift_alert": bool(summary.get("significant_shift_alert", False)),
        "top_drift_feature_1": str(summary.get("top_drift_feature_1") or ""),
        "top_drift_feature_2": str(summary.get("top_drift_feature_2") or ""),
        "top_drift_feature_3": str(summary.get("top_drift_feature_3") or ""),
        "model_version": model_version,
        "window_timestamp": window_timestamp,
        "environment": str(summary.get("environment") or Configuracoes.APP_ENV),
        "service_name": str(summary.get("service_name") or Configuracoes.SERVICE_NAME),
    }


def _get_application():
    try:
        application = newrelic_agent.application()
        if application is None:
            application = newrelic_agent.register_application(timeout=2.0)
        return application
    except Exception as erro:
        logger.warning(f"Falha ao obter application do New Relic: {erro}")
        return None


def _emit_custom_event(event_type: str, event_payload: dict, application) -> None:
    if not event_type or not isinstance(event_payload, dict) or application is None:
        return
    try:
        newrelic_agent.record_custom_event(event_type, event_payload, application=application)
    except Exception as erro:  # pragma: no cover - protecao operacional
        logger.warning(f"Falha ao enviar evento {event_type} para New Relic: {erro}")


def _normalize_threshold_row(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {
        "strategy": str(raw.get("strategy") or raw.get("name") or raw.get("metodo") or "n/d"),
        "recall": _to_float(raw.get("recall"), 0.0),
        "f1_score": _to_float(raw.get("f1_score", raw.get("f1")), 0.0),
        "threshold": _to_float(raw.get("threshold"), 0.0),
        "high_risk_estimated": _to_float(
            raw.get("high_risk_estimated", raw.get("alto_risco_estimado")),
            0.0,
        ),
        "human_reviews": _to_float(raw.get("human_reviews", raw.get("revisoes_humanas")), 0.0),
    }


def _resolve_threshold_rows(train_metrics: dict[str, Any], cmp_data: Any) -> list[dict[str, Any]]:
    rows = []
    if isinstance(cmp_data, dict):
        if isinstance(cmp_data.get("strategies"), list):
            rows = [_normalize_threshold_row(item) for item in cmp_data.get("strategies", [])]
        elif isinstance(cmp_data.get("rows"), list):
            rows = [_normalize_threshold_row(item) for item in cmp_data.get("rows", [])]
        elif cmp_data:
            rows = [_normalize_threshold_row(cmp_data)]
    elif isinstance(cmp_data, list):
        rows = [_normalize_threshold_row(item) for item in cmp_data]

    rows = [row for row in rows if row]
    if rows:
        return rows

    return [
        _normalize_threshold_row(
            {
                "strategy": train_metrics.get("threshold_strategy", "n/d"),
                "recall": train_metrics.get("recall"),
                "f1_score": train_metrics.get("f1_score"),
                "threshold": train_metrics.get("risk_threshold"),
            }
        )
    ]


def _build_performance_event(train_metrics: dict[str, Any], common: dict) -> dict:
    return {
        **common,
        "recall": _to_float(train_metrics.get("recall"), 0.0),
        "precision": _to_float(train_metrics.get("precision"), 0.0),
        "f1_score": _to_float(train_metrics.get("f1_score"), 0.0),
        "auc": _to_float(train_metrics.get("auc"), 0.0),
        "brier_score": _to_float(train_metrics.get("brier_score"), 0.0),
        "risk_threshold": _to_float(train_metrics.get("risk_threshold"), 0.0),
        "threshold_strategy": str(train_metrics.get("threshold_strategy") or "n/d"),
        "overfitting_gap_f1": _to_float(train_metrics.get("overfitting_gap_f1"), 0.0),
        "train_size": _to_int(train_metrics.get("train_size"), 0),
        "test_size": _to_int(train_metrics.get("test_size"), 0),
    }


def _build_psi_feature_events(psi_records: list[dict], common: dict) -> list[dict]:
    events = []
    for row in psi_records:
        if not isinstance(row, dict):
            continue
        events.append(
            {
                **common,
                "feature": str(row.get("feature") or "n/d"),
                "psi": _to_float(row.get("psi"), 0.0),
                "reference_mean": _to_float(row.get("reference_mean"), 0.0),
                "current_mean": _to_float(row.get("current_mean"), 0.0),
                "delta_mean": _to_float(row.get("delta_mean"), 0.0),
                "direction": str(row.get("direction") or "n/d"),
                "drift_flag": bool(row.get("drift_flag", False)),
            }
        )
    return events


def _build_threshold_events(
    train_metrics: dict[str, Any],
    cmp_data: Any,
    common: dict,
) -> list[dict]:
    active = str(train_metrics.get("threshold_strategy") or "n/d")
    rows = _resolve_threshold_rows(train_metrics, cmp_data)
    return [
        {
            **common,
            "strategy": str(row.get("strategy") or "n/d"),
            "is_active": str(row.get("strategy") or "n/d") == active,
            "recall": _to_float(row.get("recall"), 0.0),
            "f1_score": _to_float(row.get("f1_score"), 0.0),
            "threshold": _to_float(row.get("threshold"), 0.0),
            "high_risk_estimated": _to_float(row.get("high_risk_estimated"), 0.0),
            "human_reviews": _to_float(row.get("human_reviews"), 0.0),
        }
        for row in rows
    ]


def _build_fairness_events(train_metrics: dict[str, Any], common: dict) -> list[dict]:
    group_metrics = train_metrics.get("group_metrics")
    if not isinstance(group_metrics, dict):
        return []

    group_name = Configuracoes.FAIRNESS_GROUP_COL
    target_groups = group_metrics.get(group_name)
    if not isinstance(target_groups, dict):
        return []

    events = []
    for group_value, values in target_groups.items():
        if not isinstance(values, dict):
            continue
        events.append(
            {
                **common,
                "group_column": group_name,
                "group_value": str(group_value),
                "recall": _to_float(values.get("recall"), 0.0),
                "precision": _to_float(values.get("precision"), 0.0),
                "f1_score": _to_float(values.get("f1_score"), 0.0),
                "support": _to_int(values.get("support"), 0),
            }
        )
    return events


def _build_feature_importance_events(train_metrics: dict[str, Any], common: dict) -> list[dict]:
    ranking = train_metrics.get("feature_importance_ranking")
    parsed: list[tuple[str, float]] = []

    if isinstance(ranking, list):
        for item in ranking:
            if isinstance(item, dict):
                feature = str(item.get("feature") or "n/d")
                parsed.append((feature, _to_float(item.get("importance"), 0.0)))

    if not parsed and isinstance(train_metrics.get("feature_importance"), dict):
        for feature, importance in train_metrics.get("feature_importance", {}).items():
            parsed.append((str(feature), _to_float(importance, 0.0)))

    if not parsed:
        return []

    parsed.sort(key=lambda row: row[1], reverse=True)
    return [
        {
            **common,
            "feature": feature,
            "importance": importance,
            "rank": idx + 1,
        }
        for idx, (feature, importance) in enumerate(parsed[:_MAX_FEATURE_IMPORTANCE_EVENTS])
    ]


def publish_model_metrics(
    summary: dict,
    psi_records: list[dict] | None = None,
) -> None:
    """Publica evento customizado com metricas agregadas do monitoramento."""
    if not isinstance(summary, dict):
        return

    if not _newrelic_enabled():
        logger.info("Custom metrics New Relic desativadas: variaveis de ambiente ausentes.")
        return

    try:
        application = _get_application()
        if application is None:
            logger.warning("New Relic ativo, mas sem application registrada para custom event.")
            return

        summary_event = _build_event(summary)
        common = {
            "model_version": str(summary_event.get("model_version") or Configuracoes.MODEL_VERSION),
            "window_timestamp": str(summary_event.get("window_timestamp") or datetime.now(timezone.utc).isoformat()),
            "environment": str(summary_event.get("environment") or Configuracoes.APP_ENV),
            "service_name": str(summary_event.get("service_name") or Configuracoes.SERVICE_NAME),
        }

        _emit_custom_event(_EVENT_TYPE_SNAPSHOT, summary_event, application)

        psi_data = psi_records if isinstance(psi_records, list) else []
        for event in _build_psi_feature_events(psi_data, common):
            _emit_custom_event(_EVENT_TYPE_PSI_FEATURE, event, application)

        train_metrics = _safe_load_json(Configuracoes.METRICS_FILE)
        threshold_comparison_path = os.path.join(
            Configuracoes.MONITORING_DIR, "threshold_strategy_comparison.json"
        )
        threshold_comparison = _safe_load_json(threshold_comparison_path)

        if train_metrics:
            performance_event = _build_performance_event(train_metrics, common)
            _emit_custom_event(_EVENT_TYPE_PERFORMANCE, performance_event, application)

            for event in _build_threshold_events(train_metrics, threshold_comparison, common):
                _emit_custom_event(_EVENT_TYPE_THRESHOLD, event, application)

            for event in _build_fairness_events(train_metrics, common):
                _emit_custom_event(_EVENT_TYPE_FAIRNESS, event, application)

            for event in _build_feature_importance_events(train_metrics, common):
                _emit_custom_event(_EVENT_TYPE_FEATURE_IMPORTANCE, event, application)
    except Exception as erro:  # pragma: no cover - protecao operacional
        logger.warning(f"Falha ao publicar metricas de modelo para New Relic: {erro}")
