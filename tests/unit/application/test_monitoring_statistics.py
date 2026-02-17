"""Testes de estatisticas adicionais de monitoramento."""

import pandas as pd

from src.application.monitoring_service import ServicoMonitoramento


def test_psi_computation():
    referencia = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    atual = pd.Series([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    psi = ServicoMonitoramento._calcular_psi(referencia, atual, bins=5)

    assert isinstance(psi, float)
    assert psi >= 0.0


def test_feature_drift_flagging():
    psi_df = pd.DataFrame(
        [
            {"feature": "INDE_ANTERIOR", "psi": 0.05, "drift_flag": False, "direction": "decrease_or_stable"},
            {"feature": "IDADE", "psi": 0.35, "drift_flag": True, "direction": "increase"},
        ]
    )

    alertas = ServicoMonitoramento._gerar_alertas_drift(psi_df)

    assert len(alertas) >= 1
    assert any("IDADE" in alerta for alerta in alertas)


def test_monitoramento_estrategico_alerta():
    referencia = pd.DataFrame({"prediction": [0, 0, 0, 1]})
    atual = pd.DataFrame({"prediction": [1, 1, 1, 1]})

    resultado = ServicoMonitoramento._calcular_monitoramento_estrategico(referencia, atual)

    assert "reference_high_risk_rate_pct" in resultado
    assert "current_high_risk_rate_pct" in resultado
    assert "significant_shift_alert" in resultado
    assert isinstance(resultado["significant_shift_alert"], bool)


def test_monitoramento_estrategico_html():
    html = ServicoMonitoramento._gerar_monitoramento_estrategico_html(
        {
            "reference_high_risk_rate_pct": 20.0,
            "current_high_risk_rate_pct": 35.0,
            "delta_high_risk_pp": 15.0,
            "high_risk_change_pct": 75.0,
            "significant_shift_alert": True,
            "alert_message": "teste",
        }
    )
    assert "Monitoramento Estratégico" in html
    assert "ALERTA" in html
