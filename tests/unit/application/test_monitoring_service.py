"""Testes do serviço de monitoramento."""

import json
from unittest.mock import Mock, mock_open

import pandas as pd

from src.application.monitoring_service import ServicoMonitoramento
from src.config.settings import Configuracoes


def test_atualizar_snapshot_sem_referencia(monkeypatch):
    monkeypatch.setattr(
        "src.application.monitoring_service.os.path.exists",
        lambda path: False if path == Configuracoes.REFERENCE_PATH else True,
    )

    resultado = ServicoMonitoramento.atualizar_snapshot_monitoramento()

    assert resultado == {"updated": False, "reason": "reference_not_found"}


def test_atualizar_snapshot_sem_logs(monkeypatch):
    monkeypatch.setattr(
        "src.application.monitoring_service.os.path.exists",
        lambda path: False if path == Configuracoes.LOG_PATH else True,
    )

    resultado = ServicoMonitoramento.atualizar_snapshot_monitoramento()

    assert resultado == {"updated": False, "reason": "logs_not_found"}


def test_atualizar_snapshot_logs_invalidos(monkeypatch):
    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", lambda *_: True)
    monkeypatch.setattr("src.application.monitoring_service.pd.read_csv", lambda *_: pd.DataFrame({"prediction": [1]}))
    monkeypatch.setattr(
        "src.application.monitoring_service.ServicoMonitoramento._carregar_logs",
        lambda: pd.DataFrame(),
    )

    resultado = ServicoMonitoramento.atualizar_snapshot_monitoramento()

    assert resultado == {"updated": False, "reason": "empty_or_invalid_logs"}


def test_atualizar_snapshot_insufficient_samples(monkeypatch):
    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", lambda *_: True)
    monkeypatch.setattr(
        "src.application.monitoring_service.pd.read_csv",
        lambda *_: pd.DataFrame({"prediction": [0, 1, 0, 1, 0], "IDADE": [10, 11, 12, 13, 14]}),
    )
    monkeypatch.setattr(
        "src.application.monitoring_service.ServicoMonitoramento._carregar_logs",
        lambda: pd.DataFrame(
            {
                "input_features": [{"IDADE": 10}, {"IDADE": 11}, {"IDADE": 12}, {"IDADE": 13}],
                "prediction_result": [{"class": 0}, {"class": 1}, {"class": 0}, {"class": 1}],
            }
        ),
    )

    resultado = ServicoMonitoramento.atualizar_snapshot_monitoramento()

    assert resultado == {"updated": False, "reason": "insufficient_samples"}


def test_atualizar_snapshot_sucesso(monkeypatch):
    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", lambda *_: True)
    monkeypatch.setattr(
        "src.application.monitoring_service.pd.read_csv",
        lambda *_: pd.DataFrame({"prediction": [0, 1, 0, 1, 0], "IDADE": [10, 11, 12, 13, 14]}),
    )
    monkeypatch.setattr(
        "src.application.monitoring_service.ServicoMonitoramento._carregar_logs",
        lambda: pd.DataFrame(
            {
                "input_features": [
                    {"IDADE": 10},
                    {"IDADE": 11},
                    {"IDADE": 12},
                    {"IDADE": 13},
                    {"IDADE": 14},
                ],
                "prediction_result": [
                    {"class": 0},
                    {"class": 1},
                    {"class": 0},
                    {"class": 1},
                    {"class": 0},
                ],
            }
        ),
    )
    persistir_mock = Mock()
    monkeypatch.setattr(
        "src.application.monitoring_service.ServicoMonitoramento._persistir_relatorio_drift",
        persistir_mock,
    )

    resultado = ServicoMonitoramento.atualizar_snapshot_monitoramento()

    assert resultado["updated"] is True
    assert resultado["samples"] == 5
    persistir_mock.assert_called_once()


def test_carregar_logs_retorna_dataframe_vazio_para_arquivo_vazio(monkeypatch):
    monkeypatch.setattr("src.application.monitoring_service.ServicoMonitoramento._ler_ultimas_linhas", lambda *_: [])

    retorno = ServicoMonitoramento._carregar_logs()

    assert isinstance(retorno, pd.DataFrame)
    assert retorno.empty


def test_carregar_logs_retorna_dataframe_vazio_para_json_invalido(monkeypatch):
    monkeypatch.setattr("src.application.monitoring_service.ServicoMonitoramento._ler_ultimas_linhas", lambda *_: ["{}"])

    def levantar_erro(*args, **kwargs):
        raise ValueError("invalid")

    monkeypatch.setattr("src.application.monitoring_service.pd.read_json", levantar_erro)

    retorno = ServicoMonitoramento._carregar_logs()

    assert isinstance(retorno, pd.DataFrame)
    assert retorno.empty


def test_obter_importancia_global_sem_metricas(monkeypatch):
    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", lambda path: False)
    resultado = ServicoMonitoramento.obter_importancia_global()
    assert resultado["feature_importance_ranking"] == []


def test_obter_importancia_global_com_metricas(monkeypatch):
    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", lambda path: True)
    dados = {"feature_importance": {"IDADE": 0.4, "INDE_ANTERIOR": 0.3}}
    monkeypatch.setattr("builtins.open", mock_open(read_data=json.dumps(dados)))
    resultado = ServicoMonitoramento.obter_importancia_global()
    assert len(resultado["feature_importance_ranking"]) >= 1


def test_persistir_relatorio_drift(monkeypatch):
    monkeypatch.setattr("src.application.monitoring_service.os.makedirs", Mock())
    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", lambda *_: False)
    publish_mock = Mock()
    monkeypatch.setattr("src.application.monitoring_service.publish_model_metrics", publish_mock)
    arquivo_mock = mock_open()
    monkeypatch.setattr("builtins.open", arquivo_mock)

    ServicoMonitoramento._persistir_relatorio_drift(
        psi_df=pd.DataFrame(
            [
                {
                    "feature": "IDADE",
                    "psi": 0.1,
                    "reference_distribution": [0.5, 0.5],
                    "current_distribution": [0.4, 0.6],
                    "bins": [0.0, 0.5, 1.0],
                }
            ]
        ),
        alertas_psi=["ok"],
        monitoramento_estrategico={"significant_shift_alert": False, "current_high_risk_rate_pct": 42.0},
    )

    escrito = "".join(chamada.args[0] for chamada in arquivo_mock().write.call_args_list)
    payload = json.loads(escrito)
    assert "psi_by_feature" in payload
    assert "top_drift_features" in payload
    assert "drift_status" in payload
    assert "drift_history" in payload
    assert "high_risk_rate_history" in payload
    publish_mock.assert_called_once()
