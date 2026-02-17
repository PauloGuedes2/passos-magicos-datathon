"""Testes do serviço de monitoramento."""

from unittest.mock import Mock, mock_open
import json

import pandas as pd

from src.application.monitoring_service import ServicoMonitoramento
from src.config.settings import Configuracoes


def test_gerar_dashboard_sem_referencia(monkeypatch):
    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", lambda path: False)

    html = ServicoMonitoramento.gerar_dashboard()

    assert "Dataset de Referência" in html


def test_gerar_dashboard_sem_logs(monkeypatch):
    def exists(path):
        if path == Configuracoes.REFERENCE_PATH:
            return True
        return False

    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", exists)

    html = ServicoMonitoramento.gerar_dashboard()

    assert "Nenhum dado de produção" in html


def test_gerar_dashboard_logs_invalidos(monkeypatch):
    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", lambda path: True)
    monkeypatch.setattr("src.application.monitoring_service.pd.read_csv", lambda path: pd.DataFrame({"prediction": [1]}))

    def levantar_erro(*args, **kwargs):
        raise ValueError("invalid")

    monkeypatch.setattr("src.application.monitoring_service.ServicoMonitoramento._ler_ultimas_linhas", lambda *_: ["{}"])
    monkeypatch.setattr("src.application.monitoring_service.pd.read_json", levantar_erro)

    html = ServicoMonitoramento.gerar_dashboard()

    assert "arquivo de logs vazio" in html.lower()


def test_gerar_dashboard_logs_vazios(monkeypatch):
    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", lambda path: True)
    monkeypatch.setattr("src.application.monitoring_service.pd.read_csv", lambda path: pd.DataFrame({"prediction": [1]}))
    monkeypatch.setattr("src.application.monitoring_service.ServicoMonitoramento._ler_ultimas_linhas", lambda *_: [])

    html = ServicoMonitoramento.gerar_dashboard()

    assert "arquivo de logs vazio" in html.lower()


def test_gerar_dashboard_aguarda_dados(monkeypatch):
    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", lambda path: True)

    referencia = pd.DataFrame({"prediction": [0, 1, 0]})
    atual_raw = pd.DataFrame({
        "input_features": [{"IDADE": 10}, {"IDADE": 11}],
        "prediction_result": [{"class": 0}, {"class": 1}],
    })

    monkeypatch.setattr("src.application.monitoring_service.pd.read_csv", lambda path: referencia)
    monkeypatch.setattr("src.application.monitoring_service.ServicoMonitoramento._ler_ultimas_linhas", lambda *_: ["{}"])
    monkeypatch.setattr("src.application.monitoring_service.pd.read_json", lambda *args, **kwargs: atual_raw)

    html = ServicoMonitoramento.gerar_dashboard()

    assert "Aguardando mais dados" in html


def test_gerar_dashboard_sucesso(monkeypatch):
    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", lambda path: True)

    referencia = pd.DataFrame(
        {
            "prediction": [0, 1, 0, 1, 0],
            "IDADE": [10, 11, 12, 13, 14],
            "GENERO": ["Masculino", "Feminino", "Masculino", "Feminino", "Masculino"],
            Configuracoes.TARGET_COL: [0, 1, 0, 1, 0],
        }
    )
    atual_raw = pd.DataFrame({
        "input_features": [
            {"IDADE": 10, "GENERO": "Masculino"},
            {"IDADE": 11, "GENERO": "Feminino"},
            {"IDADE": 12, "GENERO": "Masculino"},
            {"IDADE": 13, "GENERO": "Outro"},
            {"IDADE": 14, "GENERO": "Masculino"},
        ],
        "prediction_result": [
            {"class": 0}, {"class": 1}, {"class": 0}, {"class": 1}, {"class": 0}
        ],
    })

    monkeypatch.setattr("src.application.monitoring_service.pd.read_csv", lambda path: referencia)
    monkeypatch.setattr("src.application.monitoring_service.ServicoMonitoramento._ler_ultimas_linhas", lambda *_: ["{}"])
    monkeypatch.setattr("src.application.monitoring_service.pd.read_json", lambda *args, **kwargs: atual_raw)

    relatorio = Mock()
    relatorio.get_html.return_value = "<html>ok</html>"

    def fabrica_relatorio(*args, **kwargs):
        return relatorio

    monkeypatch.setattr("src.application.monitoring_service.Report", fabrica_relatorio)

    html = ServicoMonitoramento.gerar_dashboard()

    assert "<html>ok</html>" in html
    assert "Fairness por Grupo" in html
    assert "Gap de FPR" in html
    relatorio.run.assert_called_once()


def test_gerar_dashboard_trata_excecao(monkeypatch):
    monkeypatch.setattr("src.application.monitoring_service.os.path.exists", lambda path: True)
    monkeypatch.setattr(
        "src.application.monitoring_service.pd.read_csv",
        lambda path: pd.DataFrame({"prediction": [1, 0, 1, 0, 1], "IDADE": [10, 11, 12, 13, 14]}),
    )
    monkeypatch.setattr(
        "src.application.monitoring_service.pd.read_json",
        lambda *args, **kwargs: pd.DataFrame(
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
    monkeypatch.setattr("src.application.monitoring_service.ServicoMonitoramento._ler_ultimas_linhas", lambda *_: ["{}"])

    def levantar_erro(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("src.application.monitoring_service.Report", levantar_erro)

    html = ServicoMonitoramento.gerar_dashboard()

    assert "Erro interno" in html


def test_resumir_gaps_fairness_badges():
    metricas = pd.DataFrame(
        {
            "GENERO": ["A", "B"],
            "false_positive_rate_pct": [3.0, 10.0],
            "false_negative_rate_pct": [2.0, 15.0],
            "support": [10, 12],
        }
    )

    html = ServicoMonitoramento._resumir_gaps_fairness(metricas)

    assert "FPR med" in html
    assert "FNR high" in html


def test_renderizar_barras_fairness():
    metricas = pd.DataFrame(
        {
            "GENERO": ["A", "B"],
            "false_positive_rate_pct": [10.0, 20.0],
            "false_negative_rate_pct": [5.0, 15.0],
            "support": [10, 12],
        }
    )

    html = ServicoMonitoramento._renderizar_barras_fairness(metricas)

    assert "fairness-bars" in html
    assert "A · FPR" in html
    assert "B · FNR" in html
    assert "10.00%" in html


def test_gerar_fairness_html_inclui_estilos_e_barras():
    referencia = pd.DataFrame(
        {
            "GENERO": ["Feminino", "Masculino", "Feminino", "Masculino"],
            Configuracoes.TARGET_COL: [0, 1, 0, 1],
            "prediction": [0, 1, 1, 0],
        }
    )
    atual = referencia.copy()

    html = ServicoMonitoramento._gerar_fairness_html(referencia, atual)

    assert "fairness-wrapper" in html
    assert "fairness-table" in html
    assert "fairness-bars" in html


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

    arquivo_mock.assert_called_once()
    escrito = "".join(chamada.args[0] for chamada in arquivo_mock().write.call_args_list)
    payload = json.loads(escrito)
    assert "psi_by_feature" in payload
    assert "top_drift_features" in payload
    assert "drift_status" in payload
    assert "drift_history" in payload
    assert "high_risk_rate_history" in payload
