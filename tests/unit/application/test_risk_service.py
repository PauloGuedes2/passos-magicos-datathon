"""Testes do serviço de risco."""

from unittest.mock import Mock

import numpy as np

from src.application.risk_service import ServicoRisco
from src.domain.student import EntradaEstudante
from src.config.settings import Configuracoes


def test_prever_risco_sucesso(estudante_exemplo):
    modelo = Mock()
    modelo.predict_proba.return_value = np.array([[0.2, 0.8]])

    logger_mock = Mock()
    repo_mock = Mock()

    servico = ServicoRisco(modelo=modelo)
    servico.logger = logger_mock
    servico.repositorio = repo_mock

    resultado = servico.prever_risco(estudante_exemplo)

    assert resultado["prediction"] == 1
    assert resultado["risk_label"] == "ALTO RISCO"
    assert isinstance(resultado["requires_human_review"], bool)
    assert resultado["risk_segment"] == "ALTO_RISCO"
    logger_mock.registrar_predicao.assert_called_once()


def test_prever_risco_limite_threshold(estudante_exemplo):
    modelo = Mock()
    modelo.predict_proba.return_value = np.array([[0.6, Configuracoes.RISK_THRESHOLD]])

    servico = ServicoRisco(modelo=modelo)
    servico.logger = Mock()
    servico._obter_threshold = Mock(return_value=Configuracoes.RISK_THRESHOLD)

    resultado = servico.prever_risco(estudante_exemplo)

    assert resultado["prediction"] == 1
    assert resultado["risk_label"] == "ALTO RISCO"


def test_mudanca_threshold_altera_classificacao(estudante_exemplo):
    modelo = Mock()
    modelo.predict_proba.return_value = np.array([[0.51, 0.49]])

    servico = ServicoRisco(modelo=modelo)
    servico.logger = Mock()
    servico._obter_threshold = Mock(return_value=0.5)
    resultado_threshold_05 = servico.prever_risco(estudante_exemplo)

    servico._obter_threshold = Mock(return_value=0.7)
    resultado_threshold_07 = servico.prever_risco(estudante_exemplo)

    assert resultado_threshold_05["prediction"] == 0
    assert resultado_threshold_07["prediction"] == 0

    modelo.predict_proba.return_value = np.array([[0.35, 0.65]])
    servico._obter_threshold = Mock(return_value=0.6)
    resultado_alto = servico.prever_risco(estudante_exemplo)
    servico._obter_threshold = Mock(return_value=0.7)
    resultado_baixo = servico.prever_risco(estudante_exemplo)
    assert resultado_alto["prediction"] == 1
    assert resultado_baixo["prediction"] == 0


def test_prever_risco_modelo_nulo():
    servico = ServicoRisco(modelo=None)
    servico.logger = Mock()

    try:
        servico.prever_risco({})
    except RuntimeError as erro:
        assert "Modelo não inicializado" in str(erro)
    else:
        raise AssertionError("RuntimeError esperado")


def test_prever_risco_inteligente_com_historico(entrada_estudante_exemplo):
    modelo = Mock()
    modelo.predict_proba.return_value = np.array([[0.3, 0.7]])

    servico = ServicoRisco(modelo=modelo)
    servico.logger = Mock()
    servico.repositorio = Mock()
    servico.repositorio.obter_historico_estudante.return_value = {
        "INDE_ANTERIOR": 1.0,
        "IAA_ANTERIOR": 1.0,
        "IEG_ANTERIOR": 1.0,
        "IPS_ANTERIOR": 1.0,
        "IDA_ANTERIOR": 1.0,
        "IPP_ANTERIOR": 1.0,
        "IPV_ANTERIOR": 1.0,
        "IAN_ANTERIOR": 1.0,
        "ALUNO_NOVO": 0,
    }

    resultado = servico.prever_risco_inteligente(EntradaEstudante(**entrada_estudante_exemplo))

    assert resultado["prediction"] == 1
    assert resultado["requires_human_review"] is False
    assert "top_risk_drivers" in resultado
    assert isinstance(resultado["top_risk_drivers"], list)
    assert len(resultado["top_risk_drivers"]) <= 3
    servico.repositorio.obter_historico_estudante.assert_called_once()


def test_prever_risco_inteligente_sem_historico(entrada_estudante_exemplo):
    modelo = Mock()
    modelo.predict_proba.return_value = np.array([[0.7, 0.2]])

    servico = ServicoRisco(modelo=modelo)
    servico.logger = Mock()
    servico.repositorio = Mock()
    servico.repositorio.obter_historico_estudante.return_value = None

    resultado = servico.prever_risco_inteligente(EntradaEstudante(**entrada_estudante_exemplo))

    assert resultado["prediction"] == 0
    assert resultado["requires_human_review"] is True
    assert "top_risk_drivers" in resultado


def test_prever_risco_inteligente_repasse_ano_referencia(entrada_estudante_exemplo):
    modelo = Mock()
    modelo.predict_proba.return_value = np.array([[0.3, 0.7]])

    servico = ServicoRisco(modelo=modelo)
    servico.logger = Mock()
    servico.repositorio = Mock()
    servico.repositorio.obter_historico_estudante.return_value = {
        "INDE_ANTERIOR": 1.0,
        "IAA_ANTERIOR": 1.0,
        "IEG_ANTERIOR": 1.0,
        "IPS_ANTERIOR": 1.0,
        "IDA_ANTERIOR": 1.0,
        "IPP_ANTERIOR": 1.0,
        "IPV_ANTERIOR": 1.0,
        "IAN_ANTERIOR": 1.0,
        "ALUNO_NOVO": 0,
    }

    entrada = EntradaEstudante(**entrada_estudante_exemplo)
    servico.prever_risco_inteligente(entrada)

    servico.repositorio.obter_historico_estudante.assert_called_once_with(
        entrada.RA, entrada.ANO_REFERENCIA
    )


def test_segmentacao_risco_limites():
    assert ServicoRisco._segmentar_risco(0.29, 0.5) == "BAIXO_RISCO"
    assert ServicoRisco._segmentar_risco(0.3, 0.5) == "MEDIO_RISCO"
    assert ServicoRisco._segmentar_risco(0.5, 0.5) == "ALTO_RISCO"
