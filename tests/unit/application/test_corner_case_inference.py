"""Cenarios extremos de inferencia para detectar comportamento patologico."""

import numpy as np

from src.application.risk_service import ServicoRisco
from src.domain.student import EntradaEstudante


class ModeloConstante:
    def __init__(self, prob: float):
        self.prob = float(prob)

    def predict_proba(self, dados):
        return np.tile([1.0 - self.prob, self.prob], (len(dados), 1))


def _servico(prob=0.6):
    servico = ServicoRisco(modelo=ModeloConstante(prob))
    servico.logger = type("LoggerNoop", (), {"registrar_predicao": lambda *args, **kwargs: None})()
    return servico


def test_cenario_zero_history_exceto_uma_feature():
    servico = _servico(0.55)
    payload = {
        "RA": "X1",
        "IDADE": 12,
        "ANO_INGRESSO": 2023,
        "GENERO": "Masculino",
        "TURMA": "1A",
        "INSTITUICAO_ENSINO": "Publica",
        "FASE": "1A",
        "ANO_REFERENCIA": 2024,
        "INDE_ANTERIOR": 0.0,
        "IAA_ANTERIOR": 0.0,
        "IEG_ANTERIOR": 0.0,
        "IPS_ANTERIOR": 0.0,
        "IDA_ANTERIOR": 0.0,
        "IPP_ANTERIOR": 0.0,
        "IPV_ANTERIOR": 0.0,
        "IAN_ANTERIOR": 10.0,
        "ALUNO_NOVO": 1,
    }
    resposta = servico.prever_risco(payload)
    assert "prediction" in resposta
    assert "risk_probability" in resposta


def test_cenario_historico_apenas_ano_mais_antigo():
    servico = _servico(0.2)
    servico.repositorio = type(
        "RepoAntigo",
        (),
        {"obter_historico_estudante": lambda *args, **kwargs: None},
    )()
    entrada = EntradaEstudante(
        RA="X2",
        IDADE=11,
        ANO_INGRESSO=2024,
        GENERO="Feminino",
        TURMA="1B",
        INSTITUICAO_ENSINO="Publica",
        FASE="1B",
        ANO_REFERENCIA=2024,
    )
    resposta = servico.prever_risco_inteligente(entrada)
    assert resposta["requires_human_review"] is True


def test_cenario_categorias_inesperadas_no_servico_bruto():
    servico = _servico(0.4)
    payload = {
        "RA": "X3",
        "IDADE": 14,
        "ANO_INGRESSO": 2022,
        "GENERO": "Nao Informado",
        "TURMA": "???",
        "INSTITUICAO_ENSINO": "Outro",
        "FASE": "fase*estranha",
        "ANO_REFERENCIA": 2024,
        "INDE_ANTERIOR": 5.0,
        "IAA_ANTERIOR": 5.0,
        "IEG_ANTERIOR": 5.0,
        "IPS_ANTERIOR": 5.0,
        "IDA_ANTERIOR": 5.0,
        "IPP_ANTERIOR": 5.0,
        "IPV_ANTERIOR": 5.0,
        "IAN_ANTERIOR": 5.0,
        "ALUNO_NOVO": 0,
    }
    resposta = servico.prever_risco(payload)
    assert resposta["risk_label"] in ("ALTO RISCO", "BAIXO RISCO")


def test_cenario_tail_risk_combinacao_extrema():
    servico = _servico(0.95)
    payload = {
        "RA": "X4",
        "IDADE": 25,
        "ANO_INGRESSO": 2010,
        "GENERO": "Outro",
        "TURMA": "9",
        "INSTITUICAO_ENSINO": "Privada",
        "FASE": "9",
        "ANO_REFERENCIA": 2024,
        "INDE_ANTERIOR": 0.0,
        "IAA_ANTERIOR": 0.0,
        "IEG_ANTERIOR": 0.0,
        "IPS_ANTERIOR": 0.0,
        "IDA_ANTERIOR": 0.0,
        "IPP_ANTERIOR": 0.0,
        "IPV_ANTERIOR": 0.0,
        "IAN_ANTERIOR": 0.0,
        "ALUNO_NOVO": 1,
    }
    resposta = servico.prever_risco(payload)
    assert resposta["prediction"] == 1
