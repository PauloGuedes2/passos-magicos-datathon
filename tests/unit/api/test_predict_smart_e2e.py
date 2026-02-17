"""Testes E2E para endpoint /predict/smart."""

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.controller import ControladorPredicao, obter_servico_risco
from src.application.risk_service import ServicoRisco


class ModeloProbConstante:
    """Modelo simples para testes E2E."""

    def __init__(self, prob_risco: float):
        self.prob_risco = prob_risco

    def predict_proba(self, dados):
        n = len(dados)
        prob = float(self.prob_risco)
        return np.tile([1.0 - prob, prob], (n, 1))


def _app_com_servico(servico: ServicoRisco) -> TestClient:
    app = FastAPI()
    controlador = ControladorPredicao()
    app.dependency_overrides[obter_servico_risco] = lambda: servico
    app.include_router(controlador.roteador, prefix="/api/v1")
    return TestClient(app)


def _payload_smart():
    return {
        "RA": "RA999",
        "IDADE": 12,
        "ANO_INGRESSO": 2023,
        "GENERO": "Feminino",
        "TURMA": "1A",
        "INSTITUICAO_ENSINO": "Publica",
        "FASE": "1A",
        "ANO_REFERENCIA": 2024,
    }


def test_predict_smart_with_history():
    servico = ServicoRisco(modelo=ModeloProbConstante(0.9))
    servico.logger = type("LoggerNoop", (), {"registrar_predicao": lambda *args, **kwargs: None})()
    servico.repositorio = type(
        "RepoComHistorico",
        (),
        {
            "obter_historico_estudante": lambda *args, **kwargs: {
                "INDE_ANTERIOR": 7.0,
                "IAA_ANTERIOR": 7.0,
                "IEG_ANTERIOR": 7.0,
                "IPS_ANTERIOR": 7.0,
                "IDA_ANTERIOR": 7.0,
                "IPP_ANTERIOR": 7.0,
                "IPV_ANTERIOR": 7.0,
                "IAN_ANTERIOR": 7.0,
                "ALUNO_NOVO": 0,
            }
        },
    )()
    client = _app_com_servico(servico)

    resposta = client.post("/api/v1/predict/smart", json=_payload_smart())

    assert resposta.status_code == 200
    corpo = resposta.json()
    for chave in (
        "risk_probability",
        "risk_label",
        "prediction",
        "requires_human_review",
        "risk_segment",
        "top_risk_drivers",
    ):
        assert chave in corpo
    assert corpo["requires_human_review"] is False
    assert isinstance(corpo["top_risk_drivers"], list)


def test_predict_smart_without_history():
    servico = ServicoRisco(modelo=ModeloProbConstante(0.2))
    servico.logger = type("LoggerNoop", (), {"registrar_predicao": lambda *args, **kwargs: None})()
    servico.repositorio = type(
        "RepoSemHistorico",
        (),
        {"obter_historico_estudante": lambda *args, **kwargs: None},
    )()
    client = _app_com_servico(servico)

    resposta = client.post("/api/v1/predict/smart", json=_payload_smart())

    assert resposta.status_code == 200
    corpo = resposta.json()
    for chave in (
        "risk_probability",
        "risk_label",
        "prediction",
        "requires_human_review",
        "risk_segment",
        "top_risk_drivers",
    ):
        assert chave in corpo
    assert corpo["requires_human_review"] is True
    assert isinstance(corpo["top_risk_drivers"], list)
