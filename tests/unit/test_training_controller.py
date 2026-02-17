"""Testes do controlador de treinamento."""

from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.training_controller import ControladorTreinamento, obter_servico_treinamento


def test_retreino_sucesso():
    aplicacao = FastAPI()
    controlador = ControladorTreinamento()

    servico = Mock()
    servico.executar_treinamento.return_value = {"status": "ok"}

    def override():
        return servico

    aplicacao.dependency_overrides[obter_servico_treinamento] = override
    aplicacao.include_router(controlador.roteador, prefix="/api/v1")

    cliente = TestClient(aplicacao)
    resposta = cliente.post("/api/v1/train/retrain")

    assert resposta.status_code == 200
    assert resposta.json()["status"] == "ok"


def test_retreino_erro_runtime():
    aplicacao = FastAPI()
    controlador = ControladorTreinamento()

    servico = Mock()
    servico.executar_treinamento.side_effect = RuntimeError("boom")

    def override():
        return servico

    aplicacao.dependency_overrides[obter_servico_treinamento] = override
    aplicacao.include_router(controlador.roteador, prefix="/api/v1")

    cliente = TestClient(aplicacao)
    resposta = cliente.post("/api/v1/train/retrain")

    assert resposta.status_code == 503
    assert "boom" in resposta.json()["detail"]
