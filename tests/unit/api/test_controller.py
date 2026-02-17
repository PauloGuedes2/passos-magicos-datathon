"""Testes do controlador de predição."""

from unittest.mock import Mock

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.api.controller import ControladorPredicao, obter_servico_risco
from src.domain.student import Estudante, EntradaEstudante


def test_predicao_completa_sucesso(estudante_exemplo):
    aplicacao = FastAPI()
    controlador = ControladorPredicao()

    servico = Mock()
    servico.prever_risco.return_value = {"prediction": 1}

    def override():
        return servico

    aplicacao.dependency_overrides[obter_servico_risco] = override
    aplicacao.include_router(controlador.roteador, prefix="/api/v1")

    cliente = TestClient(aplicacao)
    resposta = cliente.post("/api/v1/predict/full", json=estudante_exemplo)

    assert resposta.status_code == 200
    assert resposta.json()["prediction"] == 1


def test_predicao_completa_erro(estudante_exemplo):
    aplicacao = FastAPI()
    controlador = ControladorPredicao()

    servico = Mock()
    servico.prever_risco.side_effect = RuntimeError("boom")

    def override():
        return servico

    aplicacao.dependency_overrides[obter_servico_risco] = override
    aplicacao.include_router(controlador.roteador, prefix="/api/v1")

    cliente = TestClient(aplicacao)
    resposta = cliente.post("/api/v1/predict/full", json=estudante_exemplo)

    assert resposta.status_code == 503
    assert "boom" in resposta.json()["detail"]


def test_predicao_inteligente_sucesso(entrada_estudante_exemplo):
    aplicacao = FastAPI()
    controlador = ControladorPredicao()

    servico = Mock()
    servico.prever_risco_inteligente.return_value = {"prediction": 0}

    def override():
        return servico

    aplicacao.dependency_overrides[obter_servico_risco] = override
    aplicacao.include_router(controlador.roteador, prefix="/api/v1")

    cliente = TestClient(aplicacao)
    resposta = cliente.post("/api/v1/predict/smart", json=entrada_estudante_exemplo)

    assert resposta.status_code == 200
    assert resposta.json()["prediction"] == 0


def test_obter_servico_risco_sem_modelo(monkeypatch):
    from src.api import controller as modulo_controlador

    def _falhar_obter_modelo():
        raise RuntimeError("missing")

    monkeypatch.setattr(modulo_controlador, "obter_modelo_runtime", _falhar_obter_modelo)

    try:
        modulo_controlador.obter_servico_risco()
    except HTTPException as erro:
        assert erro.status_code == 503
    else:
        raise AssertionError("HTTPException esperada")


def test_modelos_pydantic_validam(estudante_exemplo, entrada_estudante_exemplo):
    estudante = Estudante(**estudante_exemplo)
    entrada = EntradaEstudante(**entrada_estudante_exemplo)

    assert estudante.RA == "123"
    assert entrada.RA == "123"
