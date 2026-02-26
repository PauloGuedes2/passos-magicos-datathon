"""Testes do controlador de predicao."""

from unittest.mock import Mock

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.api.controller import ControladorPredicao, obter_servico_risco
from src.domain.student import EntradaEstudante


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


def test_listar_versoes_modelo_sucesso(monkeypatch):
    from src.api import controller as modulo_controlador

    monkeypatch.setattr(
        modulo_controlador,
        "listar_versoes_modelo_runtime",
        lambda: ["v2026.02.25-120000", "v2026.02.24-120000"],
    )

    aplicacao = FastAPI()
    controlador = ControladorPredicao()
    aplicacao.include_router(controlador.roteador, prefix="/api/v1")
    cliente = TestClient(aplicacao)

    resposta = cliente.get("/api/v1/models/versions")
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["available_model_versions"][0] == "v2026.02.25-120000"
    assert corpo["default_model_version"] == "v2026.02.25-120000"


def test_listar_versoes_modelo_erro(monkeypatch):
    from src.api import controller as modulo_controlador

    def _falhar():
        raise RuntimeError("boom")

    monkeypatch.setattr(modulo_controlador, "listar_versoes_modelo_runtime", _falhar)

    aplicacao = FastAPI()
    controlador = ControladorPredicao()
    aplicacao.include_router(controlador.roteador, prefix="/api/v1")
    cliente = TestClient(aplicacao)

    resposta = cliente.get("/api/v1/models/versions")
    assert resposta.status_code == 503
    assert "Falha ao listar versoes de modelo" in resposta.json()["detail"]


def test_obter_servico_risco_sem_modelo(monkeypatch):
    from src.api import controller as modulo_controlador

    def _falhar_obter_modelo(**kwargs):
        raise RuntimeError("missing")

    monkeypatch.setattr(modulo_controlador, "obter_modelo_runtime", _falhar_obter_modelo)

    try:
        modulo_controlador.obter_servico_risco()
    except HTTPException as erro:
        assert erro.status_code == 503
    else:
        raise AssertionError("HTTPException esperada")


def test_modelos_pydantic_validam(entrada_estudante_exemplo):
    entrada = EntradaEstudante(**entrada_estudante_exemplo)

    assert entrada.RA == "123"


def test_obter_servico_risco_com_versao_especifica(monkeypatch):
    from src.api import controller as modulo_controlador

    modelo_fake = object()

    def _obter_modelo_runtime(model_version=None):
        assert model_version == "v2026.02.25-120000"
        return modelo_fake, "v2026.02.25-120000"

    monkeypatch.setattr(modulo_controlador, "obter_modelo_runtime", _obter_modelo_runtime)

    servico = modulo_controlador.obter_servico_risco(model_version="v2026.02.25-120000")
    assert servico.modelo is modelo_fake
    assert servico.model_version == "v2026.02.25-120000"


def test_obter_servico_risco_modelo_nao_encontrado(monkeypatch):
    from src.api import controller as modulo_controlador

    def _falhar_obter_modelo_runtime(model_version=None):
        raise FileNotFoundError("nao encontrado")

    monkeypatch.setattr(modulo_controlador, "obter_modelo_runtime", _falhar_obter_modelo_runtime)

    try:
        modulo_controlador.obter_servico_risco(model_version="v-invalida")
    except HTTPException as erro:
        assert erro.status_code == 404
    else:
        raise AssertionError("HTTPException esperada")
