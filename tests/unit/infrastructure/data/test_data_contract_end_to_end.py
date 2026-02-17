"""Testes de contrato de dados em carga e API."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.controller import ControladorPredicao, obter_servico_risco
from src.infrastructure.data.data_contract import CONTRATO_TREINO
from src.infrastructure.data.data_loader import CarregadorDados


def test_data_contract_on_train_load():
    dados = CarregadorDados().carregar_dados()
    # Nao deve lançar exceção
    CONTRATO_TREINO.validar(dados)
    assert len(dados) > 0


def test_data_contract_on_api_input():
    app = FastAPI()
    controlador = ControladorPredicao()
    app.include_router(controlador.roteador, prefix="/api/v1")
    app.dependency_overrides[obter_servico_risco] = lambda: None
    client = TestClient(app)

    payload_invalido = {
        "IDADE": 12,
        "ANO_INGRESSO": 2023,
        "GENERO": "Feminino",
        "TURMA": "1A",
        "INSTITUICAO_ENSINO": "Publica",
        "FASE": "1A",
    }
    resposta = client.post("/api/v1/predict/smart", json=payload_invalido)

    assert resposta.status_code == 422
