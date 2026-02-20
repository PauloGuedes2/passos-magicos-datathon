"""Testes do controlador de monitoramento."""

from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.monitoring_controller import ControladorMonitoramento, obter_servico_monitoramento


def test_endpoint_importancia_global():
    aplicacao = FastAPI()
    controlador = ControladorMonitoramento()

    servico = Mock()
    servico.obter_importancia_global.return_value = {
        "feature_importance_ranking": [{"feature": "IDADE", "importance": 0.4}]
    }

    def override():
        return servico

    aplicacao.dependency_overrides[obter_servico_monitoramento] = override
    aplicacao.include_router(controlador.roteador, prefix="/api/v1/monitoring")

    cliente = TestClient(aplicacao)
    resposta = cliente.get("/api/v1/monitoring/feature-importance")

    assert resposta.status_code == 200
    assert "feature_importance_ranking" in resposta.json()
