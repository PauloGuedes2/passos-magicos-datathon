"""Testes do controlador de monitoramento."""

from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.monitoring_controller as monitoring_controller
from src.api.monitoring_controller import ControladorMonitoramento, obter_servico_monitoramento


def test_endpoint_painel_gera_dashboard_se_arquivo_nao_existe(tmp_path, monkeypatch):
    aplicacao = FastAPI()
    controlador = ControladorMonitoramento()
    aplicacao.include_router(controlador.roteador, prefix="/api/v1/monitoring")

    monitoring_dir = tmp_path / "monitoring"
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(monitoring_controller.Configuracoes, "MONITORING_DIR", str(monitoring_dir))

    output_file = monitoring_dir / "professional_dashboard.html"

    service_mock = Mock()

    def _generate():
        output_file.write_text("<html><body>novo dashboard</body></html>", encoding="utf-8")
        return str(output_file)

    service_mock.generate_dashboard.side_effect = _generate
    monkeypatch.setattr(monitoring_controller, "ProfessionalDashboardService", lambda: service_mock)

    cliente = TestClient(aplicacao)
    resposta = cliente.get("/api/v1/monitoring/dashboard")

    assert resposta.status_code == 200
    assert "novo dashboard" in resposta.text
    assert "text/html" in resposta.headers["content-type"].lower()
    assert "content-disposition" not in resposta.headers
    assert resposta.headers["cache-control"].startswith("no-store")


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
