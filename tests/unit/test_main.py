"""Testes do ponto de entrada FastAPI."""

from unittest.mock import Mock

from fastapi.testclient import TestClient

import main


def test_checar_saude():
    cliente = TestClient(main.app)
    resposta = cliente.get("/health")

    assert resposta.status_code == 200
    assert resposta.json() == {"status": "ok"}


def test_evento_inicializacao_carrega_modelo(monkeypatch):
    carregar_modelo = Mock()
    monkeypatch.setattr(main, "carregar_modelo_runtime", carregar_modelo)

    cliente = TestClient(main.app)
    with cliente:
        pass

    carregar_modelo.assert_called_once()
