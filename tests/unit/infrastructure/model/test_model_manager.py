"""Testes do gerenciador de modelo."""

from unittest.mock import Mock

import pytest

from src.infrastructure.model.model_manager import GerenciadorModelo


def resetar_gerenciador():
    GerenciadorModelo._instancia = None
    GerenciadorModelo._modelo = None


def test_gerenciador_singleton():
    resetar_gerenciador()
    primeiro = GerenciadorModelo()
    segundo = GerenciadorModelo()
    assert primeiro is segundo


def test_carregar_modelo_arquivo_inexistente(monkeypatch):
    resetar_gerenciador()
    monkeypatch.setattr("src.infrastructure.model.model_manager.os.path.exists", lambda path: False)

    gerenciador = GerenciadorModelo()
    with pytest.raises(FileNotFoundError):
        gerenciador.carregar_modelo()


def test_carregar_modelo_sucesso(monkeypatch):
    resetar_gerenciador()
    monkeypatch.setattr("src.infrastructure.model.model_manager.os.path.exists", lambda path: True)
    monkeypatch.setattr("src.infrastructure.model.model_manager.Configuracoes.MODEL_SHA256_REQUIRED", False)
    modelo = Mock()
    monkeypatch.setattr("src.infrastructure.model.model_manager.load", lambda path: modelo)

    gerenciador = GerenciadorModelo()
    gerenciador.carregar_modelo()

    assert gerenciador.obter_modelo() is modelo


def test_carregar_modelo_falha(monkeypatch):
    resetar_gerenciador()
    monkeypatch.setattr("src.infrastructure.model.model_manager.os.path.exists", lambda path: True)
    monkeypatch.setattr("src.infrastructure.model.model_manager.Configuracoes.MODEL_SHA256_REQUIRED", False)

    def levantar_erro(path):
        raise RuntimeError("boom")

    monkeypatch.setattr("src.infrastructure.model.model_manager.load", levantar_erro)

    gerenciador = GerenciadorModelo()
    with pytest.raises(RuntimeError):
        gerenciador.carregar_modelo()


def test_obter_modelo_indisponivel(monkeypatch):
    resetar_gerenciador()
    monkeypatch.setattr("src.infrastructure.model.model_manager.os.path.exists", lambda path: False)

    gerenciador = GerenciadorModelo()

    with pytest.raises(FileNotFoundError):
        gerenciador.obter_modelo()


def test_carregar_modelo_sem_recarregar(monkeypatch):
    resetar_gerenciador()
    monkeypatch.setattr("src.infrastructure.model.model_manager.os.path.exists", lambda path: True)
    monkeypatch.setattr("src.infrastructure.model.model_manager.Configuracoes.MODEL_SHA256_REQUIRED", False)
    modelo = Mock()
    monkeypatch.setattr("src.infrastructure.model.model_manager.load", lambda path: modelo)

    gerenciador = GerenciadorModelo()
    gerenciador.carregar_modelo()
    gerenciador.carregar_modelo()

    assert gerenciador.obter_modelo() is modelo
