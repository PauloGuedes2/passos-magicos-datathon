"""Testes do gerenciador de modelo."""

from unittest.mock import Mock

import pytest

from src.infrastructure.model.model_manager import GerenciadorModelo


def resetar_gerenciador():
    GerenciadorModelo._instancia = None
    GerenciadorModelo._modelo = None
    GerenciadorModelo._modelos_por_versao = {}


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
    modelo = Mock()
    monkeypatch.setattr("src.infrastructure.model.model_manager.load", lambda path: modelo)

    gerenciador = GerenciadorModelo()
    gerenciador.carregar_modelo()

    assert gerenciador.obter_modelo() is modelo


def test_carregar_modelo_falha(monkeypatch):
    resetar_gerenciador()
    monkeypatch.setattr("src.infrastructure.model.model_manager.os.path.exists", lambda path: True)

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
    modelo = Mock()
    monkeypatch.setattr("src.infrastructure.model.model_manager.load", lambda path: modelo)

    gerenciador = GerenciadorModelo()
    gerenciador.carregar_modelo()
    gerenciador.carregar_modelo()

    assert gerenciador.obter_modelo() is modelo


def test_carregar_modelo_por_versao_sucesso(monkeypatch):
    resetar_gerenciador()
    monkeypatch.setattr("src.infrastructure.model.model_manager.os.path.exists", lambda path: True)
    modelo = Mock()
    monkeypatch.setattr("src.infrastructure.model.model_manager.load", lambda path: modelo)

    gerenciador = GerenciadorModelo()
    versao = gerenciador.carregar_modelo(model_version="v2026.02.25-120000")

    assert versao == "v2026.02.25-120000"
    assert gerenciador.obter_modelo(model_version="v2026.02.25-120000") is modelo


def test_aplicar_retencao_modelos_remove_excedentes(monkeypatch):
    resetar_gerenciador()

    monkeypatch.setattr("src.infrastructure.model.model_manager.Configuracoes.MAX_MODEL_VERSIONS", 3)
    monkeypatch.setattr("src.infrastructure.model.model_manager.os.path.isdir", lambda path: True)
    monkeypatch.setattr(
        "src.infrastructure.model.model_manager.os.listdir",
        lambda path: ["model_v1.joblib", "model_v2.joblib", "model_v3.joblib", "model_v4.joblib"],
    )
    monkeypatch.setattr(
        "src.infrastructure.model.model_manager.os.path.getmtime",
        lambda path: {"v1": 1, "v2": 2, "v3": 3, "v4": 4}[path.split("_")[-1].split(".")[0]],
    )

    removidos = []
    monkeypatch.setattr("src.infrastructure.model.model_manager.os.remove", lambda path: removidos.append(path))

    gerenciador = GerenciadorModelo()
    gerenciador._aplicar_retencao_modelos()

    assert len(removidos) == 1
    assert removidos[0].endswith("model_v1.joblib")
