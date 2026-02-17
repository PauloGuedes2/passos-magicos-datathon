"""Testes do script de treinamento."""

from unittest.mock import Mock

import runpy


def test_treinamento_sucesso(monkeypatch):
    carregador_mock = Mock()
    carregador_mock.carregar_dados.return_value = Mock()

    treinador_mock = Mock()

    monkeypatch.setattr("src.infrastructure.data.data_loader.CarregadorDados", lambda: carregador_mock)
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.treinador", treinador_mock)

    runpy.run_module("train", run_name="__main__")

    carregador_mock.carregar_dados.assert_called_once()
    treinador_mock.treinar.assert_called_once_with(carregador_mock.carregar_dados.return_value)


def test_treinamento_falha(monkeypatch):
    carregador_mock = Mock()
    carregador_mock.carregar_dados.side_effect = RuntimeError("boom")

    treinador_mock = Mock()

    monkeypatch.setattr("src.infrastructure.data.data_loader.CarregadorDados", lambda: carregador_mock)
    monkeypatch.setattr("src.infrastructure.model.ml_pipeline.treinador", treinador_mock)

    exit_mock = Mock()
    monkeypatch.setattr("builtins.exit", exit_mock)

    runpy.run_module("train", run_name="__main__")

    exit_mock.assert_called_once_with(1)
