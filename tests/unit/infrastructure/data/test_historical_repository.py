"""Testes do repositório histórico."""

from unittest.mock import Mock

import pandas as pd

from src.infrastructure.data.historical_repository import RepositorioHistorico


def resetar_repositorio():
    RepositorioHistorico._instancia = None
    RepositorioHistorico._dados = None


def test_repositorio_carrega_referencia_com_ra(monkeypatch):
    resetar_repositorio()

    monkeypatch.setattr(
        "src.infrastructure.data.historical_repository.Configuracoes.HISTORICAL_PATH",
        "/tmp/historico.csv",
    )
    monkeypatch.setattr("src.infrastructure.data.historical_repository.os.path.exists", lambda path: True)
    monkeypatch.setattr(
        "src.infrastructure.data.historical_repository.pd.read_csv",
        lambda path: pd.DataFrame(
            {
                "RA": ["1", "1"],
                "ANO_REFERENCIA": [2022, 2023],
                "INDE": [5.0, 6.0],
            }
        ),
    )

    repo = RepositorioHistorico()
    historico = repo.obter_historico_estudante("1")

    assert historico["INDE_ANTERIOR"] == 6.0


def test_repositorio_recarrega_quando_referencia_sem_ra(monkeypatch):
    resetar_repositorio()

    monkeypatch.setattr(
        "src.infrastructure.data.historical_repository.Configuracoes.HISTORICAL_PATH",
        "/tmp/historico.csv",
    )
    monkeypatch.setattr("src.infrastructure.data.historical_repository.os.path.exists", lambda path: True)
    monkeypatch.setattr(
        "src.infrastructure.data.historical_repository.pd.read_csv",
        lambda path: pd.DataFrame({"ANO_REFERENCIA": [2023]}),
    )

    carregador_mock = Mock()
    carregador_mock.carregar_dados.return_value = pd.DataFrame(
        {"RA": ["2", "2"], "ANO_REFERENCIA": [2022, 2023], "INDE": [3.0, 4.0]}
    )
    monkeypatch.setattr("src.infrastructure.data.data_loader.CarregadorDados", lambda: carregador_mock)

    repo = RepositorioHistorico()
    historico = repo.obter_historico_estudante("2")

    assert historico["INDE_ANTERIOR"] == 4.0


def test_repositorio_retorna_vazio_sem_dados(monkeypatch):
    resetar_repositorio()

    monkeypatch.setattr(
        "src.infrastructure.data.historical_repository.Configuracoes.HISTORICAL_PATH",
        "/tmp/historico.csv",
    )
    monkeypatch.setattr("src.infrastructure.data.historical_repository.os.path.exists", lambda path: False)

    carregador_mock = Mock()
    carregador_mock.carregar_dados.side_effect = RuntimeError("boom")
    monkeypatch.setattr("src.infrastructure.data.data_loader.CarregadorDados", lambda: carregador_mock)

    repo = RepositorioHistorico()

    assert repo.obter_historico_estudante("1") == {}


def test_repositorio_retorna_none_quando_estudante_nao_existe(monkeypatch):
    resetar_repositorio()

    monkeypatch.setattr(
        "src.infrastructure.data.historical_repository.Configuracoes.HISTORICAL_PATH",
        "/tmp/historico.csv",
    )
    monkeypatch.setattr("src.infrastructure.data.historical_repository.os.path.exists", lambda path: True)
    monkeypatch.setattr(
        "src.infrastructure.data.historical_repository.pd.read_csv",
        lambda path: pd.DataFrame(
            {
                "RA": ["1", "1"],
                "ANO_REFERENCIA": [2022, 2023],
                "INDE": [5.0, 6.0],
            }
        ),
    )

    repo = RepositorioHistorico()

    assert repo.obter_historico_estudante("999") is None


def test_repositorio_respeita_ano_referencia(monkeypatch):
    resetar_repositorio()

    monkeypatch.setattr(
        "src.infrastructure.data.historical_repository.Configuracoes.HISTORICAL_PATH",
        "/tmp/historico.csv",
    )
    monkeypatch.setattr("src.infrastructure.data.historical_repository.os.path.exists", lambda path: True)
    monkeypatch.setattr(
        "src.infrastructure.data.historical_repository.pd.read_csv",
        lambda path: pd.DataFrame(
            {
                "RA": ["1", "1", "1"],
                "ANO_REFERENCIA": [2022, 2023, 2024],
                "INDE": [5.0, 6.0, 7.0],
            }
        ),
    )

    repo = RepositorioHistorico()
    historico_2024 = repo.obter_historico_estudante("1", ano_referencia=2024)
    assert historico_2024["INDE_ANTERIOR"] == 6.0

    historico_2023 = repo.obter_historico_estudante("1", ano_referencia=2023)
    assert historico_2023["INDE_ANTERIOR"] == 5.0


def test_repositorio_trata_valores_invalidos(monkeypatch):
    resetar_repositorio()

    monkeypatch.setattr(
        "src.infrastructure.data.historical_repository.Configuracoes.HISTORICAL_PATH",
        "/tmp/historico.csv",
    )
    monkeypatch.setattr("src.infrastructure.data.historical_repository.os.path.exists", lambda path: True)
    monkeypatch.setattr(
        "src.infrastructure.data.historical_repository.pd.read_csv",
        lambda path: pd.DataFrame(
            {
                "RA": ["1", "1"],
                "ANO_REFERENCIA": [2022, 2023],
                "INDE": ["x", 1.0],
                "IAA": [None, 2.0],
            }
        ),
    )

    repo = RepositorioHistorico()
    historico = repo.obter_historico_estudante("1", ano_referencia=2023)

    assert historico["INDE_ANTERIOR"] == 0.0
    assert historico["IAA_ANTERIOR"] == 0.0
