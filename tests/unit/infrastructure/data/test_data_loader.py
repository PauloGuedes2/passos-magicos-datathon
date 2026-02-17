"""Testes do carregador de dados."""

import pandas as pd
import pytest

from src.infrastructure.data.data_loader import CarregadorDados


def test_carregar_dados_sem_arquivos(monkeypatch):
    monkeypatch.setattr("src.infrastructure.data.data_loader.glob.glob", lambda path: [])
    monkeypatch.setattr("src.infrastructure.data.data_loader.os.listdir", lambda path: ["file.txt"])

    carregador = CarregadorDados()

    with pytest.raises(FileNotFoundError):
        carregador.carregar_dados()


def test_carregar_dados_excel_invalido(monkeypatch):
    monkeypatch.setattr("src.infrastructure.data.data_loader.glob.glob", lambda path: ["file.xlsx"])

    def levantar_erro(*args, **kwargs):
        raise RuntimeError("invalid")

    monkeypatch.setattr("src.infrastructure.data.data_loader.pd.read_excel", levantar_erro)

    carregador = CarregadorDados()

    with pytest.raises(RuntimeError):
        carregador.carregar_dados()


def test_carregar_dados_ignora_abas_sem_ano(monkeypatch):
    monkeypatch.setattr("src.infrastructure.data.data_loader.glob.glob", lambda path: ["file.xlsx"])

    dados_abas = {
        "Resumo": pd.DataFrame({
            "RA": ["1"],
            "IDADE": [12],
            "ANO_INGRESSO": [2020],
            "GENERO": ["Masculino"],
            "TURMA": ["6A"],
            "INSTITUICAO_ENSINO": ["MUNICIPAL"],
            "FASE": ["6A"],
        }),
        "2023": pd.DataFrame({
            "RA": ["1"],
            "IDADE": [12],
            "ANO_INGRESSO": [2020],
            "GENERO": ["Masculino"],
            "TURMA": ["6A"],
            "INSTITUICAO_ENSINO": ["MUNICIPAL"],
            "FASE": ["6A"],
        }),
    }

    monkeypatch.setattr("src.infrastructure.data.data_loader.pd.read_excel", lambda *args, **kwargs: dados_abas)

    carregador = CarregadorDados()
    df = carregador.carregar_dados()

    assert "ANO_REFERENCIA" in df.columns
    assert df["ANO_REFERENCIA"].iloc[0] == 2023


def test_carregar_dados_erro_concat(monkeypatch):
    monkeypatch.setattr("src.infrastructure.data.data_loader.glob.glob", lambda path: ["file.xlsx"])

    dados_abas = {
        "2023": pd.DataFrame({"RA": ["1"], "ANO_INGRESSO": [2020]}),
        "2024": pd.DataFrame({"RA": ["2"], "ANO_INGRESSO": [2021]}),
    }

    monkeypatch.setattr("src.infrastructure.data.data_loader.pd.read_excel", lambda *args, **kwargs: dados_abas)

    def levantar_concat(*args, **kwargs):
        raise RuntimeError("concat")

    monkeypatch.setattr("src.infrastructure.data.data_loader.pd.concat", levantar_concat)

    carregador = CarregadorDados()

    with pytest.raises(RuntimeError):
        carregador.carregar_dados()


def test_processar_dataframe_normaliza_colunas():
    df = pd.DataFrame({
        "ra": ["1"],
        "matematica 2023": [10],
        "port 23": [9],
        "ing": [8],
        "defasagem": [0],
        "ano ingresso": [2020],
        "inst ensino": ["Escola"],
    })

    processado = CarregadorDados._processar_dataframe(df, 2023)

    assert "RA" in processado.columns
    assert "NOTA_MAT" in processado.columns
    assert "NOTA_PORT" in processado.columns
    assert "NOTA_ING" in processado.columns
    assert "DEFASAGEM" in processado.columns
    assert "ANO_INGRESSO" in processado.columns
    assert "INSTITUICAO_ENSINO" in processado.columns
