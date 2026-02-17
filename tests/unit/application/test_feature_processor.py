"""Testes do processador de features."""

from datetime import datetime
import pandas as pd

from src.application.feature_processor import ProcessadorFeatures
from src.config.settings import Configuracoes


def test_processar_preenche_colunas_e_usa_snapshot():
    df = pd.DataFrame([
        {
            "ANO_INGRESSO": 2020,
            "IDADE": "11",
            "GENERO": None,
        }
    ])

    data_snapshot = datetime(2024, 1, 1)
    processado = ProcessadorFeatures.processar(df, data_snapshot=data_snapshot)

    assert "TEMPO_NA_ONG" in processado.columns
    assert processado.loc[0, "TEMPO_NA_ONG"] == 4
    assert processado.loc[0, "IDADE"] == 11
    valor_genero = processado.loc[0, "GENERO"]
    assert valor_genero == "Outro"
    for coluna in Configuracoes.FEATURES_NUMERICAS + Configuracoes.FEATURES_CATEGORICAS:
        assert coluna in processado.columns


def test_processar_ano_ingresso_nulo_com_estatisticas():
    df = pd.DataFrame([
        {
            "ANO_INGRESSO": None,
            "IDADE": 10,
            "GENERO": "Masculino",
            "TURMA": "A",
            "INSTITUICAO_ENSINO": "Escola",
            "FASE": "1A",
        }
    ])

    data_snapshot = datetime(2024, 1, 1)
    processado = ProcessadorFeatures.processar(
        df, data_snapshot=data_snapshot, estatisticas={"mediana_ano_ingresso": 2019}
    )
    assert processado.loc[0, "TEMPO_NA_ONG"] >= 0


def test_processar_sem_ano_ingresso():
    df = pd.DataFrame([
        {
            "IDADE": 9,
            "GENERO": "menina",
            "TURMA": "B",
            "INSTITUICAO_ENSINO": "Escola",
            "FASE": "fase 2b",
        }
    ])

    data_snapshot = datetime(2024, 1, 1)
    processado = ProcessadorFeatures.processar(df, data_snapshot=data_snapshot)
    assert processado.loc[0, "TEMPO_NA_ONG"] == 0
    assert processado.loc[0, "GENERO"] == "Feminino"
    assert processado.loc[0, "FASE"] == "FASE2B"


def test_processar_idade_fora_faixa_aplica_mediana():
    df = pd.DataFrame([
        {
            "IDADE": 0,
            "ANO_INGRESSO": 2022,
            "GENERO": "Masculino",
            "TURMA": "A",
            "INSTITUICAO_ENSINO": "Escola",
            "FASE": "1A",
        }
    ])

    processado = ProcessadorFeatures.processar(
        df,
        data_snapshot=datetime(2024, 1, 1),
        estatisticas={"mediana_idade": 13, "mediana_ano_ingresso": 2021},
    )
    assert processado.loc[0, "IDADE"] == 13
