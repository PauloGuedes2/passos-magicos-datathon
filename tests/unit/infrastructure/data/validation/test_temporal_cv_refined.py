"""Testes refinados para validacao cruzada temporal."""

import numpy as np
import pandas as pd

from src.infrastructure.data.validation.temporal_cv import TemporalCVSplit, TemporalCVValidator


def _dataset_temporal_sintetico() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    linhas = []
    for ano in [2022, 2023, 2024]:
        for i in range(80):
            ra = f"RA{i:04d}"
            inde = float(rng.uniform(4.0, 9.5))
            linhas.append(
                {
                    "RA": ra,
                    "ANO_REFERENCIA": ano,
                    "ANO_INGRESSO": int(rng.integers(2019, 2024)),
                    "IDADE": int(rng.integers(8, 18)),
                    "GENERO": "Feminino" if i % 2 == 0 else "Masculino",
                    "TURMA": f"{(i % 8) + 1}A",
                    "INSTITUICAO_ENSINO": "Publica",
                    "FASE": f"{(i % 8) + 1}A",
                    "INDE": inde,
                    "DEFASAGEM": -1 if inde < 6.2 else 0,
                }
            )
    return pd.DataFrame(linhas)


def test_temporal_cv_no_leakage():
    dados = _dataset_temporal_sintetico()
    splitter = TemporalCVSplit(dados)
    folds = splitter.gerar_folds()

    for idx_treino, idx_validacao, _, _ in folds:
        anos_treino = dados.iloc[idx_treino]["ANO_REFERENCIA"].unique()
        anos_val = dados.iloc[idx_validacao]["ANO_REFERENCIA"].unique()
        assert np.all(anos_treino < anos_val.min())
        assert splitter.validar_integridade_temporal(idx_treino, idx_validacao)


def test_temporal_cv_folds_are_correct():
    dados = _dataset_temporal_sintetico()
    splitter = TemporalCVSplit(dados)
    folds = splitter.gerar_folds()

    assert len(folds) == 2
    assert folds[0][2] == 2022 and folds[0][3] == 2023
    assert folds[1][2] == 2023 and folds[1][3] == 2024


def test_temporal_cv_metrics_shape():
    dados = _dataset_temporal_sintetico()
    relatorio = TemporalCVValidator(dados, random_state=42).executar()

    assert "agregacao" in relatorio
    for chave in ("mean_f1", "std_f1", "mean_auc", "std_auc"):
        assert chave in relatorio["agregacao"]
    assert relatorio["num_folds"] == 2


def test_temporal_cv_all_folds_pass():
    dados = _dataset_temporal_sintetico()
    relatorio = TemporalCVValidator(dados, random_state=42).executar()

    assert len(relatorio["folds"]) == relatorio["num_folds"]
    for fold in relatorio["folds"]:
        assert 0.0 <= fold["f1"] <= 1.0
        assert 0.0 <= fold["precision"] <= 1.0
        assert 0.0 <= fold["recall"] <= 1.0
        assert "auc" in fold
