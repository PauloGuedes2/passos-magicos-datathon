"""Testes de calibracao, bootstrap e regressao estatistica."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from src.config.settings import Configuracoes
from src.infrastructure.model.ml_pipeline import PipelineML


def _dados_modelo_sinteticos(n: int = 400) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(7)
    dados = pd.DataFrame(
        {
            "IDADE": rng.integers(8, 19, n),
            "TEMPO_NA_ONG": rng.integers(0, 6, n),
            "INDE_ANTERIOR": rng.uniform(0, 10, n),
            "IAA_ANTERIOR": rng.uniform(0, 10, n),
            "IEG_ANTERIOR": rng.uniform(0, 10, n),
            "IPS_ANTERIOR": rng.uniform(0, 10, n),
            "IDA_ANTERIOR": rng.uniform(0, 10, n),
            "IPP_ANTERIOR": rng.uniform(0, 10, n),
            "IPV_ANTERIOR": rng.uniform(0, 10, n),
            "IAN_ANTERIOR": rng.uniform(0, 10, n),
            "ALUNO_NOVO": rng.integers(0, 2, n),
        }
    )
    score = (
            0.6 * dados["INDE_ANTERIOR"].values
            + 0.3 * dados["IEG_ANTERIOR"].values
            - 0.2 * dados["IDADE"].values
            + rng.normal(0, 1.2, n)
    )
    alvo = (score < 3.5).astype(int)
    return dados, alvo


def test_calibration_effect_positive():
    pipeline = PipelineML()
    X, y = _dados_modelo_sinteticos()
    X_treino = X.iloc[:300]
    y_treino = y[:300]

    modelo_base = pipeline._criar_modelo(X_treino)
    modelo_base.fit(X_treino, y_treino)

    _, info = pipeline._calibrar_modelo(
        modelo_base, X_treino, y_treino
    )

    assert "method" in info
    assert "brier_base" in info
    assert "brier_calibrated" in info
    assert info.get("selection_source") is not None
    if info["brier_base"] is not None:
        assert info["brier_calibrated"] <= info["brier_base"] + 1e-6


def test_calibration_metrics_computed():
    y_true = pd.Series([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 0, 0])
    y_prob = np.array([0.2, 0.8, 0.72, 0.15, 0.41, 0.21])
    X_train = pd.DataFrame({"IDADE": [10, 11, 12, 13]})
    X_test = pd.DataFrame({"IDADE": [14, 15, 16, 17, 18, 19]})

    metricas = PipelineML._calcular_metricas(
        y_true,
        y_pred,
        threshold=0.5,
        threshold_info={"strategy": "f1", "justification": "teste", "tradeoff": "teste"},
        dados_teste=pd.DataFrame({"GENERO": ["Masculino"] * 6}),
        matriz_treino=X_train,
        matriz_teste=X_test,
        probabilidades=y_prob,
        alvo_treino=pd.Series([0, 1, 1, 0]),
        probabilidades_treino=np.array([0.2, 0.8, 0.7, 0.1]),
        calibration={"method": "sigmoid", "brier_base": 0.2, "brier_calibrated": 0.18},
    )

    assert "brier_score" in metricas
    assert "auc" in metricas
    assert "calibration" in metricas
    assert "calibration_curve" in metricas
    assert metricas["threshold_strategy"] == "f1"
    assert metricas["threshold_justification"] == "teste"
    assert metricas["baseline"]["source"] in (
        "train_majority_class",
        "test_majority_class_fallback",
    )


def test_bootstrap_f1_computation():
    y_true = pd.Series([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, 1])
    resultado = PipelineML._bootstrap_f1(y_true, y_pred, n_bootstrap=100)

    for chave in (
            "bootstrap_f1_mean",
            "bootstrap_f1_std",
            "bootstrap_interval_low",
            "bootstrap_interval_high",
    ):
        assert chave in resultado
    assert resultado["bootstrap_interval_low"] <= resultado["bootstrap_f1_mean"]
    assert resultado["bootstrap_interval_high"] >= resultado["bootstrap_f1_mean"]


def _carregar_metricas_regressao():
    if not os.path.exists(Configuracoes.METRICS_FILE):
        pytest.skip("Arquivo de métricas atual não encontrado.")
    if not os.path.exists(Configuracoes.BASELINE_METRICS_FILE):
        pytest.skip("Arquivo de baseline não encontrado.")
    with open(Configuracoes.METRICS_FILE, "r", encoding="utf-8") as atual_file:
        atual = json.load(atual_file)
    with open(Configuracoes.BASELINE_METRICS_FILE, "r", encoding="utf-8") as base_file:
        baseline = json.load(base_file)
    return atual, baseline
