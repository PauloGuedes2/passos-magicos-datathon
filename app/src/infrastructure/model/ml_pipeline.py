"""Pipeline de treinamento do modelo de ML.

Responsabilidades:
- Criar variável alvo
- Gerar features históricas
- Treinar modelo e avaliar métricas
- Comparar com baseline
- Promover modelo com base em critérios
"""

import json
import os
import shutil
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    recall_score,
    f1_score,
    precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from src.application.feature_processor import ProcessadorFeatures
from src.config.settings import Configuracoes
from src.util.logger import logger


class PipelineML:
    """Pipeline de treinamento do modelo.

    Responsabilidades:
    - Preparar dados
    - Treinar e avaliar o modelo
    - Promover modelo quando aplicável
    """

    def __init__(self):
        """Inicializa o pipeline.

        Responsabilidades:
        - Instanciar o processador de features
        """
        self.processador = ProcessadorFeatures()

    @staticmethod
    def criar_target(dados: pd.DataFrame) -> pd.DataFrame:
        """Cria a variável alvo RISCO_DEFASAGEM.

        Parâmetros:
        - dados (pd.DataFrame): dados de entrada

        Retorno:
        - pd.DataFrame: dados com coluna alvo
        """
        if "DEFASAGEM" in dados.columns:
            defasagem = pd.to_numeric(dados["DEFASAGEM"], errors="coerce")
            dados[Configuracoes.TARGET_COL] = (defasagem < 0).fillna(False).astype(int)
        elif "INDE" in dados.columns:
            dados["INDE"] = pd.to_numeric(dados["INDE"], errors="coerce")
            dados[Configuracoes.TARGET_COL] = (dados["INDE"] < 6.0).astype(int)
        elif "PEDRA" in dados.columns:
            dados[Configuracoes.TARGET_COL] = dados["PEDRA"].astype(str).str.upper().apply(
                lambda valor: 1 if "QUARTZO" in valor else 0
            )
        else:
            raise ValueError("Colunas de target ausentes para criação do rótulo.")

        return dados

    @staticmethod
    def criar_features_lag(dados: pd.DataFrame) -> pd.DataFrame:
        """Gera features históricas (Lag).

        Parâmetros:
        - dados (pd.DataFrame): dados de entrada

        Retorno:
        - pd.DataFrame: dados com features de histórico
        """
        logger.info("Gerando Features Históricas (Lag)...")

        if "RA" not in dados.columns or "ANO_REFERENCIA" not in dados.columns:
            return dados

        dados = dados.sort_values(by=["RA", "ANO_REFERENCIA"])
        metricas = ["INDE", "IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN"]

        for coluna in metricas:
            if coluna in dados.columns:
                nome_coluna = f"{coluna}_ANTERIOR"
                dados[nome_coluna] = dados.groupby("RA")[coluna].shift(1).fillna(0)

        if "INDE_ANTERIOR" in dados.columns:
            dados["ALUNO_NOVO"] = (dados["INDE_ANTERIOR"] == 0).astype(int)
        else:
            dados["ALUNO_NOVO"] = 1

        return dados

    def treinar(self, dados: pd.DataFrame):
        """Executa o treinamento do modelo.

        Parâmetros:
        - dados (pd.DataFrame): dados para treinamento

        Exceções:
        - ValueError: quando ANO_REFERENCIA não está disponível
        """
        logger.info("Iniciando pipeline de treinamento Enterprise (Anti-Leakage)...")

        dados = self.criar_target(dados)
        dados = self.criar_features_lag(dados)
        dados = self._remover_colunas_proibidas(dados)

        mascara_treino, mascara_teste = self._definir_particao_temporal(dados)

        estatisticas = self._calcular_estatisticas_treino(dados, mascara_treino)
        logger.info(f"Estatísticas de Treino calculadas: {estatisticas}")
        self._salvar_estatisticas(estatisticas)

        dados_processados = self.processador.processar(dados, estatisticas=estatisticas)
        dados_processados[Configuracoes.TARGET_COL] = dados[Configuracoes.TARGET_COL]
        dados_processados["ANO_REFERENCIA"] = dados["ANO_REFERENCIA"]

        features_uso = [
            f
            for f in Configuracoes.FEATURES_MODELO_NUMERICAS + Configuracoes.FEATURES_MODELO_CATEGORICAS
            if f in dados_processados.columns
        ]

        matriz_treino = dados_processados.loc[mascara_treino, features_uso]
        alvo_treino = dados_processados.loc[mascara_treino, Configuracoes.TARGET_COL]

        matriz_teste = dados_processados.loc[mascara_teste, features_uso]
        alvo_teste = dados_processados.loc[mascara_teste, Configuracoes.TARGET_COL]

        logger.info(f"Treino: {matriz_treino.shape}, Teste: {matriz_teste.shape}")

        modelo_base = self._criar_modelo(matriz_treino)
        sample_weight = self._calcular_pesos_grupo(dados.loc[mascara_treino])
        if sample_weight is None:
            modelo_base.fit(matriz_treino, alvo_treino)
        else:
            try:
                modelo_base.fit(matriz_treino, alvo_treino, classifier__sample_weight=sample_weight)
            except TypeError:
                # Fallback para pipelines simples usados em testes.
                modelo_base.fit(matriz_treino, alvo_treino)

        modelo, info_calibracao = self._calibrar_modelo(
            modelo_base, matriz_treino, alvo_treino
        )
        probabilidades = modelo.predict_proba(matriz_teste)[:, 1]
        threshold, threshold_info = self._selecionar_threshold_estrategico(
            alvo_teste,
            probabilidades,
            dados.loc[mascara_teste],
        )
        predicoes = (probabilidades >= threshold).astype(int)

        novas_metricas = self._calcular_metricas(
            alvo_teste,
            predicoes,
            threshold,
            threshold_info,
            dados.loc[mascara_teste],
            matriz_treino,
            matriz_teste,
            probabilidades=probabilidades,
            alvo_treino=alvo_treino,
            probabilidades_treino=modelo.predict_proba(matriz_treino)[:, 1],
            calibration=info_calibracao,
        )
        logger.info(f"Métricas: {novas_metricas}")

        if self._deve_promover_modelo(novas_metricas):
            self._promover_modelo(
                modelo,
                novas_metricas,
                dados_processados.loc[mascara_teste],
                alvo_teste,
                predicoes,
                modelo_importancia=modelo_base,
            )

    @staticmethod
    def _definir_particao_temporal(dados: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Define máscaras de treino e teste com base em ANO_REFERENCIA.

        Parâmetros:
        - dados (pd.DataFrame): dados de entrada

        Retorno:
        - tuple[np.ndarray, np.ndarray]: máscaras de treino e teste

        Exceções:
        - ValueError: quando ANO_REFERENCIA não está disponível
        """
        if "ANO_REFERENCIA" not in dados.columns:
            raise ValueError("Coluna ANO_REFERENCIA necessária para treino.")

        anos_disponiveis = sorted(dados["ANO_REFERENCIA"].unique())
        if len(anos_disponiveis) > 1:
            ano_teste = anos_disponiveis[-1]
            logger.info(f"Split Temporal: Treino < {ano_teste} | Teste == {ano_teste}")
            mascara_treino = dados["ANO_REFERENCIA"] < ano_teste
            mascara_teste = dados["ANO_REFERENCIA"] == ano_teste
            return mascara_treino, mascara_teste

        logger.warning("Split temporal com apenas um ano disponível. Aplicando split aleatório 80/20.")
        indices = np.arange(len(dados))
        if len(indices) < 2:
            mascara_treino = np.ones(len(dados), dtype=bool)
            mascara_teste = np.ones(len(dados), dtype=bool)
            return mascara_treino, mascara_teste
        rng = np.random.default_rng(Configuracoes.RANDOM_STATE)
        rng.shuffle(indices)
        corte = int(len(indices) * 0.8)
        mascara_treino = np.zeros(len(dados), dtype=bool)
        mascara_teste = np.zeros(len(dados), dtype=bool)
        mascara_treino[indices[:corte]] = True
        mascara_teste[indices[corte:]] = True
        return mascara_treino, mascara_teste

    @staticmethod
    def _calcular_estatisticas_treino(dados: pd.DataFrame, mascara_treino: np.ndarray) -> Dict[str, Any]:
        """Calcula estatísticas do conjunto de treino.

        Parâmetros:
        - dados (pd.DataFrame): dados de entrada
        - mascara_treino (np.ndarray): máscara de treino

        Retorno:
        - dict: estatísticas calculadas
        """
        ano_ingresso = pd.to_numeric(dados.loc[mascara_treino, "ANO_INGRESSO"], errors="coerce")
        mediana = ano_ingresso.median()
        idade_treino = pd.to_numeric(dados.loc[mascara_treino, "IDADE"], errors="coerce")
        mediana_idade = idade_treino.median()
        if pd.isna(mediana):
            ano_ref = pd.to_numeric(dados.loc[mascara_treino, "ANO_REFERENCIA"], errors="coerce")
            mediana = ano_ref.median()
        if pd.isna(mediana):
            raise ValueError("ANO_INGRESSO e ANO_REFERENCIA indisponiveis para estatisticas.")
        if pd.isna(mediana_idade):
            mediana_idade = 12
        return {"mediana_ano_ingresso": int(mediana), "mediana_idade": int(mediana_idade)}

    @staticmethod
    def _salvar_estatisticas(estatisticas: Dict[str, Any]) -> None:
        """Salva estatísticas de treino para uso em inferência.

        Parâmetros:
        - estatisticas (dict): estatísticas calculadas
        """
        try:
            os.makedirs(os.path.dirname(Configuracoes.FEATURE_STATS_PATH), exist_ok=True)
            with open(Configuracoes.FEATURE_STATS_PATH, "w") as arquivo:
                json.dump(estatisticas, arquivo)
        except Exception as erro:
            logger.warning(f"Falha ao salvar estatísticas de treino: {erro}")

    @staticmethod
    def _remover_colunas_proibidas(dados: pd.DataFrame) -> pd.DataFrame:
        """Remove colunas proibidas para evitar vazamento de informação.

        Parâmetros:
        - dados (pd.DataFrame): dados de entrada

        Retorno:
        - pd.DataFrame: dados sem colunas proibidas
        """
        colunas_remover = [
            coluna for coluna in Configuracoes.COLUNAS_PROIBIDAS_NO_TREINO if coluna in dados.columns
        ]
        if colunas_remover:
            dados = dados.drop(columns=colunas_remover)
        return dados

    @staticmethod
    def _criar_modelo(matriz_treino: pd.DataFrame) -> Pipeline:
        """Cria pipeline de pré-processamento e modelo.

        Parâmetros:
        - matriz_treino (pd.DataFrame): dados de treino

        Retorno:
        - Pipeline: pipeline do modelo
        """
        features_numericas = [
            f for f in Configuracoes.FEATURES_MODELO_NUMERICAS if f in matriz_treino.columns
        ]
        features_categoricas = [
            f for f in Configuracoes.FEATURES_MODELO_CATEGORICAS if f in matriz_treino.columns
        ]

        transformador_numerico = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        transformador_categorico = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", transformador_numerico, features_numericas),
                ("cat", transformador_categorico, features_categoricas),
            ]
        )

        return Pipeline(steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    random_state=Configuracoes.RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=Configuracoes.N_JOBS,
                ),
            ),
        ])

    @staticmethod
    def _calibrar_modelo(
        modelo_base: Pipeline,
        matriz_treino: pd.DataFrame,
        alvo_treino,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Avalia calibracao (Platt/Isotonic) sem usar holdout.

        Seleciona metodo com base em um split interno de calibracao no
        conjunto de treino para evitar leakage no conjunto de teste final.
        """
        y = np.asarray(alvo_treino)
        if len(np.unique(y)) < 2 or len(y) < 50:
            return modelo_base, {
                "method": "none",
                "brier_base": None,
                "brier_calibrated": None,
                "selection_source": "train_only_insufficient_data",
            }

        try:
            X_fit, X_calib, y_fit, y_calib = train_test_split(
                matriz_treino,
                y,
                test_size=0.2,
                random_state=Configuracoes.RANDOM_STATE,
                stratify=y,
            )
        except ValueError:
            X_fit, X_calib, y_fit, y_calib = train_test_split(
                matriz_treino,
                y,
                test_size=0.2,
                random_state=Configuracoes.RANDOM_STATE,
                stratify=None,
            )

        modelo_base_selecao = clone(modelo_base)
        modelo_base_selecao.fit(X_fit, y_fit)
        prob_base = modelo_base_selecao.predict_proba(X_calib)[:, 1]
        brier_base = brier_score_loss(y_calib, prob_base)
        candidatos: Dict[str, float] = {"none": float(brier_base)}

        for metodo in ("sigmoid", "isotonic"):
            try:
                modelo_calibrado = CalibratedClassifierCV(
                    estimator=clone(modelo_base),
                    method=metodo,
                    cv=3,
                )
                modelo_calibrado.fit(X_fit, y_fit)
                prob_cal = modelo_calibrado.predict_proba(X_calib)[:, 1]
                candidatos[metodo] = float(brier_score_loss(y_calib, prob_cal))
            except Exception as erro:
                logger.warning(f"Falha ao calibrar modelo ({metodo}): {erro}")

        metodo_escolhido = min(candidatos.items(), key=lambda item: item[1])[0]
        melhor_brier = candidatos[metodo_escolhido]
        melhoria_brier = float(brier_base - melhor_brier)
        melhoria_minima = 1e-4

        if metodo_escolhido != "none" and melhoria_brier >= melhoria_minima:
            modelo_final = CalibratedClassifierCV(
                estimator=clone(modelo_base),
                method=metodo_escolhido,
                cv=3,
            )
            modelo_final.fit(matriz_treino, y)
            logger.info(
                f"Calibracao selecionada: {metodo_escolhido} "
                f"(Brier {brier_base:.4f} -> {melhor_brier:.4f})"
            )
            return modelo_final, {
                "method": metodo_escolhido,
                "brier_base": round(float(brier_base), 4),
                "brier_calibrated": round(float(melhor_brier), 4),
                "selection_source": "train_split",
            }

        return modelo_base, {
            "method": "none",
            "brier_base": round(float(brier_base), 4),
            "brier_calibrated": round(float(brier_base), 4),
            "selection_source": "train_split",
        }

    @staticmethod
    def _calcular_curva_calibracao(alvo_teste, probabilidades, n_bins: int = 10) -> Dict[str, Any]:
        """Calcula pontos da curva de calibracao para auditoria."""
        if len(np.unique(alvo_teste)) < 2:
            return {
                "mean_predicted_value": [],
                "fraction_of_positives": [],
                "n_bins": int(n_bins),
            }
        frac_pos, mean_pred = calibration_curve(alvo_teste, probabilidades, n_bins=n_bins)
        return {
            "mean_predicted_value": [round(float(v), 4) for v in mean_pred.tolist()],
            "fraction_of_positives": [round(float(v), 4) for v in frac_pos.tolist()],
            "n_bins": int(n_bins),
        }

    @staticmethod
    def _bootstrap_f1(
        alvo_teste,
        predicoes,
        n_bootstrap: int = 500,
    ) -> Dict[str, float]:
        """Estima intervalo de confianca de F1 no holdout via bootstrap simples."""
        if len(alvo_teste) == 0:
            return {
                "bootstrap_f1_mean": 0.0,
                "bootstrap_f1_std": 0.0,
                "bootstrap_interval_low": 0.0,
                "bootstrap_interval_high": 0.0,
            }

        rng = np.random.default_rng(Configuracoes.RANDOM_STATE)
        y_true = np.asarray(alvo_teste)
        y_pred = np.asarray(predicoes)
        scores = []

        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(y_true), len(y_true))
            score = f1_score(y_true[idx], y_pred[idx], zero_division=0)
            scores.append(float(score))

        arr = np.asarray(scores)
        return {
            "bootstrap_f1_mean": round(float(np.mean(arr)), 4),
            "bootstrap_f1_std": round(float(np.std(arr)), 4),
            "bootstrap_interval_low": round(float(np.percentile(arr, 2.5)), 4),
            "bootstrap_interval_high": round(float(np.percentile(arr, 97.5)), 4),
        }

    @staticmethod
    def _calcular_metricas(
        alvo_teste,
        predicoes,
        threshold: float,
        threshold_info: Dict[str, Any],
        dados_teste: pd.DataFrame,
        matriz_treino: pd.DataFrame,
        matriz_teste: pd.DataFrame,
        probabilidades=None,
        alvo_treino=None,
        probabilidades_treino=None,
        calibration: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Calcula métricas do modelo.

        Parâmetros:
        - alvo_teste (pd.Series): valores reais
        - predicoes (np.ndarray): predições
        - matriz_treino (pd.DataFrame): dados de treino
        - matriz_teste (pd.DataFrame): dados de teste

        Retorno:
        - dict: métricas calculadas
        """
        auc_holdout = (
            roc_auc_score(alvo_teste, probabilidades)
            if probabilidades is not None and len(np.unique(alvo_teste)) > 1
            else None
        )
        brier_holdout = (
            brier_score_loss(alvo_teste, probabilidades) if probabilidades is not None else None
        )

        metricas = {
            "timestamp": datetime.now().isoformat(),
            "recall": round(recall_score(alvo_teste, predicoes, zero_division=0), 4),
            "f1_score": round(f1_score(alvo_teste, predicoes, zero_division=0), 4),
            "precision": round(precision_score(alvo_teste, predicoes, zero_division=0), 4),
            "auc": round(float(auc_holdout), 4) if auc_holdout is not None else None,
            "brier_score": round(float(brier_holdout), 4) if brier_holdout is not None else None,
            "train_size": int(len(matriz_treino)),
            "test_size": int(len(matriz_teste)),
            "model_version": "candidate",
            "risk_threshold": round(float(threshold), 4),
            "threshold_strategy": threshold_info.get("strategy", "f1"),
            "threshold_justification": threshold_info.get("justification"),
            "threshold_tradeoff": threshold_info.get("tradeoff"),
        }
        if calibration:
            metricas["calibration"] = calibration
        if probabilidades is not None:
            metricas["calibration_curve"] = PipelineML._calcular_curva_calibracao(
                alvo_teste, probabilidades
            )
            metricas["roc_curve"] = PipelineML._calcular_curva_roc(
                alvo_teste, probabilidades
            )
            metricas["pr_curve"] = PipelineML._calcular_curva_pr(
                alvo_teste, probabilidades
            )
            metricas["confusion_matrix"] = PipelineML._calcular_matriz_confusao(
                alvo_teste, predicoes
            )
            metricas["probability_distribution"] = PipelineML._calcular_distribuicao_probabilidades(
                alvo_teste, probabilidades
            )
            metricas["threshold_tradeoff_curve"] = PipelineML._calcular_curva_tradeoff_threshold(
                alvo_teste, probabilidades
            )
        metricas.update(PipelineML._bootstrap_f1(alvo_teste, predicoes))

        if alvo_treino is not None and probabilidades_treino is not None:
            pred_treino = (np.asarray(probabilidades_treino) >= threshold).astype(int)
            metricas["train_metrics"] = {
                "f1_score": round(f1_score(alvo_treino, pred_treino, zero_division=0), 4),
                "precision": round(precision_score(alvo_treino, pred_treino, zero_division=0), 4),
                "recall": round(recall_score(alvo_treino, pred_treino, zero_division=0), 4),
                "auc": round(float(roc_auc_score(alvo_treino, probabilidades_treino)), 4)
                if len(np.unique(alvo_treino)) > 1
                else None,
            }
            if metricas["train_metrics"]["f1_score"] is not None:
                metricas["overfitting_gap_f1"] = round(
                    float(metricas["train_metrics"]["f1_score"] - metricas["f1_score"]), 4
                )

        metricas["group_metrics"] = PipelineML._calcular_metricas_grupo(
            dados_teste, alvo_teste, predicoes
        )

        # Calcular baseline (prever classe majoritária)
        if alvo_treino is not None and len(pd.Series(alvo_treino).mode()) > 0:
            classe_majoritaria = int(pd.Series(alvo_treino).mode().iloc[0])
            baseline_source = "train_majority_class"
        else:
            classe_majoritaria = int(pd.Series(alvo_teste).mode().iloc[0]) if len(pd.Series(alvo_teste).mode()) > 0 else 0
            baseline_source = "test_majority_class_fallback"
        baseline_predicoes = np.full(len(alvo_teste), classe_majoritaria, dtype=int)
        metricas["baseline"] = {
            "f1_score": round(f1_score(alvo_teste, baseline_predicoes, zero_division=0), 4),
            "recall": round(recall_score(alvo_teste, baseline_predicoes, zero_division=0), 4),
            "precision": round(precision_score(alvo_teste, baseline_predicoes, zero_division=0), 4),
            "strategy": "predict_majority_class",
            "source": baseline_source,
        }

        return metricas

    @staticmethod
    def _calcular_curva_roc(alvo_teste, probabilidades) -> Dict[str, Any]:
        """Calcula curva ROC para persistencia de auditoria."""
        y_true = np.asarray(alvo_teste).astype(int)
        y_proba = np.asarray(probabilidades, dtype=float)
        if y_true.size == 0 or len(np.unique(y_true)) < 2:
            return {"fpr": [], "tpr": []}
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        return {
            "fpr": [round(float(valor), 6) for valor in fpr.tolist()],
            "tpr": [round(float(valor), 6) for valor in tpr.tolist()],
        }

    @staticmethod
    def _calcular_curva_pr(alvo_teste, probabilidades) -> Dict[str, Any]:
        """Calcula curva Precision-Recall para persistencia de auditoria."""
        y_true = np.asarray(alvo_teste).astype(int)
        y_proba = np.asarray(probabilidades, dtype=float)
        if y_true.size == 0:
            return {"precision": [], "recall": []}
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        return {
            "precision": [round(float(valor), 6) for valor in precision.tolist()],
            "recall": [round(float(valor), 6) for valor in recall.tolist()],
        }

    @staticmethod
    def _calcular_matriz_confusao(alvo_teste, predicoes) -> Dict[str, int]:
        """Calcula matriz de confusao no threshold ativo."""
        y_true = np.asarray(alvo_teste).astype(int)
        y_pred = np.asarray(predicoes).astype(int)
        if y_true.size == 0:
            return {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
        matriz = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = matriz.ravel().tolist()
        return {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

    @staticmethod
    def _calcular_distribuicao_probabilidades(alvo_teste, probabilidades) -> Dict[str, Any]:
        """Calcula distribuicao de probabilidades por classe real."""
        y_true = np.asarray(alvo_teste).astype(int)
        y_proba = np.asarray(probabilidades, dtype=float)
        if y_true.size == 0 or y_proba.size == 0:
            bins = np.linspace(0.0, 1.0, 21)
            return {
                "bins": [round(float(valor), 6) for valor in bins.tolist()],
                "positive": [0] * 20,
                "negative": [0] * 20,
            }

        bins = np.linspace(0.0, 1.0, 21)
        positivos = y_proba[y_true == 1]
        negativos = y_proba[y_true == 0]
        hist_pos, edges = np.histogram(positivos, bins=bins)
        hist_neg, _ = np.histogram(negativos, bins=bins)
        return {
            "bins": [round(float(valor), 6) for valor in edges.tolist()],
            "positive": [int(valor) for valor in hist_pos.tolist()],
            "negative": [int(valor) for valor in hist_neg.tolist()],
        }

    @staticmethod
    def _calcular_curva_tradeoff_threshold(alvo_teste, probabilidades) -> list[Dict[str, float]]:
        """Calcula trade-off de metricas em grade fixa de thresholds."""
        y_true = np.asarray(alvo_teste).astype(int)
        y_proba = np.asarray(probabilidades, dtype=float)
        if y_true.size == 0 or y_proba.size == 0:
            return []

        curva = []
        for threshold in np.linspace(0.0, 1.0, 50):
            y_pred = (y_proba >= threshold).astype(int)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            curva.append(
                {
                    "threshold": round(float(threshold), 6),
                    "recall": round(float(recall), 6),
                    "precision": round(float(precision), 6),
                    "f1": round(float(f1), 6),
                }
            )
        return curva

    @staticmethod
    def _calcular_threshold_f1(alvo_teste, probabilidades) -> float:
        """Calcula o melhor threshold baseado em F1.

        Retorno:
        - float: threshold selecionado
        """
        precisions, recalls, thresholds = precision_recall_curve(alvo_teste, probabilidades)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
        if thresholds.size == 0:
            return Configuracoes.RISK_THRESHOLD
        melhor_indice = int(np.nanargmax(f1_scores[:-1]))
        return float(thresholds[melhor_indice])

    @staticmethod
    def _selecionar_threshold_recall(alvo_teste, probabilidades) -> float:
        """Seleciona threshold para maximizar recall respeitando precisao minima."""
        precisions, recalls, thresholds = precision_recall_curve(alvo_teste, probabilidades)
        if thresholds.size == 0:
            return Configuracoes.RISK_THRESHOLD
        candidatos = []
        for idx, threshold in enumerate(thresholds):
            precision = float(precisions[idx])
            recall = float(recalls[idx])
            if precision < Configuracoes.MIN_PRECISION:
                continue
            f1 = (2 * precision * recall) / (precision + recall + 1e-9)
            candidatos.append((threshold, recall, f1))
        if candidatos:
            candidatos.sort(key=lambda item: (-item[1], -item[2]))
            return float(candidatos[0][0])
        return float(thresholds[0])

    @staticmethod
    def _selecionar_threshold_custo(alvo_teste, probabilidades) -> float:
        """Seleciona threshold que minimiza custo ponderado de FN/FP."""
        thresholds = np.unique(
            np.concatenate(
                [
                    np.asarray([Configuracoes.RISK_THRESHOLD], dtype=float),
                    np.asarray(probabilidades, dtype=float),
                ]
            )
        )
        if thresholds.size == 0:
            return Configuracoes.RISK_THRESHOLD

        y_true = np.asarray(alvo_teste).astype(int)
        melhor_threshold = float(Configuracoes.RISK_THRESHOLD)
        melhor_custo = None
        for threshold in thresholds:
            y_pred = (np.asarray(probabilidades) >= threshold).astype(int)
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            custo = (Configuracoes.COST_FN_WEIGHT * fn) + (Configuracoes.COST_FP_WEIGHT * fp)
            if melhor_custo is None or custo < melhor_custo:
                melhor_custo = custo
                melhor_threshold = float(threshold)
        return melhor_threshold

    def _selecionar_threshold_estrategico(
        self,
        alvo_teste,
        probabilidades,
        dados_teste: pd.DataFrame,
    ) -> Tuple[float, Dict[str, Any]]:
        """Seleciona threshold conforme estrategia configurada."""
        estrategia = Configuracoes.THRESHOLD_STRATEGY

        if estrategia == "recall":
            threshold = self._selecionar_threshold_recall(alvo_teste, probabilidades)
            return threshold, {
                "strategy": "recall",
                "justification": (
                    "Threshold otimizado para maximizar recall com precisao minima, "
                    "priorizando identificar alunos de risco para intervencao precoce."
                ),
                "tradeoff": "Aumenta captura de risco potencial, com possivel alta de falsos positivos.",
            }

        if estrategia == "cost":
            threshold = self._selecionar_threshold_custo(alvo_teste, probabilidades)
            return threshold, {
                "strategy": "cost",
                "justification": (
                    "Threshold minimiza custo ponderado de falsos negativos e falsos positivos "
                    f"(FN={Configuracoes.COST_FN_WEIGHT}, FP={Configuracoes.COST_FP_WEIGHT})."
                ),
                "tradeoff": "A decisao depende dos pesos de custo definidos por negocio.",
            }

        if estrategia == "fairness_f1":
            threshold_fair = self._selecionar_threshold_justo(
                alvo_teste,
                probabilidades,
                dados_teste,
            )
            if threshold_fair is not None:
                return float(threshold_fair), {
                    "strategy": "fairness_f1",
                    "justification": (
                        "Threshold escolhido com restricoes de fairness e desempenho minimo; "
                        "entre candidatos validos, prioriza recall."
                    ),
                    "tradeoff": "Pode sacrificar F1 global para reduzir gaps entre grupos.",
                }

        threshold = self._calcular_threshold_f1(alvo_teste, probabilidades)
        return threshold, {
            "strategy": "f1",
            "justification": "Threshold otimizado para melhor equilibrio entre precision e recall (F1).",
            "tradeoff": "Equilibrio geral pode deixar casos de risco limiar sem classificacao positiva.",
        }

    @staticmethod
    def _selecionar_threshold_justo(alvo_teste, probabilidades, dados_teste: pd.DataFrame):
        """Seleciona threshold com foco em minimizar dano e vies.

        Estrategia:
        - Exige recall e precision minimos
        - Limita gap de FPR/FNR entre grupos sensiveis
        - Entre candidatos validos, maximiza recall, depois minimiza gap de FPR
        """
        precisions, recalls, thresholds = precision_recall_curve(alvo_teste, probabilidades)
        if thresholds.size == 0:
            return None

        grupo_coluna = Configuracoes.FAIRNESS_GROUP_COL
        if grupo_coluna not in dados_teste.columns:
            return None

        candidatos = []
        for idx, threshold in enumerate(thresholds):
            y_pred = (probabilidades >= threshold).astype(int)
            precision = float(precisions[idx])
            recall = float(recalls[idx])

            if recall < Configuracoes.MIN_RECALL or precision < Configuracoes.MIN_PRECISION:
                continue

            gaps = PipelineML._calcular_gaps_fairness(dados_teste, alvo_teste, y_pred)
            if gaps is None:
                continue
            gap_fpr, gap_fnr, max_fpr, max_fnr = gaps

            if gap_fpr > Configuracoes.MAX_FPR_GAP or gap_fnr > Configuracoes.MAX_FNR_GAP:
                continue
            if max_fpr > Configuracoes.MAX_FPR or max_fnr > Configuracoes.MAX_FNR:
                continue

            candidatos.append((threshold, recall, precision, gap_fpr, gap_fnr))

        if not candidatos:
            return None

        candidatos.sort(key=lambda item: (-item[1], item[3], -item[2]))
        return float(candidatos[0][0])

    @staticmethod
    def _calcular_gaps_fairness(
        dados_teste: pd.DataFrame,
        alvo_teste,
        predicoes,
    ) -> Tuple[float, float, float, float] | None:
        """Calcula gaps de FPR/FNR entre grupos."""
        grupo_coluna = Configuracoes.FAIRNESS_GROUP_COL
        if grupo_coluna not in dados_teste.columns:
            return None

        df = dados_teste[[grupo_coluna]].copy()
        df["y_true"] = alvo_teste.values
        df["y_pred"] = predicoes

        fpr_vals = []
        fnr_vals = []
        for _, subset in df.groupby(grupo_coluna):
            y_true = subset["y_true"]
            y_pred = subset["y_pred"]
            falso_positivo = int(((y_pred == 1) & (y_true == 0)).sum())
            falso_negativo = int(((y_pred == 0) & (y_true == 1)).sum())
            verdadeiro_positivo = int(((y_pred == 1) & (y_true == 1)).sum())
            verdadeiro_negativo = int(((y_pred == 0) & (y_true == 0)).sum())

            fpr_denom = falso_positivo + verdadeiro_negativo
            fnr_denom = falso_negativo + verdadeiro_positivo

            fpr = (falso_positivo / fpr_denom) if fpr_denom else 0.0
            fnr = (falso_negativo / fnr_denom) if fnr_denom else 0.0
            fpr_vals.append(fpr * 100)
            fnr_vals.append(fnr * 100)

        if not fpr_vals or not fnr_vals:
            return None

        max_fpr = max(fpr_vals)
        max_fnr = max(fnr_vals)
        gap_fpr = max_fpr - min(fpr_vals)
        gap_fnr = max_fnr - min(fnr_vals)
        return gap_fpr, gap_fnr, max_fpr, max_fnr

    @staticmethod
    def _calcular_metricas_grupo(dados_teste: pd.DataFrame, alvo_teste, predicoes) -> Dict[str, Any]:
        """Calcula métricas por grupos sensíveis para auditoria.

        Retorno:
        - dict: métricas agregadas por grupo
        """
        metricas_grupo = {}
        for coluna in Configuracoes.FEATURES_CATEGORICAS:
            if coluna not in dados_teste.columns:
                continue
            metricas_coluna = {}
            for valor, indices in dados_teste.groupby(coluna).groups.items():
                y_true = alvo_teste.loc[indices]
                y_pred = pd.Series(predicoes, index=alvo_teste.index).loc[indices]
                metricas_coluna[str(valor)] = {
                    "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
                    "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
                    "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
                    "support": int(len(indices)),
                }
            metricas_grupo[coluna] = metricas_coluna
        return metricas_grupo

    @staticmethod
    def _calcular_pesos_grupo(dados_treino: pd.DataFrame):
        """Calcula pesos por grupo para reduzir vies de representacao.

        Retorno:
        - np.ndarray | None: pesos alinhados ao treino
        """
        grupo_coluna = Configuracoes.FAIRNESS_GROUP_COL
        if grupo_coluna not in dados_treino.columns:
            return None
        grupos = dados_treino[grupo_coluna].astype(str)
        contagem = grupos.value_counts()
        if contagem.empty:
            return None
        media = contagem.mean()
        pesos = grupos.map(lambda g: media / contagem.get(g, media)).astype(float)
        return pesos.values

    @staticmethod
    def _deve_promover_modelo(novas_metricas: Dict[str, Any]) -> bool:
        """Avalia se o modelo deve ser promovido.

        Parâmetros:
        - novas_metricas (dict): métricas do modelo candidato

        Retorno:
        - bool: True se deve promover
        """
        if novas_metricas.get("recall", 0) < Configuracoes.MIN_RECALL:
            logger.warning("Recall abaixo do mínimo configurado. Modelo não promovido.")
            return False
        if not os.path.exists(Configuracoes.METRICS_FILE):
            return True
        try:
            with open(Configuracoes.METRICS_FILE, "r") as arquivo:
                atual = json.load(arquivo)
            return novas_metricas["f1_score"] >= (atual.get("f1_score", 0) * 0.95)
        except Exception:
            return True

    @staticmethod
    def _promover_modelo(
        modelo,
        metricas,
        dados_teste_original,
        alvo_teste,
        predicoes,
        modelo_importancia=None,
    ):
        """Promove o modelo e salva dados de referência.

        Parâmetros:
        - modelo (Any): modelo treinado
        - metricas (dict): métricas do modelo
        - dados_teste_original (pd.DataFrame): dados originais de teste
        - alvo_teste (pd.Series): valores reais
        - predicoes (np.ndarray): predições
        """
        logger.info("Promovendo Modelo...")

        if os.path.exists(Configuracoes.MODEL_PATH):
            shutil.copy(Configuracoes.MODEL_PATH, f"{Configuracoes.MODEL_PATH}.bak")

        os.makedirs(os.path.dirname(Configuracoes.MODEL_PATH), exist_ok=True)
        dump(modelo, Configuracoes.MODEL_PATH)

        metricas["model_version"] = datetime.now().strftime("v%Y.%m.%d")
        modelo_ref = modelo_importancia if modelo_importancia is not None else modelo

        # Extrair importância de features do RandomForest
        try:
            if hasattr(modelo_ref, "named_steps") and "classifier" in modelo_ref.named_steps:
                classifier = modelo_ref.named_steps["classifier"]
                if hasattr(classifier, "feature_importances_"):
                    importancias = classifier.feature_importances_
                    # Obter nomes das features após transformação
                    preprocessor = modelo_ref.named_steps["preprocessor"]
                    feature_names = []
                    if hasattr(preprocessor, "get_feature_names_out"):
                        feature_names = list(preprocessor.get_feature_names_out())
                    else:
                        # Fallback: usar nomes das features originais
                        feature_names = [f"feature_{i}" for i in range(len(importancias))]

                    # Criar ranking de importância
                    feature_importance_dict = {
                        name: round(float(imp), 4)
                        for name, imp in zip(feature_names, importancias)
                    }
                    # Ordenar por importância decrescente
                    feature_importance_dict = dict(
                        sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
                    )
                    metricas["feature_importance"] = feature_importance_dict
                    metricas["feature_importance_ranking"] = [
                        {"feature": nome, "importance": valor}
                        for nome, valor in list(feature_importance_dict.items())[:20]
                    ]
                    logger.info(f"Top 5 features: {list(feature_importance_dict.items())[:5]}")
        except Exception as erro:
            logger.warning(f"Não foi possível extrair importância de features: {erro}")

        with open(Configuracoes.METRICS_FILE, "w") as arquivo:
            json.dump(metricas, arquivo, indent=2)

        referencia_df = dados_teste_original.copy()
        referencia_df["prediction"] = predicoes
        referencia_df[Configuracoes.TARGET_COL] = alvo_teste

        referencia_df.to_csv(Configuracoes.REFERENCE_PATH, index=False)
        logger.info(f"Reference Data salvo com colunas processadas: {Configuracoes.REFERENCE_PATH}")


treinador = PipelineML()
