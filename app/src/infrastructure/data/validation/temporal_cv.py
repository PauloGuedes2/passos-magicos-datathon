"""Validação Cruzada Temporal (Rolling Window).

Responsabilidades:
- Criar folds temporais sem vazamento de dados
- Treinar modelo em cada fold
- Calcular métricas por fold
- Analisar estabilidade estatística
- Gerar relatório de robustez

Garantias:
- Nunca mistura anos futuros no treino
- Mantém integridade temporal
- Reutiliza pipeline existente
- Não altera dados originais
"""

import json
from typing import Dict, List, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from src.util.logger import logger
from src.config.settings import Configuracoes
from src.infrastructure.model.ml_pipeline import PipelineML


def salvar_relatorio_temporal(relatorio: Dict[str, Any], caminho_saida: str) -> None:
    """Persistir relatorio temporal em JSON.

    Parâmetros:
    - relatorio (dict): resultados da validação temporal
    - caminho_saida (str): caminho do arquivo de saída
    """
    with open(caminho_saida, "w", encoding="utf-8") as arquivo:
        json.dump(relatorio, arquivo, ensure_ascii=False, indent=2)


class TemporalCVSplit:
    """Gerenciador de splits temporais com rolling window.
    
    Garante que:
    - Treino sempre usa anos anteriores
    - Validação usa ano posterior
    - Sem vazamento temporal
    """

    def __init__(self, dados: pd.DataFrame):
        """Inicializa o splitter temporal.

        Parâmetros:
        - dados (pd.DataFrame): dataset com coluna ANO_REFERENCIA
        """
        if "ANO_REFERENCIA" not in dados.columns:
            raise ValueError("Dataset deve conter coluna ANO_REFERENCIA")

        self.dados = dados.copy()
        self.anos_unicos = sorted(self.dados["ANO_REFERENCIA"].unique())
        logger.info(f"Anos disponíveis: {self.anos_unicos}")

    def gerar_folds(self) -> List[Tuple[np.ndarray, np.ndarray, int, int]]:
        """Gera folds temporais com rolling window.

        Retorno:
        - list: [(indices_treino, indices_validacao, ano_treino_max, ano_validacao), ...]

        Garantia:
        - Cada fold: treino com anos < ano_validacao
        - Validação com ano_validacao
        - Sem sobreposição temporal
        """
        folds = []

        # Fold 1: Treino com primeiro ano, validação com segundo
        if len(self.anos_unicos) >= 2:
            ano_treino = self.anos_unicos[0]
            ano_validacao = self.anos_unicos[1]

            idx_treino = self.dados[self.dados["ANO_REFERENCIA"] == ano_treino].index.values
            idx_validacao = self.dados[
                self.dados["ANO_REFERENCIA"] == ano_validacao
            ].index.values

            folds.append((idx_treino, idx_validacao, ano_treino, ano_validacao))
            logger.info(
                f"Fold 1: Treino={ano_treino} ({len(idx_treino)} amostras), "
                f"Validação={ano_validacao} ({len(idx_validacao)} amostras)"
            )

        # Folds adicionais: Treino acumulativo
        for i in range(1, len(self.anos_unicos) - 1):
            ano_validacao = self.anos_unicos[i + 1]
            anos_treino = self.anos_unicos[: i + 1]

            idx_treino = self.dados[
                self.dados["ANO_REFERENCIA"].isin(anos_treino)
            ].index.values
            idx_validacao = self.dados[
                self.dados["ANO_REFERENCIA"] == ano_validacao
            ].index.values

            ano_treino_max = anos_treino[-1]
            folds.append((idx_treino, idx_validacao, ano_treino_max, ano_validacao))
            logger.info(
                f"Fold {i + 1}: Treino={anos_treino} ({len(idx_treino)} amostras), "
                f"Validação={ano_validacao} ({len(idx_validacao)} amostras)"
            )

        if not folds:
            raise ValueError("Insuficientes anos para criar folds temporais")

        return folds

    def validar_integridade_temporal(self, idx_treino: np.ndarray, idx_validacao: np.ndarray) -> bool:
        """Valida que treino não contém dados futuros.

        Parâmetros:
        - idx_treino (np.ndarray): índices de treino
        - idx_validacao (np.ndarray): índices de validação

        Retorno:
        - bool: True se válido, False caso contrário
        """
        anos_treino = self.dados.loc[idx_treino, "ANO_REFERENCIA"].unique()
        anos_validacao = self.dados.loc[idx_validacao, "ANO_REFERENCIA"].unique()

        # Garantir que nenhum ano de treino é >= ano de validação
        if np.any(anos_treino >= anos_validacao.min()):
            logger.error(
                f"Vazamento temporal detectado! "
                f"Treino: {anos_treino}, Validação: {anos_validacao}"
            )
            return False

        return True


class TemporalCVEvaluator:
    """Avaliador de métricas por fold temporal.
    
    Calcula:
    - Recall, Precision, F1, AUC por fold
    - Matriz de confusão
    - Média, desvio padrão, gaps
    """

    @staticmethod
    def calcular_metricas_fold(
        alvo_verdadeiro: np.ndarray,
        predicoes: np.ndarray,
        probabilidades: np.ndarray = None,
    ) -> Dict[str, Any]:
        """Calcula métricas para um fold.

        Parâmetros:
        - alvo_verdadeiro (np.ndarray): valores reais
        - predicoes (np.ndarray): predições binárias
        - probabilidades (np.ndarray): probabilidades (opcional para AUC)

        Retorno:
        - dict: métricas do fold
        """
        recall = recall_score(alvo_verdadeiro, predicoes, zero_division=0)
        precision = precision_score(alvo_verdadeiro, predicoes, zero_division=0)
        f1 = f1_score(alvo_verdadeiro, predicoes, zero_division=0)

        metricas = {
            "recall": round(float(recall), 4),
            "precision": round(float(precision), 4),
            "f1": round(float(f1), 4),
            "confusion_matrix": confusion_matrix(alvo_verdadeiro, predicoes).tolist(),
        }

        # AUC se probabilidades disponíveis
        if probabilidades is not None:
            try:
                auc = roc_auc_score(alvo_verdadeiro, probabilidades)
                metricas["auc"] = round(float(auc), 4)
            except Exception as e:
                logger.warning(f"Não foi possível calcular AUC: {e}")
                metricas["auc"] = None

        return metricas

    @staticmethod
    def agregar_metricas(metricas_folds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Agrega métricas de todos os folds.

        Parâmetros:
        - metricas_folds (list): lista de dicts com métricas por fold

        Retorno:
        - dict: estatísticas agregadas (média, std, min, max)
        """
        metricas_nomes = ["recall", "precision", "f1", "auc"]
        agregacao = {}

        for metrica in metricas_nomes:
            valores = [
                m[metrica]
                for m in metricas_folds
                if metrica in m and m[metrica] is not None
            ]

            if valores:
                agregacao[f"mean_{metrica}"] = round(float(np.mean(valores)), 4)
                agregacao[f"std_{metrica}"] = round(float(np.std(valores)), 4)
                agregacao[f"min_{metrica}"] = round(float(np.min(valores)), 4)
                agregacao[f"max_{metrica}"] = round(float(np.max(valores)), 4)
                agregacao[f"gap_{metrica}"] = round(
                    float(np.max(valores) - np.min(valores)), 4
                )

        return agregacao


class TemporalCVValidator:
    """Executor principal de validação cruzada temporal.
    
    Orquestra:
    - Criação de folds
    - Treinamento em cada fold
    - Cálculo de métricas
    - Análise de estabilidade
    - Geração de relatório
    """

    def __init__(self, dados: pd.DataFrame, random_state: int = 42):
        """Inicializa o validador temporal.

        Parâmetros:
        - dados (pd.DataFrame): dataset completo
        - random_state (int): seed para reprodutibilidade
        """
        self.dados = dados.copy()
        self.random_state = random_state
        np.random.seed(random_state)
        self.pipeline = PipelineML()
        self.dados_modelagem = self._precomputar_dados_modelagem(self.dados)
        self.splitter = TemporalCVSplit(self.dados_modelagem)
        self.evaluator = TemporalCVEvaluator()

    def _precomputar_dados_modelagem(self, dados: pd.DataFrame) -> pd.DataFrame:
        """Precomputa target e lags uma única vez no dataset completo.

        Isso mantém o mesmo fluxo conceitual do pipeline principal:
        criar target, criar lag e remover colunas proibidas antes de
        qualquer split temporal.
        """
        dados_pre = dados.copy()
        try:
            dados_pre = self.pipeline.criar_target(dados_pre)
            dados_pre = self.pipeline.criar_features_lag(dados_pre)
            dados_pre = self.pipeline._remover_colunas_proibidas(dados_pre)
        except ValueError:
            # Permite instanciacao para testes utilitarios sem target.
            logger.warning(
                "Dataset sem colunas de target para precomputacao no init do TemporalCVValidator."
            )
        return dados_pre

    def executar(self) -> Dict[str, Any]:
        """Executa validação cruzada temporal completa.

        Retorno:
        - dict: relatório com folds, métricas agregadas e interpretação

        Garantias:
        - Sem vazamento temporal
        - Reprodutível (random_state)
        - Reutiliza pipeline existente
        """
        logger.info("Iniciando Validação Cruzada Temporal...")

        folds = self.splitter.gerar_folds()
        metricas_folds = []

        for fold_idx, (idx_treino, idx_validacao, ano_treino_max, ano_validacao) in enumerate(
            folds, 1
        ):
            # Validar integridade temporal
            if not self.splitter.validar_integridade_temporal(idx_treino, idx_validacao):
                raise RuntimeError(f"Vazamento temporal no fold {fold_idx}")

            logger.info(f"\n--- Fold {fold_idx} ---")
            logger.info(f"Treino: até {ano_treino_max}, Validação: {ano_validacao}")

            # Preparar dados do fold
            dados_treino = self.dados_modelagem.iloc[idx_treino].copy()
            dados_validacao = self.dados_modelagem.iloc[idx_validacao].copy()

            # Treinar modelo
            try:
                modelo, metricas = self._treinar_fold(dados_treino, dados_validacao)
                metricas["fold"] = fold_idx
                metricas["ano_treino_max"] = int(ano_treino_max)
                metricas["ano_validacao"] = int(ano_validacao)
                metricas_folds.append(metricas)

                logger.info(f"Fold {fold_idx} - F1: {metricas['f1']}, Recall: {metricas['recall']}")

            except Exception as e:
                logger.error(f"Erro ao processar fold {fold_idx}: {e}")
                raise

        # Agregar resultados
        agregacao = self.evaluator.agregar_metricas(metricas_folds)

        # Interpretar estabilidade
        interpretacao = self._interpretar_estabilidade(metricas_folds, agregacao)

        relatorio = {
            "timestamp": datetime.now().isoformat(),
            "random_state": self.random_state,
            "num_folds": len(folds),
            "folds": metricas_folds,
            "agregacao": agregacao,
            "interpretacao": interpretacao,
        }

        logger.info("\nValidação Cruzada Temporal Concluída!")
        logger.info(f"Relatório: {json.dumps(relatorio, indent=2)}")

        return relatorio

    def _treinar_fold(
        self, dados_treino: pd.DataFrame, dados_validacao: pd.DataFrame
    ) -> Tuple[Any, Dict[str, Any]]:
        """Treina modelo em um fold específico.

        Parâmetros:
        - dados_treino (pd.DataFrame): dados de treino
        - dados_validacao (pd.DataFrame): dados de validação

        Retorno:
        - tuple: (modelo treinado, métricas do fold)
        """
        # Reutilizar o mesmo pipeline base para todos os folds.
        pipeline = self.pipeline

        # Calcular estatísticas apenas com treino
        mascara_treino = np.ones(len(dados_treino), dtype=bool)
        estatisticas = pipeline._calcular_estatisticas_treino(dados_treino, mascara_treino)

        # Processar dados
        dados_treino_proc = pipeline.processador.processar(
            dados_treino, estatisticas=estatisticas
        )
        dados_treino_proc[Configuracoes.TARGET_COL] = dados_treino[Configuracoes.TARGET_COL]

        dados_validacao_proc = pipeline.processador.processar(
            dados_validacao, estatisticas=estatisticas
        )
        dados_validacao_proc[Configuracoes.TARGET_COL] = dados_validacao[
            Configuracoes.TARGET_COL
        ]

        # Selecionar features
        features_uso = [
            f
            for f in Configuracoes.FEATURES_MODELO_NUMERICAS
            + Configuracoes.FEATURES_MODELO_CATEGORICAS
            if f in dados_treino_proc.columns
        ]

        X_treino = dados_treino_proc[features_uso]
        y_treino = dados_treino_proc[Configuracoes.TARGET_COL]

        X_validacao = dados_validacao_proc[features_uso]
        y_validacao = dados_validacao_proc[Configuracoes.TARGET_COL]

        # Criar e treinar modelo
        modelo = pipeline._criar_modelo(X_treino)
        sample_weight = pipeline._calcular_pesos_grupo(dados_treino)

        if sample_weight is None:
            modelo.fit(X_treino, y_treino)
        else:
            try:
                modelo.fit(X_treino, y_treino, classifier__sample_weight=sample_weight)
            except TypeError:
                modelo.fit(X_treino, y_treino)

        # Predições
        probabilidades = modelo.predict_proba(X_validacao)[:, 1]
        threshold = pipeline._calcular_threshold_f1(y_validacao, probabilidades)
        predicoes = (probabilidades >= threshold).astype(int)

        # Calcular métricas
        metricas = self.evaluator.calcular_metricas_fold(
            y_validacao.values, predicoes, probabilidades
        )
        metricas["threshold"] = round(float(threshold), 4)
        metricas["train_size"] = len(X_treino)
        metricas["val_size"] = len(X_validacao)

        return modelo, metricas

    def _interpretar_estabilidade(
        self, metricas_folds: List[Dict[str, Any]], agregacao: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Interpreta estabilidade estatística do modelo.

        Parâmetros:
        - metricas_folds (list): métricas de cada fold
        - agregacao (dict): estatísticas agregadas

        Retorno:
        - dict: interpretação com conclusões
        """
        interpretacao = {
            "variabilidade_alta": False,
            "f1_estavel": False,
            "degradacao_severa": False,
            "conclusoes": [],
        }

        # Verificar variabilidade
        if "std_f1" in agregacao:
            std_f1 = agregacao["std_f1"]
            mean_f1 = agregacao["mean_f1"]

            # Coeficiente de variação
            cv = (std_f1 / mean_f1) if mean_f1 > 0 else 0
            interpretacao["coeficiente_variacao_f1"] = round(cv, 4)

            if cv > 0.15:  # > 15% é considerado alta variabilidade
                interpretacao["variabilidade_alta"] = True
                interpretacao["conclusoes"].append(
                    f"⚠️ Variabilidade ALTA no F1 (CV={cv:.2%}). "
                    f"Modelo pode ser sensível a mudanças temporais."
                )
            else:
                interpretacao["f1_estavel"] = True
                interpretacao["conclusoes"].append(
                    f"✓ Variabilidade BAIXA no F1 (CV={cv:.2%}). "
                    f"Modelo é robusto temporalmente."
                )

        # Verificar degradação
        if len(metricas_folds) > 1:
            f1_valores = [m["f1"] for m in metricas_folds]
            degradacao = f1_valores[0] - f1_valores[-1]

            if degradacao > 0.15:  # > 15% de queda
                interpretacao["degradacao_severa"] = True
                interpretacao["conclusoes"].append(
                    f"⚠️ Degradação SEVERA detectada: "
                    f"F1 caiu {degradacao:.2%} do primeiro para o último fold."
                )
            else:
                interpretacao["conclusoes"].append(
                    f"✓ Degradação ACEITÁVEL: "
                    f"F1 variou {degradacao:.2%} entre folds."
                )

        # Verificar recall (métrica crítica)
        if "mean_recall" in agregacao:
            mean_recall = agregacao["mean_recall"]
            if mean_recall < 0.70:
                interpretacao["conclusoes"].append(
                    f"⚠️ Recall BAIXO ({mean_recall:.2%}). "
                    f"Modelo pode estar perdendo casos positivos."
                )
            else:
                interpretacao["conclusoes"].append(
                    f"✓ Recall ADEQUADO ({mean_recall:.2%})."
                )

        return interpretacao

    def comparar_com_holdout(
        self, metricas_holdout: Dict[str, Any], agregacao: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compara métricas do holdout com intervalo de confiança do CV.

        Parâmetros:
        - metricas_holdout (dict): métricas do holdout 2024
        - agregacao (dict): estatísticas agregadas do CV

        Retorno:
        - dict: análise comparativa
        """
        comparacao = {
            "holdout_dentro_intervalo": False,
            "analise": {},
        }
        todos_dentro = True
        teve_metrica = False

        for metrica in ["f1", "recall", "precision"]:
            mean_key = f"mean_{metrica}"
            std_key = f"std_{metrica}"

            if mean_key in agregacao and std_key in agregacao:
                mean = agregacao[mean_key]
                std = agregacao[std_key]
                holdout_valor = metricas_holdout.get(metrica)

                if holdout_valor is not None:
                    teve_metrica = True
                    intervalo_inf = mean - std
                    intervalo_sup = mean + std
                    dentro = intervalo_inf <= holdout_valor <= intervalo_sup

                    comparacao["analise"][metrica] = {
                        "cv_mean": mean,
                        "cv_std": std,
                        "intervalo": [round(intervalo_inf, 4), round(intervalo_sup, 4)],
                        "holdout": holdout_valor,
                        "dentro_intervalo": dentro,
                    }

                    if not dentro:
                        todos_dentro = False

        comparacao["holdout_dentro_intervalo"] = teve_metrica and todos_dentro

        return comparacao


def executar_validacao_temporal(dados: pd.DataFrame, random_state: int = 42) -> Dict[str, Any]:
    """Função de conveniência para executar validação cruzada temporal.

    Parâmetros:
    - dados (pd.DataFrame): dataset completo
    - random_state (int): seed para reprodutibilidade

    Retorno:
    - dict: relatório completo
    """
    validador = TemporalCVValidator(dados, random_state=random_state)
    return validador.executar()
