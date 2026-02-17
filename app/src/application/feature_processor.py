"""Processamento de features de entrada.

Responsabilidades:
- Calcular tempo na ONG
- Garantir colunas obrigatórias
- Normalizar tipos numéricos e categóricos
"""

from datetime import datetime
import re
from typing import Optional, Dict, Any

import pandas as pd

from src.config.settings import Configuracoes


class ProcessadorFeatures:
    """Processa DataFrames de entrada para o formato esperado pelo modelo.

    Responsabilidades:
    - Aplicar regras de cálculo de tempo na ONG
    - Preencher colunas ausentes
    - Normalizar tipos de dados
    """

    @staticmethod
    def processar(
        dados: pd.DataFrame,
        data_snapshot: Optional[datetime] = None,
        estatisticas: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Processa o DataFrame de entrada.

        Parâmetros:
        - dados (pd.DataFrame): dados de entrada
        - data_snapshot (datetime | None): data de referência para cálculos
        - estatisticas (dict | None): estatísticas para preenchimento de nulos

        Retorno:
        - pd.DataFrame: DataFrame com features normalizadas
        """
        dados_copia = dados.copy()

        referencia = ProcessadorFeatures._obter_ano_referencia(dados_copia, data_snapshot)
        dados_copia = ProcessadorFeatures._calcular_tempo_ong(dados_copia, referencia, estatisticas)
        dados_copia = ProcessadorFeatures._garantir_colunas_obrigatorias(dados_copia)

        colunas = Configuracoes.FEATURES_NUMERICAS + Configuracoes.FEATURES_CATEGORICAS
        dados_processados = dados_copia[colunas].copy()
        dados_processados = ProcessadorFeatures._normalizar_numericos(
            dados_processados, estatisticas=estatisticas
        )
        dados_processados = ProcessadorFeatures._normalizar_categoricos(dados_processados)

        return dados_processados

    @staticmethod
    def _obter_ano_referencia(dados: pd.DataFrame, data_snapshot: Optional[datetime]):
        """Obtém o ano de referência para cálculos.

        Parâmetros:
        - dados (pd.DataFrame): dados de entrada
        - data_snapshot (datetime | None): data de referência

        Retorno:
        - int | pd.Series: ano de referência
        """
        if "ANO_REFERENCIA" in dados.columns:
            return pd.to_numeric(dados["ANO_REFERENCIA"], errors="coerce")

        if data_snapshot is None:
            raise ValueError("ANO_REFERENCIA ausente e data_snapshot nula para cálculo de tempo na ONG.")

        return data_snapshot.year

    @staticmethod
    def _calcular_tempo_ong(
        dados: pd.DataFrame, referencia, estatisticas: Optional[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Calcula a coluna TEMPO_NA_ONG.

        Parâmetros:
        - dados (pd.DataFrame): dados de entrada
        - referencia (int | pd.Series): ano de referência
        - estatisticas (dict | None): estatísticas para preenchimento

        Retorno:
        - pd.DataFrame: DataFrame com TEMPO_NA_ONG atualizado
        """
        referencia_ano = None
        if isinstance(referencia, pd.Series):
            referencia_validos = pd.to_numeric(referencia, errors="coerce")
            if referencia_validos.notna().any():
                referencia_ano = int(referencia_validos.median())
        else:
            try:
                referencia_ano = int(referencia)
            except (TypeError, ValueError):
                referencia_ano = None

        if "ANO_INGRESSO" in dados.columns:
            ano_ingresso = pd.to_numeric(dados["ANO_INGRESSO"], errors="coerce")
            ano_ingresso = ProcessadorFeatures._preencher_ano_ingresso(
                ano_ingresso, estatisticas, referencia_ano
            )
            dados["TEMPO_NA_ONG"] = referencia - ano_ingresso
            dados["TEMPO_NA_ONG"] = dados["TEMPO_NA_ONG"].clip(lower=0)
        else:
            dados["TEMPO_NA_ONG"] = 0

        return dados

    @staticmethod
    def _preencher_ano_ingresso(
        serie: pd.Series, estatisticas: Optional[Dict[str, Any]], referencia_ano: Optional[int]
    ) -> pd.Series:
        """Preenche valores nulos de ANO_INGRESSO.

        Parâmetros:
        - serie (pd.Series): série de ano de ingresso
        - estatisticas (dict | None): estatísticas para preenchimento

        Retorno:
        - pd.Series: série com nulos preenchidos
        """
        if not serie.isnull().any():
            return serie

        if estatisticas and "mediana_ano_ingresso" in estatisticas:
            mediana = estatisticas["mediana_ano_ingresso"]
        elif not serie.isnull().all():
            mediana = serie.median()
        elif referencia_ano is not None:
            mediana = referencia_ano
        else:
            raise ValueError("ANO_INGRESSO nulo e sem referência de ano para imputar.")

        return serie.fillna(mediana)

    @staticmethod
    def _garantir_colunas_obrigatorias(dados: pd.DataFrame) -> pd.DataFrame:
        """Garante a presença de colunas obrigatórias.

        Parâmetros:
        - dados (pd.DataFrame): dados de entrada

        Retorno:
        - pd.DataFrame: DataFrame com colunas garantidas
        """
        colunas_obrigatorias = Configuracoes.FEATURES_NUMERICAS + Configuracoes.FEATURES_CATEGORICAS
        for coluna in colunas_obrigatorias:
            if coluna not in dados.columns:
                dados[coluna] = 0 if coluna in Configuracoes.FEATURES_NUMERICAS else "N/A"
        return dados

    @staticmethod
    def _normalizar_numericos(
        dados: pd.DataFrame, estatisticas: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Normaliza colunas numéricas.

        Parâmetros:
        - dados (pd.DataFrame): DataFrame com features

        Retorno:
        - pd.DataFrame: DataFrame com numéricos normalizados
        """
        idade_default = 12
        if estatisticas and "mediana_idade" in estatisticas:
            try:
                idade_default = int(estatisticas["mediana_idade"])
            except (TypeError, ValueError):
                idade_default = 12

        for coluna in Configuracoes.FEATURES_NUMERICAS:
            serie = pd.to_numeric(dados[coluna], errors="coerce")
            if coluna == "IDADE":
                serie = serie.where((serie >= 4) & (serie <= 25))
                serie = serie.fillna(idade_default)
                serie = serie.clip(lower=4, upper=25)
            elif coluna == "TEMPO_NA_ONG":
                serie = serie.fillna(0).clip(lower=0, upper=30)
            elif coluna.endswith("_ANTERIOR"):
                serie = serie.fillna(0).clip(lower=0, upper=10)
            elif coluna == "ALUNO_NOVO":
                serie = serie.fillna(0).astype(int).clip(lower=0, upper=1)
            else:
                serie = serie.fillna(0)
            dados[coluna] = serie
        return dados

    @staticmethod
    def _normalizar_categoricos(dados: pd.DataFrame) -> pd.DataFrame:
        """Normaliza colunas categóricas.

        Parâmetros:
        - dados (pd.DataFrame): DataFrame com features

        Retorno:
        - pd.DataFrame: DataFrame com categóricos normalizados
        """
        for coluna in Configuracoes.FEATURES_CATEGORICAS:
            if coluna == "GENERO":
                dados[coluna] = dados[coluna].apply(ProcessadorFeatures._limpar_genero)
            elif coluna == "FASE":
                dados[coluna] = dados[coluna].apply(ProcessadorFeatures._limpar_fase)
            else:
                dados[coluna] = dados[coluna].astype(str).replace("nan", "N/A")
        return dados

    @staticmethod
    def _limpar_genero(valor) -> str:
        """Normaliza valores de gênero para o padrão do modelo.

        Parâmetros:
        - valor (Any): valor original

        Retorno:
        - str: gênero normalizado
        """
        if pd.isna(valor):
            return "Outro"
        texto = str(valor).lower().strip()

        if any(item in texto for item in ["fem", "menina", "mulher", "garota", "feminino", "f"]):
            return "Feminino"
        if any(item in texto for item in ["masc", "menino", "homem", "garoto", "masculino", "m"]):
            return "Masculino"
        return "Outro"

    @staticmethod
    def _limpar_fase(valor) -> str:
        """Normaliza valores de fase para o padrão do modelo.

        Parâmetros:
        - valor (Any): valor original

        Retorno:
        - str: fase normalizada
        """
        if pd.isna(valor):
            return "0"
        limpo = re.sub(r"[^A-Z0-9]", "", str(valor).upper())
        return limpo if limpo else "0"

