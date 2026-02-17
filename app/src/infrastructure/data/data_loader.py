"""Carregamento e preparação de dados históricos.

Responsabilidades:
- Localizar arquivos Excel
- Normalizar colunas
- Unificar abas por ano
- Validar contrato de dados
"""

import glob
import os
import re

import pandas as pd
import unicodedata

from src.config.settings import Configuracoes
from src.infrastructure.data.data_contract import CONTRATO_TREINO
from src.util.logger import logger


class CarregadorDados:
    """Responsável pelo carregamento, limpeza e unificação dos dados históricos.

    Responsabilidades:
    - Buscar arquivos na pasta de dados
    - Processar abas por ano
    - Concatenar datasets
    """

    def carregar_dados(self) -> pd.DataFrame:
        """Busca arquivos Excel na pasta de dados e unifica as abas por ano.

        Retorno:
        - pd.DataFrame: dataset consolidado

        Exceções:
        - FileNotFoundError: quando não há arquivos .xlsx
        - RuntimeError: quando nenhuma aba válida é encontrada
        """
        padroes = ["*.xlsx", "*.csv"]
        arquivos = []
        for padrao in padroes:
            caminho_busca = os.path.join(Configuracoes.DATA_DIR, padrao)
            arquivos.extend(glob.glob(caminho_busca))
            logger.info(f"Buscando arquivos em: {caminho_busca}")

        if not arquivos:
            self._registrar_conteudo_pasta()
            raise FileNotFoundError(
                "Nenhum arquivo de dados encontrado em "
                f"{Configuracoes.DATA_DIR}. Defina DATA_DIR ou monte o volume de dados "
                "no container para habilitar retreino."
            )

        dados_unificados = []
        for caminho_arquivo in arquivos:
            if caminho_arquivo.endswith(".xlsx"):
                logger.info(f"Carregando arquivo Excel: {caminho_arquivo}")
                abas = self._ler_excel(caminho_arquivo)
                dados_unificados.extend(self._processar_abas(abas))
            else:
                logger.info(f"Carregando arquivo CSV: {caminho_arquivo}")
                df_csv = self._ler_csv(caminho_arquivo)
                if df_csv is not None:
                    if "ANO_REFERENCIA" not in df_csv.columns:
                        raise ValueError(
                            "CSV sem ANO_REFERENCIA. Defina a coluna para evitar vazamento temporal."
                        )
                    dados_unificados.append(
                        self._processar_dataframe(df_csv, int(df_csv["ANO_REFERENCIA"].iloc[0]))
                    )

        if not dados_unificados:
            raise RuntimeError("Nenhuma aba válida carregada do Excel.")

        try:
            df_final = pd.concat(dados_unificados, ignore_index=True)
        except Exception as erro:
            logger.error(f"Erro ao concatenar os dados: {erro}")
            raise erro

        logger.info(f"Dataset Total Unificado: {df_final.shape}")

        # Validar contrato de dados
        try:
            CONTRATO_TREINO.validar(df_final)
        except ValueError as erro:
            logger.error(f"Falha na validação do contrato de dados: {erro}")
            raise erro

        return df_final

    def _registrar_conteudo_pasta(self) -> None:
        """Registra o conteúdo da pasta de dados no log.

        Retorno:
        - None: não retorna valor
        """
        try:
            conteudo = os.listdir(Configuracoes.DATA_DIR)
            logger.error(f"Conteúdo encontrado em {Configuracoes.DATA_DIR}: {conteudo}")
        except Exception:
            return

    @staticmethod
    def _ler_excel(caminho_arquivo: str):
        """Lê todas as abas de um arquivo Excel.

        Parâmetros:
        - caminho_arquivo (str): caminho do arquivo

        Retorno:
        - dict: abas e DataFrames

        Exceções:
        - Exception: quando a leitura falha
        """
        try:
            return pd.read_excel(caminho_arquivo, sheet_name=None)
        except Exception as erro:
            logger.error(f"Erro crítico ao ler o Excel: {erro}")
            raise erro

    @staticmethod
    def _ler_csv(caminho_arquivo: str):
        """Lê um arquivo CSV.

        Parâmetros:
        - caminho_arquivo (str): caminho do arquivo

        Retorno:
        - pd.DataFrame | None: DataFrame lido
        """
        try:
            try:
                df = pd.read_csv(caminho_arquivo, sep=";")
                if len(df.columns) <= 1:
                    df = pd.read_csv(caminho_arquivo, sep=",")
            except Exception:
                df = pd.read_csv(caminho_arquivo, sep=",")
            return df
        except Exception as erro:
            logger.error(f"Erro crítico ao ler o CSV: {erro}")
            raise erro

    def _processar_abas(self, abas: dict):
        """Processa abas válidas e retorna lista de DataFrames.

        Parâmetros:
        - abas (dict): dicionário de abas

        Retorno:
        - list[pd.DataFrame]: dados processados
        """
        dados = []

        for nome_aba, df_aba in abas.items():
            ano_match = re.search(r"202\d", nome_aba)

            if not ano_match:
                logger.warning(f"Aba '{nome_aba}' ignorada (não contém ano no nome).")
                continue

            ano_completo = int(ano_match.group())
            logger.info(f"Processando aba: {nome_aba} (Ano {ano_completo})")

            df_processado = self._processar_dataframe(df_aba, ano_completo)
            df_processado["ANO_REFERENCIA"] = ano_completo

            dados.append(df_processado)

        return dados

    @staticmethod
    def _processar_dataframe(df: pd.DataFrame, ano_completo: int) -> pd.DataFrame:
        """Normaliza colunas e dados de uma aba.

        Parâmetros:
        - df (pd.DataFrame): dados da aba
        - ano_completo (int): ano da referência

        Retorno:
        - pd.DataFrame: DataFrame processado
        """
        novas_colunas = []
        ano_curto = int(str(ano_completo)[-2:])

        for coluna in df.columns:
            coluna_limpa = str(coluna).upper().strip()
            coluna_limpa = unicodedata.normalize("NFKD", coluna_limpa).encode("ASCII", "ignore").decode("utf-8")
            coluna_limpa = re.sub(f"[ _]{ano_completo}", "", coluna_limpa)
            coluna_limpa = re.sub(f"[ _]{ano_curto}$", "", coluna_limpa)

            if coluna_limpa in ["RA", "ID_ALUNO", "CODIGO_ALUNO", "MATRICULA"]:
                coluna_limpa = "RA"
            elif coluna_limpa in ["MAT", "MATEM", "MATEMATICA"]:
                coluna_limpa = "NOTA_MAT"
            elif coluna_limpa in ["POR", "PORT", "PORTUG", "PORTUGUES"]:
                coluna_limpa = "NOTA_PORT"
            elif coluna_limpa in ["ING", "INGL", "INGLES"]:
                coluna_limpa = "NOTA_ING"
            elif coluna_limpa in ["DEFAS", "DEFASAGEM"]:
                coluna_limpa = "DEFASAGEM"
            elif "ANO" in coluna_limpa and "INGRESSO" in coluna_limpa:
                coluna_limpa = "ANO_INGRESSO"

            if "INST" in coluna_limpa and "ENSINO" in coluna_limpa:
                coluna_limpa = "INSTITUICAO_ENSINO"
            if "PONTO" in coluna_limpa and "VIRADA" in coluna_limpa:
                coluna_limpa = "PONTO_VIRADA"
            if "PSICOLOGIA" in coluna_limpa and "REC" in coluna_limpa:
                coluna_limpa = "REC_PSICOLOGIA"

            novas_colunas.append(coluna_limpa)

        df.columns = novas_colunas

        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        if "RA" in df.columns:
            df["RA"] = df["RA"].astype(str).str.strip()

        return df
