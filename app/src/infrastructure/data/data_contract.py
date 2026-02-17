"""Validação de contrato de dados.

Responsabilidades:
- Validar presença de colunas obrigatórias
- Validar tipos de dados
- Falhar explicitamente se contrato for violado
"""

from typing import List, Dict

import pandas as pd

from src.util.logger import logger


class ContratoDataFrame:
    """Define e valida contrato de dados para DataFrames.

    Responsabilidades:
    - Especificar colunas obrigatórias
    - Validar tipos esperados
    - Falhar com mensagem clara se violado
    """

    def __init__(self, colunas_obrigatorias: List[str], tipos_esperados: Dict[str, type] = None):
        """Inicializa o contrato.

        Parâmetros:
        - colunas_obrigatorias (list): colunas que devem estar presentes
        - tipos_esperados (dict): mapeamento coluna -> tipo esperado
        """
        self.colunas_obrigatorias = colunas_obrigatorias
        self.tipos_esperados = tipos_esperados or {}

    def validar(self, df: pd.DataFrame) -> None:
        """Valida o DataFrame contra o contrato.

        Parâmetros:
        - df (pd.DataFrame): DataFrame a validar

        Exceções:
        - ValueError: quando contrato é violado
        """
        if df is None or df.empty:
            raise ValueError("DataFrame vazio ou nulo. Impossível validar contrato.")

        # Validar presença de colunas obrigatórias
        colunas_faltantes = [c for c in self.colunas_obrigatorias if c not in df.columns]
        if colunas_faltantes:
            raise ValueError(
                f"Contrato de dados violado: colunas obrigatórias ausentes: {colunas_faltantes}. "
                f"Colunas disponíveis: {list(df.columns)}"
            )

        # Validar tipos esperados
        for coluna, tipo_esperado in self.tipos_esperados.items():
            if coluna not in df.columns:
                continue

            try:
                # Para strings, apenas converter sem coerce
                if tipo_esperado is str:
                    df[coluna] = df[coluna].astype(str)
                else:
                    # Para numéricos, usar coerce para preencher NaNs
                    df[coluna] = pd.to_numeric(df[coluna], errors="coerce")
                    # Verifica se houve perda de dados na conversão
                    if df[coluna].isnull().sum() > 0:
                        logger.warning(
                            f"Coluna '{coluna}' contém valores que não podem ser convertidos para {tipo_esperado.__name__}. "
                            f"Valores nulos foram preenchidos com 0."
                        )
                        df[coluna] = df[coluna].fillna(0)
            except Exception as erro:
                raise ValueError(
                    f"Contrato de dados violado: coluna '{coluna}' não pode ser convertida para {tipo_esperado.__name__}. "
                    f"Erro: {erro}"
                )

        logger.info(f"Contrato de dados validado com sucesso. {len(df)} registros.")


# Contrato para dados de treinamento
CONTRATO_TREINO = ContratoDataFrame(
    colunas_obrigatorias=[
        "RA",
        "IDADE",
        "ANO_INGRESSO",
        "GENERO",
        "TURMA",
        "INSTITUICAO_ENSINO",
        "FASE",
        "ANO_REFERENCIA",
    ],
    tipos_esperados={
        "RA": str,
        "IDADE": int,
        "ANO_INGRESSO": int,
        "ANO_REFERENCIA": int,
    },
)

# Contrato para dados de referência (após processamento)
CONTRATO_REFERENCIA = ContratoDataFrame(
    colunas_obrigatorias=[
        "RA",
        "IDADE",
        "ANO_INGRESSO",
        "GENERO",
        "TURMA",
        "INSTITUICAO_ENSINO",
        "FASE",
        "ANO_REFERENCIA",
        "INDE_ANTERIOR",
        "ALUNO_NOVO",
        "RISCO_DEFASAGEM",
    ],
)
