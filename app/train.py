"""Ponto de entrada do pipeline de treinamento.

Responsabilidades:
- Orquestrar carregamento de dados
- Disparar treinamento do pipeline
- Tratar falhas e finalizar com código de saída
"""

from src.infrastructure.data.data_loader import CarregadorDados
from src.infrastructure.model.ml_pipeline import treinador
from src.util.logger import logger


if __name__ == "__main__":
    logger.info("Iniciando Pipeline de Treinamento...")

    try:
        carregador = CarregadorDados()
        df_bruto = carregador.carregar_dados()

        treinador.treinar(df_bruto)

        logger.info("Processo concluído com sucesso!")

    except Exception as erro:
        logger.exception(f"Ocorreu um erro fatal durante o pipeline de treinamento: {str(erro)}")
        exit(1)
