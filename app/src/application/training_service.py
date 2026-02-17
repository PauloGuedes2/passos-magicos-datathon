"""Servico de treinamento/re-treinamento do modelo.

Responsabilidades:
- Orquestrar carregamento dos dados de treino
- Executar pipeline de treinamento
- Recarregar modelo promovido em memoria
"""

from src.infrastructure.data.data_loader import CarregadorDados
from src.infrastructure.model.ml_pipeline import treinador
from src.infrastructure.model.model_manager import GerenciadorModelo


class ServicoTreinamento:
    """Servico responsavel pelo ciclo de treinamento.

    Responsabilidades:
    - Carregar dados historicos
    - Executar treinamento do pipeline
    - Atualizar modelo em memoria apos promocao
    """

    @staticmethod
    def executar_treinamento() -> dict:
        """Executa treinamento/re-treinamento do modelo.

        Retorno:
        - dict: status de execucao
        """
        carregador = CarregadorDados()
        dados = carregador.carregar_dados()
        treinador.treinar(dados)
        GerenciadorModelo().carregar_modelo(force=True)
        return {"status": "ok", "message": "Treinamento concluido com sucesso."}
