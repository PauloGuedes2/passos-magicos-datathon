"""Servicos de runtime para acesso ao modelo em producao.

Responsabilidades:
- Encapsular acesso ao gerenciador de modelo da infraestrutura
- Expor funcoes de aplicacao para carga e recuperacao de modelo
"""

from src.infrastructure.model.model_manager import GerenciadorModelo


def carregar_modelo_runtime() -> None:
    """Carrega o modelo em memoria via gerenciador de infraestrutura."""
    GerenciadorModelo().carregar_modelo()


def obter_modelo_runtime():
    """Retorna instancia de modelo pronta para inferencia."""
    return GerenciadorModelo().obter_modelo()
