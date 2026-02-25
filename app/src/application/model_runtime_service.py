"""Servicos de runtime para acesso ao modelo em producao.

Responsabilidades:
- Encapsular acesso ao gerenciador de modelo da infraestrutura
- Expor funcoes de aplicacao para carga e recuperacao de modelo
"""

from src.infrastructure.model.model_manager import GerenciadorModelo


def carregar_modelo_runtime(model_version: str | None = None) -> None:
    """Carrega o modelo em memoria via gerenciador de infraestrutura."""
    GerenciadorModelo().carregar_modelo(model_version=model_version)


def obter_modelo_runtime(model_version: str | None = None):
    """Retorna instancia de modelo pronta para inferencia."""
    return GerenciadorModelo().obter_modelo_com_versao(model_version=model_version)


def listar_versoes_modelo_runtime() -> list[str]:
    """Lista versoes de modelo disponiveis para inferencia."""
    return GerenciadorModelo().listar_versoes_disponiveis(incluir_atual=True)
