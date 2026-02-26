"""Controlador de predicao da API."""

from fastapi import APIRouter, Depends, HTTPException

from src.application.model_runtime_service import listar_versoes_modelo_runtime, obter_modelo_runtime
from src.application.risk_service import ServicoRisco
from src.domain.student import EntradaEstudante


def obter_servico_risco(model_version: str | None = None):
    """Dependencia para obter uma instancia do servico de risco."""
    try:
        modelo, versao_resolvida = obter_modelo_runtime(model_version=model_version)
        return ServicoRisco(modelo=modelo, model_version=versao_resolvida)
    except FileNotFoundError as erro:
        raise HTTPException(status_code=404, detail=str(erro))
    except RuntimeError as erro:
        raise HTTPException(status_code=503, detail=f"Modelo de ML nao inicializado. {str(erro)}")


class ControladorPredicao:
    """Controlador de predicao."""

    def __init__(self):
        self.roteador = APIRouter()
        self._registrar_rotas()

    def _registrar_rotas(self):
        self.roteador.add_api_route(
            path="/models/versions",
            endpoint=self._listar_versoes_modelo,
            methods=["GET"],
            response_model=dict,
            summary="Lista versoes disponiveis do modelo",
        )

        self.roteador.add_api_route(
            path="/predict/smart",
            endpoint=self._predizer_inteligente,
            methods=["POST"],
            response_model=dict,
            summary="Predicao com busca automatica de historico",
        )

    @staticmethod
    async def _listar_versoes_modelo():
        """Lista versoes disponiveis para selecao no endpoint de predicao."""
        try:
            versoes = listar_versoes_modelo_runtime()
            return {
                "available_model_versions": versoes,
                "default_model_version": versoes[0] if versoes else None,
            }
        except Exception as erro:
            raise HTTPException(status_code=503, detail=f"Falha ao listar versoes de modelo. {erro}")

    @staticmethod
    async def _predizer_inteligente(
        entrada: EntradaEstudante,
        servico: ServicoRisco = Depends(obter_servico_risco),
    ):
        """Predicao inteligente com busca automatica de historico."""
        try:
            return servico.prever_risco_inteligente(entrada)
        except (ValueError, TypeError, KeyError) as erro:
            raise HTTPException(status_code=400, detail=str(erro))
        except RuntimeError as erro:
            raise HTTPException(status_code=503, detail=str(erro))
