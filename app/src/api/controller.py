"""Controlador de predição da API.

Responsabilidades:
- Definir rotas de predição
- Resolver dependências do serviço de risco
- Traduzir erros em respostas HTTP
"""

from fastapi import APIRouter, HTTPException, Depends

from src.application.model_runtime_service import obter_modelo_runtime
from src.application.risk_service import ServicoRisco
from src.domain.student import Estudante, EntradaEstudante


def obter_servico_risco():
    """Dependência para obter uma instância do serviço de risco.

    Responsabilidades:
    - Validar se o modelo está carregado
    - Criar o serviço com o modelo em memória

    Retorno:
    - ServicoRisco: instância pronta para uso

    Exceções:
    - HTTPException: quando o modelo não está disponível
    """
    try:
        modelo = obter_modelo_runtime()
        return ServicoRisco(modelo=modelo)
    except RuntimeError as erro:
        raise HTTPException(status_code=503, detail=f"Modelo de ML não inicializado. {str(erro)}")


class ControladorPredicao:
    """Controlador de predição.

    Responsabilidades:
    - Registrar rotas de predição
    - Expor endpoints de predição completa e inteligente
    """

    def __init__(self):
        """Inicializa o controlador.

        Responsabilidades:
        - Instanciar o roteador
        - Registrar as rotas disponíveis
        """
        self.roteador = APIRouter()
        self._registrar_rotas()

    def _registrar_rotas(self):
        """Registra as rotas de predição.

        Responsabilidades:
        - Configurar endpoint de predição completa
        - Configurar endpoint de predição inteligente
        """
        self.roteador.add_api_route(
            path="/predict/full",
            endpoint=self._predizer,
            methods=["POST"],
            response_model=dict,
        )

        self.roteador.add_api_route(
            path="/predict/smart",
            endpoint=self._predizer_inteligente,
            methods=["POST"],
            response_model=dict,
            summary="Predição com busca automática de histórico",
        )

    @staticmethod
    async def _predizer(estudante: Estudante, servico: ServicoRisco = Depends(obter_servico_risco)):
        """Predição tradicional com modelo completo do aluno.

        Parâmetros:
        - estudante (Estudante): dados completos do aluno
        - servico (ServicoRisco): serviço de risco injetado

        Retorno:
        - dict: resultado da predição

        Exceções:
        - HTTPException: erro interno durante a predição
        """
        try:
            return servico.prever_risco(estudante.model_dump())
        except (ValueError, TypeError, KeyError) as erro:
            raise HTTPException(status_code=400, detail=str(erro))
        except RuntimeError as erro:
            raise HTTPException(status_code=503, detail=str(erro))

    @staticmethod
    async def _predizer_inteligente(
        entrada: EntradaEstudante, servico: ServicoRisco = Depends(obter_servico_risco)
    ):
        """Predição inteligente com busca automática de histórico.

        Parâmetros:
        - entrada (EntradaEstudante): dados básicos do aluno
        - servico (ServicoRisco): serviço de risco injetado

        Retorno:
        - dict: resultado da predição

        Exceções:
        - HTTPException: erro interno durante a predição
        """
        try:
            return servico.prever_risco_inteligente(entrada)
        except (ValueError, TypeError, KeyError) as erro:
            raise HTTPException(status_code=400, detail=str(erro))
        except RuntimeError as erro:
            raise HTTPException(status_code=503, detail=str(erro))
