"""Controlador de treinamento da API.

Responsabilidades:
- Definir rota de treinamento/re-treinamento
- Resolver dependencia do servico de treinamento
- Traduzir erros em respostas HTTP
"""

from fastapi import APIRouter, Depends, HTTPException

from src.application.training_service import ServicoTreinamento


def obter_servico_treinamento():
    """Dependencia para obter uma instancia do servico de treinamento.

    Retorno:
    - ServicoTreinamento: instancia pronta para uso
    """
    return ServicoTreinamento()


class ControladorTreinamento:
    """Controlador para endpoint de treinamento.

    Responsabilidades:
    - Registrar rota de treinamento/re-treinamento
    - Expor endpoint de execucao do pipeline
    """

    def __init__(self):
        """Inicializa o controlador.

        Responsabilidades:
        - Instanciar o roteador
        - Registrar as rotas disponiveis
        """
        self.roteador = APIRouter()
        self.roteador.add_api_route(
            path="/train/retrain",
            endpoint=self._retreinar_modelo,
            methods=["POST"],
            response_model=dict,
            summary="Executa treinamento/re-treinamento do modelo",
        )

    @staticmethod
    async def _retreinar_modelo(
        servico: ServicoTreinamento = Depends(obter_servico_treinamento),
    ):
        """Executa treinamento/re-treinamento do modelo.

        Parametros:
        - servico (ServicoTreinamento): servico de treinamento injetado

        Retorno:
        - dict: status da execucao

        Excecoes:
        - HTTPException: erro durante a execucao do treinamento
        """
        try:
            return servico.executar_treinamento()
        except (ValueError, TypeError, KeyError, FileNotFoundError) as erro:
            raise HTTPException(status_code=400, detail=str(erro))
        except RuntimeError as erro:
            raise HTTPException(status_code=503, detail=str(erro))
        except Exception as erro:
            raise HTTPException(status_code=500, detail=str(erro))
