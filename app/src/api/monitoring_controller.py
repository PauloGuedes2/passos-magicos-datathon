"""Controlador de monitoramento da API.

Responsabilidades:
- Fornecer dependencia do servico de monitoramento
"""

from fastapi import APIRouter, Depends

from src.application.monitoring_service import ServicoMonitoramento


def obter_servico_monitoramento():
    """Dependencia para obter uma instancia do servico de monitoramento.

    Retorno:
    - ServicoMonitoramento: instancia pronta para uso
    """
    return ServicoMonitoramento()


class ControladorMonitoramento:
    """Controlador para endpoints de monitoramento.

    Responsabilidades:
    - Registrar rotas de observabilidade
    """

    def __init__(self):
        """Inicializa o controlador.

        Responsabilidades:
        - Criar o roteador
        - Registrar rotas de observabilidade
        """
        self.roteador = APIRouter()
        self.roteador.add_api_route(
            "/feature-importance",
            self._obter_importancia_global,
            methods=["GET"],
            response_model=dict,
        )

    @staticmethod
    async def _obter_importancia_global(
        servico: ServicoMonitoramento = Depends(obter_servico_monitoramento),
    ):
        """Retorna explicabilidade global via ranking de importancia de features."""
        return servico.obter_importancia_global()
