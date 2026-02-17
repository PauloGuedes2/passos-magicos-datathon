"""Controlador de monitoramento da API.

Responsabilidades:
- Expor endpoint de dashboard de monitoramento
- Fornecer dependencia do servico de monitoramento
"""

import os

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from src.application.monitoring_service import ServicoMonitoramento
from src.application.professional_dashboard_service import ProfessionalDashboardService
from src.config.settings import Configuracoes


def obter_servico_monitoramento():
    """Dependencia para obter uma instancia do servico de monitoramento.

    Retorno:
    - ServicoMonitoramento: instancia pronta para uso
    """
    return ServicoMonitoramento()


class ControladorMonitoramento:
    """Controlador para endpoints de monitoramento.

    Responsabilidades:
    - Registrar rota do dashboard
    - Retornar dashboard profissional
    """

    def __init__(self):
        """Inicializa o controlador.

        Responsabilidades:
        - Criar o roteador
        - Registrar a rota do dashboard
        """
        self.roteador = APIRouter()
        self.roteador.add_api_route(
            "/dashboard",
            self._obter_dashboard,
            methods=["GET"],
            response_class=HTMLResponse,
        )
        self.roteador.add_api_route(
            "/feature-importance",
            self._obter_importancia_global,
            methods=["GET"],
            response_model=dict,
        )

    @staticmethod
    async def _obter_dashboard():
        """Retorna o dashboard de monitoramento.

        Retorno:
        - HTMLResponse: HTML do dashboard
        """
        dashboard_path = os.path.join(Configuracoes.MONITORING_DIR, "professional_dashboard.html")
        if not os.path.exists(dashboard_path):
            dashboard_path = ProfessionalDashboardService().generate_dashboard()

        with open(dashboard_path, "r", encoding="utf-8") as file:
            dashboard_html = file.read()

        return HTMLResponse(
            content=dashboard_html,
            media_type="text/html",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    @staticmethod
    async def _obter_importancia_global(
        servico: ServicoMonitoramento = Depends(obter_servico_monitoramento),
    ):
        """Retorna explicabilidade global via ranking de importancia de features."""
        return servico.obter_importancia_global()
