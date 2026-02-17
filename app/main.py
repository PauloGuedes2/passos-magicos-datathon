"""Ponto de entrada da API FastAPI.

Responsabilidades:
- Configurar a aplicação FastAPI
- Registrar rotas e eventos
- Inicializar recursos no startup
"""

import os

import uvicorn
from fastapi import FastAPI, HTTPException

from src.api.controller import ControladorPredicao
from src.api.monitoring_controller import ControladorMonitoramento
from src.api.training_controller import ControladorTreinamento
from src.application.model_runtime_service import carregar_modelo_runtime, obter_modelo_runtime
from src.util.logger import logger

app = FastAPI(
    title="Passos Mágicos",
    description="API com Monitoramento e Treinamento para predição de risco de defasagem escolar",
    version="1.0.0",
)


@app.on_event("startup")
async def evento_inicializacao():
    """Executa ações de inicialização da aplicação.

    Responsabilidades:
    - Registrar log de inicialização
    - Carregar o modelo na memória

    Retorno:
    - None: não retorna valor
    """
    logger.info("Inicializando recursos da API...")
    carregar_modelo_runtime()


controlador_predicao = ControladorPredicao()
app.include_router(controlador_predicao.roteador, prefix="/api/v1", tags=["Predição"])

controlador_monitoramento = ControladorMonitoramento()
app.include_router(controlador_monitoramento.roteador, prefix="/api/v1/monitoring", tags=["Observabilidade"])

controlador_treinamento = ControladorTreinamento()
app.include_router(controlador_treinamento.roteador, prefix="/api/v1", tags=["Treinamento"])


@app.get("/health", tags=["Infraestrutura"])
def checar_saude():
    """Endpoint de health check.

    Retorno:
    - dict: status da aplicação
    """
    try:
        obter_modelo_runtime()
        return {"status": "ok"}
    except Exception as erro:
        raise HTTPException(status_code=503, detail=str(erro))


if __name__ == "__main__":
    porta = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=porta)
