"""Ponto de entrada da API FastAPI.

Responsabilidades:
- Configurar a aplicação FastAPI
- Registrar rotas e eventos
- Inicializar recursos no startup
"""

import os

import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile, Request
import boto3
from src.api.controller import ControladorPredicao
from src.api.monitoring_controller import ControladorMonitoramento
from src.api.training_controller import ControladorTreinamento
from src.application.model_runtime_service import carregar_modelo_runtime, obter_modelo_runtime
from src.util.logger import logger
from fastapi.responses import HTMLResponse
from src.config.settings import Configuracoes

app = FastAPI(
    title="Passos Mágicos",
    description="API com Monitoramento e Treinamento para predição de risco de defasagem escolar",
    version="1.0.0",
)

B2_KEY_ID = os.getenv("B2_KEY_ID")
if B2_KEY_ID:
    s3_client = boto3.client(
        "s3",
        endpoint_url=os.getenv("B2_ENDPOINT_URL"),
        aws_access_key_id=B2_KEY_ID,
        aws_secret_access_key=os.getenv("B2_APPLICATION_KEY")
    )
else:
    s3_client = None

def sincronizar_dados_nuvem():
    """Descarrega ficheiros do Backblaze para a pasta local no startup para evitar perda no Render."""
    if not s3_client:
        logger.warning("Sincronização ignorada: chaves do Backblaze não configuradas.")
        return

    try:
        bucket = os.getenv("B2_BUCKET_NAME")
        objetos = s3_client.list_objects_v2(Bucket=bucket).get('Contents', [])
        
        for obj in objetos:
            nome_arquivo = obj['Key']
            caminho_local = os.path.join(Configuracoes.DATA_DIR, nome_arquivo)
            
            if not os.path.exists(caminho_local):
                logger.info(f"Sincronizando {nome_arquivo} da nuvem...")
                s3_client.download_file(bucket, nome_arquivo, caminho_local)
    except Exception as e:
        logger.error(f"Erro na sincronização: {e}")

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


    
@app.get("/admin/upload", response_class=HTMLResponse, tags=["Interface"])
async def pagina_upload():
    """Retorna a interface visual para upload de arquivos."""
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Datathon - Upload de Dados</title>
            <style>
                body { font-family: sans-serif; margin: 40px; line-height: 1.6; }
                .container { max-width: 500px; margin: auto; padding: 20px; border: 1px solid #ccc; border-radius: 8px; }
                input[type="file"] { margin: 20px 0; }
                input[type="submit"] { background: #007bff; color: white; border: none; padding: 10px 20px; cursor: pointer; border-radius: 4px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Enviar Novo Arquivo de Dados em .CSV .XLSX</h2>
                <form action="/api/v1/upload-arquivo" enctype="multipart/form-data" method="post">
                    <input name="file" type="file" accept=".xlsx, .csv" required>
                    <br>
                    <input type="submit" value="Fazer Upload">
                </form>
            </div>
        </body>
    </html>
    """
@app.post("/api/v1/upload-arquivo", tags=["Interface"])
async def receber_arquivo(file: UploadFile = File(...)):
    """Salva o arquivo enviado na pasta data do projeto."""
    try:
        # Usa o DATA_DIR definido nas suas configurações
        from src.config.settings import Configuracoes
        
        caminho_destino = os.path.join(Configuracoes.DATA_DIR, file.filename)
        
        # Cria a pasta caso ela não exista
        os.makedirs(Configuracoes.DATA_DIR, exist_ok=True)
        
        with open(caminho_destino, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        return {"status": "sucesso", "arquivo": file.filename}
    except Exception as e:
        return {"status": "erro", "detalhes": str(e)}
    
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
