"""Ponto de entrada da API FastAPI."""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Garante que o diretório 'app' esteja no path para importações do 'src'
sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn
import boto3
from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from botocore.config import Config
import re
from urllib.parse import urlparse
from src.api.controller import ControladorPredicao
from botocore.exceptions import ClientError
from src.api.monitoring_controller import ControladorMonitoramento
from src.api.training_controller import ControladorTreinamento
from src.application.model_runtime_service import carregar_modelo_runtime, obter_modelo_runtime
from src.util.logger import logger
from src.config.settings import Configuracoes


def _obter_env_limpo(nome: str, padrao: str | None = None) -> str | None:
    """Lê variável de ambiente removendo espaços/aspas acidentais."""
    valor = os.getenv(nome, padrao)
    if valor is None:
        return None

    valor_limpo = valor.strip().strip('"').strip("'")
    return valor_limpo or None

def _normalizar_endpoint_b2(endpoint: str | None) -> str | None:
    """Garante endpoint HTTPS sem sufixos inesperados (ex.: trailing slash)."""
    if not endpoint:
        return None

    endpoint = endpoint.rstrip("/")
    if not endpoint.startswith(("http://", "https://")):
        endpoint = f"https://{endpoint}"
    return endpoint


def _obter_regiao_b2(endpoint: str | None, regiao_env: str | None) -> str:
    """Resolve região para assinatura S3v4 (prioriza variável explícita)."""
    if regiao_env:
        return regiao_env

    if endpoint:
        hostname = urlparse(endpoint).hostname or ""
        correspondencia = re.search(r"s3\.([a-z0-9-]+)\.backblazeb2\.com$", hostname)
        if correspondencia:
            return correspondencia.group(1)

    return "us-east-005"

# Configuração do cliente B2 (S3 compatível)
B2_KEY_ID = _obter_env_limpo("B2_KEY_ID")
B2_ENDPOINT_URL = _normalizar_endpoint_b2(_obter_env_limpo("B2_ENDPOINT_URL"))
B2_REGION = _obter_regiao_b2(B2_ENDPOINT_URL, _obter_env_limpo("B2_REGION"))

if B2_KEY_ID:
    s3_client = boto3.client(
        "s3",
        endpoint_url=B2_ENDPOINT_URL,
        aws_access_key_id=B2_KEY_ID,
        aws_secret_access_key=_obter_env_limpo("B2_APPLICATION_KEY"),
        region_name=B2_REGION,
        
        # Configuração extra para compatibilidade total com B2
        config=Config(
            signature_version='s3v4',
            retries={'max_attempts': 3, 'mode': 'standard'},
            s3={'addressing_style': 'path'} # Força o estilo de caminho para evitar erros de DNS
        )
    )
else:
    s3_client = None

def sincronizar_dados_nuvem():
    """Descarrega ficheiros do Backblaze para a pasta local no startup para evitar perda no Render."""
    if not s3_client:
        logger.warning("Sincronização ignorada: chaves do Backblaze não configuradas.")
        return

    try:
        bucket = _obter_env_limpo("B2_BUCKET_NAME")
        if not bucket:
            logger.warning("Sincronização ignorada: B2_BUCKET_NAME não configurado.")
            return
        objetos = s3_client.list_objects_v2(Bucket=bucket).get('Contents', [])
        
        for obj in objetos:
            nome_arquivo = obj['Key']
            caminho_local = os.path.join(Configuracoes.DATA_DIR, nome_arquivo)
            
            if not os.path.exists(caminho_local):
                logger.info(f"Sincronizando {nome_arquivo} da nuvem...")
                s3_client.download_file(bucket, nome_arquivo, caminho_local)
    except Exception as e:
        logger.error(f"Erro na sincronização: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação (Startup e Shutdown)."""
    logger.info("Inicializando recursos da API...")
    
    # Cria pasta de dados se não existir
    os.makedirs(Configuracoes.DATA_DIR, exist_ok=True)
    
    # Sincroniza arquivos da nuvem (Backblaze) para o local (Render)
    if s3_client:
        sincronizar_dados_nuvem()
    
    # Carrega o modelo de ML na memória
    carregar_modelo_runtime()
    yield

app = FastAPI(
    title="Passos Mágicos",
    description="API com Monitoramento e Treinamento para predição de risco de defasagem escolar",
    version="1.1.0",
    lifespan=lifespan
)

# Inclusão de Rotas
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
                body { font-family: sans-serif; margin: 40px; line-height: 1.6; background-color: #f4f4f9; }
                .container { max-width: 500px; margin: auto; padding: 20px; border: 1px solid #ccc; border-radius: 8px; background-color: white; }
                input[type="file"] { margin: 20px 0; }
                input[type="submit"] { background: #007bff; color: white; border: none; padding: 10px 20px; cursor: pointer; border-radius: 4px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Inserir Novos Dados (Excel/CSV)</h2>
                <form action="/api/v1/admin/upload" enctype="multipart/form-data" method="post">
                    <input name="file" type="file" accept=".xlsx, .csv" required>
                    <br>
                    <input type="submit" value="Fazer Upload">
                </form>
            </div>
        </body>
    </html>
    """

@app.post("/api/v1/admin/upload", tags=["Interface"])
async def receber_arquivo(file: UploadFile = File(...)):
    """Salva o arquivo no Backblaze (Nuvem) e localmente."""
    try:
        conteudo = await file.read()
        
        # 1. Salva na Nuvem (Persistência para o Render)
        if s3_client:
            bucket = _obter_env_limpo("B2_BUCKET_NAME")
            if bucket:
                s3_client.put_object(Bucket=bucket, Key=file.filename, Body=conteudo)
            else:
                logger.warning("Upload em nuvem ignorado: B2_BUCKET_NAME não configurado.")
        
        # 2. Salva Localmente (Para uso imediato)
        caminho_local = os.path.join(Configuracoes.DATA_DIR, file.filename)
        os.makedirs(Configuracoes.DATA_DIR, exist_ok=True)
        with open(caminho_local, "wb") as buffer:
            buffer.write(conteudo)
            
        return {"status": "sucesso", "arquivo": file.filename}
    except ClientError as e:
        codigo_erro = e.response.get("Error", {}).get("Code", "desconhecido")
        mensagem = e.response.get("Error", {}).get("Message", str(e))
        logger.error(f"Erro no upload para Backblaze (codigo={codigo_erro}): {mensagem}")
        return {
            "status": "erro",
            "detalhes": mensagem,
            "diagnostico": (
                "Falha de autenticação no Backblaze S3. Confira B2_KEY_ID, "
                "B2_APPLICATION_KEY, B2_ENDPOINT_URL e B2_REGION no Render."
            ),
        }
    except Exception as e:
        logger.error(f"Erro no upload: {e}")
        return {"status": "erro", "detalhes": str(e)}

@app.get("/health", tags=["Infraestrutura"])
def checar_saude():
    try:
        obter_modelo_runtime()
        return {"status": "ok", "cloud_storage": "ativo" if s3_client else "inativo"}
    except Exception as erro:
        raise HTTPException(status_code=503, detail=str(erro))

if __name__ == "__main__":
    porta = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=porta)