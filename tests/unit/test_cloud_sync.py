import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import os
import sys
from pathlib import Path

# Garante que a raiz do projeto esteja no sys.path para localizar a pasta 'app'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.main import app, sincronizar_dados_nuvem

client = TestClient(app)

@pytest.fixture
def mock_s3():
    """Mock do cliente boto3 para evitar chamadas reais à rede durante os testes."""
    with patch("app.main.s3_client") as mock:
        yield mock

def test_pagina_upload_carrega_corretamente():
    """
    Testa se a interface HTML de upload está acessível e contém os textos esperados.
    O teste valida se a rota /admin/upload retorna o formulário correto.
    """
    response = client.get("/admin/upload")
    assert response.status_code == 200
    # O texto deve ser idêntico ao definido no main.py para passar no assert
    assert "Inserir Novos Dados (Excel/CSV)" in response.text
    assert 'method="post"' in response.text

@patch("app.main.s3_client")
def test_realizar_upload_sucesso(mock_s3_client):
    """
    Testa o endpoint de POST para garantir que o arquivo é processado.
    Verifica se o sistema tenta salvar na nuvem e localmente.
    """
    # Simula sucesso na resposta do S3 (Backblaze)
    mock_s3_client.put_object.return_value = {}
    
    file_content = b"RA,NOTA_MAT,NOTA_PORT\n12345,8.5,9.0"
    file_name = "teste_integracao.csv"
    
    # Mock do 'open' para não criar arquivos reais no seu disco durante o teste
    with patch("builtins.open", MagicMock()):
        response = client.post(
            "/api/v1/admin/upload",
            files={"file": (file_name, file_content, "text/csv")}
        )

    assert response.status_code == 200
    assert response.json()["status"] == "sucesso"
    assert response.json()["arquivo"] == file_name
    
    # Valida se a função de upload para o bucket foi chamada
    if mock_s3_client:
        mock_s3_client.put_object.assert_called()

@patch("app.main.s3_client")
@patch("app.main.Configuracoes")
@patch("os.path.exists")
def test_sincronizar_dados_nuvem_startup(mock_exists, mock_config, mock_s3_client):
    """
    Testa a lógica de sincronização que ocorre no startup da API.
    Simula um cenário onde um arquivo existe na nuvem mas não existe localmente.
    """
    # Simula que o bucket tem um arquivo
    mock_s3_client.list_objects_v2.return_value = {
        'Contents': [{'Key': 'historico_2024.xlsx'}]
    }
    
    # Simula que o arquivo NÃO existe na pasta local (DATA_DIR)
    mock_exists.return_value = False
    mock_config.DATA_DIR = "app/data"
    
    # Simula as variáveis de ambiente necessárias
    with patch("os.getenv", side_effect=lambda k, d=None: "meu-bucket" if k == "B2_BUCKET_NAME" else d):
        sincronizar_dados_nuvem()
        
    # Verifica se o download_file foi acionado para recuperar o arquivo da nuvem
    mock_s3_client.download_file.assert_called_with(
        "meu-bucket", 
        "historico_2024.xlsx", 
        os.path.join("app/data", "historico_2024.xlsx")
    )