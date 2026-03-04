import pytest
from fastapi.testclient import TestClient
import app.main as main

# Inicializa o cliente de teste do FastAPI
client = TestClient(main.app)

def test_checar_saude():
    """
    Testa o endpoint de health check.
    O teste agora aceita o campo 'cloud_storage' que pode ser 'ativo' ou 'inativo'
    dependendo das variáveis de ambiente no ambiente de teste.
    """
    response = client.get("/health")
    
    # Verifica se o status HTTP é 200 (OK)
    assert response.status_code == 200
    
    dados = response.json()
    
    # Valida se o status principal é 'ok'
    assert dados["status"] == "ok"
    
    # Valida se a chave de monitoramento da nuvem está presente
    # Usamos 'in' para garantir que o teste passe mesmo se a nuvem estiver off nos testes
    assert "cloud_storage" in dados
    assert dados["cloud_storage"] in ["ativo", "inativo"]

def test_pagina_home_carrega_corretamente():
    """
    Verifica se a rota raiz (se existir) ou a de upload está respondendo.
    """
    response = client.get("/admin/upload")
    assert response.status_code == 200
    assert "html" in response.headers["content-type"]

def test_rotas_api_v1_registradas():
    """
    Verifica se os prefixos das rotas principais foram registrados no app.
    """
    urls_encontradas = [route.path for route in main.app.routes]
    
    # Verifica se as rotas dos controladores estão presentes
    assert any("/api/v1" in url for url in urls_encontradas)
    assert any("/api/v1/monitoring" in url for url in urls_encontradas)

# Se você tiver testes de predição aqui, mantenha-os abaixo:
# def test_predicao_exemplo():
#     ...