"""
Simulador de trafego de producao para a API de predicao.

Responsabilidades:
- Carregar dados reais do diretorio de dados
- Sanitizar e normalizar campos para o payload
- Enviar requisicoes continuas para a API
"""

import argparse
import glob
import os
import re
import sys
import time
import warnings

import pandas as pd
import requests

# Suprime avisos de pandas
warnings.simplefilter(action="ignore", category=FutureWarning)

# --- 1. Configuracao de Path ---
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
RAIZ_PROJETO = os.path.dirname(os.path.dirname(DIRETORIO_ATUAL))
APP_DIR = os.path.join(RAIZ_PROJETO, "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
if RAIZ_PROJETO not in sys.path:
    sys.path.insert(0, RAIZ_PROJETO)

from src.config.settings import Configuracoes

# --- 2. Configuracoes da API ---
PORTA = int(os.getenv("PORT", 8000))
DEFAULT_API_URL = f"http://localhost:{PORTA}/api/v1/predict/smart"
# DEFAULT_API_URL = "https://passos-magicos-datathon.onrender.com/api/v1/predict/smart"
URL_API = os.getenv("API_URL", DEFAULT_API_URL).strip()
DELAY = 0.1  # Acelerado para teste


def obter_coluna(row, nomes_possiveis):
    """
    Retorna o primeiro valor encontrado em possiveis nomes de coluna.

    Parametros:
    - row (pd.Series): linha do DataFrame
    - nomes_possiveis (list[str]): nomes de colunas possiveis

    Retorno:
    - Any: valor encontrado ou None
    """
    for nome in nomes_possiveis:
        nome_upper = nome.upper().strip()
        if nome_upper in row and pd.notnull(row[nome_upper]):
            return row[nome_upper]
    return None


def carregar_dados_reais():
    """
    Carrega dados reais do diretorio de dados.

    Retorno:
    - pd.DataFrame | None: dados consolidados ou None
    """
    diretorio_dados = Configuracoes.DATA_DIR
    print(f"[INFO] Buscando arquivos em: {diretorio_dados}")

    try:
        from src.infrastructure.data.data_loader import CarregadorDados
        return CarregadorDados().carregar_dados()
    except Exception as erro:
        print(f"[WARN] Falha ao usar CarregadorDados: {erro}. Usando fallback bruto.")

    extensoes = ["*.xlsx", "*.csv"]
    arquivos = []
    for extensao in extensoes:
        arquivos.extend(glob.glob(os.path.join(diretorio_dados, extensao)))

    if not arquivos:
        print(f"[ERROR] Nenhum arquivo encontrado em {diretorio_dados}")
        return None

    dataframes = []
    for arquivo in arquivos:
        try:
            nome_arquivo = os.path.basename(arquivo)
            if arquivo.endswith(".xlsx"):
                excel = pd.ExcelFile(arquivo)
                for nome_aba in excel.sheet_names:
                    df = pd.read_excel(arquivo, sheet_name=nome_aba)
                    df["_ORIGEM"] = f"{nome_arquivo} ({nome_aba})"
                    match_ano = re.search(r"(20\\d{2})", str(nome_aba))
                    if match_ano:
                        df["ANO_REFERENCIA"] = int(match_ano.group(1))
                    dataframes.append(df)
            else:
                try:
                    df = pd.read_csv(arquivo, sep=";")
                    if len(df.columns) <= 1:
                        df = pd.read_csv(arquivo, sep=",")
                except Exception:
                    df = pd.read_csv(arquivo, sep=",")
                df["_ORIGEM"] = nome_arquivo
                dataframes.append(df)
        except Exception as erro:
            print(f"[WARN] Ignorando {arquivo}: {erro}")

    if not dataframes:
        return None
    return pd.concat(dataframes, ignore_index=True)


def normalizar_colunas(df):
    """
    Normaliza nomes de colunas e ajustes de RA.

    Parametros:
    - df (pd.DataFrame): dados originais

    Retorno:
    - pd.DataFrame: dados normalizados
    """
    df.columns = [str(c).upper().strip() for c in df.columns]
    mapa_renomear = {
        "ID_ALUNO": "RA",
        "CODIGO_ALUNO": "RA",
        "MATRICULA": "RA",
        "ALUNO": "NOME",
        "NOME_ALUNO": "NOME",
        "ANO REFERENCIA": "ANO_REFERENCIA",
    }
    df = df.rename(columns=mapa_renomear)
    if "RA" in df.columns:
        df["RA"] = df["RA"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    return df


def obter_stream_infinito(df):
    """
    Gera um stream infinito de linhas embaralhadas.

    Parametros:
    - df (pd.DataFrame): dados de origem

    Retorno:
    - generator: stream infinito de linhas
    """
    while True:
        df_embaralhado = df.sample(frac=1).reset_index(drop=True)
        for _, row in df_embaralhado.iterrows():
            yield row


def _montar_payload(row, chaves):
    """
    Monta o payload de requisicao a partir da linha.

    Parametros:
    - row (pd.Series): linha de dados
    - chaves (dict): dicionario de chaves por campo

    Retorno:
    - dict: payload pronto para envio
    """
    ano_ref_raw = obter_coluna(row, chaves["ano_referencia"])

    ano_ref_final = None
    if ano_ref_raw:
        try:
            ano_ref_final = int(float(ano_ref_raw))
        except Exception:
            ano_ref_final = None

    payload = {
        "RA": str(row["RA"]),
        "ANO_REFERENCIA": ano_ref_final,
    }

    if payload.get("ANO_REFERENCIA") is None:
        payload.pop("ANO_REFERENCIA", None)

    return payload


def _enviar_payload(payload):
    """
    Envia o payload para a API.

    Parametros:
    - payload (dict): dados da requisicao

    Retorno:
    - requests.Response: resposta da API
    """
    inicio = time.time()
    resposta = requests.post(URL_API, json=payload, timeout=(3.05, 10))
    _ = time.time() - inicio
    return resposta


def _atualizar_snapshot_local():
    """
    Atualiza snapshot de monitoramento a partir dos artefatos persistidos.
    """
    from src.application.monitoring_service import ServicoMonitoramento

    resultado = ServicoMonitoramento.atualizar_snapshot_monitoramento()
    print(f"Snapshot de monitoramento atualizado: {resultado}")


def simular_trafego_producao(
    max_requests: int | None = None,
    delay: float = DELAY,
    atualizar_snapshot: bool = False,
    snapshot_a_cada_request: bool = True,
):
    """
    Inicia a simulacao de trafego de producao.

    Retorno:
    - None: nao retorna valor
    """
    print("--- [START] Iniciando simulacao BLINDADA (sanitizacao ativa) ---")

    dados_brutos = carregar_dados_reais()
    if dados_brutos is None or dados_brutos.empty:
        return

    dados = normalizar_colunas(dados_brutos)

    if "RA" not in dados.columns:
        print("Erro: Coluna RA nao encontrada.")
        return

    if "ANO_REFERENCIA" in dados.columns:
        ano_referencia = pd.to_numeric(dados["ANO_REFERENCIA"], errors="coerce")
        validos = ano_referencia.notna()
        if validos.any():
            ano_max = int(ano_referencia[validos].max())
            dados = dados[validos & (ano_referencia == ano_max)]
            print(f"Filtrando producao para ANO_REFERENCIA == {ano_max}")
        else:
            print("Nenhuma linha com ANO_REFERENCIA valido. Prosseguindo sem filtro.")

    print(f"[OK] Dados carregados: {len(dados)} linhas.")

    chaves = {
        "ano_referencia": ["ANO_REFERENCIA", "ANO REFERENCIA", "ANO_REF"],
    }

    stream = obter_stream_infinito(dados)
    contador = 0
    sucesso = 0
    erros = 0
    ignorados = 0

    for row in stream:
        contador += 1
        try:
            payload = _montar_payload(row, chaves)
            if payload is None:
                ignorados += 1
                continue
            resposta = _enviar_payload(payload)

            origem = str(row.get("_ORIGEM", "BD"))[:15]

            if resposta.status_code == 200:
                dados_resposta = resposta.json()
                sucesso += 1
                print(
                    f"#{contador} | [OK] {origem} | RA={payload['RA']} | "
                    f"ANO_REF={payload.get('ANO_REFERENCIA', 'N/A')} | "
                    f"{dados_resposta.get('risk_label')}"
                )
                if snapshot_a_cada_request:
                    _atualizar_snapshot_local()
            else:
                erros += 1
                print(f"#{contador} | [ERROR] {resposta.status_code} | {resposta.text}")

        except requests.exceptions.ConnectionError:
            erros += 1
            print("[WARN] API Offline...")
            time.sleep(2)
        except Exception as erro:
            erros += 1
            print(f"[WARN] Erro no script: {erro}")

        if max_requests is not None and sucesso >= max_requests:
            break

        time.sleep(delay)

    print(
        f"Resumo simulacao | sucesso={sucesso} | erros={erros} | ignorados={ignorados} | "
        f"max_requests={max_requests if max_requests is not None else 'infinito'}"
    )

    if atualizar_snapshot:
        _atualizar_snapshot_local()


def _parse_args():
    """
    Parse de argumentos CLI para facilitar testes de snapshot de monitoramento.
    """
    parser = argparse.ArgumentParser(
        description="Simula trafego de producao para alimentar monitoramento."
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Numero de respostas 200 para encerrar (padrao: infinito). Ex: 120",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DELAY,
        help="Delay entre requests em segundos (padrao: 0.05)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=URL_API,
        help="URL completa do endpoint de predicao smart.",
    )
    parser.add_argument(
        "--atualizar-snapshot",
        action="store_true",
        help="Atualiza snapshot de monitoramento ao final da simulacao.",
    )
    parser.set_defaults(snapshot_a_cada_request=True)
    parser.add_argument(
        "--snapshot-a-cada-request",
        dest="snapshot_a_cada_request",
        action="store_true",
        help="Atualiza snapshot a cada resposta 200 da API.",
    )
    parser.add_argument(
        "--sem-snapshot-a-cada-request",
        dest="snapshot_a_cada_request",
        action="store_false",
        help="Desativa atualizacao de snapshot a cada resposta 200 da API.",
    )
    return parser.parse_args()


def main() -> int:
    global URL_API
    args = _parse_args()
    URL_API = str(args.api_url).strip()
    print(f"[INFO] Endpoint alvo: {URL_API}")
    simular_trafego_producao(
        max_requests=args.max_requests,
        delay=args.delay,
        atualizar_snapshot=args.atualizar_snapshot,
        snapshot_a_cada_request=args.snapshot_a_cada_request,
    )
    return 0


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEncerrado.")

