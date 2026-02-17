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

# --- 1. ConfiguraAAo de Path ---
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
RAIZ_PROJETO = os.path.dirname(DIRETORIO_ATUAL)
APP_DIR = os.path.join(RAIZ_PROJETO, "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
if RAIZ_PROJETO not in sys.path:
    sys.path.insert(0, RAIZ_PROJETO)

# ruff: noqa: E402
from src.config.settings import Configuracoes

# --- 2. ConfiguraAAes da API ---
PORTA = int(os.getenv("PORT", 8000))
URL_API = f"http://localhost:{PORTA}/api/v1/predict/smart"
DELAY = 0.05  # Acelerado para teste


def limpar_genero(valor):
    """
    Converte Menino/Menina/Garota para o padrao da API.

    ParAmetros:
    - valor (Any): valor original do genero

    Retorno:
    - str: genero normalizado
    """
    if pd.isna(valor):
        return "Outro"
    texto = str(valor).lower().strip()

    if any(item in texto for item in ["fem", "menina", "mulher", "garota"]):
        return "Feminino"
    if any(item in texto for item in ["masc", "menino", "homem", "garoto"]):
        return "Masculino"
    return "Outro"


def limpar_fase(valor):
    """
    Remove espacos e caracteres especiais da FASE (Ex: 'FASE 5' -> 'FASE5').

    ParAmetros:
    - valor (Any): valor original da fase

    Retorno:
    - str: fase sanitizada
    """
    if pd.isna(valor):
        return "0"
    texto = str(valor).upper().strip()
    if re.fullmatch(r"[0-9A-Z]+", texto):
        return texto
    match = re.search(r"\b(ALFA)\b", texto)
    if match:
        return match.group(1)
    match = re.search(r"\b([0-9]{1,2}[A-Z])\b", texto)
    if match:
        return match.group(1)
    match = re.search(r"\b([0-9]{1,2})\b", texto)
    if match:
        return match.group(1)
    limpo = re.sub(r"[^A-Z0-9]", "", texto)
    return limpo if limpo else "0"


def obter_coluna(row, nomes_possiveis):
    """
    Retorna o primeiro valor encontrado em possiveis nomes de coluna.

    ParAmetros:
    - row (pd.Series): linha do DataFrame
    - nomes_possiveis (list[str]): nomes de colunas possAveis

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

    ParAmetros:
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

    ParAmetros:
    - df (pd.DataFrame): dados de origem

    Retorno:
    - generator: stream infinito de linhas
    """
    while True:
        df_embaralhado = df.sample(frac=1).reset_index(drop=True)
        for _, row in df_embaralhado.iterrows():
            yield row


def _normalizar_idade(idade_raw, ano_referencia=None):
    """
    Normaliza a idade a partir de valores brutos.

    ParAmetros:
    - idade_raw (Any): valor original

    Retorno:
    - int: idade normalizada
    """
    if idade_raw:
        try:
            valor = float(idade_raw)
            if 4 <= valor <= 25:
                return int(valor)
            if valor > 1900:
                ano_base = ano_referencia if ano_referencia else 2024
                valor = ano_base - valor
                if 4 <= valor <= 25:
                    return int(valor)
        except Exception:
            pass

    try:
        data_nasc = pd.to_datetime(idade_raw, errors="coerce")
        if pd.notna(data_nasc):
            ano_base = ano_referencia if ano_referencia else 2024
            idade_calc = int(ano_base - data_nasc.year)
            if 4 <= idade_calc <= 25:
                return idade_calc
    except Exception:
        pass
    return None


def _normalizar_ano_ingresso(ano_raw):
    """
    Normaliza o ano de ingresso.

    ParAmetros:
    - ano_raw (Any): valor original

    Retorno:
    - int: ano de ingresso normalizado
    """
    if ano_raw:
        try:
            valor = int(float(ano_raw))
            if 2000 <= valor <= 2026:
                return valor
        except Exception:
            pass
    return None


def _montar_payload(row, chaves):
    """
    Monta o payload de requisicao a partir da linha.

    Parametros:
    - row (pd.Series): linha de dados
    - chaves (dict): dicionario de chaves por campo

    Retorno:
    - dict: payload pronto para envio
    """
    idade_raw = obter_coluna(row, chaves["idade"])
    ano_raw = obter_coluna(row, chaves["ano_ingresso"])
    genero_raw = obter_coluna(row, chaves["genero"])
    fase_raw = obter_coluna(row, chaves["fase"])
    ano_ref_raw = obter_coluna(row, chaves["ano_referencia"])

    ano_ref_final = None
    if ano_ref_raw:
        try:
            ano_ref_final = int(float(ano_ref_raw))
        except Exception:
            ano_ref_final = None

    idade_final = _normalizar_idade(idade_raw, ano_ref_final)
    ano_final = _normalizar_ano_ingresso(ano_raw)
    genero_final = limpar_genero(genero_raw)
    if pd.isna(fase_raw):
        fase_final = "N/A"
    else:
        fase_final = str(fase_raw)

    if idade_final is None:
        return None

    payload = {
        "RA": str(row["RA"]),
        "NOME": str(row.get("NOME", f"Aluno {row['RA']}")),
        "IDADE": idade_final,
        "ANO_INGRESSO": ano_final,
        "GENERO": genero_final,
        "TURMA": str(obter_coluna(row, chaves["turma"]) or "N/A"),
        "INSTITUICAO_ENSINO": str(obter_coluna(row, chaves["instituicao"]) or "N/A"),
        "FASE": fase_final,
        "ANO_REFERENCIA": ano_ref_final,
    }

    if payload.get("ANO_REFERENCIA") is None:
        payload.pop("ANO_REFERENCIA", None)
    if payload.get("IDADE") is None:
        payload.pop("IDADE", None)
    if payload.get("ANO_INGRESSO") is None:
        payload.pop("ANO_INGRESSO", None)

    for chave, valor in payload.items():
        if str(valor).lower() in ["nan", "nat", "none"]:
            payload[chave] = "N/A"

    return payload


def _enviar_payload(payload):
    """
    Envia o payload para a API.

    ParAmetros:
    - payload (dict): dados da requisicao

    Retorno:
    - requests.Response: resposta da API
    """
    inicio = time.time()
    resposta = requests.post(URL_API, json=payload, timeout=(3.05, 10))
    _ = time.time() - inicio
    return resposta


def _gerar_dashboard_local():
    """
    Atualiza drift e gera dashboard profissional a partir dos artefatos persistidos.
    """
    from src.application.monitoring_service import ServicoMonitoramento
    from src.application.professional_dashboard_service import ProfessionalDashboardService

    # Atualiza drift_report.json com base nos logs mais recentes.
    _ = ServicoMonitoramento.gerar_dashboard()
    caminho = ProfessionalDashboardService().generate_dashboard()
    print(f"Dashboard profissional gerado em: {caminho}")


def simular_trafego_producao(
    max_requests: int | None = None,
    delay: float = DELAY,
    gerar_dashboard: bool = False,
    dashboard_a_cada_request: bool = True,
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
        "idade": ["IDADE", "IDADE 2024", "IDADE_ALUNO", "ANO_NASC", "ANO NASC", "DATA DE NASC"],
        "ano_ingresso": ["ANO_INGRESSO", "ANO INGRESSO"],
        "genero": ["GENERO", "SEXO"],
        "turma": ["TURMA", "TURMA 2024"],
        "instituicao": ["INSTITUICAO_ENSINO", "INSTITUICAO DE ENSINO", "ESCOLA", "INSTITUICAO"],
        "fase": ["FASE", "FASE 2024", "FASE_TURMA"],
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
                    f"#{contador} | [OK] {origem} | {payload['RA']} | {payload['GENERO']} | {payload['FASE']} | "
                    f"{dados_resposta.get('risk_label')}"
                )
                if dashboard_a_cada_request:
                    _gerar_dashboard_local()
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

    if gerar_dashboard:
        _gerar_dashboard_local()


def _parse_args():
    """
    Parse de argumentos CLI para facilitar testes do dashboard.
    """
    parser = argparse.ArgumentParser(
        description="Simula trafego de producao para alimentar monitoramento/dashboard."
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
        "--gerar-dashboard",
        action="store_true",
        help="Gera o professional_dashboard.html ao final da simulacao.",
    )
    parser.add_argument(
        "--dashboard-a-cada-request",
        action="store_true",
        default=True,
        help="Atualiza drift e dashboard a cada resposta 200 da API (sempre ativo por padrao).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    simular_trafego_producao(
        max_requests=args.max_requests,
        delay=args.delay,
        gerar_dashboard=args.gerar_dashboard,
        dashboard_a_cada_request=args.dashboard_a_cada_request,
    )
    return 0


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEncerrado.")
