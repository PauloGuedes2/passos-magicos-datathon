"""Serviço de monitoramento de drift.

Responsabilidades:
- Ler dados de referência e produção
- Gerar relatório Evidently em HTML
- Tratar cenários de erro e dados insuficientes
"""

import json
import os
from collections import deque
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report

from src.config.settings import Configuracoes
from src.util.logger import logger


class ServicoMonitoramento:
    """Serviço para monitoramento de Data Drift e Target Drift.

    Responsabilidades:
    - Validar existência de arquivos necessários
    - Preparar dados atuais e de referência
    - Executar e retornar relatório Evidently
    """

    @staticmethod
    def gerar_dashboard() -> str:
        """Gera o relatório HTML comparando referência vs produção.

        Retorno:
        - str: HTML do relatório ou mensagens de aviso/erro
        """
        if not os.path.exists(Configuracoes.REFERENCE_PATH):
            return "<h1>Erro: Dataset de Referência não encontrado. Treine o modelo primeiro.</h1>"

        if not os.path.exists(Configuracoes.LOG_PATH):
            return "<h1>Aviso: Nenhum dado de produção ainda. Faça algumas predições na API primeiro.</h1>"

        try:
            referencia = pd.read_csv(Configuracoes.REFERENCE_PATH)
            dados_atual_raw = ServicoMonitoramento._carregar_logs()
            if isinstance(dados_atual_raw, str):
                return dados_atual_raw

            if dados_atual_raw.empty:
                return "<h1>Aviso: Arquivo de logs sem dados.</h1>"

            dados_atual = ServicoMonitoramento._montar_dados_atual(dados_atual_raw)
            referencia, dados_atual = ServicoMonitoramento._filtrar_predicoes_validas(referencia, dados_atual)

            colunas_comuns = list(set(referencia.columns) & set(dados_atual.columns))
            if len(dados_atual) < 5:
                return "<h1>Aguardando mais dados... (Mínimo 5 requisições para gerar relatório confiável)</h1>"

            mapeamento_colunas = ServicoMonitoramento._criar_mapeamento(colunas_comuns, dados_atual)
            relatorio = ServicoMonitoramento._executar_relatorio(
                referencia, dados_atual, mapeamento_colunas
            )
            fairness_html = ServicoMonitoramento._gerar_fairness_html(referencia, dados_atual)
            psi_df = ServicoMonitoramento._calcular_psi_features(referencia, dados_atual)
            drift_alertas = ServicoMonitoramento._gerar_alertas_drift(psi_df)
            psi_html = ServicoMonitoramento._gerar_psi_html(psi_df, drift_alertas)
            monitoramento_estrategico = ServicoMonitoramento._calcular_monitoramento_estrategico(
                referencia, dados_atual
            )
            estrategico_html = ServicoMonitoramento._gerar_monitoramento_estrategico_html(
                monitoramento_estrategico
            )
            ServicoMonitoramento._persistir_relatorio_drift(
                psi_df=psi_df,
                alertas_psi=drift_alertas,
                monitoramento_estrategico=monitoramento_estrategico,
            )
            return f"{relatorio.get_html()}{fairness_html}{psi_html}{estrategico_html}"

        except Exception as erro:
            logger.error(f"Erro ao gerar dashboard: {erro}")
            return f"<h1>Erro interno ao gerar relatório: {str(erro)}</h1>"

    @staticmethod
    def _carregar_logs():
        """Carrega os logs de produção em JSONL.

        Retorno:
        - pd.DataFrame | str: DataFrame com logs ou mensagem HTML de aviso
        """
        try:
            linhas = ServicoMonitoramento._ler_ultimas_linhas(
                Configuracoes.LOG_PATH, Configuracoes.LOG_SAMPLE_LIMIT
            )
            if not linhas:
                return "<h1>Aviso: Arquivo de logs vazio ou inválido.</h1>"
            buffer = StringIO("".join(linhas))
            return pd.read_json(buffer, lines=True)
        except ValueError:
            return "<h1>Aviso: Arquivo de logs vazio ou inválido.</h1>"
        except FileNotFoundError:
            return "<h1>Aviso: Nenhum dado de produção ainda. Faça algumas predições na API primeiro.</h1>"

    @staticmethod
    def _ler_ultimas_linhas(caminho: str, limite: int):
        """Lê as últimas linhas de um arquivo de log.

        Parâmetros:
        - caminho (str): caminho do arquivo
        - limite (int): quantidade máxima de linhas

        Retorno:
        - list[str]: linhas lidas
        """
        if limite <= 0:
            return []
        with open(caminho, "r", encoding="utf-8") as arquivo:
            return list(deque(arquivo, maxlen=limite))

    @staticmethod
    def _montar_dados_atual(dados_raw: pd.DataFrame) -> pd.DataFrame:
        """Monta o DataFrame atual combinando features e predições.

        Parâmetros:
        - dados_raw (pd.DataFrame): logs brutos

        Retorno:
        - pd.DataFrame: dados atuais preparados
        """
        features_df = pd.json_normalize(dados_raw["input_features"])
        preds_df = pd.json_normalize(dados_raw["prediction_result"])
        dados_atual = pd.concat([features_df, preds_df], axis=1)

        if "class" in dados_atual.columns:
            dados_atual.rename(columns={"class": "prediction"}, inplace=True)

        return dados_atual

    @staticmethod
    def _calcular_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Calcula PSI (Population Stability Index) para uma feature numerica."""
        ref = pd.to_numeric(reference, errors="coerce").dropna()
        cur = pd.to_numeric(current, errors="coerce").dropna()
        if ref.empty or cur.empty:
            return 0.0

        quantis = np.linspace(0, 1, bins + 1)
        cortes = ref.quantile(quantis).values
        cortes = np.unique(cortes)
        if len(cortes) < 3:
            return 0.0

        ref_bins = pd.cut(ref, bins=cortes, include_lowest=True)
        cur_bins = pd.cut(cur, bins=cortes, include_lowest=True)

        ref_dist = ref_bins.value_counts(normalize=True, sort=False)
        cur_dist = cur_bins.value_counts(normalize=True, sort=False)

        epsilon = 1e-6
        psi = 0.0
        for intervalo in ref_dist.index:
            r = float(ref_dist.get(intervalo, 0.0)) + epsilon
            c = float(cur_dist.get(intervalo, 0.0)) + epsilon
            psi += (c - r) * np.log(c / r)
        return float(psi)

    @staticmethod
    def _calcular_psi_detalhado(
        reference: pd.Series,
        current: pd.Series,
        bins: int = 10,
    ) -> dict:
        """Calcula PSI e distribuicoes normalizadas de referencia/atual."""
        ref = pd.to_numeric(reference, errors="coerce").dropna()
        cur = pd.to_numeric(current, errors="coerce").dropna()
        if ref.empty or cur.empty:
            return {
                "psi": 0.0,
                "reference_distribution": [],
                "current_distribution": [],
                "bins": [],
            }

        quantis = np.linspace(0, 1, bins + 1)
        cortes = ref.quantile(quantis).values
        cortes = np.unique(cortes)
        if len(cortes) < 3:
            return {
                "psi": 0.0,
                "reference_distribution": [],
                "current_distribution": [],
                "bins": [round(float(v), 6) for v in cortes.tolist()],
            }

        ref_bins = pd.cut(ref, bins=cortes, include_lowest=True)
        cur_bins = pd.cut(cur, bins=cortes, include_lowest=True)
        ref_dist = ref_bins.value_counts(normalize=True, sort=False)
        cur_dist = cur_bins.value_counts(normalize=True, sort=False)

        epsilon = 1e-6
        psi = 0.0
        ref_values = []
        cur_values = []
        for intervalo in ref_dist.index:
            r = float(ref_dist.get(intervalo, 0.0))
            c = float(cur_dist.get(intervalo, 0.0))
            ref_values.append(round(r, 6))
            cur_values.append(round(c, 6))
            psi += ((c + epsilon) - (r + epsilon)) * np.log((c + epsilon) / (r + epsilon))

        return {
            "psi": round(float(psi), 6),
            "reference_distribution": ref_values,
            "current_distribution": cur_values,
            "bins": [round(float(v), 6) for v in cortes.tolist()],
        }

    @staticmethod
    def _calcular_psi_features(referencia: pd.DataFrame, atual: pd.DataFrame) -> pd.DataFrame:
        """Calcula PSI para features numericas prioritarias."""
        features = [
            f for f in Configuracoes.FEATURES_NUMERICAS
            if f in referencia.columns and f in atual.columns
        ][: Configuracoes.PSI_TOP_FEATURES]

        linhas = []
        for feature in features:
            psi_detalhado = ServicoMonitoramento._calcular_psi_detalhado(
                referencia[feature], atual[feature]
            )
            psi_valor = float(psi_detalhado["psi"])
            linhas.append(
                {
                    "feature": feature,
                    "psi": round(float(psi_valor), 4),
                    "reference_mean": round(float(pd.to_numeric(referencia[feature], errors="coerce").mean()), 4),
                    "current_mean": round(float(pd.to_numeric(atual[feature], errors="coerce").mean()), 4),
                    "delta_mean": round(
                        float(
                            pd.to_numeric(atual[feature], errors="coerce").mean()
                            - pd.to_numeric(referencia[feature], errors="coerce").mean()
                        ),
                        4,
                    ),
                    "direction": (
                        "increase"
                        if pd.to_numeric(atual[feature], errors="coerce").mean()
                        > pd.to_numeric(referencia[feature], errors="coerce").mean()
                        else "decrease_or_stable"
                    ),
                    "drift_flag": bool(psi_valor >= Configuracoes.PSI_THRESHOLD),
                    "reference_distribution": psi_detalhado.get("reference_distribution", []),
                    "current_distribution": psi_detalhado.get("current_distribution", []),
                    "bins": psi_detalhado.get("bins", []),
                }
            )
        return pd.DataFrame(linhas)

    @staticmethod
    def _gerar_alertas_drift(psi_df: pd.DataFrame) -> list[str]:
        """Gera alertas textuais para features acima do threshold de drift."""
        if psi_df.empty:
            return []
        alertas = []
        for _, row in psi_df.iterrows():
            if float(row["psi"]) >= Configuracoes.DRIFT_WARNING_THRESHOLD:
                alertas.append(
                    f"Feature {row['feature']} com PSI={float(row['psi']):.4f} acima do limite "
                    f"(direcao: {row.get('direction', 'n/a')})."
                )
        return alertas

    @staticmethod
    def _gerar_psi_html(psi_df: pd.DataFrame, alertas: list[str]) -> str:
        """Renderiza bloco de PSI e alertas de drift."""
        if psi_df.empty:
            return "<section><h2>PSI</h2><p>Sem features numericas em comum para calcular PSI.</p></section>"
        tabela = psi_df.to_html(index=False)
        if alertas:
            itens = "".join([f"<li>{a}</li>" for a in alertas])
            alertas_html = f"<ul>{itens}</ul>"
        else:
            alertas_html = "<p>Sem alertas de drift por PSI.</p>"
        return (
            "<section>"
            "<h2>PSI - Population Stability Index</h2>"
            f"{tabela}"
            "<h3>Alertas de Drift</h3>"
            f"{alertas_html}"
            "</section>"
        )

    @staticmethod
    def _calcular_monitoramento_estrategico(
        referencia: pd.DataFrame,
        atual: pd.DataFrame,
    ) -> dict:
        """Calcula indicador de taxa de ALTO_RISCO e alerta de mudanca."""
        taxa_referencia = ServicoMonitoramento._calcular_taxa_alto_risco(referencia)
        taxa_atual = ServicoMonitoramento._calcular_taxa_alto_risco(atual)
        delta_pp = round(taxa_atual - taxa_referencia, 2)
        variacao_pct = 0.0
        if taxa_referencia > 0:
            variacao_pct = round(((taxa_atual - taxa_referencia) / taxa_referencia) * 100.0, 2)
        else:
            variacao_pct = 100.0 if taxa_atual > 0 else 0.0

        limite = Configuracoes.HIGH_RISK_SHIFT_THRESHOLD_PCT
        alerta = abs(variacao_pct) >= limite
        mensagem = (
            "Mudanca significativa na taxa de ALTO_RISCO detectada; "
            "revisar capacidade de atendimento e possivel recalibracao."
            if alerta
            else "Taxa de ALTO_RISCO dentro da variacao esperada."
        )
        return {
            "reference_high_risk_rate_pct": taxa_referencia,
            "current_high_risk_rate_pct": taxa_atual,
            "delta_high_risk_pp": delta_pp,
            "high_risk_change_pct": variacao_pct,
            "significant_shift_alert": bool(alerta),
            "alert_message": mensagem,
            "threshold_pct": limite,
        }

    @staticmethod
    def _calcular_taxa_alto_risco(df: pd.DataFrame) -> float:
        """Calcula percentual de alunos classificados como ALTO_RISCO."""
        if "prediction" in df.columns:
            serie = pd.to_numeric(df["prediction"], errors="coerce").fillna(0)
            return round(float((serie == 1).mean() * 100.0), 2)
        if "risk_segment" in df.columns:
            serie = df["risk_segment"].astype(str)
            return round(float((serie == "ALTO_RISCO").mean() * 100.0), 2)
        return 0.0

    @staticmethod
    def _gerar_monitoramento_estrategico_html(monitoramento: dict) -> str:
        """Renderiza bloco HTML do monitoramento estrategico."""
        alerta = monitoramento.get("significant_shift_alert", False)
        badge = "ALERTA" if alerta else "ESTAVEL"
        cor = "#c53030" if alerta else "#2f855a"
        return (
            "<section>"
            "<h2>Monitoramento Estratégico</h2>"
            f"<p><strong>Taxa ALTO_RISCO Referência:</strong> {monitoramento.get('reference_high_risk_rate_pct', 0):.2f}%</p>"
            f"<p><strong>Taxa ALTO_RISCO Produção:</strong> {monitoramento.get('current_high_risk_rate_pct', 0):.2f}%</p>"
            f"<p><strong>Variação:</strong> {monitoramento.get('high_risk_change_pct', 0):.2f}% "
            f"(Δ {monitoramento.get('delta_high_risk_pp', 0):.2f} pp)</p>"
            f"<p><strong>Status:</strong> <span style='color:{cor}'>{badge}</span></p>"
            f"<p>{monitoramento.get('alert_message', '')}</p>"
            "</section>"
        )

    @staticmethod
    def _persistir_relatorio_drift(
        psi_df: pd.DataFrame,
        alertas_psi: list[str],
        monitoramento_estrategico: dict,
    ) -> None:
        """Persistir relatorio consolidado de drift com PSI e indicadores estrategicos."""
        try:
            os.makedirs(os.path.dirname(Configuracoes.DRIFT_REPORT_FILE), exist_ok=True)
            historico_anterior = {}
            if os.path.exists(Configuracoes.DRIFT_REPORT_FILE):
                try:
                    with open(Configuracoes.DRIFT_REPORT_FILE, "r", encoding="utf-8") as arquivo:
                        historico_anterior = json.load(arquivo)
                except Exception:
                    historico_anterior = {}

            psi_records = psi_df.to_dict(orient="records")
            psi_by_feature = {}
            for row in psi_records:
                feature = str(row.get("feature", ""))
                if not feature:
                    continue
                psi_by_feature[feature] = {
                    "psi": round(float(row.get("psi", 0.0) or 0.0), 4),
                    "reference_distribution": row.get("reference_distribution", []),
                    "current_distribution": row.get("current_distribution", []),
                    "bins": row.get("bins", []),
                }

            ordered = sorted(
                psi_by_feature.items(),
                key=lambda item: float(item[1].get("psi", 0.0)),
                reverse=True,
            )
            top_drift_features = [feature for feature, _ in ordered[:5]]
            avg_psi = round(float(psi_df["psi"].mean()), 4) if not psi_df.empty else 0.0
            high_risk_rate = round(
                float(monitoramento_estrategico.get("current_high_risk_rate_pct", 0.0) or 0.0),
                2,
            )
            drift_status = "Estavel"
            if avg_psi >= max(Configuracoes.PSI_THRESHOLD, 0.2):
                drift_status = "Moderado"
            if avg_psi >= 0.35 or bool(monitoramento_estrategico.get("significant_shift_alert", False)):
                drift_status = "Critico"

            drift_history = historico_anterior.get("drift_history", [])
            if not isinstance(drift_history, list):
                drift_history = []
            high_risk_rate_history = historico_anterior.get("high_risk_rate_history", [])
            if not isinstance(high_risk_rate_history, list):
                high_risk_rate_history = []

            timestamp = datetime.now().isoformat()
            drift_history.append(
                {
                    "timestamp": timestamp,
                    "high_risk_rate": high_risk_rate,
                    "avg_psi": avg_psi,
                }
            )
            high_risk_rate_history.append(
                {
                    "timestamp": timestamp,
                    "rate": high_risk_rate,
                }
            )

            relatorio = {
                "timestamp": timestamp,
                "psi_metrics": psi_records,
                "psi_alerts": alertas_psi,
                "strategic_monitoring": monitoramento_estrategico,
                "psi_by_feature": psi_by_feature,
                "top_drift_features": top_drift_features,
                "drift_status": drift_status,
                "drift_history": drift_history,
                "high_risk_rate_history": high_risk_rate_history,
            }
            with open(Configuracoes.DRIFT_REPORT_FILE, "w", encoding="utf-8") as arquivo:
                json.dump(relatorio, arquivo, indent=2, ensure_ascii=False)
        except Exception as erro:
            logger.warning(f"Falha ao persistir relatorio de drift consolidado: {erro}")

    @staticmethod
    def _filtrar_predicoes_validas(referencia: pd.DataFrame, atual: pd.DataFrame):
        """Remove linhas sem coluna de predição.

        Parâmetros:
        - referencia (pd.DataFrame): dados de referência
        - atual (pd.DataFrame): dados atuais

        Retorno:
        - tuple[pd.DataFrame, pd.DataFrame]: referência e atual filtrados
        """
        referencia_filtrada = referencia.dropna(subset=["prediction"])
        atual_filtrado = atual.dropna(subset=["prediction"])
        return referencia_filtrada, atual_filtrado

    @staticmethod
    def _criar_mapeamento(colunas_comuns, dados_atual: pd.DataFrame) -> ColumnMapping:
        """Cria o mapeamento de colunas para o Evidently.

        Parâmetros:
        - colunas_comuns (list): colunas presentes em ambos os conjuntos
        - dados_atual (pd.DataFrame): dados atuais

        Retorno:
        - ColumnMapping: configuração de colunas
        """
        mapeamento = ColumnMapping()
        mapeamento.numerical_features = [c for c in Configuracoes.FEATURES_NUMERICAS if c in colunas_comuns]
        mapeamento.categorical_features = [c for c in Configuracoes.FEATURES_CATEGORICAS if c in colunas_comuns]

        if "prediction" in dados_atual.columns:
            mapeamento.prediction = "prediction"

        return mapeamento

    @staticmethod
    def _executar_relatorio(referencia: pd.DataFrame, atual: pd.DataFrame, mapeamento: ColumnMapping) -> Report:
        """Executa o relatório de drift.

        Parâmetros:
        - referencia (pd.DataFrame): dados de referência
        - atual (pd.DataFrame): dados atuais
        - mapeamento (ColumnMapping): configuração de colunas

        Retorno:
        - Report: relatório Evidently executado
        """
        metricas = [DataDriftPreset()]
        if ServicoMonitoramento._tem_target_valido(referencia, atual):
            metricas.append(TargetDriftPreset())
        relatorio = Report(metrics=metricas)
        relatorio.run(
            reference_data=referencia,
            current_data=atual,
            column_mapping=mapeamento,
        )
        return relatorio

    @staticmethod
    def _gerar_fairness_html(referencia: pd.DataFrame, atual: pd.DataFrame) -> str:
        """Gera um bloco HTML com métricas de fairness por grupo.

        Parâmetros:
        - referencia (pd.DataFrame): dados de referência com target/prediction
        - atual (pd.DataFrame): dados atuais

        Retorno:
        - str: HTML com métricas de fairness ou mensagem de aviso
        """
        grupos = []

        def adicionar_metricas(dataset_nome: str, dados: pd.DataFrame):
            metricas = ServicoMonitoramento._calcular_metricas_fairness(dados)
            if isinstance(metricas, str):
                return f"<div class='fairness-section'><h3>{dataset_nome}</h3><p>{metricas}</p></div>"
            tabela_html = metricas.to_html(index=False, classes="fairness-table")
            resumo = ServicoMonitoramento._resumir_gaps_fairness(metricas)
            barras = ServicoMonitoramento._renderizar_barras_fairness(metricas)
            return f"<div class='fairness-section'><h3>{dataset_nome}</h3>{resumo}{barras}{tabela_html}</div>"

        grupos.append(adicionar_metricas("Referência (Treino/Teste)", referencia))
        if Configuracoes.TARGET_COL in atual.columns:
            grupos.append(adicionar_metricas("Produção (com target)", atual))

        conteudo = "".join(grupos)
        estilos = (
            "<style>"
            ".fairness-wrapper{"
            "--fair-bg:var(--euiColorEmptyShade,#ffffff);"
            "--fair-bg-soft:var(--euiColorLightestShade,#f7f7f9);"
            "--fair-card:var(--euiColorLightestShade,#f9fafc);"
            "--fair-border:var(--euiColorLightShade,#e6e6ef);"
            "--fair-text:var(--euiTextColor,#1d1f2a);"
            "--fair-muted:var(--euiTextSubduedColor,#5b6070);"
            "--fair-chip:var(--euiColorPrimaryTint,#eef2ff);"
            "--fair-chip-border:var(--euiColorPrimaryLightShade,#d8e1ff);"
            "--fair-table-head:var(--euiColorLightestShade,#f3f4f6);"
            "--fair-hover:var(--euiColorLightestShade,#fafafa);"
            "--fair-shadow:rgba(20,20,40,0.08);"
            "--fair-success:var(--euiColorSuccess,#2f855a);"
            "--fair-warning:var(--euiColorWarning,#b7791f);"
            "--fair-danger:var(--euiColorDanger,#c53030);"
            "--fair-primary:var(--euiColorPrimary,#3b82f6);"
            "}"
            ".fairness-wrapper{font-family:Arial,Helvetica,sans-serif;margin:24px 0;padding:20px;"
            "background:linear-gradient(135deg,var(--fair-bg-soft) 0%,var(--fair-bg) 70%);"
            "border:1px solid var(--fair-border);border-radius:14px;box-shadow:0 8px 20px var(--fair-shadow)}"
            ".fairness-header{display:flex;align-items:center;justify-content:space-between;gap:16px}"
            ".fairness-title{font-size:22px;margin:0;color:var(--fair-text)}"
            ".fairness-subtitle{margin:6px 0 0;color:var(--fair-muted);font-size:13px}"
            ".fairness-chip{font-size:12px;font-weight:600;color:var(--fair-text);background:var(--fair-chip);"
            "border:1px solid var(--fair-chip-border);padding:6px 10px;border-radius:999px}"
            ".fairness-section{margin-top:18px;padding:14px 16px;background:var(--fair-bg);"
            "border:1px solid var(--fair-border);border-radius:12px}"
            ".fairness-section h3{margin:0 0 8px;font-size:16px;color:var(--fair-text)}"
            ".fairness-summary{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0 4px}"
            ".fairness-card{flex:1 1 180px;background:var(--fair-card);border:1px solid var(--fair-border);"
            "border-radius:10px;padding:10px 12px}"
            ".fairness-card .label{font-size:12px;color:var(--fair-muted);margin-bottom:4px}"
            ".fairness-card .value{font-size:18px;font-weight:700;color:var(--fair-text)}"
            ".fairness-badges{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 2px}"
            ".fairness-badge{font-size:11px;font-weight:700;letter-spacing:0.3px;padding:4px 8px;"
            "border-radius:999px;border:1px solid var(--fair-border);text-transform:uppercase}"
            ".fairness-badge.low{background:color-mix(in srgb,var(--fair-success) 20%, transparent);"
            "color:var(--fair-success);border-color:color-mix(in srgb,var(--fair-success) 35%, transparent)}"
            ".fairness-badge.med{background:color-mix(in srgb,var(--fair-warning) 20%, transparent);"
            "color:var(--fair-warning);border-color:color-mix(in srgb,var(--fair-warning) 35%, transparent)}"
            ".fairness-badge.high{background:color-mix(in srgb,var(--fair-danger) 20%, transparent);"
            "color:var(--fair-danger);border-color:color-mix(in srgb,var(--fair-danger) 35%, transparent)}"
            ".fairness-bars{margin:10px 0 6px;display:flex;flex-direction:column;gap:8px}"
            ".fairness-bar-row{display:grid;grid-template-columns:140px 1fr 60px;gap:10px;align-items:center}"
            ".fairness-bar-label{font-size:12px;color:var(--fair-muted)}"
            ".fairness-bar-track{height:8px;border-radius:999px;background:var(--fair-table-head);overflow:hidden}"
            ".fairness-bar-fill{height:100%;border-radius:999px}"
            ".fairness-bar-fill.fpr{background:var(--fair-warning)}"
            ".fairness-bar-fill.fnr{background:var(--fair-primary)}"
            ".fairness-bar-value{font-size:12px;color:var(--fair-text);text-align:right}"
            ".fairness-table{width:100%;border-collapse:collapse;font-size:13px;margin-top:10px;color:var(--fair-text)}"
            ".fairness-table th,.fairness-table td{padding:8px 10px;border-bottom:1px solid var(--fair-border)}"
            ".fairness-table th{text-align:left;color:var(--fair-text);background:var(--fair-table-head);"
            "position:sticky;top:0}"
            ".fairness-table tr:hover{background:var(--fair-hover)}"
            "</style>"
        )
        return (
            f"{estilos}"
            "<section class='fairness-wrapper'>"
            "<div class='fairness-header'>"
            "<div>"
            "<h2 class='fairness-title'>Fairness por Grupo</h2>"
            "<p class='fairness-subtitle'>Taxas em %, diferencas altas indicam risco de vies.</p>"
            "</div>"
            "<div class='fairness-chip'>Auditoria de Equidade</div>"
            "</div>"
            f"{conteudo}"
            "</section>"
        )

    @staticmethod
    def _calcular_metricas_fairness(dados: pd.DataFrame):
        """Calcula FPR/FNR por grupo para análise de fairness.

        Parâmetros:
        - dados (pd.DataFrame): dados com target e prediction

        Retorno:
        - pd.DataFrame | str: métricas por grupo ou mensagem de aviso
        """
        grupo_coluna = Configuracoes.FAIRNESS_GROUP_COL
        target_col = Configuracoes.TARGET_COL

        colunas_necessarias = {grupo_coluna, target_col, "prediction"}
        if not colunas_necessarias.issubset(dados.columns):
            return "Dados insuficientes para calcular fairness (grupo, target ou prediction ausentes)."

        metricas = []
        for grupo, subset in dados.groupby(grupo_coluna):
            y_true = subset[target_col]
            y_pred = subset["prediction"]

            falso_positivo = int(((y_pred == 1) & (y_true == 0)).sum())
            falso_negativo = int(((y_pred == 0) & (y_true == 1)).sum())
            verdadeiro_positivo = int(((y_pred == 1) & (y_true == 1)).sum())
            verdadeiro_negativo = int(((y_pred == 0) & (y_true == 0)).sum())

            fpr_denom = falso_positivo + verdadeiro_negativo
            fnr_denom = falso_negativo + verdadeiro_positivo

            fpr = round(falso_positivo / fpr_denom, 4) if fpr_denom else 0.0
            fnr = round(falso_negativo / fnr_denom, 4) if fnr_denom else 0.0
            metricas.append(
                {
                    grupo_coluna: grupo,
                    "false_positive_rate_pct": round(fpr * 100, 2),
                    "false_negative_rate_pct": round(fnr * 100, 2),
                    "support": int(len(subset)),
                }
            )

        return pd.DataFrame(metricas)

    @staticmethod
    def _resumir_gaps_fairness(metricas: pd.DataFrame) -> str:
        """Resume os gaps de fairness com base nas taxas por grupo.

        Parâmetros:
        - metricas (pd.DataFrame): métricas por grupo

        Retorno:
        - str: HTML com resumo de gaps
        """
        if metricas.empty:
            return "<p>Sem dados suficientes para resumir gaps de fairness.</p>"

        gap_fpr = metricas["false_positive_rate_pct"].max() - metricas["false_positive_rate_pct"].min()
        gap_fnr = metricas["false_negative_rate_pct"].max() - metricas["false_negative_rate_pct"].min()
        def _nivel_gap(valor: float) -> str:
            if valor <= 5:
                return "low"
            if valor <= 10:
                return "med"
            return "high"

        nivel_fpr = _nivel_gap(gap_fpr)
        nivel_fnr = _nivel_gap(gap_fnr)
        return (
            "<div class='fairness-summary'>"
            "<div class='fairness-card'>"
            "<div class='label'>Gap de FPR</div>"
            f"<div class='value'>{gap_fpr:.2f} pp</div>"
            "<div class='fairness-badges'>"
            f"<span class='fairness-badge {nivel_fpr}'>FPR {nivel_fpr}</span>"
            "</div>"
            "</div>"
            "<div class='fairness-card'>"
            "<div class='label'>Gap de FNR</div>"
            f"<div class='value'>{gap_fnr:.2f} pp</div>"
            "<div class='fairness-badges'>"
            f"<span class='fairness-badge {nivel_fnr}'>FNR {nivel_fnr}</span>"
            "</div>"
            "</div>"
            "</div>"
        )

    @staticmethod
    def _renderizar_barras_fairness(metricas: pd.DataFrame) -> str:
        """Renderiza barras simples de FPR/FNR por grupo.

        Parâmetros:
        - metricas (pd.DataFrame): métricas por grupo

        Retorno:
        - str: HTML das barras
        """
        if metricas.empty:
            return ""

        max_valor = max(
            metricas["false_positive_rate_pct"].max(),
            metricas["false_negative_rate_pct"].max(),
        )
        if not max_valor or pd.isna(max_valor):
            max_valor = 1.0

        linhas = []
        for _, row in metricas.iterrows():
            grupo = row.iloc[0]
            fpr = float(row["false_positive_rate_pct"])
            fnr = float(row["false_negative_rate_pct"])
            fpr_pct = max(0.0, min(100.0, (fpr / max_valor) * 100.0))
            fnr_pct = max(0.0, min(100.0, (fnr / max_valor) * 100.0))
            linhas.append(
                "<div class='fairness-bar-row'>"
                f"<div class='fairness-bar-label'>{grupo} · FPR</div>"
                "<div class='fairness-bar-track'>"
                f"<div class='fairness-bar-fill fpr' style='width:{fpr_pct:.1f}%'></div>"
                "</div>"
                f"<div class='fairness-bar-value'>{fpr:.2f}%</div>"
                "</div>"
            )
            linhas.append(
                "<div class='fairness-bar-row'>"
                f"<div class='fairness-bar-label'>{grupo} · FNR</div>"
                "<div class='fairness-bar-track'>"
                f"<div class='fairness-bar-fill fnr' style='width:{fnr_pct:.1f}%'></div>"
                "</div>"
                f"<div class='fairness-bar-value'>{fnr:.2f}%</div>"
                "</div>"
            )

        return "<div class='fairness-bars'>" + "".join(linhas) + "</div>"

    @staticmethod
    def _tem_target_valido(referencia: pd.DataFrame, atual: pd.DataFrame) -> bool:
        """Verifica se há coluna de target com dados válidos.

        Retorno:
        - bool: True se o target está disponível nos dois conjuntos
        """
        target_col = Configuracoes.TARGET_COL
        if target_col not in referencia.columns or target_col not in atual.columns:
            return False
        return referencia[target_col].notna().any() and atual[target_col].notna().any()

    @staticmethod
    def obter_importancia_global() -> dict:
        """Retorna ranking global de importancia de features salvo no treino."""
        if not os.path.exists(Configuracoes.METRICS_FILE):
            return {"feature_importance_ranking": [], "source": "metrics_not_found"}

        try:
            with open(Configuracoes.METRICS_FILE, "r", encoding="utf-8") as arquivo:
                metricas = json.load(arquivo)
            ranking = metricas.get("feature_importance_ranking")
            if ranking:
                return {"feature_importance_ranking": ranking, "source": "feature_importance_ranking"}

            importancia = metricas.get("feature_importance", {})
            ranking_convertido = [
                {"feature": nome, "importance": valor}
                for nome, valor in list(importancia.items())[:20]
            ]
            return {"feature_importance_ranking": ranking_convertido, "source": "feature_importance"}
        except Exception as erro:
            logger.warning(f"Falha ao carregar importancia global: {erro}")
            return {"feature_importance_ranking": [], "source": "error"}
