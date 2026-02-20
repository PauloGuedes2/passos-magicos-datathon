"""Serviço de monitoramento de drift.

Responsabilidades:
- Ler dados de referência e produção
- Consolidar snapshots de monitoramento
- Publicar métricas no New Relic
"""

import json
import os
from collections import deque
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd

from src.config.settings import Configuracoes
from src.infrastructure.monitoring.newrelic_model_metrics import publish_model_metrics
from src.util.logger import logger


class ServicoMonitoramento:
    """Serviço para monitoramento de data drift operacional.

    Responsabilidades:
    - Validar existência de arquivos necessários
    - Preparar dados atuais e de referência
    - Consolidar e publicar snapshots de observabilidade
    """

    @staticmethod
    def atualizar_snapshot_monitoramento() -> dict:
        """Atualiza snapshot de monitoramento (drift + métricas customizadas)."""
        if not os.path.exists(Configuracoes.REFERENCE_PATH):
            return {"updated": False, "reason": "reference_not_found"}

        if not os.path.exists(Configuracoes.LOG_PATH):
            return {"updated": False, "reason": "logs_not_found"}

        try:
            referencia = pd.read_csv(Configuracoes.REFERENCE_PATH)
            dados_atual_raw = ServicoMonitoramento._carregar_logs()
            if dados_atual_raw.empty:
                return {"updated": False, "reason": "empty_or_invalid_logs"}

            dados_atual = ServicoMonitoramento._montar_dados_atual(dados_atual_raw)
            referencia, dados_atual = ServicoMonitoramento._filtrar_predicoes_validas(referencia, dados_atual)
            if len(dados_atual) < 5:
                return {"updated": False, "reason": "insufficient_samples"}

            psi_df = ServicoMonitoramento._calcular_psi_features(referencia, dados_atual)
            drift_alertas = ServicoMonitoramento._gerar_alertas_drift(psi_df)
            monitoramento_estrategico = ServicoMonitoramento._calcular_monitoramento_estrategico(
                referencia, dados_atual
            )
            missing_ratio = ServicoMonitoramento._calcular_missing_ratio(dados_atual)
            ServicoMonitoramento._persistir_relatorio_drift(
                psi_df=psi_df,
                alertas_psi=drift_alertas,
                monitoramento_estrategico=monitoramento_estrategico,
                missing_ratio=missing_ratio,
            )
            return {"updated": True, "samples": int(len(dados_atual))}
        except Exception as erro:
            logger.warning(f"Falha ao atualizar snapshot de monitoramento: {erro}")
            return {"updated": False, "reason": "internal_error"}

    @staticmethod
    def _carregar_logs():
        """Carrega os logs de produção em JSONL.

        Retorno:
        - pd.DataFrame: DataFrame com logs (vazio em caso de erro)
        """
        try:
            linhas = ServicoMonitoramento._ler_ultimas_linhas(
                Configuracoes.LOG_PATH, Configuracoes.LOG_SAMPLE_LIMIT
            )
            if not linhas:
                return pd.DataFrame()
            buffer = StringIO("".join(linhas))
            return pd.read_json(buffer, lines=True)
        except ValueError:
            return pd.DataFrame()
        except FileNotFoundError:
            return pd.DataFrame()

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
    def _calcular_missing_ratio(df: pd.DataFrame) -> float:
        """Calcula proporcao de valores ausentes no snapshot atual."""
        try:
            if df.empty:
                return 0.0
            total_celulas = float(df.shape[0] * df.shape[1])
            if total_celulas <= 0:
                return 0.0
            total_missing = float(df.isna().sum().sum())
            return round(total_missing / total_celulas, 6)
        except Exception:
            return 0.0

    @staticmethod
    def _obter_model_version() -> str:
        """Obtém versao do modelo via metricas de treino ou variavel de ambiente."""
        try:
            if os.path.exists(Configuracoes.METRICS_FILE):
                with open(Configuracoes.METRICS_FILE, "r", encoding="utf-8") as arquivo:
                    metricas = json.load(arquivo)
                versao = metricas.get("model_version")
                if versao:
                    return str(versao)
        except Exception as erro:
            logger.warning(f"Falha ao obter versao do modelo para observabilidade: {erro}")
        return str(Configuracoes.MODEL_VERSION)

    @staticmethod
    def _persistir_relatorio_drift(
        psi_df: pd.DataFrame,
        alertas_psi: list[str],
        monitoramento_estrategico: dict,
        missing_ratio: float = 0.0,
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

            number_of_drifted_features = 0
            if not psi_df.empty and "drift_flag" in psi_df.columns:
                number_of_drifted_features = int(psi_df["drift_flag"].fillna(False).astype(bool).sum())

            summary = {
                "drift_score": avg_psi,
                "number_of_drifted_features": number_of_drifted_features,
                "risk_rate": high_risk_rate,
                "missing_ratio": round(float(missing_ratio), 6),
                "drift_status": drift_status,
                "high_risk_change_pct": float(
                    monitoramento_estrategico.get("high_risk_change_pct", 0.0) or 0.0
                ),
                "significant_shift_alert": bool(
                    monitoramento_estrategico.get("significant_shift_alert", False)
                ),
                "top_drift_feature_1": top_drift_features[0] if len(top_drift_features) > 0 else "",
                "top_drift_feature_2": top_drift_features[1] if len(top_drift_features) > 1 else "",
                "top_drift_feature_3": top_drift_features[2] if len(top_drift_features) > 2 else "",
                "model_version": ServicoMonitoramento._obter_model_version(),
                "window_timestamp": timestamp,
                "environment": Configuracoes.APP_ENV,
                "service_name": Configuracoes.SERVICE_NAME,
            }
            publish_model_metrics(summary, psi_records=psi_records)
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
