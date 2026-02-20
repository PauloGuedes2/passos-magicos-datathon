"""Serviço de predição de risco.

Responsabilidades:
- Preparar dados de entrada
- Executar predições com o modelo
- Registrar logs de predição
"""

import json
import os
import time
import pandas as pd

from src.application.feature_processor import ProcessadorFeatures
from src.application.monitoring_service import ServicoMonitoramento
from src.config.settings import Configuracoes
from src.domain.student import EntradaEstudante, Estudante
from src.infrastructure.data.historical_repository import RepositorioHistorico
from src.infrastructure.logging.prediction_logger import LoggerPredicao
from src.util.logger import logger


class ServicoRisco:
    """Serviço para predição de risco de defasagem acadêmica.

    Responsabilidades:
    - Converter entradas em DataFrame
    - Aplicar processamento de features
    - Calcular probabilidade e classe de risco
    - Persistir logs de predição
    """

    def __init__(self, modelo):
        """Inicializa o serviço com o modelo.

        Parâmetros:
        - modelo (Any): modelo de ML carregado
        """
        self.modelo = modelo
        self.processador = ProcessadorFeatures()
        self.logger = LoggerPredicao()
        self.repositorio = RepositorioHistorico()

    _ultimo_snapshot_monitoramento = 0.0

    def prever_risco(self, dados_estudante: dict) -> dict:
        """Realiza a predição de risco.

        Parâmetros:
        - dados_estudante (dict): dados completos do aluno

        Retorno:
        - dict: resultado da predição

        Exceções:
        - RuntimeError: quando o modelo não está inicializado
        - Exception: quando ocorre erro na inferência
        """
        if not self.modelo:
            raise RuntimeError("Serviço indisponível: Modelo não inicializado.")

        try:
            dados_brutos = pd.DataFrame([dados_estudante])
            estatisticas = self._carregar_estatisticas()
            dados_features = self.processador.processar(dados_brutos, estatisticas=estatisticas)

            features_modelo = [
                c
                for c in Configuracoes.FEATURES_MODELO_NUMERICAS
                + Configuracoes.FEATURES_MODELO_CATEGORICAS
                if c in dados_features.columns
            ]
            dados_modelo = dados_features[features_modelo]

            prob_risco = self.modelo.predict_proba(dados_modelo)[:, 1][0]
            threshold = self._obter_threshold()
            classe_predicao = int(prob_risco >= threshold)
            rotulo_risco = "ALTO RISCO" if classe_predicao == 1 else "BAIXO RISCO"
            segmento_risco = self._segmentar_risco(prob_risco, threshold)
            requires_review = False
            if Configuracoes.REVIEW_ENABLED and Configuracoes.REVIEW_MARGIN > 0:
                genero = dados_features.get("GENERO")
                if isinstance(genero, dict):
                    genero = genero.get("GENERO")
                if isinstance(dados_features, pd.DataFrame):
                    try:
                        genero = str(dados_features.loc[0, "GENERO"])
                    except Exception:
                        genero = None
                if isinstance(genero, str):
                    genero = genero.strip()
                margem = Configuracoes.REVIEW_MARGIN_BY_GROUP.get(genero, Configuracoes.REVIEW_MARGIN)
                limite_inferior = max(0.0, threshold - margem)
                limite_superior = min(1.0, threshold + margem)
                requires_review = limite_inferior <= prob_risco <= limite_superior

            resultado = {
                "risk_probability": round(float(prob_risco), 4),
                "risk_label": rotulo_risco,
                "prediction": classe_predicao,
                "requires_human_review": bool(requires_review),
                "risk_segment": segmento_risco,
            }

            features = dados_features.to_dict(orient="records")[0]
            self.logger.registrar_predicao(features=features, dados_predicao=resultado)
            self._atualizar_snapshot_monitoramento()

            return resultado

        except Exception as erro:
            logger.error(f"Erro na inferência: {erro}")
            raise erro

    @staticmethod
    def _atualizar_snapshot_monitoramento() -> None:
        """Atualiza snapshot de monitoramento e publica eventos de observabilidade."""
        try:
            intervalo = max(0, int(Configuracoes.MONITORING_SNAPSHOT_MIN_INTERVAL_SECONDS))
            agora = time.time()
            if intervalo > 0 and (agora - ServicoRisco._ultimo_snapshot_monitoramento) < intervalo:
                return

            snapshot = ServicoMonitoramento.atualizar_snapshot_monitoramento()
            if snapshot.get("updated"):
                ServicoRisco._ultimo_snapshot_monitoramento = agora
        except Exception as erro:
            logger.warning(f"Falha ao atualizar snapshot de monitoramento: {erro}")

    @staticmethod
    def _obter_threshold() -> float:
        """Obtém o threshold configurado ou salvo em métricas.

        Retorno:
        - float: threshold de risco
        """
        try:
            if Configuracoes.METRICS_FILE and os.path.exists(Configuracoes.METRICS_FILE):
                with open(Configuracoes.METRICS_FILE, "r") as arquivo:
                    metricas = json.load(arquivo)
                return float(metricas.get("risk_threshold", Configuracoes.RISK_THRESHOLD))
        except Exception as erro:
            logger.warning(f"Falha ao carregar threshold salvo: {erro}")
        return Configuracoes.RISK_THRESHOLD

    @staticmethod
    def _carregar_estatisticas() -> dict:
        """Carrega estatísticas de treino salvas.

        Retorno:
        - dict: estatísticas ou vazio se indisponível
        """
        try:
            if os.path.exists(Configuracoes.FEATURE_STATS_PATH):
                with open(Configuracoes.FEATURE_STATS_PATH, "r") as arquivo:
                    return json.load(arquivo)
        except Exception as erro:
            logger.warning(f"Falha ao carregar estatísticas de treino: {erro}")
        return {}

    @staticmethod
    def _segmentar_risco(prob_risco: float, threshold: float) -> str:
        """Segmenta risco em baixo/medio/alto sem alterar predicao binaria."""
        limite_baixo = Configuracoes.RISK_LOW_SEGMENT_MAX
        if prob_risco < limite_baixo:
            return "BAIXO_RISCO"
        if prob_risco < threshold:
            return "MEDIO_RISCO"
        return "ALTO_RISCO"

    @staticmethod
    def _gerar_top_risk_drivers(dados: dict) -> list[str]:
        """Gera explicabilidade textual simples por aluno (top 3 drivers)."""
        drivers = []
        ieg_anterior = float(dados.get("IEG_ANTERIOR", 0.0) or 0.0)
        inde_anterior = float(dados.get("INDE_ANTERIOR", 0.0) or 0.0)
        aluno_novo = int(dados.get("ALUNO_NOVO", 0) or 0)
        idade = float(dados.get("IDADE", 0) or 0)
        ano_ref = float(dados.get("ANO_REFERENCIA", 0) or 0)
        ano_ingresso = float(dados.get("ANO_INGRESSO", 0) or 0)
        tempo_ong = max(0.0, ano_ref - ano_ingresso) if ano_ref and ano_ingresso else 0.0

        if ieg_anterior <= 4.0:
            drivers.append("queda no IEG_ANTERIOR")
        if inde_anterior <= 5.0:
            drivers.append("desempenho histórico baixo no INDE_ANTERIOR")
        if aluno_novo == 1 or tempo_ong <= 1:
            drivers.append("baixo tempo de vínculo na ONG")
        if idade >= 15:
            drivers.append("idade acima da média da série")

        if not drivers:
            drivers.append("histórico acadêmico estável sem sinal crítico dominante")
        return drivers[:3]

    def prever_risco_inteligente(self, entrada: EntradaEstudante) -> dict:
        """Predição inteligente que busca histórico automaticamente.

        Parâmetros:
        - entrada (EntradaEstudante): dados básicos do aluno

        Retorno:
        - dict: resultado da predição
        """
        historico = self.repositorio.obter_historico_estudante(
            entrada.RA, entrada.ANO_REFERENCIA
        )
        requer_revisao_humana = False

        if historico:
            logger.info(f"Histórico encontrado para RA: {entrada.RA}")
        else:
            logger.info(f"Aluno novo ou sem histórico (RA: {entrada.RA})")
            requer_revisao_humana = True
            historico = {
                "INDE_ANTERIOR": 0.0,
                "IAA_ANTERIOR": 0.0,
                "IEG_ANTERIOR": 0.0,
                "IPS_ANTERIOR": 0.0,
                "IDA_ANTERIOR": 0.0,
                "IPP_ANTERIOR": 0.0,
                "IPV_ANTERIOR": 0.0,
                "IAN_ANTERIOR": 0.0,
                "ALUNO_NOVO": 1,
            }

        dados_completos = entrada.model_dump()
        dados_completos.update(historico)

        estudante = Estudante(**dados_completos)
        resultado = self.prever_risco(estudante.model_dump())
        resultado["requires_human_review"] = (
            resultado.get("requires_human_review", False) or requer_revisao_humana
        )
        resultado["requires_human_review"] = bool(resultado["requires_human_review"])
        resultado["top_risk_drivers"] = self._gerar_top_risk_drivers(dados_completos)
        return resultado
