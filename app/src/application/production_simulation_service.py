"""Servico de simulacao de cenario de producao em batch.

Responsabilidades:
- Processar lote sintetico de alunos
- Executar predicoes em batch
- Gerar relatorio estruturado para planejamento operacional
"""

import json
import os
from datetime import datetime
from typing import Any

import pandas as pd

from src.application.feature_processor import ProcessadorFeatures
from src.application.risk_service import ServicoRisco
from src.config.settings import Configuracoes


class ServicoSimulacaoProducao:
    """Simula comportamento de producao sem alterar pipeline principal."""

    def __init__(self, modelo: Any):
        """Inicializa o servico com o modelo ja carregado para inferencia em lote."""
        self.modelo = modelo
        self.processador = ProcessadorFeatures()

    def simular_lote(self, lote_sintetico: pd.DataFrame, salvar_em: str | None = None) -> dict:
        """Executa simulacao batch e persiste relatorio JSON."""
        if lote_sintetico is None or lote_sintetico.empty:
            raise ValueError("Lote sintetico vazio para simulacao.")
        if self.modelo is None:
            raise RuntimeError("Modelo indisponivel para simulacao.")

        estatisticas = ServicoRisco._carregar_estatisticas()
        dados_processados = self.processador.processar(lote_sintetico, estatisticas=estatisticas)
        features_modelo = [
            c
            for c in Configuracoes.FEATURES_MODELO_NUMERICAS + Configuracoes.FEATURES_MODELO_CATEGORICAS
            if c in dados_processados.columns
        ]
        dados_modelo = dados_processados[features_modelo]
        probabilidades = self.modelo.predict_proba(dados_modelo)[:, 1]

        threshold = ServicoRisco._obter_threshold()
        predicoes = (probabilidades >= threshold).astype(int)
        segmentos = [ServicoRisco._segmentar_risco(float(p), threshold) for p in probabilidades]

        margem = max(Configuracoes.REVIEW_MARGIN, 0.0)
        revisoes = int(((probabilidades >= (threshold - margem)) & (probabilidades <= (threshold + margem))).sum())
        total = int(len(lote_sintetico))
        distribuicao = pd.Series(segmentos).value_counts().to_dict()

        relatorio = {
            "timestamp": datetime.now().isoformat(),
            "batch_size": total,
            "risk_threshold": round(float(threshold), 4),
            "risk_distribution": {
                "BAIXO_RISCO": int(distribuicao.get("BAIXO_RISCO", 0)),
                "MEDIO_RISCO": int(distribuicao.get("MEDIO_RISCO", 0)),
                "ALTO_RISCO": int(distribuicao.get("ALTO_RISCO", 0)),
            },
            "risk_percentages": {
                "BAIXO_RISCO": round((distribuicao.get("BAIXO_RISCO", 0) / total) * 100.0, 2),
                "MEDIO_RISCO": round((distribuicao.get("MEDIO_RISCO", 0) / total) * 100.0, 2),
                "ALTO_RISCO": round((distribuicao.get("ALTO_RISCO", 0) / total) * 100.0, 2),
            },
            "estimated_human_reviews": revisoes,
            "estimated_human_reviews_pct": round((revisoes / total) * 100.0, 2),
            "predicted_positive_count": int(predicoes.sum()),
        }

        caminho_saida = salvar_em or os.path.join(
            Configuracoes.MONITORING_DIR, "production_simulation_report.json"
        )
        os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
        with open(caminho_saida, "w", encoding="utf-8") as arquivo:
            json.dump(relatorio, arquivo, indent=2, ensure_ascii=False)

        return relatorio
