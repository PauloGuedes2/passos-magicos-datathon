"""Teste de integracao do simulador de cenario de producao."""

import json

import numpy as np
import pandas as pd

from src.application.production_simulation_service import ServicoSimulacaoProducao


class ModeloBatchTeste:
    """Modelo simples de probabilidade para simulacao batch."""

    def predict_proba(self, dados):
        probs = np.clip(pd.to_numeric(dados["INDE_ANTERIOR"], errors="coerce").fillna(0) / 10.0, 0, 1)
        return np.column_stack((1.0 - probs, probs))


def test_simulacao_batch_gera_relatorio(tmp_path):
    lote = pd.DataFrame(
        [
            {
                "RA": "1",
                "IDADE": 10,
                "ANO_INGRESSO": 2022,
                "ANO_REFERENCIA": 2024,
                "GENERO": "Masculino",
                "TURMA": "A",
                "INSTITUICAO_ENSINO": "Escola",
                "FASE": "1A",
                "INDE_ANTERIOR": 2.0,
                "IAA_ANTERIOR": 2.0,
                "IEG_ANTERIOR": 2.0,
                "IPS_ANTERIOR": 2.0,
                "IDA_ANTERIOR": 2.0,
                "IPP_ANTERIOR": 2.0,
                "IPV_ANTERIOR": 2.0,
                "IAN_ANTERIOR": 2.0,
                "ALUNO_NOVO": 0,
            },
            {
                "RA": "2",
                "IDADE": 14,
                "ANO_INGRESSO": 2021,
                "ANO_REFERENCIA": 2024,
                "GENERO": "Feminino",
                "TURMA": "B",
                "INSTITUICAO_ENSINO": "Escola",
                "FASE": "2B",
                "INDE_ANTERIOR": 9.0,
                "IAA_ANTERIOR": 9.0,
                "IEG_ANTERIOR": 9.0,
                "IPS_ANTERIOR": 9.0,
                "IDA_ANTERIOR": 9.0,
                "IPP_ANTERIOR": 9.0,
                "IPV_ANTERIOR": 9.0,
                "IAN_ANTERIOR": 9.0,
                "ALUNO_NOVO": 0,
            },
        ]
    )

    destino = tmp_path / "simulation_report.json"
    servico = ServicoSimulacaoProducao(modelo=ModeloBatchTeste())

    resultado = servico.simular_lote(lote, salvar_em=str(destino))

    assert resultado["batch_size"] == 2
    assert "risk_distribution" in resultado
    assert "estimated_human_reviews" in resultado
    assert destino.exists()

    conteudo = json.loads(destino.read_text(encoding="utf-8"))
    assert conteudo["batch_size"] == 2
