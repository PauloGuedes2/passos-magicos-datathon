"""
Testes de integracao end-to-end do pipeline de ML.

Responsabilidades:
- Validar pipeline completo sem mocks
- Testar carregamento de modelo real
- Testar predicao end-to-end
"""

import pandas as pd
import pytest

from src.config.settings import Configuracoes
from src.infrastructure.model.ml_pipeline import PipelineML


@pytest.fixture
def dados_teste_simples():
    """Cria um DataFrame simples para testes."""
    return pd.DataFrame({
        "RA": ["1001", "1002", "1003", "1004"],
        "IDADE": [12, 13, 12, 14],
        "ANO_INGRESSO": [2022, 2021, 2022, 2020],
        "GENERO": ["Masculino", "Feminino", "Masculino", "Feminino"],
        "TURMA": ["6A", "7B", "6A", "8C"],
        "INSTITUICAO_ENSINO": ["MUNICIPAL", "ESTADUAL", "MUNICIPAL", "ESTADUAL"],
        "FASE": ["6A", "7B", "6A", "8C"],
        "ANO_REFERENCIA": [2023, 2023, 2023, 2023],
        "INDE": [7.5, 6.0, 5.5, 8.0],
        "DEFASAGEM": [0, 0, -1, 0],
    })


@pytest.fixture
def dados_treino_multiplos_anos():
    """Cria um DataFrame com multiplos anos para validar split temporal."""
    dados_2022 = pd.DataFrame({
        "RA": ["1001", "1002", "1003"],
        "IDADE": [11, 12, 11],
        "ANO_INGRESSO": [2021, 2020, 2021],
        "GENERO": ["Masculino", "Feminino", "Masculino"],
        "TURMA": ["5A", "6B", "5A"],
        "INSTITUICAO_ENSINO": ["MUNICIPAL", "ESTADUAL", "MUNICIPAL"],
        "FASE": ["5A", "6B", "5A"],
        "ANO_REFERENCIA": [2022, 2022, 2022],
        "INDE": [7.0, 6.5, 5.0],
        "DEFASAGEM": [0, 0, -1],
    })

    dados_2023 = pd.DataFrame({
        "RA": ["1001", "1002", "1003"],
        "IDADE": [12, 13, 12],
        "ANO_INGRESSO": [2021, 2020, 2021],
        "GENERO": ["Masculino", "Feminino", "Masculino"],
        "TURMA": ["6A", "7B", "6A"],
        "INSTITUICAO_ENSINO": ["MUNICIPAL", "ESTADUAL", "MUNICIPAL"],
        "FASE": ["6A", "7B", "6A"],
        "ANO_REFERENCIA": [2023, 2023, 2023],
        "INDE": [7.5, 6.0, 5.5],
        "DEFASAGEM": [0, 0, -1],
    })

    return pd.concat([dados_2022, dados_2023], ignore_index=True)


class TestPipelineIntegration:
    """Testes de integracao do pipeline de ML."""

    def test_pipeline_completo_sem_mocks(self, dados_treino_multiplos_anos, tmp_path, monkeypatch):
        """
        Testa o pipeline completo sem mocks.

        Valida:
        - Criacao de target
        - Criacao de lag features
        - Split temporal
        - Treinamento
        - Persistencia de artefatos
        """
        model_path = tmp_path / "model.joblib"
        metrics_path = tmp_path / "train_metrics.json"
        reference_path = tmp_path / "reference_data.csv"
        feature_stats_path = tmp_path / "feature_stats.json"

        monkeypatch.setattr(Configuracoes, "MODEL_PATH", str(model_path))
        monkeypatch.setattr(Configuracoes, "METRICS_FILE", str(metrics_path))
        monkeypatch.setattr(Configuracoes, "REFERENCE_PATH", str(reference_path))
        monkeypatch.setattr(Configuracoes, "FEATURE_STATS_PATH", str(feature_stats_path))

        pipeline = PipelineML()
        pipeline.treinar(dados_treino_multiplos_anos)

        assert model_path.exists()
        assert metrics_path.exists()
        assert reference_path.exists()
        assert feature_stats_path.exists()

        metricas = pd.read_json(metrics_path, typ="series")
        assert "f1_score" in metricas.index
        assert "risk_threshold" in metricas.index
        assert "threshold_strategy" in metricas.index
        assert "threshold_justification" in metricas.index
        assert float(metricas["f1_score"]) >= 0.0

    def test_lag_features_sao_criadas_corretamente(self, dados_treino_multiplos_anos):
        """
        Testa se lag features sao criadas corretamente.

        Valida:
        - shift(1) funciona corretamente
        - ALUNO_NOVO e calculado
        - Sem vazamento de dados futuros
        """
        pipeline = PipelineML()
        dados_com_lag = pipeline.criar_features_lag(dados_treino_multiplos_anos)

        assert "INDE_ANTERIOR" in dados_com_lag.columns
        assert "ALUNO_NOVO" in dados_com_lag.columns
        primeiro_por_ra = dados_com_lag.groupby("RA").first()
        assert (primeiro_por_ra["INDE_ANTERIOR"] == 0).all()

    def test_split_temporal_correto(self, dados_treino_multiplos_anos):
        """
        Testa se split temporal e feito corretamente.

        Valida:
        - Treino contem anos anteriores
        - Teste contem ano mais recente
        - Sem sobreposicao
        """
        pipeline = PipelineML()
        mascara_treino, mascara_teste = pipeline._definir_particao_temporal(
            dados_treino_multiplos_anos
        )

        dados_treino = dados_treino_multiplos_anos[mascara_treino]
        dados_teste = dados_treino_multiplos_anos[mascara_teste]

        assert 2022 in dados_treino["ANO_REFERENCIA"].values
        assert 2023 in dados_teste["ANO_REFERENCIA"].values
        assert len(dados_treino) + len(dados_teste) == len(dados_treino_multiplos_anos)

    def test_baseline_comparacao(self, dados_treino_multiplos_anos):
        """
        Testa se baseline e calculado e comparado.

        Valida:
        - Baseline e criado
        - Metricas do baseline sao salvas
        """
        pipeline = PipelineML()
        dados = pipeline.criar_target(dados_treino_multiplos_anos)
        dados = pipeline.criar_features_lag(dados)

        mascara_treino, mascara_teste = pipeline._definir_particao_temporal(dados)
        alvo_teste = dados.loc[mascara_teste, "RISCO_DEFASAGEM"]

        predicoes = [0, 0, 1]
        metricas = pipeline._calcular_metricas(
            alvo_teste,
            predicoes,
            0.5,
            {
                "strategy": "f1",
                "justification": "teste",
                "tradeoff": "teste",
            },
            dados.loc[mascara_teste],
            dados.loc[mascara_treino],
            dados.loc[mascara_teste],
            alvo_treino=dados.loc[mascara_treino, "RISCO_DEFASAGEM"],
        )

        assert "baseline" in metricas
        assert "f1_score" in metricas["baseline"]
        assert "strategy" in metricas["baseline"]
        assert metricas["baseline"]["strategy"] == "predict_majority_class"
        assert metricas["baseline"]["source"] == "train_majority_class"

    def test_contrato_dados_validado(self, dados_teste_simples):
        """
        Testa se contrato de dados e validado.

        Valida:
        - Colunas obrigatorias sao verificadas
        - Erro e lancado se coluna falta
        """
        from src.infrastructure.data.data_contract import CONTRATO_TREINO

        CONTRATO_TREINO.validar(dados_teste_simples)

        dados_invalidos = dados_teste_simples.drop(columns=["RA"])
        with pytest.raises(ValueError, match="colunas obrigat"):
            CONTRATO_TREINO.validar(dados_invalidos)

    def test_feature_importance_extraida(self, dados_treino_multiplos_anos):
        """
        Testa se importancia de features e extraida.

        Valida:
        - Feature importance e calculada
        - Vetor de importancias nao e vazio
        """
        pipeline = PipelineML()
        dados = pipeline.criar_target(dados_treino_multiplos_anos)
        dados = pipeline.criar_features_lag(dados)
        dados = pipeline._remover_colunas_proibidas(dados)

        mascara_treino, _ = pipeline._definir_particao_temporal(dados)

        estatisticas = pipeline._calcular_estatisticas_treino(dados, mascara_treino)
        dados_processados = pipeline.processador.processar(dados, estatisticas=estatisticas)
        dados_processados["RISCO_DEFASAGEM"] = dados["RISCO_DEFASAGEM"]

        features_uso = [
            f
            for f in Configuracoes.FEATURES_MODELO_NUMERICAS + Configuracoes.FEATURES_MODELO_CATEGORICAS
            if f in dados_processados.columns
        ]

        matriz_treino = dados_processados.loc[mascara_treino, features_uso]
        alvo_treino = dados_processados.loc[mascara_treino, "RISCO_DEFASAGEM"]

        modelo = pipeline._criar_modelo(matriz_treino)
        modelo.fit(matriz_treino, alvo_treino)

        assert hasattr(modelo, "named_steps")
        assert "classifier" in modelo.named_steps
        assert hasattr(modelo.named_steps["classifier"], "feature_importances_")
        assert len(modelo.named_steps["classifier"].feature_importances_) > 0
