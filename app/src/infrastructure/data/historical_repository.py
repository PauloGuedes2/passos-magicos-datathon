"""Repositório de histórico acadêmico.

Responsabilidades:
- Carregar dataset de referência
- Fornecer histórico do aluno
- Aplicar normalizações de RA
"""

import os
import pandas as pd

from src.config.settings import Configuracoes
from src.util.logger import logger


class RepositorioHistorico:
    """Repositório singleton para consulta de histórico.

    Responsabilidades:
    - Manter dados carregados em memória
    - Recarregar quando necessário
    - Buscar métricas do ano anterior
    """

    _instancia = None
    _dados = None

    def __new__(cls):
        """Cria ou reutiliza a instância única.

        Retorno:
        - RepositorioHistorico: instância singleton
        """
        if cls._instancia is None:
            cls._instancia = super(RepositorioHistorico, cls).__new__(cls)
            cls._instancia._carregar_dados()
        return cls._instancia

    def _carregar_dados(self):
        """Carrega o dataset de referência ou o arquivo bruto.

        Retorno:
        - None: não retorna valor
        """
        try:
            logger.info("Carregando base histórica para Feature Store...")

            if Configuracoes.HISTORICAL_PATH and os.path.exists(Configuracoes.HISTORICAL_PATH):
                self._dados = pd.read_csv(Configuracoes.HISTORICAL_PATH)
                if "RA" not in self._dados.columns:
                    logger.warning("CSV histórico sem RA. Recarregando do Excel...")
                    from src.infrastructure.data.data_loader import CarregadorDados
                    self._dados = CarregadorDados().carregar_dados()
            else:
                from src.infrastructure.data.data_loader import CarregadorDados
                self._dados = CarregadorDados().carregar_dados()

            if "RA" not in self._dados.columns:
                logger.warning("Coluna RA não encontrada no histórico! A busca smart falhará.")
                return

            self._dados["RA"] = (
                self._dados["RA"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            )
            self._dados = self._dados.sort_values(by=["RA", "ANO_REFERENCIA"])

            logger.info(f"Feature Store carregada com {len(self._dados)} registros.")
        except Exception as erro:
            logger.error(f"Erro ao carregar Feature Store: {erro}")
            self._dados = pd.DataFrame()

    def obter_historico_estudante(self, ra_estudante: str, ano_referencia: int | None = None) -> dict:
        """Busca métricas do ano anterior para um aluno.

        Parâmetros:
        - ra_estudante (str): RA do aluno

        Retorno:
        - dict: métricas históricas ou vazio para aluno novo
        """
        if self._dados is None or self._dados.empty:
            return {}

        ra_alvo = str(ra_estudante).strip()
        historico = self._dados[self._dados["RA"] == ra_alvo]

        if historico.empty:
            return None

        if ano_referencia is not None and "ANO_REFERENCIA" in historico.columns:
            anos_validos = pd.to_numeric(historico["ANO_REFERENCIA"], errors="coerce")
            historico = historico.loc[anos_validos < int(ano_referencia)]
            if historico.empty:
                return None
            ultimo_registro = historico.iloc[-1]
        else:
            # Fallback: usa ultimo registro conhecido.
            ultimo_registro = historico.iloc[-1]

        def _obter_seguro(nome_coluna):
            valor = ultimo_registro.get(nome_coluna, 0.0)
            try:
                convertido = float(valor)
                if pd.isna(convertido):
                    return 0.0
                return convertido
            except (ValueError, TypeError):
                return 0.0

        return {
            "INDE_ANTERIOR": _obter_seguro("INDE"),
            "IAA_ANTERIOR": _obter_seguro("IAA"),
            "IEG_ANTERIOR": _obter_seguro("IEG"),
            "IPS_ANTERIOR": _obter_seguro("IPS"),
            "IDA_ANTERIOR": _obter_seguro("IDA"),
            "IPP_ANTERIOR": _obter_seguro("IPP"),
            "IPV_ANTERIOR": _obter_seguro("IPV"),
            "IAN_ANTERIOR": _obter_seguro("IAN"),
            "ALUNO_NOVO": 0,
        }
