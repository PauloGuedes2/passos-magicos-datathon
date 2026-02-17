"""Logger de predições em JSONL.

Responsabilidades:
- Registrar predições com segurança de thread
- Garantir estrutura padronizada do log
"""

import json
import os
import threading
import uuid
from datetime import datetime

from src.config.settings import Configuracoes
from src.util.logger import logger


class LoggerPredicao:
    """Logger thread-safe para persistir predições.

    Responsabilidades:
    - Garantir instância única
    - Serializar dados de predição
    - Escrever logs com segurança
    """

    _instancia = None
    _lock = threading.Lock()

    def __new__(cls):
        """Cria ou reutiliza a instância única.

        Retorno:
        - LoggerPredicao: instância singleton
        """
        if cls._instancia is None:
            with cls._lock:
                if cls._instancia is None:
                    cls._instancia = super(LoggerPredicao, cls).__new__(cls)
        return cls._instancia

    def registrar_predicao(self, features: dict, dados_predicao: dict, versao_modelo: str = "2.1.0"):
        """Escreve um registro de predição de forma atômica.

        Parâmetros:
        - features (dict): features de entrada
        - dados_predicao (dict): dados da predição
        - versao_modelo (str): versão do modelo

        Retorno:
        - None: não retorna valor
        """
        entrada_log = {
            "prediction_id": str(uuid.uuid4()),
            "correlation_id": dados_predicao.get("correlation_id", str(uuid.uuid4())),
            "timestamp": datetime.now().isoformat(),
            "model_version": versao_modelo,
            "input_features": features,
            "prediction_result": {
                "class": dados_predicao.get("prediction"),
                "probability": dados_predicao.get("risk_probability"),
                "label": dados_predicao.get("risk_label"),
            },
        }

        try:
            linha_json = json.dumps(entrada_log, ensure_ascii=False)
        except Exception as erro:
            logger.error(f"Falha ao serializar log: {erro}")
            return

        with self._lock:
            try:
                os.makedirs(os.path.dirname(Configuracoes.LOG_PATH), exist_ok=True)
                self._rotacionar_se_necessario()
                with open(Configuracoes.LOG_PATH, "a", encoding="utf-8") as arquivo:
                    arquivo.write(linha_json + "\n")
            except Exception as erro:
                logger.error(f"Falha Crítica ao escrever no log de predição: {erro}")

    @staticmethod
    def _rotacionar_se_necessario() -> None:
        """Rotaciona o arquivo de log quando atinge o tamanho máximo."""
        try:
            if not os.path.exists(Configuracoes.LOG_PATH):
                return
            tamanho_atual = os.path.getsize(Configuracoes.LOG_PATH)
            if tamanho_atual < Configuracoes.LOG_MAX_BYTES:
                return
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            novo_nome = f"{Configuracoes.LOG_PATH}.{timestamp}.bak"
            os.replace(Configuracoes.LOG_PATH, novo_nome)
        except Exception as erro:
            logger.warning(f"Falha ao rotacionar log: {erro}")
