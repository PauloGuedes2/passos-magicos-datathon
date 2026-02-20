"""Gerenciador singleton para o modelo de ML.

Responsabilidades:
- Carregar o modelo do disco
- Expor o modelo carregado
- Garantir thread-safety
"""

import os
from joblib import load
from threading import Lock, RLock
from typing import Any, Optional

from src.config.settings import Configuracoes
from src.util.logger import logger


class GerenciadorModelo:
    """Singleton thread-safe para gerenciamento do modelo.

    Responsabilidades:
    - Controlar a instância única
    - Manter o modelo em memória
    - Evitar recargas desnecessárias
    """

    _instancia = None
    _lock = Lock()
    _model_lock = RLock()
    _modelo: Optional[Any] = None

    def __new__(cls):
        """Cria ou reutiliza a instância única.

        Retorno:
        - GerenciadorModelo: instância singleton
        """
        if cls._instancia is None:
            with cls._lock:
                if cls._instancia is None:
                    cls._instancia = super(GerenciadorModelo, cls).__new__(cls)
        return cls._instancia

    def carregar_modelo(self, force: bool = False) -> None:
        """Carrega o modelo do disco para a memória.

        Parâmetros:
        - force (bool): recarrega o modelo mesmo quando já existe em memória

        Retorno:
        - None: não retorna valor
        """
        with self._model_lock:
            if self._modelo is not None and not force:
                logger.info("Modelo já carregado em memória. Reutilizando.")
                return

            if not os.path.exists(Configuracoes.MODEL_PATH):
                logger.critical(f"Arquivo de modelo não encontrado em: {Configuracoes.MODEL_PATH}")
                raise FileNotFoundError(f"Modelo não encontrado em {Configuracoes.MODEL_PATH}")

            try:
                logger.info(f"Carregando modelo do disco: {Configuracoes.MODEL_PATH}...")
                self._modelo = load(Configuracoes.MODEL_PATH)
                logger.info("Modelo carregado com sucesso!")
            except Exception as erro:
                logger.critical(f"Falha fatal ao carregar o modelo: {erro}")
                raise erro

    def obter_modelo(self) -> Any:
        """Retorna o modelo carregado.

        Retorno:
        - Any: modelo em memória

        Exceções:
        - RuntimeError: quando o modelo não está disponível
        """
        if self._modelo is None:
            self.carregar_modelo()

        if self._modelo is None:
            raise RuntimeError("Modelo indisponível para inferência.")

        return self._modelo
