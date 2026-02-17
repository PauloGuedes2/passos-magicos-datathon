"""Fábrica de logger da aplicação.

Responsabilidades:
- Configurar loggers de forma padronizada
- Evitar duplicação de handlers
- Direcionar saída para stdout
"""

import logging
import sys

from src.config.settings import Configuracoes


class FabricaLogger:
    """Responsável por configurar e fornecer instâncias de Logger.

    Responsabilidades:
    - Configuração única de logger
    - Formatação padronizada
    - Handler para console/Docker
    """

    _configurado = False

    @classmethod
    def configurar(cls, nome: str = "PASSOS_MAGICOS_APP"):
        """Configura o logger se ainda não estiver configurado.

        Parâmetros:
        - nome (str): nome do logger

        Retorno:
        - logging.Logger: logger configurado
        """
        logger_instancia = logging.getLogger(nome)

        if not logger_instancia.handlers:
            nivel_log = getattr(Configuracoes, "LOG_LEVEL", "INFO")
            logger_instancia.setLevel(nivel_log)

            formatador = logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            handler_console = logging.StreamHandler(sys.stdout)
            handler_console.setFormatter(formatador)
            logger_instancia.addHandler(handler_console)

            logger_instancia.propagate = False

        return logger_instancia


logger = FabricaLogger.configurar()
