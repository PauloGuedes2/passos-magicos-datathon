"""Testes da fábrica de logger."""

import logging

from src.util.logger import FabricaLogger


def test_fabrica_logger_configuracao_unica():
    nome_logger = "TEST_LOGGER"
    logger = logging.getLogger(nome_logger)
    logger.handlers = []

    primeiro = FabricaLogger.configurar(nome_logger)
    total_handlers = len(primeiro.handlers)
    segundo = FabricaLogger.configurar(nome_logger)

    assert primeiro is segundo
    assert len(segundo.handlers) == total_handlers
