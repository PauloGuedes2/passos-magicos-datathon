"""Testes do logger de predições."""

from unittest.mock import Mock, mock_open

from src.infrastructure.logging.prediction_logger import LoggerPredicao


def resetar_logger():
    LoggerPredicao._instancia = None


def test_logger_predicao_singleton():
    resetar_logger()
    primeiro = LoggerPredicao()
    segundo = LoggerPredicao()
    assert primeiro is segundo


def test_registrar_predicao_sucesso(monkeypatch):
    resetar_logger()

    class LockDummy:
        """Lock falso para validar entrada em contexto."""
        def __init__(self):
            """Inicializa o lock falso."""
            self.entrou = False
        def __enter__(self):
            """Marca a entrada no contexto."""
            self.entrou = True
        def __exit__(self, exc_type, exc, tb):
            """Mantém comportamento padrão de saída."""
            return False

    logger_predicao = LoggerPredicao()
    logger_predicao._lock = LockDummy()

    monkeypatch.setattr("src.infrastructure.logging.prediction_logger.uuid.uuid4", lambda: "uuid")

    class DataFixa:
        """Classe de data fixa para testes."""
        @classmethod
        def now(cls):
            """Retorna um objeto com data fixa."""
            class _Agora:
                """Objeto simples com data fixa."""
                def isoformat(self):
                    """Retorna data fixa em ISO."""
                    return "2024-01-01T00:00:00"
            return _Agora()

    monkeypatch.setattr("src.infrastructure.logging.prediction_logger.datetime", DataFixa)
    monkeypatch.setattr("src.infrastructure.logging.prediction_logger.os.makedirs", Mock())

    arquivo_mock = mock_open()
    monkeypatch.setattr("builtins.open", arquivo_mock)

    logger_predicao.registrar_predicao({"IDADE": 10}, {"prediction": 1, "risk_probability": 0.5, "risk_label": "ALTO"})

    assert logger_predicao._lock.entrou is True


def test_registrar_predicao_falha_serializacao(monkeypatch):
    resetar_logger()
    logger_predicao = LoggerPredicao()

    monkeypatch.setattr(
        "src.infrastructure.logging.prediction_logger.json.dumps",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad")),
    )

    erro_mock = Mock()
    monkeypatch.setattr("src.infrastructure.logging.prediction_logger.logger", erro_mock)

    logger_predicao.registrar_predicao({"bad": object()}, {"prediction": 1})

    erro_mock.error.assert_called_once()


def test_registrar_predicao_falha_escrita(monkeypatch):
    resetar_logger()
    logger_predicao = LoggerPredicao()

    monkeypatch.setattr("src.infrastructure.logging.prediction_logger.uuid.uuid4", lambda: "uuid")

    class DataFixa:
        """Classe de data fixa para testes."""
        @classmethod
        def now(cls):
            """Retorna um objeto com data fixa."""
            class _Agora:
                """Objeto simples com data fixa."""
                def isoformat(self):
                    """Retorna data fixa em ISO."""
                    return "2024-01-01T00:00:00"
            return _Agora()

    monkeypatch.setattr("src.infrastructure.logging.prediction_logger.datetime", DataFixa)
    monkeypatch.setattr("src.infrastructure.logging.prediction_logger.os.makedirs", Mock())

    def levantar_erro(*args, **kwargs):
        raise OSError("nope")

    monkeypatch.setattr("builtins.open", levantar_erro)

    erro_mock = Mock()
    monkeypatch.setattr("src.infrastructure.logging.prediction_logger.logger", erro_mock)

    logger_predicao.registrar_predicao({"IDADE": 10}, {"prediction": 1})

    erro_mock.error.assert_called_once()
