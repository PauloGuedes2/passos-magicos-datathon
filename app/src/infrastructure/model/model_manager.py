"""Gerenciador singleton para o modelo de ML.

Responsabilidades:
- Carregar o modelo do disco
- Expor o modelo carregado
- Garantir thread-safety
- Gerenciar versoes do modelo com retencao
"""

import json
import os
import re
from joblib import dump, load
from threading import Lock, RLock
from typing import Any, Optional

from src.config.settings import Configuracoes
from src.util.logger import logger


class GerenciadorModelo:
    """Singleton thread-safe para gerenciamento do modelo."""

    _instancia = None
    _lock = Lock()
    _model_lock = RLock()
    _modelo: Optional[Any] = None
    _modelos_por_versao: dict[str, Any] = {}

    def __new__(cls):
        if cls._instancia is None:
            with cls._lock:
                if cls._instancia is None:
                    cls._instancia = super(GerenciadorModelo, cls).__new__(cls)
        return cls._instancia

    @staticmethod
    def _sanitizar_versao(versao: str) -> str:
        versao_limpa = str(versao).strip()
        return re.sub(r"[^0-9A-Za-z._-]", "_", versao_limpa)

    @staticmethod
    def _path_modelo_versionado(versao: str) -> str:
        nome = GerenciadorModelo._sanitizar_versao(versao)
        return os.path.join(Configuracoes.MODEL_VERSIONS_DIR, f"model_{nome}.joblib")

    @staticmethod
    def _obter_versao_modelo_atual() -> str:
        try:
            if Configuracoes.METRICS_FILE and os.path.exists(Configuracoes.METRICS_FILE):
                with open(Configuracoes.METRICS_FILE, "r", encoding="utf-8") as arquivo:
                    metricas = json.load(arquivo)
                versao = metricas.get("model_version")
                if versao:
                    return str(versao)
        except Exception as erro:
            logger.warning(f"Falha ao ler versao atual em metricas: {erro}")
        return str(Configuracoes.MODEL_VERSION)

    def listar_versoes_disponiveis(self, incluir_atual: bool = True) -> list[str]:
        versoes: list[str] = []
        if os.path.isdir(Configuracoes.MODEL_VERSIONS_DIR):
            arquivos = [
                os.path.join(Configuracoes.MODEL_VERSIONS_DIR, nome)
                for nome in os.listdir(Configuracoes.MODEL_VERSIONS_DIR)
                if nome.startswith("model_") and nome.endswith(".joblib")
            ]
            arquivos.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            for caminho in arquivos:
                nome = os.path.basename(caminho)
                versao = nome[len("model_") : -len(".joblib")]
                versoes.append(versao)

        if incluir_atual:
            atual = self._obter_versao_modelo_atual()
            if atual not in versoes:
                versoes.insert(0, atual)

        return versoes

    def _resolver_path_e_versao(self, model_version: str | None) -> tuple[str, str]:
        if model_version is None:
            versao_atual = self._obter_versao_modelo_atual()
            return Configuracoes.MODEL_PATH, versao_atual

        versao = self._sanitizar_versao(model_version)
        caminho = self._path_modelo_versionado(versao)
        return caminho, versao

    def carregar_modelo(self, force: bool = False, model_version: str | None = None) -> str:
        """Carrega o modelo do disco para memoria e retorna versao resolvida."""
        with self._model_lock:
            caminho_modelo, versao_resolvida = self._resolver_path_e_versao(model_version)

            if not force and versao_resolvida in self._modelos_por_versao:
                if model_version is None:
                    self._modelo = self._modelos_por_versao[versao_resolvida]
                logger.info(f"Modelo {versao_resolvida} ja carregado em memoria. Reutilizando.")
                return versao_resolvida

            if not os.path.exists(caminho_modelo):
                disponiveis = self.listar_versoes_disponiveis(incluir_atual=True)
                raise FileNotFoundError(
                    f"Modelo nao encontrado para versao '{versao_resolvida}'. Disponiveis: {disponiveis}"
                )

            try:
                logger.info(f"Carregando modelo do disco: {caminho_modelo}...")
                modelo = load(caminho_modelo)
                self._modelos_por_versao[versao_resolvida] = modelo
                if model_version is None:
                    self._modelo = modelo
                logger.info(f"Modelo {versao_resolvida} carregado com sucesso!")
                return versao_resolvida
            except Exception as erro:
                logger.critical(f"Falha fatal ao carregar o modelo: {erro}")
                raise erro

    def obter_modelo(self, model_version: str | None = None) -> Any:
        """Retorna o modelo carregado."""
        versao_resolvida = self.carregar_modelo(model_version=model_version)
        modelo = self._modelos_por_versao.get(versao_resolvida)
        if modelo is None:
            raise RuntimeError("Modelo indisponivel para inferencia.")
        return modelo

    def obter_modelo_com_versao(self, model_version: str | None = None) -> tuple[Any, str]:
        """Retorna modelo e versao efetiva utilizada na inferencia."""
        versao_resolvida = self.carregar_modelo(model_version=model_version)
        modelo = self._modelos_por_versao.get(versao_resolvida)
        if modelo is None:
            raise RuntimeError("Modelo indisponivel para inferencia.")
        return modelo, versao_resolvida

    def registrar_modelo_versionado(self, modelo: Any, versao_modelo: str) -> str:
        """Persiste snapshot versionado e aplica retencao maxima configurada."""
        versao = self._sanitizar_versao(versao_modelo)
        os.makedirs(Configuracoes.MODEL_VERSIONS_DIR, exist_ok=True)
        caminho = self._path_modelo_versionado(versao)
        dump(modelo, caminho)
        self._aplicar_retencao_modelos()
        return caminho

    def _aplicar_retencao_modelos(self) -> None:
        limite = max(1, int(Configuracoes.MAX_MODEL_VERSIONS))
        if not os.path.isdir(Configuracoes.MODEL_VERSIONS_DIR):
            return

        arquivos = [
            os.path.join(Configuracoes.MODEL_VERSIONS_DIR, nome)
            for nome in os.listdir(Configuracoes.MODEL_VERSIONS_DIR)
            if nome.startswith("model_") and nome.endswith(".joblib")
        ]
        if len(arquivos) <= limite:
            return

        arquivos.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for antigo in arquivos[limite:]:
            try:
                os.remove(antigo)
            except OSError as erro:
                logger.warning(f"Falha ao remover modelo antigo {antigo}: {erro}")
