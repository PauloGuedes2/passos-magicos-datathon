"""Modelos de dominio para representar dados do aluno."""

from typing import Optional

from pydantic import BaseModel, Field


class EntradaEstudante(BaseModel):
    """Dados minimos do aluno enviados pelo cliente para predicao smart."""

    RA: str = Field(..., min_length=1, description="Registro Academico Unico do Aluno")
    ANO_REFERENCIA: Optional[int] = Field(None, ge=2010, le=2030)
