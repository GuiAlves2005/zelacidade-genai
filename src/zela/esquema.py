"""Contrato de saída da IA (JSON) e validação."""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from .config import CATEGORIAS, RISCOS

MSG_RECUSA = (
    "Não consta na base de conhecimento da ZelaCidade. "
    "Para esse assunto, consulte os canais oficiais da Prefeitura (SP156)."
)


class RespostaZela(BaseModel):
    resposta: str = Field(min_length=1, max_length=1500)
    categoria: Literal[tuple(CATEGORIAS)]  # type: ignore[valid-type]
    risco: Literal[tuple(RISCOS)]  # type: ignore[valid-type]
    fontes: list[str] = Field(default_factory=list)
    recusou: bool

    @field_validator("fontes")
    @classmethod
    def ids_validos(cls, v: list[str]) -> list[str]:
        for f in v:
            if not re.fullmatch(r"T\d{3}", f):
                raise ValueError(f"fonte com formato inválido: {f!r}")
        return v


def extrair_json(texto: str) -> str | None:
    """Pega o primeiro objeto JSON balanceado do texto gerado (o modelo às vezes escreve antes/depois)."""
    texto = re.sub(r"```(?:json)?", "", texto)
    inicio = texto.find("{")
    if inicio < 0:
        return None
    nivel, em_str, escape = 0, False, False
    for i, c in enumerate(texto[inicio:], start=inicio):
        if em_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                em_str = False
            continue
        if c == '"':
            em_str = True
        elif c == "{":
            nivel += 1
        elif c == "}":
            nivel -= 1
            if nivel == 0:
                return texto[inicio : i + 1]
    return None


def validar(texto: str) -> tuple[RespostaZela | None, str | None]:
    """Retorna (objeto, None) se válido ou (None, motivo_do_erro)."""
    bruto = extrair_json(texto)
    if bruto is None:
        return None, "sem_json"
    try:
        dados = json.loads(bruto)
    except json.JSONDecodeError as e:
        return None, f"json_invalido: {e.msg}"
    try:
        return RespostaZela.model_validate(dados), None
    except ValidationError as e:
        return None, f"esquema: {e.errors()[0]['loc']} {e.errors()[0]['msg']}"


def resposta_de_recusa(motivo: str = "") -> RespostaZela:
    return RespostaZela(resposta=MSG_RECUSA, categoria="fora_do_escopo", risco="nao_se_aplica", fontes=[], recusou=True)
