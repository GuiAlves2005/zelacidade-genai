"""Normalização de texto usada na busca léxica e na correção automática."""
from __future__ import annotations

import re
import unicodedata


def normalizar(texto: str) -> str:
    """Minúsculas, sem acento, espaços colapsados."""
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.lower()
    return re.sub(r"\s+", " ", texto).strip()


def contem_pontos_chave(resposta: str, pontos_chave: list[list[str]]) -> bool:
    """Cada grupo é uma lista de alternativas; a resposta precisa conter ao menos uma de cada grupo."""
    r = normalizar(resposta)
    return all(any(normalizar(alt) in r for alt in grupo) for grupo in pontos_chave)
