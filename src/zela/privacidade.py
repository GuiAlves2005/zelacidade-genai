"""LGPD: minimização de dados pessoais antes de enviar ao modelo e antes de gravar logs.

A pergunta do morador pode trazer CPF, telefone, e-mail ou placa sem necessidade.
Esses dados não ajudam a orientar sobre o problema urbano, então são mascarados.
"""
from __future__ import annotations

import re

_PADROES = [
    ("[CPF]", re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")),
    ("[EMAIL]", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("[TELEFONE]", re.compile(r"(?:\+?55\s?)?\(?\b\d{2}\)?\s?9?\d{4}[-\s]?\d{4}\b")),
    ("[PLACA]", re.compile(r"\b[A-Z]{3}-?\d[A-Z0-9]\d{2}\b", re.IGNORECASE)),
]

# Números de serviço públicos não são dados pessoais e não devem ser mascarados.
_PERMITIDOS = re.compile(r"\b(?:0800\s?\d{3}\s?\d{4}|\(11\)\s?3230-5156)\b")


def mascarar(texto: str) -> str:
    protegidos: dict[str, str] = {}

    def _guardar(m: re.Match) -> str:
        chave = f"__P{len(protegidos)}__"
        protegidos[chave] = m.group(0)
        return chave

    texto = _PERMITIDOS.sub(_guardar, texto)
    for rotulo, padrao in _PADROES:
        texto = padrao.sub(rotulo, texto)
    for chave, original in protegidos.items():
        texto = texto.replace(chave, original)
    return texto
