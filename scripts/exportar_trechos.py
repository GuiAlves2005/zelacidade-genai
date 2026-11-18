"""Exporta a base de conhecimento em um JSON simples para o modo offline do assistente no MVP.

Saída: base/trechos.json — lista de {id, titulo, texto, categoria, orgao, link, fonte_id}.
Rode de novo sempre que editar base/trechos.jsonl ou base/fontes.csv.

Uso:  python scripts/exportar_trechos.py
"""
from __future__ import annotations

import json

import _caminho  # noqa: F401

from zela.base import carregar_fontes, carregar_trechos
from zela.config import DIR_BASE


def main() -> None:
    fontes = carregar_fontes()
    saida = []
    for t in carregar_trechos():
        f = fontes[t.fonte]
        link = f["url"] if f["url"].startswith("http") else None  # documento interno não tem link público
        saida.append({"id": t.id, "titulo": t.titulo, "texto": t.texto, "categoria": t.categoria,
                      "orgao": f["orgao"], "link": link, "fonte_id": t.fonte})
    destino = DIR_BASE / "trechos.json"
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(saida)} trechos exportados para {destino}")


if __name__ == "__main__":
    main()
