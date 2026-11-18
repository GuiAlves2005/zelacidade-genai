"""Leitura da base de conhecimento, das fontes e dos conjuntos de pergunta."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .config import DIR_AVALIACAO, DIR_BASE, DIR_TREINO


@dataclass(frozen=True)
class Trecho:
    id: str
    fonte: str
    categoria: str
    titulo: str
    texto: str
    perguntas_treino: tuple[str, ...] = ()

    @property
    def para_indexar(self) -> str:
        return f"{self.titulo}. {self.texto}"


def ler_jsonl(caminho: Path) -> list[dict]:
    with open(caminho, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def escrever_jsonl(caminho: Path, linhas: list[dict]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        for l in linhas:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")


def carregar_trechos(caminho: Path | None = None) -> list[Trecho]:
    linhas = ler_jsonl(caminho or DIR_BASE / "trechos.jsonl")
    trechos = [
        Trecho(
            id=l["id"],
            fonte=l["fonte"],
            categoria=l["categoria"],
            titulo=l["titulo"],
            texto=l["texto"],
            perguntas_treino=tuple(l.get("perguntas_treino", [])),
        )
        for l in linhas
    ]
    ids = [t.id for t in trechos]
    assert len(ids) == len(set(ids)), "IDs de trecho duplicados na base"
    return trechos


def carregar_fontes(caminho: Path | None = None) -> dict[str, dict]:
    with open(caminho or DIR_BASE / "fontes.csv", encoding="utf-8") as f:
        return {r["id_fonte"]: r for r in csv.DictReader(f)}


def carregar_teste(caminho: Path | None = None) -> list[dict]:
    return ler_jsonl(caminho or DIR_AVALIACAO / "perguntas_teste.jsonl")


def carregar_recusas_treino(caminho: Path | None = None) -> list[dict]:
    return ler_jsonl(caminho or DIR_TREINO / "recusas_treino.jsonl")
