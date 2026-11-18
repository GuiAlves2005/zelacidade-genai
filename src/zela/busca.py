"""Busca na base de conhecimento.

Dois backends com a mesma interface:
- "e5":    embeddings densos multilíngues (intfloat/multilingual-e5-small), similaridade de cosseno.
- "tfidf": TF-IDF com n-gramas de palavra, sem acento. Roda em qualquer máquina, sem GPU;
           serve de linha de base para a busca e de fallback se o modelo de embeddings não carregar.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .base import Trecho
from .config import DIR_BASE
from .texto import normalizar


STOPWORDS = sorted(set(normalizar(
    "a o as os um uma uns umas de do da dos das em no na nos nas por pelo pela pelos pelas para pra com sem "
    "que se e é ou ao aos à às mas mais muito já ter tem tenho ser está estou foi são como qual quais quando "
    "onde quem eu meu minha meus minhas me mim você seu sua isso isto esse essa este esta lá aqui não sim "
    "devo posso preciso fazer faço vai vou só também depois antes há"
).split()))


@dataclass
class Resultado:
    trecho: Trecho
    score: float


class Buscador:
    def __init__(self, trechos: list[Trecho], backend: str = "e5", modelo_emb: str | None = None):
        self.trechos = trechos
        self.backend = backend
        if backend == "e5":
            self._iniciar_e5(modelo_emb or "intfloat/multilingual-e5-small")
        elif backend == "tfidf":
            self._iniciar_tfidf()
        else:
            raise ValueError(f"backend desconhecido: {backend}")

    # ---------- e5 ----------
    def _iniciar_e5(self, nome: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._modelo = SentenceTransformer(nome)
        textos = [f"passage: {t.para_indexar}" for t in self.trechos]
        chave = hashlib.md5(("|".join(textos) + nome).encode()).hexdigest()[:10]
        cache = DIR_BASE / f".emb_cache_{chave}.npy"
        if cache.exists():
            self._matriz = np.load(cache)
        else:
            self._matriz = self._modelo.encode(textos, normalize_embeddings=True, show_progress_bar=False)
            np.save(cache, self._matriz)

    def _scores_e5(self, consulta: str) -> np.ndarray:
        q = self._modelo.encode([f"query: {consulta}"], normalize_embeddings=True, show_progress_bar=False)[0]
        return self._matriz @ q

    # ---------- tfidf ----------
    def _iniciar_tfidf(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vet = TfidfVectorizer(
            preprocessor=normalizar, stop_words=STOPWORDS, ngram_range=(1, 2), sublinear_tf=True, min_df=1
        )
        self._matriz = self._vet.fit_transform([t.para_indexar for t in self.trechos])

    def _scores_tfidf(self, consulta: str) -> np.ndarray:
        q = self._vet.transform([consulta])
        return (self._matriz @ q.T).toarray().ravel()

    # ---------- API ----------
    def buscar(self, consulta: str, k: int = 3) -> list[Resultado]:
        scores = self._scores_e5(consulta) if self.backend == "e5" else self._scores_tfidf(consulta)
        ordem = np.argsort(-scores)[:k]
        return [Resultado(self.trechos[i], float(scores[i])) for i in ordem]


def recall_at_k(buscador: Buscador, perguntas: list[dict], k: int) -> float:
    """Fração das perguntas respondíveis em que algum trecho relevante aparece no top-k."""
    respondiveis = [p for p in perguntas if p["trechos_relevantes"]]
    acertos = 0
    for p in respondiveis:
        ids = {r.trecho.id for r in buscador.buscar(p["pergunta"], k)}
        acertos += bool(ids & set(p["trechos_relevantes"]))
    return acertos / len(respondiveis)
