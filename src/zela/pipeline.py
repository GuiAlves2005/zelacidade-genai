"""Pipeline de resposta das três configurações (A, B, C)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .busca import Buscador, Resultado
from .esquema import RespostaZela, resposta_de_recusa, validar
from .llm import Gerador
from .privacidade import mascarar
from .prompts import mensagens_rag, mensagens_sem_contexto

DESCRICAO = {
    "A": "Modelo sem contexto",
    "B": "RAG simples (busca + limiar + citação)",
    "C": "RAG + LoRA (modelo especializado)",
}


@dataclass
class Execucao:
    pergunta: str
    config: str
    saida: RespostaZela | None
    texto_bruto: str
    erro_json: str | None
    contexto: list[dict] = field(default_factory=list)  # [{"id":..., "score":...}]
    recusa_por_limiar: bool = False
    latencia_s: float = 0.0

    def para_dict(self) -> dict:
        return {
            "pergunta": self.pergunta,
            "config": self.config,
            "saida": self.saida.model_dump() if self.saida else None,
            "texto_bruto": self.texto_bruto,
            "erro_json": self.erro_json,
            "contexto": self.contexto,
            "recusa_por_limiar": self.recusa_por_limiar,
            "latencia_s": round(self.latencia_s, 3),
        }


class PipelineZela:
    def __init__(self, config: str, gerador: Gerador, buscador: Buscador | None = None, top_k: int = 3, limiar: float = 0.0):
        if config not in DESCRICAO:
            raise ValueError("config deve ser A, B ou C")
        if config in ("B", "C") and buscador is None:
            raise ValueError("configurações B e C precisam de buscador")
        self.config, self.gerador, self.buscador = config, gerador, buscador
        self.top_k, self.limiar = top_k, limiar

    def responder(self, pergunta: str) -> Execucao:
        inicio = time.perf_counter()
        pergunta_segura = mascarar(pergunta)  # LGPD: nada de CPF/telefone/e-mail vai para o modelo

        if self.config == "A":
            bruto = self.gerador.gerar(mensagens_sem_contexto(pergunta_segura))
            saida, erro = validar(bruto)
            return Execucao(pergunta, "A", saida, bruto, erro, latencia_s=time.perf_counter() - inicio)

        resultados: list[Resultado] = self.buscador.buscar(pergunta_segura, self.top_k)
        contexto = [{"id": r.trecho.id, "score": round(r.score, 4)} for r in resultados]

        # Limiar: se nem o melhor trecho é parecido o bastante, recusamos sem chamar o modelo.
        if not resultados or resultados[0].score < self.limiar:
            recusa = resposta_de_recusa()
            return Execucao(
                pergunta, self.config, recusa, recusa.model_dump_json(), None, contexto, True, time.perf_counter() - inicio
            )

        bruto = self.gerador.gerar(mensagens_rag(pergunta_segura, resultados))
        saida, erro = validar(bruto)
        return Execucao(pergunta, self.config, saida, bruto, erro, contexto, False, time.perf_counter() - inicio)
