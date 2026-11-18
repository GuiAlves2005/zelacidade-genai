"""Configuração central. Tudo pode ser sobrescrito por variáveis de ambiente (.env)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DIR_BASE = RAIZ / "base"
DIR_AVALIACAO = RAIZ / "avaliacao"
DIR_RESULTADOS = DIR_AVALIACAO / "resultados"
DIR_TREINO = RAIZ / "treino"
DIR_MODELOS = RAIZ / "modelos"
ARQ_LIMIAR = DIR_AVALIACAO / "limiar.json"


def _carregar_env(caminho: Path = RAIZ / ".env") -> None:
    """Lê o .env da raiz (KEY=valor) sem sobrescrever variáveis já definidas no ambiente."""
    if not caminho.exists():
        return
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        linha = linha.split("#", 1)[0].strip()
        if "=" in linha:
            chave, valor = (p.strip() for p in linha.split("=", 1))
            os.environ.setdefault(chave, valor)


_carregar_env()

CATEGORIAS = [
    "buraco_via",
    "calcada",
    "iluminacao",
    "arvore",
    "drenagem",
    "emergencia",
    "obstaculo_sinalizacao",
    "canais",
    "plataforma",
    "fora_do_escopo",
]
RISCOS = ["baixo", "medio", "alto", "nao_se_aplica"]


def _limiar_padrao(backend: str) -> float:
    """Lê o limiar calibrado por scripts/calibrar_limiar.py; usa um valor seguro se não existir."""
    if ARQ_LIMIAR.exists():
        dados = json.loads(ARQ_LIMIAR.read_text(encoding="utf-8"))
        if backend in dados:
            return float(dados[backend]["limiar"])
    return {"e5": 0.82, "tfidf": 0.12}.get(backend, 0.5)


@dataclass
class Config:
    # Modelo base: pequeno o bastante para treinar LoRA numa GPU de 8 GB.
    # Troque por uma versão mais nova da mesma família se quiser (ex.: Qwen3-1.7B).
    modelo_llm: str = os.getenv("ZELA_LLM", "Qwen/Qwen2.5-1.5B-Instruct")
    modelo_emb: str = os.getenv("ZELA_EMB", "intfloat/multilingual-e5-small")
    backend_busca: str = os.getenv("ZELA_BUSCA", "e5")  # "e5" (denso) ou "tfidf" (léxico)
    adaptador_lora: Path = Path(os.getenv("ZELA_LORA", str(DIR_MODELOS / "lora-zela")))
    top_k: int = int(os.getenv("ZELA_TOPK", "3"))
    max_novos_tokens: int = int(os.getenv("ZELA_MAX_TOKENS", "350"))
    semente: int = 42
    limiar: float = field(default=-1.0)

    def __post_init__(self) -> None:
        if self.limiar < 0:
            env = os.getenv("ZELA_LIMIAR")
            self.limiar = float(env) if env else _limiar_padrao(self.backend_busca)
