"""API da Zela, consumida pelo MVP da ZelaCidade.

Rodar:  uvicorn api.main:app --host 0.0.0.0 --port 8001
Env:    ZELA_CONFIG=C (padrão) | B | A   ·   ZELA_GERADOR=hf (padrão) | ollama

Endpoints
  GET  /saude               → status e configuração ativa
  POST /perguntar           → {"pergunta": "..."}  ⇒  RespostaZela + metadados (fontes com título e URL)

Segurança/LGPD
  - A pergunta é mascarada (CPF, e-mail, telefone, placa) antes de ir ao modelo e ao log.
  - Nada é persistido além de um log técnico mínimo (latência, config, ids citados, se recusou).
  - Limite de tamanho da pergunta para evitar abuso.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import secrets  # noqa: E402

from fastapi import Depends, FastAPI, Header, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from zela.base import carregar_fontes, carregar_trechos  # noqa: E402
from zela.busca import Buscador  # noqa: E402
from zela.config import Config  # noqa: E402
from zela.esquema import RespostaZela  # noqa: E402
from zela.llm import GeradorHF, GeradorOllama  # noqa: E402
from zela.pipeline import DESCRICAO, PipelineZela  # noqa: E402
from zela.privacidade import mascarar  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("zela-api")

cfg = Config()
CONFIG = os.getenv("ZELA_CONFIG", "C")
trechos = carregar_trechos()
por_id = {t.id: t for t in trechos}
fontes = carregar_fontes()
buscador = Buscador(trechos, cfg.backend_busca, cfg.modelo_emb)

if os.getenv("ZELA_GERADOR", "hf") == "ollama":
    gerador = GeradorOllama(os.getenv("ZELA_OLLAMA_MODELO", "qwen2.5:1.5b"))
    CONFIG = "B" if CONFIG == "C" else CONFIG  # Ollama não carrega o adaptador LoRA
else:
    gerador = GeradorHF(cfg.modelo_llm, cfg.adaptador_lora if CONFIG == "C" else None, cfg.max_novos_tokens)
pipeline = PipelineZela(CONFIG, gerador, buscador, cfg.top_k, cfg.limiar)

app = FastAPI(title="Zela — assistente da ZelaCidade", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ZELA_CORS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


API_KEY = os.getenv("ZELA_API_KEY", "")


def verificar_chave(x_api_key: str | None = Header(default=None)) -> None:
    """Se ZELA_API_KEY estiver definida, exige o cabeçalho X-API-Key igual a ela (comparação em tempo constante)."""
    if API_KEY and not (x_api_key and secrets.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(status_code=401, detail="X-API-Key ausente ou inválida.")


class Pergunta(BaseModel):
    pergunta: str = Field(min_length=3, max_length=500)


class FonteCitada(BaseModel):
    id: str
    titulo: str
    orgao: str
    url: str


class RespostaAPI(BaseModel):
    saida: RespostaZela
    fontes_detalhadas: list[FonteCitada]
    config: str
    recusa_por_limiar: bool
    latencia_s: float
    aviso: str = "Orientação informativa. A ZelaCidade não substitui os canais oficiais da Prefeitura (SP156)."


@app.get("/saude")
def saude() -> dict:
    return {"status": "ok", "config": CONFIG, "descricao": DESCRICAO[CONFIG], "modelo": gerador.nome,
            "busca": cfg.backend_busca, "limiar": cfg.limiar, "n_trechos": len(trechos)}


@app.post("/perguntar", response_model=RespostaAPI, dependencies=[Depends(verificar_chave)])
def perguntar(p: Pergunta) -> RespostaAPI:
    exe = pipeline.responder(p.pergunta)
    if exe.saida is None:
        log.warning("json_invalido config=%s erro=%s", CONFIG, exe.erro_json)
        raise HTTPException(status_code=502, detail="O modelo gerou uma resposta fora do formato. Tente reformular.")
    detalhes = []
    for fid in exe.saida.fontes:
        t = por_id.get(fid)
        if t is None:
            continue
        f = fontes[t.fonte]
        detalhes.append(FonteCitada(id=fid, titulo=t.titulo, orgao=f["orgao"], url=f["url"]))
    log.info("pergunta=%r config=%s recusou=%s fontes=%s lat=%.2fs",
             mascarar(p.pergunta)[:80], CONFIG, exe.saida.recusou, exe.saida.fontes, exe.latencia_s)
    return RespostaAPI(saida=exe.saida, fontes_detalhadas=detalhes, config=CONFIG,
                       recusa_por_limiar=exe.recusa_por_limiar, latencia_s=round(exe.latencia_s, 3))
