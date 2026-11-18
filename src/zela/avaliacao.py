"""Correção automática das respostas e placar.

Definições (mesmas usadas no relatório):
- json_valido      : a saída passou no esquema RespostaZela.
- acerto           : pergunta respondível, o modelo NÃO recusou e a resposta contém todos os pontos-chave do gabarito.
- recusa_correta   : pergunta "não consta" e o modelo recusou.
- recusa_indevida  : pergunta respondível e o modelo recusou.
- citacao_correta  : respondeu citando ao menos um trecho relevante do gabarito.
- citacao_fantasma : citou um id que não estava no contexto recebido (ou citou algo na config A, que não tem contexto).
- alucinacao       : afirmou algo sem sustentação, ou seja:
                     (a) respondeu uma pergunta "não consta"; ou
                     (b) respondeu com citação fantasma; ou
                     (c) respondeu uma pergunta respondível de forma errada (sem os pontos-chave).
                     JSON inválido conta como falha, mas não como alucinação.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .texto import contem_pontos_chave


def corrigir(execucao: dict, gabarito: dict) -> dict:
    saida = execucao["saida"]
    deve_recusar = gabarito["deve_recusar"]
    linha = {
        "id": gabarito["id"],
        "config": execucao["config"],
        "categoria": gabarito["categoria"],
        "deve_recusar": deve_recusar,
        "json_valido": saida is not None,
        "recusou": bool(saida and saida["recusou"]),
        "recusa_por_limiar": execucao.get("recusa_por_limiar", False),
        "latencia_s": execucao.get("latencia_s", 0.0),
    }
    if saida is None:
        linha.update(acerto=False, recusa_correta=False, recusa_indevida=False,
                     citacao_correta=False, citacao_fantasma=False, alucinacao=False)
        return linha

    respondeu = not saida["recusou"]
    ids_contexto = {c["id"] for c in execucao.get("contexto", [])}
    fontes = set(saida["fontes"])
    fantasma = respondeu and bool(fontes - ids_contexto)
    acerto = (not deve_recusar) and respondeu and contem_pontos_chave(saida["resposta"], gabarito["pontos_chave"])
    linha.update(
        acerto=acerto,
        recusa_correta=deve_recusar and not respondeu,
        recusa_indevida=(not deve_recusar) and not respondeu,
        citacao_correta=respondeu and bool(fontes & set(gabarito["trechos_relevantes"])),
        citacao_fantasma=fantasma,
        alucinacao=respondeu and (deve_recusar or fantasma or not acerto),
    )
    return linha


def placar(linhas: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(linhas)
    resp = df[~df.deve_recusar]
    nc = df[df.deve_recusar]
    tab = defaultdict(dict)
    for cfg in sorted(df.config.unique()):
        r, n, t = resp[resp.config == cfg], nc[nc.config == cfg], df[df.config == cfg]
        tab[cfg] = {
            "acuracia_respondiveis": r.acerto.mean(),
            "recusa_correta_nao_consta": n.recusa_correta.mean() if len(n) else float("nan"),
            "recusa_indevida": r.recusa_indevida.mean(),
            "taxa_alucinacao": t.alucinacao.mean(),
            "json_valido": t.json_valido.mean(),
            "citacao_correta": r.citacao_correta.mean(),
            "citacao_fantasma": t.citacao_fantasma.mean(),
            "latencia_media_s": t.latencia_s.mean(),
            "n_perguntas": len(t),
        }
    return pd.DataFrame(tab).T


def placar_markdown(df: pd.DataFrame, descricoes: dict[str, str]) -> str:
    pct = [c for c in df.columns if c not in ("latencia_media_s", "n_perguntas")]
    cab = "| Config | " + " | ".join(df.columns) + " |\n|---|" + "---|" * len(df.columns) + "\n"
    corpo = ""
    for cfg, row in df.iterrows():
        vals = [f"{row[c]*100:.1f}%" if c in pct else (f"{row[c]:.2f}" if c == "latencia_media_s" else str(int(row[c])))
                for c in df.columns]
        corpo += f"| {cfg} — {descricoes.get(cfg, '')} | " + " | ".join(vals) + " |\n"
    return cab + corpo
