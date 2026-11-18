"""Mede recall@1 / recall@3 da busca e calibra o limiar de recusa.

O limiar é escolhido no conjunto de TREINO (perguntas de treino dos trechos + perguntas fora do escopo),
nunca no conjunto de teste, para não "decorar" o teste. Critério: maior acurácia balanceada entre
"tem resposta na base" (score do 1º trecho >= limiar) e "não tem".

Uso:  python scripts/calibrar_limiar.py --backend e5
      python scripts/calibrar_limiar.py --backend tfidf
"""
from __future__ import annotations

import argparse
import json

import _caminho  # noqa: F401
import numpy as np
import pandas as pd

from zela.base import carregar_recusas_treino, carregar_teste, carregar_trechos
from zela.busca import Buscador, recall_at_k
from zela.config import ARQ_LIMIAR, DIR_RESULTADOS, Config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=Config().backend_busca, choices=["e5", "tfidf"])
    args = ap.parse_args()
    cfg = Config(backend_busca=args.backend)

    trechos = carregar_trechos()
    teste = carregar_teste()
    buscador = Buscador(trechos, args.backend, cfg.modelo_emb)

    # ---------------- recall no TESTE ----------------
    r1, r3 = recall_at_k(buscador, teste, 1), recall_at_k(buscador, teste, 3)
    print(f"[{args.backend}] recall@1 = {r1:.3f} | recall@3 = {r3:.3f}  (n = {sum(1 for p in teste if p['trechos_relevantes'])})")

    detalhes = []
    for p in teste:
        res = buscador.buscar(p["pergunta"], 3)
        detalhes.append({
            "id": p["id"], "deve_recusar": p["deve_recusar"],
            "relevantes": p["trechos_relevantes"], "top3": [r.trecho.id for r in res],
            "scores": [round(r.score, 4) for r in res],
            "hit@1": bool(p["trechos_relevantes"]) and res[0].trecho.id in p["trechos_relevantes"],
            "hit@3": bool(set(p["trechos_relevantes"]) & {r.trecho.id for r in res}),
        })
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detalhes).to_csv(DIR_RESULTADOS / f"busca_teste_{args.backend}.csv", index=False)

    # ---------------- calibração no TREINO ----------------
    positivos = [q for t in trechos for q in t.perguntas_treino]
    negativos = [r["pergunta"] for r in carregar_recusas_treino()]
    s_pos = np.array([buscador.buscar(q, 1)[0].score for q in positivos])
    s_neg = np.array([buscador.buscar(q, 1)[0].score for q in negativos])

    candidatos = np.unique(np.concatenate([s_pos, s_neg]))
    curva = []
    for t in candidatos:
        tpr = float((s_pos >= t).mean())  # respondíveis que passam
        tnr = float((s_neg < t).mean())   # fora do escopo barradas
        curva.append({"limiar": float(t), "tpr": tpr, "tnr": tnr, "acc_balanceada": (tpr + tnr) / 2})
    curva_df = pd.DataFrame(curva)
    melhor = curva_df.acc_balanceada.max()
    empates = curva_df[curva_df.acc_balanceada == melhor]
    limiar = float(empates.limiar.median())
    linha = curva_df.iloc[(curva_df.limiar - limiar).abs().argsort().iloc[0]]
    curva_df.to_csv(DIR_RESULTADOS / f"curva_limiar_{args.backend}.csv", index=False)

    # efeito do limiar no TESTE (só para relatar, não para escolher)
    top1_teste = {d["id"]: d["scores"][0] for d in detalhes}
    nc = [p for p in teste if p["deve_recusar"]]
    rp = [p for p in teste if not p["deve_recusar"]]
    barrou_nc = np.mean([top1_teste[p["id"]] < limiar for p in nc])
    passou_rp = np.mean([top1_teste[p["id"]] >= limiar for p in rp])

    print(f"[{args.backend}] limiar escolhido = {limiar:.4f} | treino: TPR={linha.tpr:.2f} TNR={linha.tnr:.2f}")
    print(f"[{args.backend}] no teste: {passou_rp:.0%} das respondíveis passam | {barrou_nc:.0%} das 'não consta' são barradas")

    dados = json.loads(ARQ_LIMIAR.read_text(encoding="utf-8")) if ARQ_LIMIAR.exists() else {}
    dados[args.backend] = {
        "limiar": round(limiar, 4),
        "criterio": "max acuracia balanceada no treino (mediana dos empates)",
        "treino": {"tpr": round(float(linha.tpr), 3), "tnr": round(float(linha.tnr), 3),
                   "n_pos": len(positivos), "n_neg": len(negativos),
                   "media_score_pos": round(float(s_pos.mean()), 4), "media_score_neg": round(float(s_neg.mean()), 4)},
        "teste": {"recall@1": round(r1, 3), "recall@3": round(r3, 3),
                  "respondiveis_passam": round(float(passou_rp), 3), "nao_consta_barradas": round(float(barrou_nc), 3)},
    }
    ARQ_LIMIAR.write_text(json.dumps(dados, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].hist(s_pos, bins=20, alpha=.7, label="com resposta na base")
        ax[0].hist(s_neg, bins=20, alpha=.7, label="fora do escopo")
        ax[0].axvline(limiar, color="k", ls="--", label=f"limiar {limiar:.3f}")
        ax[0].set_title(f"Score do 1º trecho (treino) — {args.backend}"); ax[0].legend()
        ax[1].plot(curva_df.limiar, curva_df.tpr, label="TPR (respondíveis passam)")
        ax[1].plot(curva_df.limiar, curva_df.tnr, label="TNR (fora do escopo barradas)")
        ax[1].plot(curva_df.limiar, curva_df.acc_balanceada, label="acurácia balanceada")
        ax[1].axvline(limiar, color="k", ls="--"); ax[1].set_title("Curva do limiar"); ax[1].legend()
        fig.tight_layout(); fig.savefig(DIR_RESULTADOS / f"limiar_{args.backend}.png", dpi=120)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
