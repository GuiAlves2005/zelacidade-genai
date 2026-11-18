"""Gera o dataset de ajuste fino (SFT) para o LoRA e verifica vazamento treino → teste.

Cada exemplo é exatamente o que o modelo verá em produção (prompt RAG com os 3 trechos recuperados)
e a saída-alvo é o JSON correto:
- perguntas_treino de cada trecho  → resposta com o texto do trecho, citando o id dele;
- recusas_treino (fora do escopo)  → recusa padrão com fontes vazias.

O que o LoRA aprende: o FORMATO (JSON válido), CITAR o trecho certo entre distratores e RECUSAR
quando o contexto não responde. Nenhuma pergunta do conjunto de teste entra no treino.

Uso:  python scripts/gerar_dataset_lora.py --backend e5
"""
from __future__ import annotations

import argparse
import json

import _caminho  # noqa: F401
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from zela.base import carregar_recusas_treino, carregar_teste, carregar_trechos, escrever_jsonl
from zela.busca import Buscador, Resultado
from zela.config import DIR_RESULTADOS, DIR_TREINO, Config
from zela.esquema import RespostaZela, resposta_de_recusa
from zela.prompts import mensagens_rag
from zela.texto import normalizar

RISCO_POR_CATEGORIA = {
    "emergencia": "alto",
    "buraco_via": "medio",
    "obstaculo_sinalizacao": "medio",
    "iluminacao": "medio",
    "arvore": "medio",
    "drenagem": "medio",
    "calcada": "baixo",
    "canais": "nao_se_aplica",
    "plataforma": "nao_se_aplica",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=Config().backend_busca, choices=["e5", "tfidf"])
    args = ap.parse_args()
    cfg = Config(backend_busca=args.backend)

    trechos = carregar_trechos()
    por_id = {t.id: t for t in trechos}
    buscador = Buscador(trechos, args.backend, cfg.modelo_emb)
    exemplos = []

    for t in trechos:
        for q in t.perguntas_treino:
            res = buscador.buscar(q, cfg.top_k)
            if t.id not in {r.trecho.id for r in res}:
                # garante que o trecho-ouro esteja no contexto (o LoRA aprende a citar, não a buscar)
                res = res[:-1] + [Resultado(t, 1.0)]
            alvo = RespostaZela(
                resposta=t.texto,
                categoria=t.categoria,
                risco=RISCO_POR_CATEGORIA[t.categoria],
                fontes=[t.id],
                recusou=False,
            )
            exemplos.append({"tipo": "resposta", "trecho": t.id,
                             "mensagens": mensagens_rag(q, res), "alvo": alvo.model_dump_json()})

    for r in carregar_recusas_treino():
        res = buscador.buscar(r["pergunta"], cfg.top_k)
        exemplos.append({"tipo": "recusa", "trecho": None,
                         "mensagens": mensagens_rag(r["pergunta"], res),
                         "alvo": resposta_de_recusa().model_dump_json()})

    escrever_jsonl(DIR_TREINO / "sft.jsonl", exemplos)
    n_resp = sum(e["tipo"] == "resposta" for e in exemplos)
    print(f"{len(exemplos)} exemplos gravados em treino/sft.jsonl ({n_resp} respostas, {len(exemplos) - n_resp} recusas)")

    # ---------------- checagem de vazamento ----------------
    perguntas_treino = [q for t in trechos for q in t.perguntas_treino] + [r["pergunta"] for r in carregar_recusas_treino()]
    perguntas_teste = [p["pergunta"] for p in carregar_teste()]
    iguais = set(map(normalizar, perguntas_treino)) & set(map(normalizar, perguntas_teste))
    assert not iguais, f"Vazamento! perguntas idênticas em treino e teste: {iguais}"

    vet = TfidfVectorizer(preprocessor=normalizar, ngram_range=(1, 2)).fit(perguntas_treino + perguntas_teste)
    sim = cosine_similarity(vet.transform(perguntas_teste), vet.transform(perguntas_treino))
    maximos = sim.max(axis=1)
    relatorio = {
        "perguntas_identicas": 0,
        "similaridade_max_media": round(float(maximos.mean()), 3),
        "similaridade_max_maior": round(float(maximos.max()), 3),
        "par_mais_parecido": {
            "teste": perguntas_teste[int(maximos.argmax())],
            "treino": perguntas_treino[int(sim[int(maximos.argmax())].argmax())],
        },
    }
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    (DIR_RESULTADOS / "vazamento.json").write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Checagem de vazamento:", json.dumps(relatorio, ensure_ascii=False))
    _ = por_id  # (mantido para inspeção interativa)


if __name__ == "__main__":
    main()
