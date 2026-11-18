"""Roda as três configurações no conjunto de teste e gera o placar.

Saídas em avaliacao/:
  resultados/execucoes_<cfg>.jsonl  — tudo o que o modelo gerou (para auditoria)
  resultados/correcao.csv           — correção pergunta a pergunta
  resultados/candidatos_erro.csv    — falhas candidatas à análise de erros (notebook 06)
  placar.csv / placar.md            — placar final

Uso:  python scripts/avaliar.py                    # A, B e C
      python scripts/avaliar.py --configs A B      # antes de treinar o LoRA
"""
from __future__ import annotations

import argparse
import json

import _caminho  # noqa: F401
import pandas as pd
from tqdm import tqdm

from zela.avaliacao import corrigir, placar, placar_markdown
from zela.base import carregar_teste, carregar_trechos, escrever_jsonl
from zela.busca import Buscador
from zela.config import DIR_AVALIACAO, DIR_RESULTADOS, Config
from zela.llm import GeradorHF
from zela.pipeline import DESCRICAO, PipelineZela


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    args = ap.parse_args()
    cfg = Config()
    teste = carregar_teste()
    buscador = Buscador(carregar_trechos(), cfg.backend_busca, cfg.modelo_emb)
    print(f"busca={cfg.backend_busca} limiar={cfg.limiar} modelo={cfg.modelo_llm}")

    base_gerador = None
    linhas = []
    for c in args.configs:
        if c == "C":
            if base_gerador is not None:  # libera a VRAM do modelo base (GPU de 8 GB não comporta dois)
                import gc

                import torch

                del base_gerador, gerador, pipe  # o pipeline anterior também segura o modelo
                base_gerador = None
                gc.collect()
                torch.cuda.empty_cache()
            gerador = GeradorHF(cfg.modelo_llm, cfg.adaptador_lora, cfg.max_novos_tokens)
        else:
            base_gerador = base_gerador or GeradorHF(cfg.modelo_llm, None, cfg.max_novos_tokens)
            gerador = base_gerador
        pipe = PipelineZela(c, gerador, buscador, cfg.top_k, cfg.limiar)
        execs = []
        for p in tqdm(teste, desc=f"config {c}"):
            e = pipe.responder(p["pergunta"]).para_dict() | {"id": p["id"]}
            execs.append(e)
            linhas.append(corrigir(e, p))
        escrever_jsonl(DIR_RESULTADOS / f"execucoes_{c}.jsonl", execs)

    df = pd.DataFrame(linhas)
    df.to_csv(DIR_RESULTADOS / "correcao.csv", index=False)
    tab = placar(linhas)
    tab.to_csv(DIR_AVALIACAO / "placar.csv")
    md = placar_markdown(tab, DESCRICAO)
    (DIR_AVALIACAO / "placar.md").write_text(
        f"# Placar — {len(teste)} perguntas (40 respondíveis + 10 'não consta')\n\n"
        f"Modelo: `{cfg.modelo_llm}` · Busca: `{cfg.backend_busca}` · top-k={cfg.top_k} · limiar={cfg.limiar}\n\n{md}",
        encoding="utf-8",
    )
    print(md)

    falhas = df[(~df.acerto & ~df.deve_recusar) | (df.deve_recusar & ~df.recusa_correta) | ~df.json_valido]
    falhas.to_csv(DIR_RESULTADOS / "candidatos_erro.csv", index=False)
    print(f"{len(falhas)} falhas listadas em avaliacao/resultados/candidatos_erro.csv")
    (DIR_RESULTADOS / "meta.json").write_text(json.dumps(vars(cfg) | {"adaptador_lora": str(cfg.adaptador_lora)},
                                                         default=str, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
