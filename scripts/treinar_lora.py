"""Treina o adaptador LoRA (config C) a partir de treino/sft.jsonl.

Pensado para GPU de 8 GB (ex.: RTX 5060 Ti) com modelo de ~1,5B parâmetros em bf16, sem quantização
(evita depender de bitsandbytes no Windows). Só os tokens da resposta entram na loss: o prompt é mascarado.

Uso:  python scripts/treinar_lora.py                 # padrão: 3 épocas
      python scripts/treinar_lora.py --epocas 4 --rank 16
"""
from __future__ import annotations

import argparse
import json
import random

import _caminho  # noqa: F401
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from zela.base import ler_jsonl
from zela.config import DIR_RESULTADOS, DIR_TREINO, Config


class DatasetSFT(torch.utils.data.Dataset):
    def __init__(self, exemplos: list[dict], tok, max_len: int = 1536):
        self.itens = []
        for e in exemplos:
            prompt = tok.apply_chat_template(e["mensagens"], tokenize=False, add_generation_prompt=True)
            ids_prompt = tok(prompt, add_special_tokens=False)["input_ids"]
            ids_alvo = tok(e["alvo"] + tok.eos_token, add_special_tokens=False)["input_ids"]
            ids = (ids_prompt + ids_alvo)[:max_len]
            rotulos = ([-100] * len(ids_prompt) + ids_alvo)[:max_len]
            self.itens.append({"input_ids": ids, "labels": rotulos})

    def __len__(self):
        return len(self.itens)

    def __getitem__(self, i):
        return self.itens[i]


def colar(lote, pad_id):
    n = max(len(x["input_ids"]) for x in lote)
    ids = torch.full((len(lote), n), pad_id)
    lab = torch.full((len(lote), n), -100)
    att = torch.zeros((len(lote), n), dtype=torch.long)
    for i, x in enumerate(lote):
        k = len(x["input_ids"])
        ids[i, :k] = torch.tensor(x["input_ids"])
        lab[i, :k] = torch.tensor(x["labels"])
        att[i, :k] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def main() -> None:
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=cfg.modelo_llm)
    ap.add_argument("--saida", default=str(cfg.adaptador_lora))
    ap.add_argument("--epocas", type=float, default=3)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    random.seed(cfg.semente)
    torch.manual_seed(cfg.semente)

    exemplos = ler_jsonl(DIR_TREINO / "sft.jsonl")
    random.shuffle(exemplos)
    n_val = max(8, len(exemplos) // 10)
    val, trein = exemplos[:n_val], exemplos[n_val:]

    tok = AutoTokenizer.from_pretrained(args.modelo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    modelo = AutoModelForCausalLM.from_pretrained(
        args.modelo, torch_dtype=torch.bfloat16 if bf16 else torch.float32, device_map="auto"
    )
    modelo.gradient_checkpointing_enable()
    modelo.enable_input_require_grads()

    lora = LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    modelo = get_peft_model(modelo, lora)
    modelo.print_trainable_parameters()

    treino_args = TrainingArguments(
        output_dir=str(DIR_TREINO / "checkpoints"),
        num_train_epochs=args.epocas,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,  # transformers 5.x: float em [0,1) = fração dos passos (antigo warmup_ratio)
        logging_steps=2,
        eval_strategy="epoch",
        save_strategy="no",
        bf16=bf16,
        report_to=[],
        label_names=["labels"],
        seed=cfg.semente,
    )
    trainer = Trainer(
        model=modelo,
        args=treino_args,
        train_dataset=DatasetSFT(trein, tok),
        eval_dataset=DatasetSFT(val, tok),
        data_collator=lambda lote: colar(lote, tok.pad_token_id),
    )
    trainer.train()
    modelo.save_pretrained(args.saida)
    tok.save_pretrained(args.saida)

    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    (DIR_RESULTADOS / "lora_historico.json").write_text(
        json.dumps({"args": vars(args), "n_treino": len(trein), "n_val": len(val), "log": trainer.state.log_history},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Adaptador salvo em {args.saida}")


if __name__ == "__main__":
    main()
