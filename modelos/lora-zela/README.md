---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-1.5B-Instruct
- lora
- transformers
---

# lora-zela — adaptador LoRA da Zela (config C)

Adaptador LoRA treinado pela equipe da ZelaCidade (Global Solution FIAP 2026.2, Generative AI for Engineering).
Repositório: https://github.com/GuiAlves2005/zelacidade-genai

**Finalidade:** especializar o `Qwen/Qwen2.5-1.5B-Instruct` no prompt RAG da Zela, para (1) responder **só em JSON**
no formato `{resposta, categoria, risco, fontes, recusou}`, (2) **citar** o id do trecho que sustenta a resposta e
(3) **recusar** quando o contexto não responde. Ele não acrescenta conhecimento: a informação vem dos trechos
recuperados pela busca E5 (`base/trechos.jsonl`).

## Treino

| Item | Valor |
|---|---|
| Modelo base | `Qwen/Qwen2.5-1.5B-Instruct` (bf16, sem quantização) |
| LoRA | r = 16, alpha = 32, dropout = 0,05, bias = none |
| Módulos-alvo | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Parâmetros treináveis | 18.464.768 (1,18% de 1.562.179.072) |
| Dados | 119 exemplos SFT (`treino/sft.jsonl`): 94 respostas + 25 recusas → 108 treino / 11 validação |
| Otimização | 3 épocas, lr 2e-4 cosseno, aquecimento 5%, batch 1 × acumulação 8, loss só nos tokens da resposta |
| Hardware / tempo | RTX 5060 Ti 8 GB, 334 s |
| Loss | treino 0,225 → 0,0065; validação 0,0298 → 0,0217 → 0,0174 (épocas 1–3) |

Script: `scripts/treinar_lora.py` · histórico: `avaliacao/resultados/lora_historico.json` · curva:
`avaliacao/resultados/curva_loss_lora.png`. Nenhuma pergunta do teste entra no treino
(`avaliacao/resultados/vazamento.json`).

## Resultado (50 perguntas de teste, `avaliacao/placar.md`)

Config C (busca E5 + este adaptador): JSON válido 100%, acurácia nas respondíveis 87,5%, citação correta 92,5%,
recusa correta nas "não consta" 80%, alucinação 14%. Limitação conhecida: a resposta tende a ser a cópia literal de
um trecho (era o alvo do SFT) — às vezes do trecho errado. Detalhes no `RELATORIO.md`.

## Como carregar

Pelo código do projeto (mesmo prompt usado no treino), rodando da raiz do repositório:

```python
import sys; sys.path.insert(0, "src")
from zela.config import Config
from zela.llm import GeradorHF

cfg = Config()
gerador = GeradorHF(cfg.modelo_llm, cfg.adaptador_lora)   # carrega o base e aplica modelos/lora-zela
```

Ou direto com PEFT:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype="auto", device_map="auto")
tok = AutoTokenizer.from_pretrained("modelos/lora-zela")
modelo = PeftModel.from_pretrained(base, "modelos/lora-zela")
# O adaptador espera o prompt RAG de src/zela/prompts.py (mensagens_rag); fora dele o comportamento não foi avaliado.
```
