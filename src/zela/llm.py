"""Geradores de texto (LLM local).

GeradorHF     : transformers rodando na GPU/CPU local, com adaptador LoRA opcional (config C).
GeradorOllama : alternativa leve para servir a API pelo Ollama (sem LoRA).
GeradorFixo   : stub determinístico usado nos testes automatizados (não carrega modelo).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Protocol


class Gerador(Protocol):
    nome: str

    def gerar(self, mensagens: list[dict]) -> str: ...


class GeradorHF:
    def __init__(self, modelo: str, adaptador_lora: Path | None = None, max_novos_tokens: int = 350):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(modelo)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )
        # Na CPU, sem device_map: "auto" mandava camadas para o disco (meta) quando a RAM livre era curta.
        self.modelo = AutoModelForCausalLM.from_pretrained(
            modelo, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None
        )
        self.nome = modelo
        if adaptador_lora is not None:
            from peft import PeftModel

            if not Path(adaptador_lora).exists():
                raise FileNotFoundError(
                    f"Adaptador LoRA não encontrado em {adaptador_lora}. Rode scripts/treinar_lora.py antes."
                )
            self.modelo = PeftModel.from_pretrained(self.modelo, str(adaptador_lora))
            self.nome = f"{modelo}+LoRA"
        self.modelo.eval()
        self.max_novos_tokens = max_novos_tokens

    def gerar(self, mensagens: list[dict]) -> str:
        prompt = self.tok.apply_chat_template(mensagens, tokenize=False, add_generation_prompt=True)
        entrada = self.tok(prompt, return_tensors="pt").to(self.modelo.device)
        with self.torch.no_grad():
            saida = self.modelo.generate(
                **entrada,
                max_new_tokens=self.max_novos_tokens,
                do_sample=False,  # decodificação gulosa = resultados reprodutíveis
                pad_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(saida[0][entrada["input_ids"].shape[1] :], skip_special_tokens=True)


class GeradorOllama:
    def __init__(self, modelo: str = "qwen2.5:1.5b", url: str = "http://localhost:11434"):
        self.nome = f"ollama:{modelo}"
        self.modelo = modelo
        self.url = url.rstrip("/")

    def gerar(self, mensagens: list[dict]) -> str:
        import urllib.request

        corpo = json.dumps(
            {"model": self.modelo, "messages": mensagens, "stream": False, "format": "json", "options": {"temperature": 0}}
        ).encode()
        req = urllib.request.Request(f"{self.url}/api/chat", data=corpo, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["message"]["content"]


class GeradorFixo:
    """Responde com uma função Python; útil para testar o pipeline sem GPU."""

    def __init__(self, funcao: Callable[[list[dict]], str], nome: str = "fixo"):
        self.funcao = funcao
        self.nome = nome

    def gerar(self, mensagens: list[dict]) -> str:
        return self.funcao(mensagens)
