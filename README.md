# Zela — assistente de IA local da ZelaCidade

> **Global Solution 2026.2 · CitizenTech · FIAP Engenharia de Software 4º ano**
> Disciplina: **Generative AI for Engineering** — *Treinamento de uma LLM local*

A **Zela** é o componente de **Inteligência** da ZelaCidade. Ela orienta o morador de São Paulo sobre problemas de
**segurança e mobilidade** em espaços públicos — buraco na rua, calçada quebrada, poste apagado, árvore com risco,
bueiro entupido, obstáculo sem sinalização — respondendo **com base em fontes oficiais**, **citando a fonte** e
**recusando** quando a informação não está na base. Tudo roda **localmente** (LLM de 1,5B + LoRA), sem enviar dados
do cidadão para APIs externas.

📄 **Relatório técnico completo:** [`RELATORIO.md`](RELATORIO.md)

🎥 **Vídeo de demonstração:** https://youtu.be/cUMB1Jk7_Eg

## Integrantes

| Nome completo | RM |
|---|---|
| Bruno Venturi Lopes Vieira | 99431 |
| Guilherme Alves de Lima | 550433 |
| Leonardo de Oliveira Ruiz | 98901 |

## Ficha Técnica do MVP

| Campo | Conteúdo |
|---|---|
| **Nome do produto** | ZelaCidade (componente de IA: **Zela**) |
| **Cidadão e território** | Moradores de São Paulo — piloto na Zona Norte (Santana/Tucuruvi) — incluindo pessoas com deficiência, idosos e pedestres em geral |
| **Problema, em uma frase** | O morador não sabe se um problema na rua é perigoso, de quem é a responsabilidade e por qual canal oficial pedir o conserto — e informações erradas o fazem desistir ou se expor a risco. |
| **Proposta de valor** | Orientação imediata, em linguagem simples e verificável (com a fonte oficial citada), integrada ao registro da ocorrência no mapa colaborativo. |
| **Arquitetura: componentes** | Base de conhecimento (47 trechos / 19 fontes) → busca semântica E5 → limiar de recusa → LLM local Qwen2.5-1.5B-Instruct + adaptador LoRA → validação JSON (Pydantic) → API FastAPI `/perguntar` consumida pelo MVP |
| **Fontes de dados** | Páginas oficiais da Prefeitura de SP (SP156, subprefeituras, SP Regula), Decreto 59.671/2020, Lei 15.442/2011, CTB art. 94, orientações da Defesa Civil (SEMIL-SP) — ver [`base/tabela_base.md`](base/tabela_base.md) |
| **Modelos de IA utilizados** | `Qwen/Qwen2.5-1.5B-Instruct` (geração) · LoRA r=16 próprio (especialização) · `intfloat/multilingual-e5-small` (embeddings) |
| **URL do MVP e repositório** | MVP: https://app-zelacidade-gs-99431.azurewebsites.net/painel · Repositório: https://github.com/GuiAlves2005/zelacidade-genai |
| **Indicadores de impacto** | % de respostas corretas e com fonte; taxa de alucinação; % de perguntas fora da base recusadas; nº de moradores orientados; % de ocorrências registradas também no SP156 após a orientação |
| **ODS conectados** | **ODS 11** (cidades sustentáveis — 11.2 transporte/mobilidade segura, 11.7 espaços públicos seguros e acessíveis) · **ODS 9** (infraestrutura resiliente) · **ODS 10** (inclusão — acessibilidade de calçadas) |

## Estrutura do repositório

```
base/            trechos.jsonl (47 trechos), fontes.csv (19 fontes + licença), tabela_base.md
avaliacao/       perguntas_teste.jsonl (50 perguntas c/ gabarito, 10 "não consta"), limiar.json, placar.md, resultados/
notebooks/       01 base · 02 busca/recall · 03 RAG · 04 LoRA · 05 placar · 06 erros  (com saídas salvas)
src/zela/        código: busca, prompts, esquema JSON, pipeline A/B/C, avaliação, LGPD
scripts/         calibrar_limiar.py · gerar_dataset_lora.py · treinar_lora.py · avaliar.py
api/main.py      API FastAPI consumida pelo MVP
treino/          recusas_treino.jsonl, sft.jsonl (gerado)
tests/           testes que rodam sem GPU
```

## Como reproduzir

Desenvolvido e testado em Windows 10 + RTX 5060 Ti 8 GB + Python 3.11; em Linux/Colab use `requirements.txt` (não testado pela equipe).

```bash
# 1) ambiente
python -m venv .venv
.venv\Scripts\activate                 # Linux/Mac: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX série 50; outras GPUs: cu121/cu124
pip install -r requirements.txt
# ou, para as MESMAS versões usadas nos resultados (inclui torch 2.11.0+cu128 e transformers 5.22.2):
# pip install -r requirements-lock.txt   # lock do WINDOWS (inclui pywinpty); em Linux/Mac use o requirements.txt
copy .env.example .env                  # Linux/Mac: cp .env.example .env

# 2) testes rápidos (sem GPU)
pytest -q

# 3) pipeline completo (ou rode os notebooks 01 → 06 na ordem)
python scripts/calibrar_limiar.py --backend tfidf   # linha de base da busca
python scripts/calibrar_limiar.py --backend e5      # recall@1, recall@3 e limiar
python scripts/gerar_dataset_lora.py --backend e5   # 119 exemplos SFT + checagem de vazamento
python scripts/treinar_lora.py                      # ~5-15 min em GPU de 8 GB
python scripts/avaliar.py --configs A B C           # placar em avaliacao/placar.md

# 4) API para o MVP
uvicorn api.main:app --host 0.0.0.0 --port 8001
```

Na primeira execução os modelos são baixados do Hugging Face (~3,5 GB no total).

### Usando a API

```bash
curl -X POST http://localhost:8001/perguntar -H "Content-Type: application/json" \
     -d "{\"pergunta\": \"Qual a largura mínima que deve ficar livre para os pedestres numa calçada em São Paulo?\"}"
```

> No Windows, o `curl` do Git Bash/PowerShell pode corromper os acentos do corpo (a API responde
> "There was an error parsing the body"). Nesse caso, envie o JSON em UTF-8 (ex.: com Python `urllib`/`requests`).

Resposta real (pergunta Q16 do teste; config C na RTX 5060 Ti; 1ª chamada após subir a API, por isso a latência inclui o
aquecimento — ver medições em [Resultados](#resultados)):

```json
{
  "saida": {
    "resposta": "A faixa livre é a parte da calçada destinada exclusivamente à circulação de pedestres. Ela deve ter no mínimo 1,20 m de largura, inclinação transversal de no máximo 3% e ocupar pelo menos metade da largura da calçada quando esta tiver mais de 2,40 m.",
    "categoria": "calcada",
    "risco": "medio",
    "fontes": [
      "T015"
    ],
    "recusou": false
  },
  "fontes_detalhadas": [
    {
      "id": "T015",
      "titulo": "Faixa livre: largura e inclinação",
      "orgao": "Prefeitura de Sao Paulo",
      "url": "https://saopaulo.sp.leg.br/iah/fulltext/decretos/D59671.pdf"
    }
  ],
  "config": "C",
  "recusa_por_limiar": false,
  "latencia_s": 13.474,
  "aviso": "Orientação informativa. A ZelaCidade não substitui os canais oficiais da Prefeitura (SP156)."
}
```

## Resultados

Placar nas 50 perguntas de teste (40 respondíveis + 10 "não consta"), de [`avaliacao/placar.md`](avaliacao/placar.md)
— busca E5, top-k=3, limiar 0,8511; latência da avaliação em lote na GPU:

| Config | Acurácia (respondíveis) | Recusa correta ("não consta") | Recusa indevida | Alucinação | JSON válido | Citação correta | Latência média |
|---|---|---|---|---|---|---|---|
| A — sem contexto | 10,0% | 0,0% | 0,0% | 64,0% | 72,0% | 5,0% | 3,15 s |
| B — RAG simples | 20,0% | 70,0% | 0,0% | 8,0% | 38,0% | 25,0% | 3,39 s |
| C — RAG + LoRA | 87,5% | 80,0% | 0,0% | 14,0% | 100,0% | 92,5% | 6,33 s |

A alucinação menor da B não a torna mais confiável: 62% das saídas dela nem foram JSON válido (o que não conta como
alucinação). A **API usa a config C**. Busca: E5 com recall@1 = 0,825 e recall@3 = 1,0 (TF-IDF: 0,725 e 0,800).
Análise completa no [`RELATORIO.md`](RELATORIO.md).

Latência da API (config C; mediana das chamadas 2 e 3, descartando a 1ª como aquecimento —
[`avaliacao/resultados/exemplos_api.json`](avaliacao/resultados/exemplos_api.json)):

| Pergunta | GPU RTX 5060 Ti (bf16) | CPU Ryzen 5 5600X (fp32) | Resultado |
|---|---|---|---|
| "Tem um buraco enorme na minha rua, o que eu faço?" | 6,49 s | 33,60 s | respondeu (T006) |
| "Qual o valor da multa por calçada irregular?" | 4,42 s | 25,92 s | recusou (modelo) |
| "Como faço para renovar minha CNH?" | 0,008 s | 0,014 s | recusou (limiar) |

Exemplo real de recusa — "Qual o valor da multa por calçada irregular?" (o valor não está na base; GPU, 2ª chamada):

```json
{
  "saida": {
    "resposta": "Não consta na base de conhecimento da ZelaCidade. Para esse assunto, consulte os canais oficiais da Prefeitura (SP156).",
    "categoria": "fora_do_escopo",
    "risco": "nao_se_aplica",
    "fontes": [],
    "recusou": true
  },
  "fontes_detalhadas": [],
  "config": "C",
  "recusa_por_limiar": false,
  "latencia_s": 4.395,
  "aviso": "Orientação informativa. A ZelaCidade não substitui os canais oficiais da Prefeitura (SP156)."
}
```

## Segurança, explicabilidade e LGPD

- **Local:** modelo e embeddings rodam na máquina/servidor da equipe; a pergunta do cidadão não vai para terceiros.
- **Minimização (LGPD):** CPF, e-mail, telefone e placa são mascarados antes de chegar ao modelo e ao log (`src/zela/privacidade.py`). Nada é persistido além de um log técnico.
- **Explicável:** toda resposta traz os ids dos trechos usados e a API devolve título, órgão e URL de cada fonte.
- **Recusa em duas camadas:** limiar de similaridade (calibrado no treino) + instrução/LoRA para recusar quando o contexto não responde.
- **Pessoa no circuito:** a Zela orienta; ela não abre protocolo nem decide nada pelo cidadão. Em risco alto, prioriza Defesa Civil (199) e Bombeiros (193).
- **Acesso protegido:** com `ZELA_API_KEY` definida no `.env`, o `/perguntar` exige o cabeçalho `X-API-Key` (o `/saude` fica aberto).
- **Sem credenciais no código:** a chave fica só no `.env` (fora do Git).
- **Modo offline do MVP:** `python scripts/exportar_trechos.py` gera `base/trechos.json` (texto, título, órgão, link e categoria de cada trecho).
