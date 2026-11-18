# Placar — 50 perguntas (40 respondíveis + 10 'não consta')

Modelo: `Qwen/Qwen2.5-1.5B-Instruct` · Busca: `e5` · top-k=3 · limiar=0.8511

| Config | acuracia_respondiveis | recusa_correta_nao_consta | recusa_indevida | taxa_alucinacao | json_valido | citacao_correta | citacao_fantasma | latencia_media_s | n_perguntas |
|---|---|---|---|---|---|---|---|---|---|
| A — Modelo sem contexto | 10.0% | 0.0% | 0.0% | 64.0% | 72.0% | 5.0% | 32.0% | 3.15 | 50 |
| B — RAG simples (busca + limiar + citação) | 20.0% | 70.0% | 0.0% | 8.0% | 38.0% | 25.0% | 0.0% | 3.39 | 50 |
| C — RAG + LoRA (modelo especializado) | 87.5% | 80.0% | 0.0% | 14.0% | 100.0% | 92.5% | 0.0% | 6.33 | 50 |
