# Reprodução do SLM-MUX no GSM8K (≤ 8B, uma RTX 3090)

Objetivo: aprender o ciclo de reprodução com um método sem treino e montar a
infraestrutura (dados, extração, tokens) que será reaproveitada nos Cenários A e B.

- Paper: SLM-MUX (Wang et al., ICLR 2026, arXiv 2510.05077)
- Modelos: `meta-llama/Llama-3.1-8B-Instruct` + `Qwen/Qwen2.5-7B-Instruct` (bf16, um por vez)
- Benchmark: GSM8K teste (1.319) + validação de 500 questões do treino
- Configuração do paper: k = 3 amostras a temperatura 0,3 (Seção 4.2); desempate pela acurácia
  na validação, coletada com 3 × 3 gerações a temperatura 0,5 (Seção 4.3); modelo sozinho em
  decodificação greedy (Apêndice B.2)

Documentação completa (objetivo, decisões, desvios, resultados):
[Reprodução 1 — SLM-MUX no GSM8K](https://claude.ai/artifact/M4FoKEE5tHzZXscWFejRYA).

O paper não reporta essa dupla no GSM8K: o alvo é a afirmação qualitativa
(MUX > melhor modelo sozinho e > Single-Best-SC). O número publicado para esta
dupla está no MATH-500 (Tabela 2) — próximo passo.

## Pool de modelos

`configs/model_pool.yaml` lista 12 modelos de até 10B com a revisão do Hugging Face fixada
(conferida em 2026-11-16) e o campo `source` dizendo de onde cada modelo veio (paper do
SLM-MUX, ToolOrchestra ou comparações de SLMs de 2026). `scripts/serve_vllm.sh` lê esse arquivo e sobe o modelo nessa revisão,
com a fração de memória da GPU definida para ele.

| Grupo | Modelos |
|---|---|
| paper (Reprodução 1) | Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct |
| especialista | Qwen2.5-Math-7B-Instruct, Qwen2.5-Coder-7B-Instruct |
| nova geração | Qwen3-8B (thinking ligado por padrão) |
| raciocínio | DeepSeek-R1-Distill-Qwen-7B (respostas longas: suba com `max_model_len` maior) |
| pequeno | Phi-4-mini-instruct (3,8B), Qwen3.5-4B, Gemma 4 E4B (~4,5B efetivos) |
| teto 10B | Qwen3.5-9B, Ministral-3-8B (BF16), Granite 4.2 8B |

O Qwen3.5-9B (~19,3 GB de pesos em bf16) deixa pouco espaço para o cache KV na 3090:
sobe com `gpu_memory_utilization: 0.95` e aguenta menos requisições simultâneas. Se der falta de
memória, reduza `num_workers` no config ou o `max_model_len`.

Para acrescentar um modelo: inclua no `model_pool.yaml`, nos dois configs `vllm_3090_gsm8k_*.yaml`
e no `slurm/vllm_servers.json`.

## Onde roda cada etapa

| Etapa | Onde |
|---|---|
| Preparar dados, combinar (offline), analisar | qualquer máquina (CPU) |
| Coletar amostras (`slm-mux collect`) | máquina do laboratório com a 3090 (Linux) |

## Passo a passo (máquina do laboratório)

```bash
git clone https://github.com/camazlucas/SLM-MUX.git && cd SLM-MUX
git checkout repro-gsm8k-8b
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]" vllm

# Llama é gated: aceite a licença na página do modelo no Hugging Face antes
huggingface-cli login

python scripts/prepare_gsm8k.py                 # -> data/gsm8k_test.json, data/gsm8k_val500.json
mkdir -p slurm && cp configs/vllm_servers.example.json slurm/vllm_servers.json
```

Para cada modelo (use `tmux`: uma janela serve, outra coleta):

```bash
# janela 1 (sobe na revisão fixada; 3º argumento opcional = max_model_len)
bash scripts/serve_vllm.sh Qwen/Qwen2.5-7B-Instruct

# janela 2 (quando o servidor disser que está pronto)
slm-mux collect --config configs/vllm_3090_gsm8k_val.yaml         --models Qwen/Qwen2.5-7B-Instruct
slm-mux collect --config configs/vllm_3090_gsm8k_test.yaml        --models Qwen/Qwen2.5-7B-Instruct
slm-mux collect --config configs/vllm_3090_gsm8k_test_greedy.yaml --models Qwen/Qwen2.5-7B-Instruct

# derrube o servidor (Ctrl+C na janela 1) e repita com meta-llama/Llama-3.1-8B-Instruct
```

Saída: `results/gsm8k_val/` (9 amostras a 0,5), `results/gsm8k_test/` (5 amostras a 0,3) e
`results/gsm8k_test_greedy/` (1 amostra greedy), um JSON por modelo, com resposta bruta,
resposta extraída e tokens de cada amostra.

Antes da coleta completa, faça um teste de fumaça: `sample_size: 20` no config de validação,
colete com um modelo e confira nas respostas brutas se o `####` aparece. Depois volte para `-1`.

## Combinação (Algoritmo 1) e baselines

```bash
python scripts/offline_mux_alg1.py \
  --test-dir results/gsm8k_test --val-dir results/gsm8k_val \
  --greedy-dir results/gsm8k_test_greedy \
  --models meta-llama/Llama-3.1-8B-Instruct,Qwen/Qwen2.5-7B-Instruct \
  --samples 3 --trials 10
```

Reporta, com média ± desvio em 10 rodadas de Monte Carlo (subamostrando 3 das 5 amostras):
`mux`, `oracle`, `greedy::<modelo>` (Single-Best do paper), `single::<modelo>`, `sc3::<modelo>`
(Single-Best-SC) e tokens por questão de cada método.

## Desvios em relação ao repositório original (registrar no relatório)

1. **Combinação offline própria.** `slm-mux offline` faz votação conjunta das amostras de
   todos os modelos (não é o Algoritmo 1), desempata com a acurácia do *teste* e compara a
   referência sem normalizar. `scripts/offline_mux_alg1.py` usa a `ConsistencyConfidence` e o
   `ValidationAccuracyTieBreaker` do próprio repositório, com acurácia da validação.
   O `slm-mux run` implementa o Algoritmo 1, mas exige os dois modelos servidos ao mesmo tempo.
2. **Extrator do GSM8K** (`src/slm_mux/extractors/gsm.py`): aceita separador de milhar e `$`
   (o original lia `#### 1,000` como `1`) e trata `18.00` como `18`.
3. **Dados**: o repositório não traz script de preparo; `scripts/prepare_gsm8k.py` grava só o
   número final em `answer`.
4. **Acurácia de validação** = acurácia média das 9 amostras (3 × 3 a temperatura 0,5) nas 500
   questões. O paper descreve a coleta (Seção 4.3), mas não dá a fórmula.
5. **`single`** (uma amostra a 0,3) é um extra; o baseline do paper é o `greedy`.

## Próximos passos

- MATH-500 com a mesma dupla, para comparar com a Tabela 2.
- Busca de seleção de modelos (`slm-mux search`) num pool ≤ 8B.
- LLM-Debate com os mesmos SLMs (subcenário "pequeno time com discussão").
