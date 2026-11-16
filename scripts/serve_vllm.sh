#!/usr/bin/env bash
# Sobe UM modelo no vLLM (na 3090 cabe um modelo de até 10B em bf16 por vez),
# na revisão e com a fração de memória definidas em configs/model_pool.yaml.
# Uso: bash scripts/serve_vllm.sh <modelo_hf> [porta] [max_model_len]
#   bash scripts/serve_vllm.sh Qwen/Qwen2.5-7B-Instruct
#   bash scripts/serve_vllm.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-7B 8000 16384
set -euo pipefail
MODEL="${1:?informe o modelo do Hugging Face}"
PORT="${2:-8000}"
MAXLEN="${3:-4096}"
POOL="$(dirname "$0")/../configs/model_pool.yaml"

read -r REVISION GPU_MEM < <(python - "$POOL" "$MODEL" <<'PY'
import sys, yaml
pool, model = sys.argv[1], sys.argv[2]
data = yaml.safe_load(open(pool, encoding="utf-8"))
for m in data.get("models") or []:
    if m["id"] == model:
        print(m.get("revision", "main"), m.get("gpu_memory_utilization", 0.90))
        break
else:
    sys.exit(f"{model} não está em {pool}")
PY
)

echo "Servindo $MODEL @ $REVISION na porta $PORT (max-model-len $MAXLEN, gpu-mem $GPU_MEM)"
vllm serve "$MODEL" \
  --revision "$REVISION" \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$GPU_MEM" \
  --seed 0
