#!/usr/bin/env python3
"""Grava, a cada intervalo, o estado da GPU e do servidor vLLM num CSV.

Rode ANTES de subir o servidor e deixe rodando durante a coleta (Ctrl+C para parar):
  python scripts/monitor_gpu.py --port 8000 --out results/gpu_log.csv

Colunas:
  unix, mem_used_mb, mem_total_mb, gpu_util_pct, power_w  -> nvidia-smi
  server_ready      -> 1 quando /health responde (a diferença até o início dá o tempo de carga)
  kv_cache_pct      -> uso do cache KV do vLLM (o que de fato varia; a VRAM reservada é fixa)
  running, waiting  -> requisições em execução e na fila
"""
import argparse
import csv
import os
import subprocess
import time
import urllib.request

FIELDS = ["unix", "mem_used_mb", "mem_total_mb", "gpu_util_pct", "power_w",
          "server_ready", "kv_cache_pct", "running", "waiting"]
# o nome da métrica do cache KV mudou entre versões do vLLM
KV_NAMES = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")


def nvidia_smi(gpu):
    out = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
    return [float(x) for x in out.split(",")] if out else [None] * 4


def http(url):
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except Exception:
        return None, ""


def vllm_metrics(text):
    kv = running = waiting = None
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        base = name.split("{")[0]
        try:
            v = float(value)
        except ValueError:
            continue
        if base in KV_NAMES:
            kv = 100 * v
        elif base == "vllm:num_requests_running":
            running = v
        elif base == "vllm:num_requests_waiting":
            waiting = v
    return kv, running, waiting


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--out", default="results/gpu_log.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new = not os.path.exists(args.out)
    base = f"http://localhost:{args.port}"
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(FIELDS)
        print(f"gravando em {args.out} a cada {args.interval}s (Ctrl+C para parar)")
        try:
            while True:
                t = time.time()
                mem_used, mem_total, util, power = nvidia_smi(args.gpu)
                status, _ = http(base + "/health")
                kv = running = waiting = None
                if status == 200:
                    _, text = http(base + "/metrics")
                    kv, running, waiting = vllm_metrics(text)
                w.writerow([round(t, 3), mem_used, mem_total, util, power,
                            int(status == 200), kv, running, waiting])
                f.flush()
                time.sleep(max(0.0, args.interval - (time.time() - t)))
        except KeyboardInterrupt:
            print("parado")


if __name__ == "__main__":
    main()
