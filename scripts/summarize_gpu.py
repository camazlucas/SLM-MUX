#!/usr/bin/env python3
"""Resume custo de cada coleta: tempo de GPU, tokens, VRAM, cache KV e energia.

Cruza o JSON de cada coleta (campo `timing`, gravado pelo slm-mux collect) com o log do
scripts/monitor_gpu.py, usando a janela start-end da coleta.

Uso:
  python scripts/summarize_gpu.py --log results/gpu_log.csv \\
      --runs results/gsm8k_val results/gsm8k_test results/gsm8k_test_greedy

Notas:
  - mem_used_mb fica quase constante: o vLLM reserva de saída a fração pedida da VRAM.
    O que diferencia modelos é o tamanho dos pesos (~2 bytes por parâmetro em bf16) e o pico do cache KV.
  - O tempo de carga do modelo aparece no log como o intervalo entre o início da memória ocupada
    e server_ready = 1; é reportado por bloco de carga, não por coleta.
"""
import argparse
import csv
import json
import os


def read_log(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if v not in ("", "None") else None) for k, v in r.items()})
    return rows


def stats(vals):
    vals = [v for v in vals if v is not None]
    return (max(vals), sum(vals) / len(vals)) if vals else (None, None)


def load_times(log):
    """Tempos de carga: de quando a VRAM passa de 1 GB até o /health responder."""
    out, t_up = [], None
    for prev, cur in zip(log, log[1:]):
        if t_up is None and (cur["mem_used_mb"] or 0) > 1024 and not cur["server_ready"]:
            t_up = cur["unix"]
        if t_up is not None and cur["server_ready"] and not prev["server_ready"]:
            out.append({"start_unix": t_up, "ready_unix": cur["unix"], "load_s": cur["unix"] - t_up})
            t_up = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="pastas com os JSON das coletas")
    ap.add_argument("--out", default="results/cost_summary.json")
    args = ap.parse_args()

    log = read_log(args.log)
    summary = []
    for d in args.runs:
        for name in sorted(os.listdir(d)):
            if not name.endswith(".json") or name.startswith(("search_", "mux_")):
                continue
            data = json.load(open(os.path.join(d, name), encoding="utf-8"))
            tm = data.get("timing")
            if not tm:
                continue
            win = [r for r in log if tm["start_unix"] <= r["unix"] <= tm["end_unix"]]
            energy_wh = sum((r["power_w"] or 0) for r in win) * (
                (win[-1]["unix"] - win[0]["unix"]) / max(1, len(win) - 1)) / 3600 if len(win) > 1 else None
            peak_mem, _ = stats([r["mem_used_mb"] for r in win])
            peak_kv, mean_kv = stats([r["kv_cache_pct"] for r in win])
            _, mean_util = stats([r["gpu_util_pct"] for r in win])
            tok = data.get("token_usage", {})
            n_q = data.get("sample_size", 0)
            lat = [s.get("latency_s", 0) for r in data["responses"] for s in r["samples"]]
            summary.append({
                "run": os.path.basename(d.rstrip("/")), "model": data["model"], "questions": n_q,
                "requests": tm["num_requests"], "gpu_wall_s": tm["wall_s"],
                "gpu_s_per_question": tm["wall_s"] / n_q if n_q else None,
                "prompt_tokens": tok.get("prompt_tokens"), "completion_tokens": tok.get("completion_tokens"),
                "completion_tokens_per_s": tok.get("completion_tokens", 0) / tm["wall_s"] if tm["wall_s"] else None,
                "mean_latency_s": sum(lat) / len(lat) if lat else None,
                "peak_mem_used_mb": peak_mem, "peak_kv_cache_pct": peak_kv, "mean_kv_cache_pct": mean_kv,
                "mean_gpu_util_pct": mean_util, "energy_wh": energy_wh, "log_points": len(win),
            })

    print(f"{'coleta':22s} {'modelo':42s} {'GPU (min)':>9s} {'s/questão':>9s} {'tok saída':>10s} "
          f"{'tok/s':>7s} {'pico KV':>7s} {'Wh':>6s}")
    for s in summary:
        fmt = lambda v, p=1: "-" if v is None else f"{v:.{p}f}"
        print(f"{s['run']:22s} {s['model']:42s} {fmt(s['gpu_wall_s']/60):>9s} {fmt(s['gpu_s_per_question'],2):>9s} "
              f"{s['completion_tokens'] or 0:>10d} {fmt(s['completion_tokens_per_s'],0):>7s} "
              f"{fmt(s['peak_kv_cache_pct'])+'%':>7s} {fmt(s['energy_wh'],0):>6s}")
    loads = load_times(log)
    for l in loads:
        print(f"carga de modelo: {l['load_s']:.0f}s")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"runs": summary, "model_loads": loads}, open(args.out, "w", encoding="utf-8"), indent=2)
    print("salvo em", args.out)


if __name__ == "__main__":
    main()
