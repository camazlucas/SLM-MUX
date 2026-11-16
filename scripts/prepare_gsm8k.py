#!/usr/bin/env python3
"""Prepara o GSM8K no formato esperado pelo SLM-MUX.

Gera:
  data/gsm8k_test.json    -> as 1.319 questões de teste
  data/gsm8k_val500.json  -> 500 questões sorteadas do treino (validação, como no paper)

Cada item: {"question": ..., "answer": "<só o número, sem vírgula>", "solution": "<solução original>"}

O campo "answer" original do Hugging Face traz a solução inteira terminando em "#### N";
aqui guardamos só N, porque o `slm-mux offline` original compara a referência sem normalizar.

Uso:  python scripts/prepare_gsm8k.py [--out-dir data] [--val-size 500] [--seed 42]
Requer: pip install datasets
"""
import argparse
import json
import os
import random

from datasets import load_dataset


def convert(ex):
    final = ex["answer"].split("####")[-1].strip().replace(",", "")
    return {"question": ex["question"], "answer": final, "solution": ex["answer"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_dataset("openai/gsm8k", "main")
    test = [convert(e) for e in ds["test"]]
    train = [convert(e) for e in ds["train"]]
    val = random.Random(args.seed).sample(train, args.val_size)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, items in (("gsm8k_test.json", test), (f"gsm8k_val{args.val_size}.json", val)):
        path = os.path.join(args.out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=1)
        print(f"{path}: {len(items)} questões")


if __name__ == "__main__":
    main()
