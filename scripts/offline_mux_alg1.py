#!/usr/bin/env python3
"""SLM-MUX offline seguindo o Algoritmo 1 do paper (arXiv 2510.05077).

Por que não usar `slm-mux offline`: aquele comando junta as amostras de todos
os modelos numa votação única, usa como "acurácia de validação" a acurácia do
próprio conjunto de teste e compara a referência sem normalizar. Aqui:

  1. Para cada questão e cada modelo, sorteia k das amostras coletadas.
  2. Confiança do modelo = frequência da resposta majoritária dele
     (ConsistencyConfidence do próprio repositório).
  3. Vence o modelo mais confiante; empate desfeito pela acurácia na
     VALIDAÇÃO (ValidationAccuracyTieBreaker do repositório).
  4. Repete por T rodadas de Monte Carlo e reporta média ± desvio.

Também calcula, com as mesmas amostras sorteadas:
  - single: acurácia de 1 amostra de cada modelo (temperatura 0,3, não greedy)
  - SC@k:   self-consistency de cada modelo (o Single-Best-SC do paper)
  - oracle: alguma das respostas majoritárias dos modelos está certa
e o custo em tokens por questão de cada método.

Uso:
  python scripts/offline_mux_alg1.py \
      --test-dir results/gsm8k_test --val-dir results/gsm8k_val \
      --models meta-llama/Llama-3.1-8B-Instruct,Qwen/Qwen2.5-7B-Instruct \
      --samples 3 --trials 10
"""
import argparse
import json
import os
import random
import statistics

from slm_mux.confidence.consistency import ConsistencyConfidence
from slm_mux.engine.tie_breaker import ModelConfidence, ValidationAccuracyTieBreaker
from slm_mux.extractors.gsm import normalize_answer_gsm


def model_file(d, model):
    return os.path.join(d, model.replace("/", "_") + ".json")


def load(path):
    """questão -> {ref, answers[], prompt_toks[], compl_toks[]}"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for r in data["responses"]:
        ref = r.get("answer", r.get("reference_answer", ""))
        out[r["question"]] = {
            "ref": normalize_answer_gsm(str(ref)),
            "answers": [normalize_answer_gsm(s.get("extracted_answer") or "") for s in r["samples"]],
            "prompt_toks": [s.get("token_usage", {}).get("prompt_tokens", 0) for s in r["samples"]],
            "compl_toks": [s.get("token_usage", {}).get("completion_tokens", 0) for s in r["samples"]],
        }
    return out


def per_sample_accuracy(qdata):
    """Acurácia média de uma amostra isolada (usada como acurácia de validação)."""
    hits = total = 0
    for d in qdata.values():
        for a in d["answers"]:
            hits += int(bool(a) and a == d["ref"])
            total += 1
    return hits / total if total else 0.0


def ms(xs):
    return (statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--models", required=True, help="IDs separados por vírgula, na ordem de desempate final")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    test = {m: load(model_file(args.test_dir, m)) for m in models}
    val_acc = {m: per_sample_accuracy(load(model_file(args.val_dir, m))) for m in models}

    questions = sorted(set.intersection(*(set(test[m]) for m in models)))
    k = args.samples
    for m in models:
        n_min = min(len(test[m][q]["answers"]) for q in questions)
        if n_min < k:
            raise SystemExit(f"{m}: só {n_min} amostras em alguma questão, mas --samples={k}")

    random.seed(args.seed)  # a ConsistencyConfidence usa o random global para empates internos
    rng = random.Random(args.seed)
    conf_eval = ConsistencyConfidence(ignore_empty=True)
    tie_breaker = ValidationAccuracyTieBreaker()

    acc = {"mux": [], "oracle": []}
    for m in models:
        acc[f"single::{m}"] = []
        acc[f"sc{k}::{m}"] = []
    toks = {m: 0 for m in models}  # tokens das k amostras de cada modelo, somados nas rodadas
    winners = {m: 0 for m in models}

    for _ in range(args.trials):
        hit = {key: 0 for key in acc}
        for q in questions:
            confs = []
            ref = test[models[0]][q]["ref"]
            for m in models:
                d = test[m][q]
                idx = rng.sample(range(len(d["answers"])), k)
                answers = [d["answers"][i] for i in idx]
                toks[m] += sum(d["prompt_toks"][i] + d["compl_toks"][i] for i in idx)
                c = conf_eval.evaluate(responses=[], extracted_answers=answers)
                confs.append(ModelConfidence(m, c.score, c.selected_answer))
                hit[f"single::{m}"] += int(bool(answers[0]) and answers[0] == ref)
                hit[f"sc{k}::{m}"] += int(bool(c.selected_answer) and c.selected_answer == ref)

            best = max(c.confidence_score for c in confs)
            tied = [c for c in confs if abs(c.confidence_score - best) < 1e-9]  # já na ordem de `models`
            win = tie_breaker.break_tie(tied, val_acc)
            winners[win.model_name] += 1
            hit["mux"] += int(bool(win.selected_answer) and win.selected_answer == ref)
            hit["oracle"] += int(any(c.selected_answer and c.selected_answer == ref for c in confs))
        for key in acc:
            acc[key].append(hit[key] / len(questions))

    nq_total = len(questions) * args.trials
    tok_per_q = {m: toks[m] / nq_total for m in models}  # k amostras de um modelo
    tok_cost = {"mux": sum(tok_per_q.values()), "oracle": sum(tok_per_q.values())}
    for m in models:
        tok_cost[f"single::{m}"] = tok_per_q[m] / k
        tok_cost[f"sc{k}::{m}"] = tok_per_q[m]

    print(f"GSM8K | {len(questions)} questões | k={k} | {args.trials} rodadas")
    print("Acurácia de validação (1 amostra):", {m: round(v, 4) for m, v in val_acc.items()})
    print(f"{'método':55s} {'acurácia':>16s} {'tokens/questão':>15s}")
    summary = {}
    for key in acc:
        mean, sd = ms(acc[key])
        summary[key] = {"accuracy_mean": mean, "accuracy_std": sd, "tokens_per_question": tok_cost[key]}
        print(f"{key:55s} {100*mean:8.2f} ± {100*sd:4.2f} {tok_cost[key]:15.0f}")
    print("Modelo escolhido pelo MUX (% das questões):",
          {m: round(100 * winners[m] / nq_total, 1) for m in models})

    out = args.out or os.path.join(args.test_dir, f"mux_alg1_k{k}_t{args.trials}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "models": models, "samples_per_model": k, "trials": args.trials, "seed": args.seed,
            "n_questions": len(questions), "validation_accuracy": val_acc,
            "results": summary,
            "mux_winner_share": {m: winners[m] / nq_total for m in models},
        }, f, ensure_ascii=False, indent=2)
    print("salvo em", out)


if __name__ == "__main__":
    main()
