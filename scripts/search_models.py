#!/usr/bin/env python3
"""Busca de seleção de modelos do SLM-MUX (Seções 3.2 e 4.3 do paper), em cima da coleta de validação.

Para cada grupo S de modelos:
    O(S) = UnionAcc(S) - lambda * Contradiction(S)          (paper: lambda = 1)
    UnionAcc      = fração das questões em que algum modelo de S acerta
    Contradiction = fração das questões em que algum modelo de S erra com confiança
                    E outro modelo de S acerta (dividida pelo total de questões, como no paper)

O paper não define "acerta" nem "erra com confiança". Aqui, como o MUX usa 3 respostas
por modelo, as 9 respostas da validação são divididas em trios e cada trio é avaliado:
    acerta            = a resposta majoritária do trio está certa
    erra com confiança = as 3 respostas do trio são iguais, não vazias e erradas
O score final é a média sobre os trios.

Além do O(S), reporta a acurácia do próprio MUX (Algoritmo 1) simulada na validação,
para conferir se o objetivo do paper aponta para os grupos que de fato funcionam melhor.

Por que não usar `slm-mux search`: usa lambda = 0,5 por padrão, divide a contradição só
pelas questões que o grupo acerta e conta extrações vazias como "erro com confiança".

Uso:
  python scripts/search_models.py --val-dir results/gsm8k_val --k-min 2 --k-max 5 --top 5
"""
import argparse
import itertools
import json
import os
from collections import Counter

from slm_mux.extractors.gsm import normalize_answer_gsm


def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for r in data["responses"]:
        ref = normalize_answer_gsm(str(r.get("answer", r.get("reference_answer", ""))))
        answers = [normalize_answer_gsm(s.get("extracted_answer") or "") for s in r["samples"]]
        out[r["question"]] = (ref, answers)
    return out


def trio_stats(ref, trio):
    """(acerta, erra_com_confianca, resposta_majoritaria, confianca) de um trio."""
    votes = Counter(a for a in trio if a)
    if not votes:
        return False, False, "", 0.0
    maj, cnt = votes.most_common(1)[0]
    tied = [a for a, c in votes.items() if c == cnt]
    if len(tied) > 1:          # 3 respostas diferentes: sem maioria
        return False, False, "", cnt / len(trio)
    correct = maj == ref
    confident_wrong = cnt == len(trio) and not correct
    return correct, confident_wrong, maj, cnt / len(trio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--models", default="", help="IDs separados por vírgula (padrão: todos os JSON da pasta)")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=5)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--trio", type=int, default=3, help="respostas por grupo (= k do MUX)")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = sorted(f[:-5].replace("_", "/", 1) for f in os.listdir(args.val_dir)
                        if f.endswith(".json") and not f.startswith("search_"))
    data = {m: load(os.path.join(args.val_dir, m.replace("/", "_") + ".json")) for m in models}
    questions = sorted(set.intersection(*(set(d) for d in data.values())))
    n_samples = min(len(data[m][q][1]) for m in models for q in questions)
    n_trios = n_samples // args.trio
    if n_trios == 0:
        raise SystemExit(f"só {n_samples} respostas por questão; precisa de pelo menos {args.trio}")

    # pré-calcula o trio t de cada modelo em cada questão
    st = {m: {q: [trio_stats(data[m][q][0], data[m][q][1][t * args.trio:(t + 1) * args.trio])
                  for t in range(n_trios)] for q in questions} for m in models}

    # descrição de cada modelo (não entra na escolha)
    per_model = {}
    for m in models:
        hits = sum(a == data[m][q][0] and a != "" for q in questions for a in data[m][q][1][:n_samples])
        per_model[m] = {
            "accuracy_per_sample": hits / (len(questions) * n_samples),
            "trio_majority_accuracy": sum(st[m][q][t][0] for q in questions for t in range(n_trios)) / (len(questions) * n_trios),
            "trio_confident_wrong_rate": sum(st[m][q][t][1] for q in questions for t in range(n_trios)) / (len(questions) * n_trios),
        }
    val_acc = {m: per_model[m]["accuracy_per_sample"] for m in models}

    def score(combo):
        U = C = MUX = 0.0
        for t in range(n_trios):
            u = c = mux = 0
            for q in questions:
                rows = [st[m][q][t] for m in combo]
                any_ok = any(r[0] for r in rows)
                u += any_ok
                c += any_ok and any(r[1] for r in rows)
                best = max(r[3] for r in rows)
                cands = [m for m, r in zip(combo, rows) if abs(r[3] - best) < 1e-9]
                win = max(cands, key=lambda m: val_acc[m])
                mux += st[win][q][t][0]
            n = len(questions)
            U += u / n; C += c / n; MUX += mux / n
        U /= n_trios; C /= n_trios; MUX /= n_trios
        return {"combo": list(combo), "score": U - args.lam * C, "union_acc": U, "contradiction": C, "mux_val_acc": MUX}

    print(f"Validação: {len(questions)} questões, {n_samples} respostas por modelo ({n_trios} trios), lambda = {args.lam}")
    print(f"{'modelo':50s} {'acc/amostra':>11s} {'acc maioria':>11s} {'erro confiante':>14s}")
    for m in sorted(models, key=lambda m: -val_acc[m]):
        p = per_model[m]
        print(f"{m:50s} {100*p['accuracy_per_sample']:10.1f}% {100*p['trio_majority_accuracy']:10.1f}% {100*p['trio_confident_wrong_rate']:13.1f}%")

    results = {}
    for k in range(max(2, args.k_min), min(len(models), args.k_max) + 1):
        ranked = sorted((score(c) for c in itertools.combinations(models, k)),
                        key=lambda r: (r["score"], r["union_acc"]), reverse=True)
        results[k] = ranked
        print(f"\nK = {k}: {len(ranked)} grupos")
        for r in ranked[:args.top]:
            print(f"  O={r['score']:.4f}  união={r['union_acc']:.4f}  contradição={r['contradiction']:.4f}  "
                  f"MUX(val)={r['mux_val_acc']:.4f}  | {', '.join(r['combo'])}")

    # salva fora da pasta de validação, para não ser lido como modelo na próxima busca
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.val_dir)),
                                   f"search_gsm8k_val_l{args.lam}_k{args.k_min}-{args.k_max}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"lambda": args.lam, "trio": args.trio, "n_questions": len(questions),
                   "n_samples": n_samples, "models": models, "per_model": per_model,
                   "ranked_by_k": {str(k): v for k, v in results.items()}}, f, ensure_ascii=False, indent=2)
    print("\nsalvo em", out)


if __name__ == "__main__":
    main()
