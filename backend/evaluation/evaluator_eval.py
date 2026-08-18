# measure how accurately each LLM performs the evaluation task
#
# py evaluator_eval.py
# py evaluator_eval.py --models llama-groq,qwen-openrouter
# py evaluator_eval.py --passes A,C --n 10

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

# make the sibling packages importable when run from this folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm import groq_client, openrouter_client

QUESTIONS_PATH = "eval_questions.csv"
RESULTS_DIR = "../../results"

MODELS = {
    "llama-groq": (groq_client, "llama-3.3-70b-versatile"),
    "qwen-openrouter": (openrouter_client, "qwen/qwen3.6-27b"),
    "mistral-openrouter": (openrouter_client, "mistralai/mistral-medium-3-5"),
    "gpt-openrouter": (openrouter_client, "openai/gpt-5.4-mini"),
    "gemini-openrouter": (openrouter_client, "google/gemini-3.7-flash"),
    "nemotron-openrouter": (openrouter_client, "nvidia/nemotron-3.5-lightning:free"),
}

PASSES = {
    "A": ("correct_answer", "correct answers", lambda row: "CORRECT"),
    "B": ("wrong_answer", "wrong answers", lambda row: "WRONG"),
    "C": ("hard_variant", "hard variants", lambda row: row["hard_expected"].strip().upper()),
}

PAUSE_S = 0.5

def load_questions(path: str = QUESTIONS_PATH) -> list[dict]:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        sys.exit(1)
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

# call evaluate_answer() on the right provider and normalize the result
def call_evaluate(model_id: str, question: str, answer: str) -> tuple[bool, str, float]:

    client_module, model_name = MODELS[model_id]

    t0 = time.time()
    result = client_module.evaluate_answer(question, answer, model=model_name)
    wall_ms = (time.time() - t0) * 1000

    if len(result) == 4:
        is_correct, explanation, _, client_ms = result
        return is_correct, explanation, client_ms

    is_correct, explanation = result
    return is_correct, explanation, wall_ms

def run_pass(model_id: str, rows: list[dict], pass_key: str) -> list[dict]:
    answer_col, label, expected_fn = PASSES[pass_key]

    print(f"\n  --- pass {pass_key}: {label} ---")
    results = []

    for row in rows:
        question = row["question"]
        answer = row[answer_col]
        expected = expected_fn(row)

        try:
            is_correct, explanation, latency_ms = call_evaluate(model_id, question, answer)
            verdict = "CORRECT" if is_correct else "WRONG"
            error = ""
        except Exception as e:
            verdict, explanation, latency_ms = "ERROR", "", 0.0
            error = f"{type(e).__name__}: {str(e)[:80]}"

        agrees = (verdict == expected)

        has_explanation = len(explanation.split()) > 2

        mark = "ok " if agrees else "MISS"
        if error:
            mark = "ERR "
        print(f"  [{row['id']:>2}] {mark} exp={expected:7s} got={verdict:7s} "
              f"{latency_ms:6.0f} ms | {answer[:28]:28s} | {explanation[:38]}")

        results.append({
            "model":            model_id,
            "pass":             pass_key,
            "id":               row["id"],
            "question":         question,
            "answer_given":     answer,
            "expected_verdict": expected,
            "actual_verdict":   verdict,
            "agrees":           agrees,
            "explanation":      explanation,
            "has_explanation":  has_explanation,
            "latency_ms":       round(latency_ms),
            "hard_type":        row.get("hard_type", "") if pass_key == "C" else "",
            "category":         row.get("category", ""),
            "difficulty":       row.get("difficulty", ""),
            "error":            error,
        })
        time.sleep(PAUSE_S)

    return results


def summarise(all_rows: list[dict], models: list[str], passes: list[str]) -> None:
    print("\n" + "-" * 74)
    print("RESULTS")
    print("" * 74)

    # ---- per-pass accuracy ----
    print(f"\n{'model':<20} " + " ".join(f"{'pass '+p:>12}" for p in passes) + f" {'overall':>10}")
    for model_id in models:
        cells = []
        total_ok = total_n = 0
        for p in passes:
            rows = [r for r in all_rows if r["model"] == model_id and r["pass"] == p]
            ok = sum(1 for r in rows if r["agrees"])
            total_ok += ok
            total_n += len(rows)
            cells.append(f"{ok:>3}/{len(rows):<3} {100*ok/len(rows) if rows else 0:>3.0f}%")
        overall = f"{100*total_ok/total_n:.0f}%" if total_n else "-"
        print(f"{model_id:<20} " + " ".join(f"{c:>12}" for c in cells) + f" {overall:>10}")

    # ---- confusion matrix (A + B only: unambiguous ground truth) ----
    if "A" in passes and "B" in passes:
        print("\n  Confusion matrix (passes A + B)")
        print(f"  {'model':<20} {'true pos':>9} {'false neg':>10} {'true neg':>9} {'false pos':>10}")
        for model_id in models:
            a = [r for r in all_rows if r["model"] == model_id and r["pass"] == "A"]
            b = [r for r in all_rows if r["model"] == model_id and r["pass"] == "B"]
            tp = sum(1 for r in a if r["actual_verdict"] == "CORRECT")
            fn = sum(1 for r in a if r["actual_verdict"] == "WRONG")
            tn = sum(1 for r in b if r["actual_verdict"] == "WRONG")
            fp = sum(1 for r in b if r["actual_verdict"] == "CORRECT")
            print(f"  {model_id:<20} {tp:>9} {fn:>10} {tn:>9} {fp:>10}")
        print("     false neg = rejected a correct answer (too strict)")
        print("     false pos = accepted a wrong answer  (too lenient)")

    # ---- pass C broken down by difficulty type ----
    if "C" in passes:
        print("\n  Pass C accuracy by variant type")
        types = sorted({r["hard_type"] for r in all_rows if r["pass"] == "C" and r["hard_type"]})
        print(f"  {'model':<20} " + " ".join(f"{t[:9]:>10}" for t in types))
        for model_id in models:
            cells = []
            for t in types:
                rows = [r for r in all_rows
                        if r["model"] == model_id and r["pass"] == "C" and r["hard_type"] == t]
                ok = sum(1 for r in rows if r["agrees"])
                cells.append(f"{ok}/{len(rows)}" if rows else "-")
            print(f"  {model_id:<20} " + " ".join(f"{c:>10}" for c in cells))

    # ---- latency and explanation compliance ----
    print("\n  Latency and output format")
    print(f"  {'model':<20} {'median ms':>10} {'p95 ms':>9} {'no explanation':>16}")
    for model_id in models:
        rows = [r for r in all_rows if r["model"] == model_id and not r["error"]]
        if not rows:
            continue
        lat = sorted(r["latency_ms"] for r in rows)
        median = lat[len(lat) // 2]
        p95 = lat[min(int(len(lat) * 0.95), len(lat) - 1)]
        bare = sum(1 for r in rows if not r["has_explanation"])
        print(f"  {model_id:<20} {median:>10} {p95:>9} {bare:>13}/{len(rows)}")
    print("     'no explanation' = verdict word only, no feedback sentence")

    print("=" * 74 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluator accuracy benchmark")
    parser.add_argument("--models", default=",".join(MODELS.keys()),
                        help="comma-separated model IDs")
    parser.add_argument("--passes", default="A,B,C", help="which passes to run")
    parser.add_argument("--n", type=int, default=0,
                        help="limit to first N questions (0 = all)")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        print(f"ERROR: unknown model IDs {unknown}. Available: {list(MODELS.keys())}")
        sys.exit(1)

    passes = [p.strip().upper() for p in args.passes.split(",") if p.strip()]

    rows = load_questions()
    if args.n:
        rows = rows[:args.n]

    total_calls = len(rows) * len(passes) * len(models)
    print(f"Questions: {len(rows)} | passes: {', '.join(passes)} | models: {len(models)}")
    print(f"Total API calls: {total_calls}")

    all_rows = []
    for model_id in models:
        print("\n" + "=" * 74)
        print(f"  {model_id}")
        print("=" * 74)
        for p in passes:
            all_rows.extend(run_pass(model_id, rows, p))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{RESULTS_DIR}/evaluator_eval_{stamp}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    summarise(all_rows, models, passes)
    print(f"Raw results saved to: {output_path}\n")


if __name__ == "__main__":
    main()