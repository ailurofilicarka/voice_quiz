# evaluation for the voice quiz pipeline
# talks to the live backend over HTTP
#
# TESTING QUESTION GENERATION
# - calls /api/start for each model and records generation latency plus the question text itself
# - testing generating only the first question
#
# py run_pipeline_eval.py --gen
# py run_pipeline_eval.py --gen --n 20 --models llama-groq,qwen-openrouter

import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime

import requests

API_BASE = "http://localhost:8000"
RESULTS_DIR = "../../results"

DEFAULT_MODELS = ["llama-groq", "qwen-openrouter", "mistral-openrouter", "gpt-openrouter", "gemini-openrouter", "nemotron-openrouter"]

DEFAULT_N = 15
DEFAULT_TOPIC = "general knowledge"
PAUSE_S = 1.0
NUM_QUESTIONS = 5

# make sure the backend is actually running
def check_backend(models: list[str]) -> None:
    try:
        health = requests.get(f"{API_BASE}/api/health", timeout=5)
        health.raise_for_status()
    except Exception as e:
        print(f"ERROR: backend not reachable at {API_BASE}")
        print(f"({e})")
        sys.exit(1)

    try:
        listed = requests.get(f"{API_BASE}/api/models", timeout=5).json()
        available = {m["id"] for m in listed.get("llm", [])}
    except Exception:
        print("WARNING: could not read /api/models — skipping model validation.")
        return

    unknown = [m for m in models if m not in available]
    if unknown:
        print(f"ERROR: these model IDs are not registered in the backend: {unknown}")
        print(f"Available: {sorted(available)}")
        sys.exit(1)

# one generation request
# only first question generation is measured, no "previously asked questions" list of a real session
def generate_once(model_id: str, topic: str) -> dict:
    t_start = time.time()
    error = ""
    question = ""
    server_llm_ms = None
    status = 0

    try:
        response = requests.post(
            f"{API_BASE}/api/start",
            json={
                "topic": topic,
                "num_questions": NUM_QUESTIONS,
                "config": {"stt": "whisper-groq", "llm": model_id, "tts": "orpheus"},
            },
            timeout=120,
        )
        status = response.status_code
        wall_ms = (time.time() - t_start) * 1000

        if response.ok:
            data = response.json()
            question = (data.get("next_question") or "").strip()
            server_llm_ms = (data.get("latency_ms") or {}).get("llm")
        else:
            error = f"HTTP {status}: {response.text[:200]}"

    except Exception as e:
        wall_ms = (time.time() - t_start) * 1000
        error = f"{type(e).__name__}: {e}"

    return {
        "model":         model_id,
        "topic":         topic,
        "question":      question,
        "wall_ms":       round(wall_ms),
        "server_llm_ms": server_llm_ms if server_llm_ms is not None else "",
        "http_status":   status,
        "error":         error,
        "question_len":  len(question),
        "word_count":    len(question.split()) if question else 0,
    }

def run_generation_eval(models: list[str], n: int, topic: str, output_path: str) -> list[dict]:
    print("-" * 62)
    print("QUESTION GENERATION")
    print("-" * 62)

    rows = []
    for model_id in models:
        print(f"\n--- {model_id} ---")
        for i in range(1, n + 1):
            row = generate_once(model_id, topic)
            row["run"] = i
            rows.append(row)

            if row["error"]:
                print(f"[{i:2d}/{n}] ERROR  {row['wall_ms']:6d} ms | {row['error'][:70]}")
            else:
                srv = row["server_llm_ms"]
                srv_txt = f"{srv:5} ms" if srv != "" else "  n/a"
                print(f"[{i:2d}/{n}] {row['wall_ms']:6d} ms wall | llm {srv_txt} | {row['question'][:58]}")

            time.sleep(PAUSE_S)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["model", "run", "topic", "question", "wall_ms", "server_llm_ms",
                  "http_status", "error", "question_len", "word_count"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nRaw results saved to: {output_path}")
    return rows

# analyse quesion quality
def analyse_generation(rows: list[dict], models: list[str]) -> None:

    def median(values):
        s = sorted(values)
        n = len(s)
        if n == 0:
            return float("nan")
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    def percentile(values, p):
        s = sorted(values)
        if not s:
            return float("nan")
        k = (len(s) - 1) * p
        f = int(k)
        c = min(f + 1, len(s) - 1)
        return s[f] + (s[c] - s[f]) * (k - f)

    prefix_pattern = re.compile(r"^\s*question\s*(number\s*)?\d*\s*[:.\-]", re.IGNORECASE)

    print("\n" + "-" * 62)
    print("RESULTS")
    print("-" * 62)

    for model_id in models:
        model_rows = [r for r in rows if r["model"] == model_id]
        ok_rows = [r for r in model_rows if not r["error"] and r["question"]]

        n_total = len(model_rows)
        n_ok = len(ok_rows)
        n_failed = n_total - n_ok

        print(f"\n{model_id}")
        print(f"successful:      {n_ok}/{n_total}" + (f"   ({n_failed} failed)" if n_failed else ""))

        if not ok_rows:
            print("(no successful generations — nothing further to report)")
            continue

        walls = [r["wall_ms"] for r in ok_rows]
        print(f"  latency (wall):  median {median(walls):>7.0f} ms | "
              f"p95 {percentile(walls, 0.95):>7.0f} ms | "
              f"min {min(walls)} | max {max(walls)}")

        server_times = [r["server_llm_ms"] for r in ok_rows if r["server_llm_ms"] != ""]
        if server_times:
            print(f"  latency (llm):   median {median(server_times):>7.0f} ms | "
                  f"p95 {percentile(server_times, 0.95):>7.0f} ms")

        # diversity
        questions = [r["question"].strip().lower() for r in ok_rows]
        unique = len(set(questions))
        print(f"  unique questions: {unique}/{n_ok}  ({100 * unique / n_ok:.0f}% distinct)")

        words = [r["word_count"] for r in ok_rows]
        print(f"  length:          median {median(words):.0f} words | "
              f"range {min(words)}-{max(words)}")

        prefixed = sum(1 for r in ok_rows if prefix_pattern.match(r["question"]))
        print(f"  format issues:   {prefixed}/{n_ok} start with a numbering prefix")

def main():
    parser = argparse.ArgumentParser(description="Voice quiz pipeline evaluation")
    parser.add_argument("--gen", action="store_true", help="run question generation test")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help=f"generations per model (default {DEFAULT_N})")
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS),
                        help="comma-separated model IDs")
    parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC, help="quiz topic")
    args = parser.parse_args()

    if not args.gen:
        parser.print_help()
        return

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    check_backend(models)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{RESULTS_DIR}/generation_eval_{stamp}.csv"

    rows = run_generation_eval(models, args.n, args.topic, output_path)
    analyse_generation(rows, models)

if __name__ == "__main__":
    main()