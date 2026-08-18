# end-to-end pipeline latency with real audio
#
# Replays pre-recorded answers through the LIVE backend so every model
# sees byte-identical input. Human speech varies between takes; frozen
# fixtures remove that variance, which is what makes a cross-model
# comparison valid
#
# Measures one full turn per fixture:
# POST /api/transcribe   audio  -> transcript   (STT)
# POST /api/evaluate     text   -> verdict      (LLM eval + next-question gen)
# POST /api/speak        text   -> audio        (TTS)
#
# Run:
# py run_pipeline_latency.py --vary llm
# py run_pipeline_latency.py --vary stt --reps 3
# py run_pipeline_latency.py --vary tts --fixtures short_correct,long_wrong

import argparse
import csv
import json
import os
import statistics as st
import sys
import time
from datetime import datetime

import requests

API_BASE = "http://localhost:8000"
FIXTURES_DIR = "fixtures"
FIXTURES_MANIFEST = "fixtures.json"
RESULTS_DIR = "../../results"

# held constant while another axis varies
BASELINE = {
    "stt": "gpt4o-transcribe",
    "llm": "gpt-openrouter",
    "tts": "minimax-speech",
}

AXES = {
    "stt": ["whisper-large-v3", "gpt4o-transcribe", "nova-3", "voxtral-stt"],
    "llm": ["qwen-openrouter", "mistral-openrouter", "gpt-openrouter", "gemini-openrouter", "nemotron-openrouter"],
    "tts": ["kokoro", "qwen-tts", "deepgram-aura", "minimax-speech"],
}

PAUSE_S = 1.0

# SETUP

def check_backend() -> None:
    try:
        requests.get(f"{API_BASE}/api/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"ERROR: backend not reachable at {API_BASE} ({e})")
        print("       cd backend && uvicorn main:app --reload --port 8000")
        sys.exit(1)


def load_fixtures(only: list[str] | None = None) -> list[dict]:
    if not os.path.exists(FIXTURES_MANIFEST):
        print(f"ERROR: {FIXTURES_MANIFEST} not found")
        sys.exit(1)

    with open(FIXTURES_MANIFEST, encoding="utf-8") as f:
        fixtures = json.load(f)["fixtures"]

    missing = []
    usable = []
    for fx in fixtures:
        name = os.path.splitext(fx["file"])[0]
        if only and name not in only:
            continue
        path = os.path.join(FIXTURES_DIR, fx["file"])
        if not os.path.exists(path):
            missing.append(fx["file"])
            continue
        fx["path"] = path
        fx["name"] = name
        fx["size_kb"] = round(os.path.getsize(path) / 1024, 1)
        usable.append(fx)

    if missing:
        print(f"  skipping {len(missing)} fixture(s) with no recording: {', '.join(missing)}")
    if not usable:
        print("ERROR: no usable fixtures found. Record some with record_fixtures.py first.")
        sys.exit(1)

    return usable


def start_session(config: dict, num_questions: int) -> bool:
    try:
        r = requests.post(f"{API_BASE}/api/start", timeout=120, json={
            "topic": "general knowledge",
            "num_questions": num_questions,
            "personality": "classic",
            "config": config,
        })
        return r.ok
    except Exception:
        return False


# ONE TURN
def run_turn(fixture: dict, config: dict) -> dict:
    row = {
        "stt_model": config["stt"], "llm_model": config["llm"], "tts_model": config["tts"],
        "fixture": fixture["name"], "length": fixture.get("length", ""),
        "audio_kb": fixture["size_kb"],
        "stt_ms": None, "llm_eval_ms": None, "llm_gen_ms": None, "tts_ms": None,
        "pipeline_ms": None, "transcript": "", "verdict": "", "speech": "",
        "error": "",
    }

    # --- STT ---
    try:
        with open(fixture["path"], "rb") as f:
            t0 = time.time()
            r = requests.post(
                f"{API_BASE}/api/transcribe",
                params={"stt_model": config["stt"]},
                files={"audio": (fixture["file"], f, "audio/webm")},
                timeout=120,
            )
        stt_wall = (time.time() - t0) * 1000
        if not r.ok:
            row["error"] = f"STT HTTP {r.status_code}: {r.text[:120]}"
            return row
        data = r.json()
        row["transcript"] = data.get("transcript", "")
        row["stt_ms"] = data.get("latency_ms", {}).get("stt", round(stt_wall))
    except Exception as e:
        row["error"] = f"STT {type(e).__name__}: {e}"
        return row

    # --- LLM ---
    try:
        r = requests.post(f"{API_BASE}/api/evaluate", timeout=180, json={
            "transcript": row["transcript"],
            "question": fixture["question"],
            "question_index": 0,
            "config": config,
        })
        if not r.ok:
            row["error"] = f"LLM HTTP {r.status_code}: {r.text[:120]}"
            return row
        data = r.json()
        lat = data.get("latency_ms", {})
        row["llm_eval_ms"] = lat.get("llm_eval")
        row["llm_gen_ms"] = lat.get("llm_gen")
        row["verdict"] = "CORRECT" if data.get("is_correct") else "WRONG"
        row["speech"] = data.get("message", "")
    except Exception as e:
        row["error"] = f"LLM {type(e).__name__}: {e}"
        return row

    # --- TTS ---
    try:
        t0 = time.time()
        r = requests.post(f"{API_BASE}/api/speak", timeout=180, json={
            "text": row["speech"],
            "tts_model": config["tts"],
        })
        tts_wall = (time.time() - t0) * 1000
        if not r.ok:
            row["error"] = f"TTS HTTP {r.status_code}: {r.text[:120]}"
            return row
        header = r.headers.get("X-TTS-Latency-Ms")
        row["tts_ms"] = int(header) if header else round(tts_wall)
    except Exception as e:
        row["error"] = f"TTS {type(e).__name__}: {e}"
        return row

    row["pipeline_ms"] = sum(v or 0 for v in
                             (row["stt_ms"], row["llm_eval_ms"], row["llm_gen_ms"], row["tts_ms"]))
    return row


# SUMMARY
def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    if not s:
        return float("nan")
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def summarise(rows: list[dict], axis: str) -> None:
    ok = [r for r in rows if not r["error"] and r["pipeline_ms"] is not None]
    failed = [r for r in rows if r["error"]]

    print("\n" + "=" * 78)
    print(f"  RESULTS - varying {axis.upper()}")
    print(f"  baseline: " + " · ".join(f"{k}={v}" for k, v in BASELINE.items() if k != axis))
    print("=" * 78)

    if failed:
        print(f"\n  {len(failed)} failed turn(s):")
        for r in failed[:8]:
            print(f"    {r[axis + '_model']:20s} {r['fixture']:18s} {r['error'][:70]}")
        if len(failed) > 8:
            print(f"    ... and {len(failed) - 8} more")

    if not ok:
        print("\n  No successful turns to summarise.\n")
        return

    key = axis + "_model"
    models = sorted({r[key] for r in ok})

    # per-stage medians
    print(f"\n  {'model':<22} {'n':>3} {'STT':>7} {'LLM ev':>7} {'LLM gen':>8} {'TTS':>7} {'TOTAL':>8} {'p95':>8}")
    for m in models:
        rs = [r for r in ok if r[key] == m]
        totals = [r["pipeline_ms"] for r in rs]
        def med(field):
            vals = [r[field] for r in rs if r[field] is not None]
            return st.median(vals) if vals else 0
        print(f"  {m:<22} {len(rs):>3} "
              f"{med('stt_ms'):>7.0f} {med('llm_eval_ms'):>7.0f} {med('llm_gen_ms'):>8.0f} "
              f"{med('tts_ms'):>7.0f} {st.median(totals):>8.0f} {percentile(totals, 0.95):>8.0f}")
    print("     all values are medians in ms, except the final p95 of total pipeline time")

    # the varying stage broken down by utterance length - only meaningful
    # for STT (input length) and TTS (output text length)
    if axis in ("stt", "tts"):
        stage = "stt_ms" if axis == "stt" else "tts_ms"
        lengths = [l for l in ("short", "medium", "long", "silence")
                   if any(r.get("length") == l for r in ok)]
        if lengths:
            print(f"\n  {stage} by utterance length (median ms)")
            print(f"  {'model':<22} " + " ".join(f"{l:>9}" for l in lengths))
            for m in models:
                cells = []
                for l in lengths:
                    vals = [r[stage] for r in ok
                            if r[key] == m and r.get("length") == l and r[stage] is not None]
                    cells.append(f"{st.median(vals):.0f}" if vals else "-")
                print(f"  {m:<22} " + " ".join(f"{c:>9}" for c in cells))

    print("=" * 78 + "\n")


# 
# MAIN
def main():
    parser = argparse.ArgumentParser(description="Pipeline latency")
    parser.add_argument("--vary", choices=["stt", "llm", "tts", "none"], default="llm",
                        help="which component to vary (default: llm)")
    parser.add_argument("--reps", type=int, default=3, help="repetitions per fixture (default 3)")
    parser.add_argument("--fixtures", type=str, default="",
                        help="comma-separated fixture names, default all")
    parser.add_argument("--models", type=str, default="",
                        help="comma-separated model IDs, default the full axis list")
    args = parser.parse_args()

    check_backend()

    only = [s.strip() for s in args.fixtures.split(",") if s.strip()] or None
    fixtures = load_fixtures(only)

    if args.vary == "none":
        models = [BASELINE[list(BASELINE)[0]]]
        variants = [dict(BASELINE)]
        axis = "llm"
    else:
        axis = args.vary
        models = [m.strip() for m in args.models.split(",") if m.strip()] or AXES[axis]
        variants = [{**BASELINE, axis: m} for m in models]

    total_turns = len(variants) * len(fixtures) * args.reps
    print("-" * 78)
    print(f"varying {axis}")
    print(f"{len(variants)} config(s) x {len(fixtures)} fixture(s) x {args.reps} rep(s) = {total_turns} turns")
    print(f"each turn = 3 API calls => {total_turns * 3} requests")
    print("-" * 78)

    rows = []
    for config in variants:
        label = config[axis]
        print(f"\n--- {axis}={label} ---")

        # a session with headroom, so it never completes during the run
        if not start_session(config, num_questions=len(fixtures) * args.reps + 5):
            print("  could not start a session - skipping this configuration")
            continue

        for rep in range(1, args.reps + 1):
            for fx in fixtures:
                row = run_turn(fx, config)
                row["rep"] = rep
                rows.append(row)

                if row["error"]:
                    print(f"  [{rep}] {fx['name']:18s} ERROR  {row['error'][:60]}")
                else:
                    print(f"  [{rep}] {fx['name']:18s} "
                          f"stt {row['stt_ms']:>5} · llm {(row['llm_eval_ms'] or 0) + (row['llm_gen_ms'] or 0):>5} · "
                          f"tts {row['tts_ms']:>5} · total {row['pipeline_ms']:>6} ms  "
                          f"| {row['transcript'][:34]}")
                time.sleep(PAUSE_S)

    if not rows:
        print("\nNo turns completed.\n")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{RESULTS_DIR}/pipeline_latency_{axis}_{stamp}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["stt_model", "llm_model", "tts_model", "fixture", "length", "rep",
                  "audio_kb", "stt_ms", "llm_eval_ms", "llm_gen_ms", "tts_ms",
                  "pipeline_ms", "transcript", "verdict", "speech", "error"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summarise(rows, axis)
    print(f"Results saved to: {output_path}\n")


if __name__ == "__main__":
    main()