# Run:
#     py screen_stt.py
#     py screen_stt.py --reps 5
#     py screen_stt.py --models whisper-groq,nova-3

import argparse
import csv
import json
import os
import re
import statistics as st
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stt import whisper_groq, stt_openrouter

FIXTURES_DIR = "fixtures"
FIXTURES_MANIFEST = "fixtures.json"
RESULTS_DIR = "../../results"

MODELS = {
    "whisper-turbo-or": (stt_openrouter, "openai/whisper-large-v3-turbo", "en"),
    "gpt4o-transcribe": (stt_openrouter,  "openai/gpt-4o-mini-transcribe",    "en"),
    "nova-3":           (stt_openrouter,  "deepgram/nova-3",                  "en"),
    "parakeet":         (stt_openrouter,  "nvidia/parakeet-tdt-0.6b-v3",      "en"),
    "voxtral-stt": (stt_openrouter, "mistralai/voxtral-mini-transcribe", "en"),
}

PAUSE_S = 0.8

# Any character outside basic ASCII. Flags the language-misdetection
# failure seen on short or hesitant audio, where a model transcribes
# English speech into Chinese or Cyrillic script.
NON_LATIN = re.compile(r"[^\x00-\x7F]")


def load_fixtures() -> list[dict]:
    with open(FIXTURES_MANIFEST, encoding="utf-8") as f:
        fixtures = json.load(f)["fixtures"]

    usable = []
    for fx in fixtures:
        path = os.path.join(FIXTURES_DIR, fx["file"])
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            fx["audio"] = f.read()          # read once, reuse for every model
        fx["name"] = os.path.splitext(fx["file"])[0]
        usable.append(fx)

    if not usable:
        print("ERROR: no fixture recordings found. Run record_fixtures.py first.")
        sys.exit(1)
    return usable


def transcribe_once(model_id: str, fixture: dict) -> dict:
    client, model_name, language = MODELS[model_id]

    t0 = time.time()
    try:
        if client is stt_openrouter:
            text = client.transcribe(fixture["audio"], filename=fixture["file"],
                                     model=model_name, language=language)
        else:
            text = client.transcribe(fixture["audio"], filename=fixture["file"],
                                     model=model_name)
        error = ""
    except Exception as e:
        text = ""
        error = f"{type(e).__name__}: {str(e)[:90]}"

    return {
        "model":      model_id,
        "fixture":    fixture["name"],
        "length":     fixture.get("length", ""),
        "expected":   fixture.get("expected_transcript", ""),
        "transcript": text,
        "latency_ms": round((time.time() - t0) * 1000),
        "empty":      (text.strip() == ""),
        "non_latin":  bool(NON_LATIN.search(text)),
        "error":      error,
    }


def percentile(values, p):
    s = sorted(values)
    if not s:
        return float("nan")
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def summarise(rows: list[dict], models: list[str], fixtures: list[dict]) -> None:
    print("\n" + "=" * 76)
    print("  LATENCY")
    print("=" * 76)
    print(f"  {'model':<20} {'n':>3} {'median':>8} {'p95':>8} {'min':>7} {'max':>8} {'fails':>7}")

    ranking = []
    for m in models:
        rs = [r for r in rows if r["model"] == m and not r["error"]]
        fails = sum(1 for r in rows if r["model"] == m and r["error"])
        if not rs:
            print(f"  {m:<20} {'-':>3} {'all failed':>8}")
            continue
        lat = [r["latency_ms"] for r in rs]
        median = st.median(lat)
        ranking.append((median, m))
        print(f"  {m:<20} {len(rs):>3} {median:>8.0f} {percentile(lat, 0.95):>8.0f} "
              f"{min(lat):>7} {max(lat):>8} {fails:>7}")

    # latency by utterance length - the axis that actually drives STT time
    lengths = [l for l in ("short", "medium", "long", "silence")
               if any(r.get("length") == l for r in rows)]
    if lengths:
        print(f"\n  median ms by utterance length")
        print(f"  {'model':<20} " + " ".join(f"{l:>9}" for l in lengths))
        for m in models:
            cells = []
            for l in lengths:
                vals = [r["latency_ms"] for r in rows
                        if r["model"] == m and r["length"] == l and not r["error"]]
                cells.append(f"{st.median(vals):.0f}" if vals else "-")
            print(f"  {m:<20} " + " ".join(f"{c:>9}" for c in cells))

    print("\n" + "=" * 76)
    print("  OUTPUT PROBLEMS")
    print("=" * 76)
    print(f"  {'model':<20} {'empty':>7} {'non-latin':>11} {'errors':>8}")
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        print(f"  {m:<20} {sum(1 for r in rs if r['empty']):>7} "
              f"{sum(1 for r in rs if r['non_latin']):>11} "
              f"{sum(1 for r in rs if r['error']):>8}")
    print("     non-latin = transcript contains characters outside ASCII,")
    print("     which on English audio indicates language misdetection")

    # one transcript per fixture per model, for eyeballing quality
    print("\n" + "=" * 76)
    print("  TRANSCRIPTS  (first repetition)")
    print("=" * 76)
    for fx in fixtures:
        print(f"\n  {fx['name']}  ({fx.get('length', '')})")
        if fx.get("expected_transcript"):
            print(f"    {'expected':<20} {fx['expected_transcript']}")
        for m in models:
            first = next((r for r in rows if r["model"] == m and r["fixture"] == fx["name"]), None)
            if first is None:
                continue
            flag = " [!]" if (first["non_latin"] or first["empty"]) else ""
            shown = first["error"] or first["transcript"] or "(empty)"
            print(f"    {m:<20} {shown[:64]}{flag}")

    if ranking:
        ranking.sort()
        print("\n" + "=" * 76)
        print("  SLOWEST BY MEDIAN LATENCY")
        print("=" * 76)
        for median, m in reversed(ranking):
            print(f"    {m:<20} {median:>8.0f} ms")
        print("\n  Speed is a usability filter for a real-time loop, not a quality")
        print("  ranking - check the transcripts above before eliminating anything.")
    print("=" * 76 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Screen STT models on latency")
    parser.add_argument("--reps", type=int, default=3, help="repetitions per fixture (default 3)")
    parser.add_argument("--models", type=str, default="", help="comma-separated model IDs")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()] or list(MODELS)
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        print(f"ERROR: unknown model IDs {unknown}. Available: {list(MODELS)}")
        sys.exit(1)

    fixtures = load_fixtures()
    total = len(models) * len(fixtures) * args.reps
    print(f"{len(models)} model(s) x {len(fixtures)} fixture(s) x {args.reps} rep(s) = {total} calls\n")

    rows = []
    for model_id in models:
        print(f"--- {model_id} ---")
        for rep in range(1, args.reps + 1):
            for fx in fixtures:
                row = transcribe_once(model_id, fx)
                row["rep"] = rep
                rows.append(row)

                mark = "ERR " if row["error"] else ("[!] " if (row["empty"] or row["non_latin"]) else "    ")
                shown = row["error"] or row["transcript"] or "(empty)"
                print(f"  [{rep}] {mark}{row['latency_ms']:>6} ms  {row['fixture']:<18} {shown[:46]}")
                time.sleep(PAUSE_S)
        print()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{RESULTS_DIR}/stt_screening_{stamp}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["model", "fixture", "length", "rep", "latency_ms",
                  "transcript", "expected", "empty", "non_latin", "error"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summarise(rows, models, fixtures)
    print(f"Raw results saved to: {output_path}\n")


if __name__ == "__main__":
    main()


# ============================================================================
#   LATENCY
# ============================================================================
#   model                  n   median      p95     min      max   fails
#   whisper-groq          30      250     3332     158    11764       0
#   whisper-groq-large    30      328     3376     243     3401       0
#   whisper-turbo-or      30      534      989     400     4784       0
#   whisper-v3-or         30      560     1396     444     2072       0
#   gpt4o-transcribe      30      526      663     412     1196       0
#   nova-3                30      258      696     229      945       0
#   parakeet              30      204      614     120     1347       0
#   voxtral-stt           30      338      460     283      653       0

#   median ms by utterance length
#   model                    short    medium      long   silence
#   whisper-groq               243       263       254       239
#   whisper-groq-large         314       339       336       271
#   whisper-turbo-or           530       656       714       426
#   whisper-v3-or              558       592       552       503
#   gpt4o-transcribe           508       525       560       477
#   nova-3                     246       247       315       230
#   parakeet                   210       218       158       135
#   voxtral-stt                320       364       374       333

# ============================================================================
#   OUTPUT PROBLEMS
# ============================================================================
#   model                  empty   non-latin   errors
#   whisper-groq               0           0        0
#   whisper-groq-large         0           0        0
#   whisper-turbo-or           0           0        0
#   whisper-v3-or              0           0        0
#   gpt4o-transcribe           3           2        0
#   nova-3                     3           0        0
#   parakeet                   6           9        0
#   voxtral-stt                3           0        0
#      non-latin = transcript contains characters outside ASCII,
#      which on English audio indicates language misdetection

# ============================================================================
#   TRANSCRIPTS  (first repetition)
# ============================================================================

#   short_correct  (short)
#     expected             Paris
#     whisper-groq         Paris
#     whisper-groq-large   Paris
#     whisper-turbo-or     Paris
#     whisper-v3-or        Paris
#     gpt4o-transcribe     Paríž [!]
#     nova-3               Paris
#     parakeet             Paris
#     voxtral-stt          Paris

#   short_correct_2  (short)
#     expected             Jupiter
#     whisper-groq         Jupiter
#     whisper-groq-large   Jupiter
#     whisper-turbo-or     Jupiter
#     whisper-v3-or        Jupiter
#     gpt4o-transcribe     Jupiter
#     nova-3               Jupiter.
#     parakeet             Джупитер. [!]
#     voxtral-stt          Jupiter

#   medium_correct  (medium)
#     expected             I think it's Paris
#     whisper-groq         I think it's Paris.
#     whisper-groq-large   I think it's Paris.
#     whisper-turbo-or     I think it's Paris.
#     whisper-v3-or        I think it's Paris.
#     gpt4o-transcribe     I think it's Paris.
#     nova-3               I think it's Paris.
#     parakeet             I think it's Paris.
#     voxtral-stt          I think it's Paris.

#   medium_correct_2  (medium)
#     expected             It's Jupiter, the biggest one
#     whisper-groq         It's Jupiter, the biggest one.
#     whisper-groq-large   It's Jupiter, the biggest one.
#     whisper-turbo-or     It's Jupiter, the biggest one.
#     whisper-v3-or        It's Jupiter, the biggest one.
#     gpt4o-transcribe     It's Jupiter, the biggest one.
#     nova-3               It's Jupiter, the biggest one.
#     parakeet             It's Jupiter, the biggest one.
#     voxtral-stt          It's Jupiter, the biggest one.

#   long_correct  (long)
#     expected             Um, I'm fairly sure it's Paris, the capital of France
#     whisper-groq         I'm fairly sure it's Paris, the capital of France.
#     whisper-groq-large   I'm fairly sure it's Paris, the capital of France.
#     whisper-turbo-or     I'm fairly sure it's Paris, the capital of France.
#     whisper-v3-or        I'm fairly sure it's Paris, the capital of France.
#     gpt4o-transcribe     I am fairly sure it's Paris, the capital of France.
#     nova-3               I'm fairly sure it's Paris, the capital of France.
#     parakeet             Mmm, I'm fairly sure it's Paris, the capital of France.
#     voxtral-stt          I'm fairly sure it's Paris, the capital of France.

#   short_wrong  (short)
#     expected             Sydney
#     whisper-groq         Sydney
#     whisper-groq-large   Sydney
#     whisper-turbo-or     Sydney
#     whisper-v3-or        Sydney
#     gpt4o-transcribe     Sydney
#     nova-3               Sydney.
#     parakeet             Сидней. [!]
#     voxtral-stt          Sydney

#   long_wrong  (long)
#     expected             I think maybe it's Sydney, or possibly Melbourne
#     whisper-groq         I think maybe it's Sydney or possibly Melbourne.
#     whisper-groq-large   I think maybe it's Sydney or possibly Melbourne
#     whisper-turbo-or     I think maybe it's Sydney or possibly Melbourne.
#     whisper-v3-or        I think maybe it's Sydney or possibly Melbourne
#     gpt4o-transcribe     I think maybe it's Sydney or possibly Melbourne.
#     nova-3               I think maybe it's Sydney or possibly Melbourne
#     parakeet             I think maybe it's Sydney or possibly Melbourne.
#     voxtral-stt          I think maybe it's Sydney or possibly Melbourne.

#   dont_know  (short)
#     expected             I don't know
#     whisper-groq         I don't know.
#     whisper-groq-large   I don't know.
#     whisper-turbo-or     I don't know.
#     whisper-v3-or        I don't know.
#     gpt4o-transcribe     I don't know.
#     nova-3               I don't know.
#     parakeet             Ай, Донт но. [!]
#     voxtral-stt          I don't know.

#   hesitation  (medium)
#     expected             Hmm... uh... let me think
#     whisper-groq         Hmm... Oh... Let me think...
#     whisper-groq-large   hmmmm oh let me think
#     whisper-turbo-or     Hmm... Oh... Let me think...
#     whisper-v3-or        hmmmm oh let me think
#     gpt4o-transcribe     Let me think.
#     nova-3               Oh, let me think.
#     parakeet             (empty) [!]
#     voxtral-stt          Hmm... Oh... Let me think...

#   silence  (silence)
#     whisper-groq         .
#     whisper-groq-large   Thank you.
#     whisper-turbo-or     .
#     whisper-v3-or        Thank you.
#     gpt4o-transcribe     (empty) [!]
#     nova-3               (empty) [!]
#     parakeet             (empty) [!]
#     voxtral-stt          (empty) [!]

# ============================================================================
#   SLOWEST BY MEDIAN LATENCY
# ============================================================================
#     whisper-v3-or             560 ms
#     whisper-turbo-or          534 ms
#     gpt4o-transcribe          526 ms
#     voxtral-stt               338 ms
#     whisper-groq-large        328 ms
#     nova-3                    258 ms
#     whisper-groq              250 ms
#     parakeet                  204 ms

#   Speed is a usability filter for a real-time loop, not a quality
#   ranking - check the transcripts above before eliminating anything.
# ============================================================================

# Raw results saved to: ../../results/stt_screening_20260816_230620.csv

# (.venv) PS C:\Users\aleks\Music\magg\voice_quiz\backend\evaluation> py .\check_parakeet_language.py

# --- language=en ---
#   [1] 'Джупитер.'
#   [2] 'Джупитер.'
#   [3] 'Джупитер.'

# --- language=None ---
#   [1] 'Джупитер.'
#   [2] 'Джупитер.'
#   [3] 'Джупитер.'
# (.venv) PS C:\Users\aleks\Music\magg\voice_quiz\backend\evaluation> 