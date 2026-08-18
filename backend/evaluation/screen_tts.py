# backend/evaluation/screen_tts.py
#
# Screening run to narrow the TTS candidate list.
#
# Unlike STT there is no ground truth to score against - quality is
# subjective. This script measures what CAN be measured objectively
# (latency, audio duration, silence, truncation) and saves every sample
# to disk so the quality judgement can be made by listening.
#
# Two failure modes seen previously are checked automatically:
#   - silent audio: a correctly formed file containing only zero samples
#   - truncation:   audio far shorter than the text warrants, caused by
#                   placeholder length headers in streamed mp3
#
# Calls the TTS clients DIRECTLY - the backend does not need to be running.
#
# Run:
#     py screen_tts.py
#     py screen_tts.py --reps 5
#     py screen_tts.py --models kokoro,deepgram-aura
# ============================================================

import argparse
import csv
import io
import os
import statistics as st
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tts import tts_groq, tts_openrouter

SAMPLES_DIR = "tts_samples"
RESULTS_DIR = "../../results"

# Mirrors TTS_MODELS in main.py: (client, model, voice, format)
MODELS = {
    # "orpheus":       (tts_groq,       "canopylabs/orpheus-v1-english",      "autumn",           "wav"),
    "orpheus-or": (tts_openrouter, "canopylabs/orpheus-3b-0.1-ft", "leo", "pcm"),
    "kokoro":        (tts_openrouter, "hexgrad/kokoro-82m",                 "af_bella",         "pcm"),
    "gemini-tts":    (tts_openrouter, "google/gemini-3.1-flash-tts-preview","Charon",           "pcm"),
    "qwen-tts":      (tts_openrouter, "qwen/qwen-audio-3.0-tts-flash",      "loongjohn",        "mp3"),
    "deepgram-aura": (tts_openrouter, "deepgram/aura-2",                    "aura-2-thalia-en", "mp3"),
    "minimax-speech": (tts_openrouter, "minimax/speech-2.8-turbo", "Friendly_Person", "mp3"),
}

# Test texts modelled on real host output, not generic sentences.
# Each targets something that has caused trouble before.
CASES = [
    ("short",     "Correct!"),
    ("verdict",   "Ooh, not quite! The correct answer was Canberra."),
    ("typical",   "That's absolutely right! Jupiter is the largest planet in our solar system. Three in a row!"),
    ("long",      "Well done indeed! You've now reached four points with two questions still to come. "
                  "The capital of Australia is Canberra, not Sydney, which surprises a lot of people. "
                  "Let's keep that momentum going into the next round!"),
    ("numbers",   "You scored 7 out of 10, which is 70 percent. The war ended in 1945."),
    ("pauses",    "Oooh... let me see... that is... absolutely correct!"),
    ("names",     "Ljubljana is the capital of Slovenia, and Triglav is its highest mountain."),
]

PAUSE_S = 1.0

# Speech runs at roughly 14-16 characters per second. Audio much shorter
# than this suggests the tail was cut off rather than spoken quickly.
CHARS_PER_SECOND = 15
TRUNCATION_THRESHOLD = 0.6      # flag below 60% of the expected duration


def decode_audio(audio_bytes: bytes, fmt: str, sample_rate: int) -> tuple[np.ndarray, int]:
    """Return (samples, sample_rate) for any of the formats in use."""
    if fmt == "pcm":
        # raw 16-bit little-endian, no header
        return np.frombuffer(audio_bytes, dtype=np.int16), (sample_rate or 24000)

    import soundfile as sf
    audio, rate = sf.read(io.BytesIO(audio_bytes))
    return audio, rate


def synthesize_once(model_id: str, case_name: str, text: str, save: bool) -> dict:
    client, model_name, voice, fmt = MODELS[model_id]

    row = {
        "model": model_id, "case": case_name, "chars": len(text),
        "latency_ms": None, "duration_s": None, "size_kb": None,
        "rtf": None, "silent": False, "truncated": False, "error": "",
    }

    t0 = time.time()
    try:
        if client is tts_openrouter:
            audio_bytes, sample_rate = client.synthesize(
                text, voice=voice, model=model_name, response_format=fmt)
        else:
            audio_bytes = client.synthesize(text, voice=voice, model=model_name)
            sample_rate = 0
        row["latency_ms"] = round((time.time() - t0) * 1000)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:90]}"
        return row

    row["size_kb"] = round(len(audio_bytes) / 1024, 1)

    try:
        samples, rate = decode_audio(audio_bytes, fmt, sample_rate)
        duration = len(samples) / rate if rate else 0
        row["duration_s"] = round(duration, 2)

        # peak of exactly zero means a well-formed file with no audio in it
        row["silent"] = bool(np.max(np.abs(samples)) == 0) if len(samples) else True

        expected = len(text) / CHARS_PER_SECOND
        row["truncated"] = duration < expected * TRUNCATION_THRESHOLD

        # real-time factor: synthesis time divided by audio length.
        # below 1.0 means it generates faster than the audio plays.
        if duration > 0:
            row["rtf"] = round((row["latency_ms"] / 1000) / duration, 2)
    except Exception as e:
        row["error"] = f"decode failed: {type(e).__name__}: {str(e)[:60]}"

    if save and not row["error"]:
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        # pcm has no header, so it is saved as .raw - open it in Audacity as
        # 16-bit signed PCM mono at the reported sample rate
        extension = "raw" if fmt == "pcm" else fmt
        path = os.path.join(SAMPLES_DIR, f"{model_id}__{case_name}.{extension}")
        with open(path, "wb") as f:
            f.write(audio_bytes)

    return row


def percentile(values, p):
    s = sorted(values)
    if not s:
        return float("nan")
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def summarise(rows: list[dict], models: list[str]) -> None:
    print("\n" + "=" * 78)
    print("  LATENCY")
    print("=" * 78)
    print(f"  {'model':<16} {'n':>3} {'median':>8} {'p95':>8} {'min':>7} {'max':>8} {'RTF':>6} {'fails':>7}")

    ranking = []
    for m in models:
        ok = [r for r in rows if r["model"] == m and not r["error"]]
        fails = sum(1 for r in rows if r["model"] == m and r["error"])
        if not ok:
            print(f"  {m:<16} {'-':>3}  all failed")
            continue
        lat = [r["latency_ms"] for r in ok]
        rtfs = [r["rtf"] for r in ok if r["rtf"] is not None]
        median = st.median(lat)
        ranking.append((median, m))
        print(f"  {m:<16} {len(ok):>3} {median:>8.0f} {percentile(lat, 0.95):>8.0f} "
              f"{min(lat):>7} {max(lat):>8} {st.median(rtfs) if rtfs else 0:>6.2f} {fails:>7}")
    print("     RTF = synthesis time / audio length. Below 1.0 means faster than real time.")

    # latency against text length - the axis that drives TTS cost
    print(f"\n  median ms by test case")
    case_names = [c[0] for c in CASES]
    header = "  " + f"{'model':<16}" + " ".join(f"{c[:8]:>9}" for c in case_names)
    print(header)
    for m in models:
        cells = []
        for c in case_names:
            vals = [r["latency_ms"] for r in rows
                    if r["model"] == m and r["case"] == c and not r["error"]]
            cells.append(f"{st.median(vals):.0f}" if vals else "-")
        print(f"  {m:<16} " + " ".join(f"{v:>9}" for v in cells))

    print("\n" + "=" * 78)
    print("  OUTPUT PROBLEMS")
    print("=" * 78)
    print(f"  {'model':<16} {'silent':>8} {'truncated':>11} {'errors':>8}")
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        print(f"  {m:<16} {sum(1 for r in rs if r['silent']):>8} "
              f"{sum(1 for r in rs if r['truncated']):>11} "
              f"{sum(1 for r in rs if r['error']):>8}")
    print("     silent    = file contains only zero samples")
    print(f"     truncated = audio shorter than {int(TRUNCATION_THRESHOLD*100)}% of the duration the text implies")

    # duration per case, so truncation is visible rather than just counted
    print("\n" + "=" * 78)
    print("  AUDIO DURATION (s)  -  expected vs produced")
    print("=" * 78)
    for name, text in CASES:
        expected = len(text) / CHARS_PER_SECOND
        print(f"\n  {name}  ({len(text)} chars, roughly {expected:.1f}s expected)")
        for m in models:
            first = next((r for r in rows
                          if r["model"] == m and r["case"] == name and not r["error"]), None)
            if first is None:
                continue
            flag = ""
            if first["silent"]:
                flag = "  [SILENT]"
            elif first["truncated"]:
                flag = "  [TRUNCATED]"
            print(f"    {m:<16} {first['duration_s']:>6.2f}s{flag}")

    if ranking:
        ranking.sort()
        print("\n" + "=" * 78)
        print("  BY MEDIAN LATENCY")
        print("=" * 78)
        for median, m in ranking:
            print(f"    {m:<16} {median:>8.0f} ms")

    print("\n" + "=" * 78)
    print("  NEXT STEP - LISTEN")
    print("=" * 78)
    print(f"  Samples saved to {SAMPLES_DIR}/  as <model>__<case>.<ext>")
    print("  Nothing above measures whether a voice sounds good. Play the same")
    print("  case across models and judge:")
    print("    - naturalness: does it sound like a person or like a machine")
    print("    - numbers:     is '7 out of 10' and '1945' read correctly")
    print("    - names:       Ljubljana, Triglav, Canberra")
    print("    - pauses:      are the '...' rendered as pauses or read aloud")
    print("    - suitability: would this voice work as a quiz host")
    print("  .raw files are headerless PCM - open in Audacity as 16-bit signed")
    print("  mono at the model's sample rate.")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Screen TTS models")
    parser.add_argument("--reps", type=int, default=3, help="repetitions per case (default 3)")
    parser.add_argument("--models", type=str, default="", help="comma-separated model IDs")
    parser.add_argument("--no-save", action="store_true", help="do not write audio samples")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()] or list(MODELS)
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        print(f"ERROR: unknown model IDs {unknown}. Available: {list(MODELS)}")
        sys.exit(1)

    total = len(models) * len(CASES) * args.reps
    print(f"{len(models)} model(s) x {len(CASES)} case(s) x {args.reps} rep(s) = {total} calls\n")

    rows = []
    for model_id in models:
        print(f"--- {model_id} ---")
        for rep in range(1, args.reps + 1):
            for case_name, text in CASES:
                # only the first repetition is saved - the rest would be
                # near-identical and just clutter the folder
                save = (rep == 1) and not args.no_save
                row = synthesize_once(model_id, case_name, text, save)
                row["rep"] = rep
                rows.append(row)

                if row["error"]:
                    print(f"  [{rep}] ERR  {case_name:<10} {row['error'][:56]}")
                else:
                    flag = "SILENT" if row["silent"] else ("TRUNC " if row["truncated"] else "      ")
                    print(f"  [{rep}] {flag} {case_name:<10} {row['latency_ms']:>6} ms  "
                          f"{row['duration_s']:>5.2f}s  {row['size_kb']:>7.1f} KB")
                time.sleep(PAUSE_S)
        print()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{RESULTS_DIR}/tts_screening_{stamp}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["model", "case", "rep", "chars", "latency_ms", "duration_s",
                  "size_kb", "rtf", "silent", "truncated", "error"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summarise(rows, models)
    print(f"Raw results saved to: {output_path}\n")


if __name__ == "__main__":
    main()


# (.venv) PS C:\Users\aleks\Music\magg\voice_quiz\backend\evaluation> py .\screen_tts.py
# 6 model(s) x 7 case(s) x 3 rep(s) = 126 calls

# --- orpheus-or ---
#   [1]        short        3380 ms   0.77s     36.0 KB
#   [1]        verdict      3029 ms   3.24s    152.0 KB
#   [1]        typical      4718 ms   5.89s    276.0 KB
#   [1]        long         5074 ms  12.54s    588.0 KB
#   [1]        numbers      2728 ms   5.21s    244.0 KB
#   [1]        pauses       3723 ms   3.84s    180.0 KB
#   [1]        names        3997 ms   4.52s    212.0 KB
#   [2]        short        1549 ms   0.68s     32.0 KB
#   [2]        verdict      3197 ms   3.33s    156.0 KB
#   [2]        typical      1916 ms   5.21s    244.0 KB
#   [2]        long         4009 ms  10.75s    504.0 KB
#   [2]        numbers      4136 ms   5.38s    252.0 KB
#   [2]        pauses       1634 ms   3.84s    180.0 KB
#   [2]        names        4448 ms   5.38s    252.0 KB
#   [3]        short        1559 ms   0.85s     40.0 KB
#   [3]        verdict      3435 ms   4.10s    192.0 KB
#   [3]        typical      2571 ms   6.06s    284.0 KB
#   [3]        long         4427 ms  12.37s    580.0 KB
#   [3]        numbers      4280 ms   4.86s    228.0 KB
#   [3]        pauses       3526 ms   4.18s    196.0 KB
#   [3]        names        3529 ms   4.52s    212.0 KB

# --- kokoro ---
#   [1]        short        1084 ms   1.15s     53.9 KB
#   [1]        verdict      1790 ms   3.75s    175.8 KB
#   [1]        typical      1032 ms   6.67s    312.9 KB
#   [1]        long         2068 ms  13.93s    652.7 KB
#   [1]        numbers      1336 ms   6.22s    291.8 KB
#   [1]        pauses       1365 ms   4.42s    207.4 KB
#   [1]        names        1155 ms   5.35s    250.8 KB
#   [2]        short         790 ms   1.15s     53.9 KB
#   [2]        verdict      1194 ms   3.75s    175.8 KB
#   [2]        typical      1354 ms   6.67s    312.9 KB
#   [2]        long         2972 ms  13.93s    652.7 KB
#   [2]        numbers      1048 ms   6.22s    291.8 KB
#   [2]        pauses        835 ms   4.42s    207.4 KB
#   [2]        names        5052 ms   5.35s    250.8 KB
#   [3]        short        1122 ms   1.15s     53.9 KB
#   [3]        verdict      1302 ms   3.75s    175.8 KB
#   [3]        typical      1038 ms   6.67s    312.9 KB
#   [3]        long         1160 ms  13.93s    652.7 KB
#   [3]        numbers      2506 ms   6.22s    291.8 KB
#   [3]        pauses       6638 ms   4.42s    207.4 KB
#   [3]        names        1496 ms   5.01s    234.8 KB

# --- gemini-tts ---
#   [1]        short        2487 ms   1.12s     52.5 KB
#   [1]        verdict      3301 ms   4.44s    208.1 KB
#   [1]        typical      8389 ms   7.04s    330.0 KB
#   [1]        long         8745 ms  14.04s    658.1 KB
#   [1]        numbers      5476 ms   6.88s    322.5 KB
#   [1]        pauses       5762 ms   5.64s    264.4 KB
#   [1]        names        5930 ms   5.32s    249.4 KB
#   [2]        short        2276 ms   1.56s     73.1 KB
#   [2]        verdict      3540 ms   4.92s    230.6 KB
#   [2]        typical      5221 ms   6.28s    294.4 KB
#   [2]        long        16578 ms  14.12s    661.9 KB
#   [2]        numbers      7563 ms   7.08s    331.9 KB
#   [2]        pauses       6504 ms   6.12s    286.9 KB
#   [2]        names        4879 ms   5.32s    249.4 KB
#   [3]        short        2218 ms   1.28s     60.0 KB
#   [3]        verdict      2884 ms   4.72s    221.2 KB
#   [3]        typical      7369 ms   6.56s    307.5 KB
#   [3]        long        15716 ms  14.52s    680.6 KB
#   [3]        numbers      5213 ms   6.76s    316.9 KB
#   [3]        pauses       5224 ms   6.80s    318.8 KB
#   [3]        names        4300 ms   5.44s    255.0 KB

# --- qwen-tts ---
#   [1]        short         722 ms   1.25s     24.5 KB
#   [1]        verdict      1333 ms   4.06s     79.3 KB
#   [1]        typical      1504 ms   7.10s    138.8 KB
#   [1]        long         1986 ms  14.69s    287.0 KB
#   [1]        numbers      1442 ms   6.77s    132.3 KB
#   [1]        pauses       1028 ms   4.54s     88.7 KB
#   [1]        names         990 ms   5.66s    110.7 KB
#   [2]        short         649 ms   1.25s     24.5 KB
#   [2]        verdict       915 ms   4.06s     79.3 KB
#   [2]        typical      1366 ms   7.10s    138.8 KB
#   [2]        long         2040 ms  14.69s    287.0 KB
#   [2]        numbers      1210 ms   6.77s    132.3 KB
#   [2]        pauses       1369 ms   4.54s     88.7 KB
#   [2]        names        1011 ms   5.66s    110.7 KB
#   [3]        short         679 ms   1.25s     24.5 KB
#   [3]        verdict      1103 ms   4.06s     79.3 KB
#   [3]        typical      1429 ms   7.10s    138.8 KB
#   [3]        long         2072 ms  14.69s    287.0 KB
#   [3]        numbers      1231 ms   6.77s    132.3 KB
#   [3]        pauses       1501 ms   4.54s     88.7 KB
#   [3]        names        1340 ms   5.66s    110.7 KB

# --- deepgram-aura ---
#   [1]        short        1142 ms   0.89s      5.2 KB
#   [1]        verdict      1829 ms   3.46s     20.2 KB
#   [1]        typical      3483 ms   7.10s     41.6 KB
#   [1]        long         6730 ms  14.21s     83.2 KB
#   [1]        numbers      3080 ms   6.10s     35.7 KB
#   [1]        pauses       2789 ms   5.62s     32.9 KB
#   [1]        names        2379 ms   4.78s     28.0 KB
#   [2]        short         577 ms   0.70s      4.1 KB
#   [2]        verdict      2214 ms   4.06s     23.8 KB
#   [2]        typical      3084 ms   5.45s     31.9 KB
#   [2]        long         6379 ms  13.49s     79.0 KB
#   [2]        numbers      2778 ms   5.57s     32.6 KB
#   [2]        pauses       2932 ms   5.90s     34.6 KB
#   [2]        names        2339 ms   4.61s     27.0 KB
#   [3]        short         636 ms   0.89s      5.2 KB
#   [3]        verdict      1988 ms   3.86s     22.6 KB
#   [3]        typical      3006 ms   5.78s     33.9 KB
#   [3]        long         6520 ms  13.82s     81.0 KB
#   [3]        numbers      2741 ms   5.50s     32.2 KB
#   [3]        pauses       3428 ms   6.98s     40.9 KB
#   [3]        names        2489 ms   4.94s     29.0 KB

# --- minimax-speech ---
# Warning: Xing stream size off by more than 1%, fuzzy seeking may be even more fuzzy than by design!
#   [1]        short        1392 ms   0.94s     15.2 KB
# Warning: Xing stream size off by more than 1%, fuzzy seeking may be even more fuzzy than by design!
#   [1]        verdict      1002 ms   5.15s     81.0 KB
# Warning: Xing stream size off by more than 1%, fuzzy seeking may be even more fuzzy than by design!
#   [1]        typical      1260 ms   6.59s    103.5 KB
# Warning: Xing stream size off by more than 1%, fuzzy seeking may be even more fuzzy than by design!
#   [1]        long         1488 ms  14.04s    220.0 KB
# Warning: Xing stream size off by more than 1%, fuzzy seeking may be even more fuzzy than by design!
#   [1]        numbers      1126 ms   7.34s    115.4 KB
# Warning: Xing stream size off by more than 1%, fuzzy seeking may be even more fuzzy than by design!
#   [1]        pauses        960 ms   4.75s     74.9 KB
# Warning: Xing stream size off by more than 1%, fuzzy seeking may be even more fuzzy than by design!
#   [1]        names        1063 ms   4.61s     72.6 KB
# Warning: Xing stream size off by more than 1%, fuzzy seeking may be even more fuzzy than by design!
#   [2]        short         522 ms   0.86s     14.1 KB
# Warning: Xing stream size off by more than 1%, fuzzy seeking may be even more fuzzy than by design!
#   [2]        verdict      1129 ms   4.75s     74.9 KB
# Warning: Xing stream size off by more than 1%, fuzzy seeking may be even more fuzzy than by design!
#   [2]        typical      1021 ms   6.30s     99.0 KB
# Warning: Xing stream size off by more than 1%, fuzzy seeking may be even more fuzzy than by design!
#   [2]        long         1545 ms  15.26s    239.1 KB
# Warning: Xing stream size off by more than 1%, fuzzy seeking may be even more fuzzy than by design!
#   [2]        numbers      1432 ms   7.34s    115.4 KB
# Warning: Xing stream size off by more than 1%, fuzzy seeking may be even more fuzzy than by design!
#   [2]        pauses        866 ms   4.79s     75.4 KB
# Warning: Xing stream size off by more than 1%, fuzzy seeking may be even more fuzzy than by design!
#   [2]        names         746 ms   4.54s     71.5 KB
# Warning: Xing stream size off by more than 1%, fuzzy seeking may be even more fuzzy than by design!
#   [3]        short         704 ms   0.72s     11.9 KB
# Warning: Xing stream size off by more than 1%, fuzzy seeking may be even more fuzzy than by design!
#   [3]        verdict       969 ms   4.68s     73.7 KB
# Warning: Xing stream size off by more than 1%, fuzzy seeking may be even more fuzzy than by design!
#   [3]        typical       879 ms   6.37s    100.2 KB
# Warning: Xing stream size off by more than 1%, fuzzy seeking may be even more fuzzy than by design!
#   [3]        long         1922 ms  14.18s    222.2 KB
# Warning: Xing stream size off by more than 1%, fuzzy seeking may be even more fuzzy than by design!
#   [3]        numbers      1236 ms   5.94s     93.4 KB
# Warning: Xing stream size off by more than 1%, fuzzy seeking may be even more fuzzy than by design!
#   [3]        pauses        973 ms   4.50s     70.9 KB
# Warning: Xing stream size off by more than 1%, fuzzy seeking may be even more fuzzy than by design!
#   [3]        names         995 ms   5.18s     81.6 KB


# ==============================================================================
#   LATENCY
# ==============================================================================
#   model              n   median      p95     min      max    RTF   fails
#   orpheus-or        21     3526     4718    1549     5074   0.83       0
#   kokoro            21     1302     5052     790     6638   0.30       0
#   gemini-tts        21     5224    15716    2218    16578   1.02       0
#   qwen-tts          21     1333     2040     649     2072   0.21       0
#   deepgram-aura     21     2778     6520     577     6730   0.50       0
#   minimax-speech    21     1021     1545     522     1922   0.19       0
#      RTF = synthesis time / audio length. Below 1.0 means faster than real time.

#   median ms by test case
#   model               short   verdict   typical      long   numbers    pauses     names
#   orpheus-or            1559      3197      2571      4427      4136      3526      3997
#   kokoro                1084      1302      1038      2068      1336      1365      1496
#   gemini-tts            2276      3301      7369     15716      5476      5762      4879
#   qwen-tts               679      1103      1429      2040      1231      1369      1011
#   deepgram-aura          636      1988      3084      6520      2778      2932      2379
#   minimax-speech         704      1002      1021      1545      1236       960       995

# ==============================================================================
#   OUTPUT PROBLEMS
# ==============================================================================
#   model              silent   truncated   errors
#   orpheus-or              0           0        0
#   kokoro                  0           0        0
#   gemini-tts              0           0        0
#   qwen-tts                0           0        0
#   deepgram-aura           0           0        0
#   minimax-speech          0           0        0
#      silent    = file contains only zero samples
#      truncated = audio shorter than 60% of the duration the text implies

# ==============================================================================
#   AUDIO DURATION (s)  -  expected vs produced
# ==============================================================================

#   short  (8 chars, roughly 0.5s expected)
#     orpheus-or         0.77s
#     kokoro             1.15s
#     gemini-tts         1.12s
#     qwen-tts           1.25s
#     deepgram-aura      0.89s
#     minimax-speech     0.94s

#   verdict  (48 chars, roughly 3.2s expected)
#     orpheus-or         3.24s
#     kokoro             3.75s
#     gemini-tts         4.44s
#     qwen-tts           4.06s
#     deepgram-aura      3.46s
#     minimax-speech     5.15s

#   typical  (91 chars, roughly 6.1s expected)
#     orpheus-or         5.89s
#     kokoro             6.67s
#     gemini-tts         7.04s
#     qwen-tts           7.10s
#     deepgram-aura      7.10s
#     minimax-speech     6.59s

#   long  (217 chars, roughly 14.5s expected)
#     orpheus-or        12.54s
#     kokoro            13.93s
#     gemini-tts        14.04s
#     qwen-tts          14.69s
#     deepgram-aura     14.21s
#     minimax-speech    14.04s

#   numbers  (67 chars, roughly 4.5s expected)
#     orpheus-or         5.21s
#     kokoro             6.22s
#     gemini-tts         6.88s
#     qwen-tts           6.77s
#     deepgram-aura      6.10s
#     minimax-speech     7.34s

#   pauses  (52 chars, roughly 3.5s expected)
#     orpheus-or         3.84s
#     kokoro             4.42s
#     gemini-tts         5.64s
#     qwen-tts           4.54s
#     deepgram-aura      5.62s
#     minimax-speech     4.75s

#   names  (74 chars, roughly 4.9s expected)
#     orpheus-or         4.52s
#     kokoro             5.35s
#     gemini-tts         5.32s
#     qwen-tts           5.66s
#     deepgram-aura      4.78s
#     minimax-speech     4.61s

# ==============================================================================
#   BY MEDIAN LATENCY
# ==============================================================================
#     minimax-speech       1021 ms
#     kokoro               1302 ms
#     qwen-tts             1333 ms
#     deepgram-aura        2778 ms
#     orpheus-or           3526 ms
#     gemini-tts           5224 ms

# ==============================================================================
#   NEXT STEP - LISTEN
# ==============================================================================
#   Samples saved to tts_samples/  as <model>__<case>.<ext>
#   Nothing above measures whether a voice sounds good. Play the same
#   case across models and judge:
#     - naturalness: does it sound like a person or like a machine
#     - numbers:     is '7 out of 10' and '1945' read correctly
#     - names:       Ljubljana, Triglav, Canberra
#     - pauses:      are the '...' rendered as pauses or read aloud
#     - suitability: would this voice work as a quiz host
#   .raw files are headerless PCM - open in Audacity as 16-bit signed
#   mono at the model's sample rate.
# ==============================================================================

# Raw results saved to: ../../results/tts_screening_20260817_132041.csv

# (.venv) PS C:\Users\aleks\Music\magg\voice_quiz\backend\evaluation> 

# KOMENTAR

# Good call moving Orpheus — and the numbers show exactly how much the rate limiting was distorting things.

# Orpheus was never that slow
# 	on Groq	on OpenRouter
# median	7494 ms	3526 ms
# p95	8795 ms	4718 ms
# RTF	0.92	0.83

# Less than half the latency for the same model. And the flat-floor pattern I flagged last time — 6603 ms for "Correct!" — is gone: it's now 1559 ms for short and scales sensibly with length. That fixed cost was rate-limit queuing, not model behaviour.

# So my earlier reading of Orpheus was wrong, and for the same reason your first STT run was wrong. That's now four times free-tier throttling has masqueraded as model performance in your data. It really is the recurring methodological hazard of this project, and worth stating plainly in the thesis: measurements taken on rate-limited free tiers are not measurements of the model.

# MiniMax is the clear winner

# Median 1021 ms, p95 1545 ms, RTF 0.19 — and crucially the tightest spread of anything you've tested. Compare tails:

# minimax-speech   522 → 1922    (3.7× spread)
# qwen-tts         649 → 2072    (3.2×)
# kokoro           790 → 6638    (8.4×)
# gemini-tts      2218 → 16578   (7.5×)

# Kokoro's median looks competitive at 1302 ms, but its p95 of 5052 and max of 6638 make it unpredictable — the same pattern that made whisper-groq a poor choice despite a good median. Almost certainly the PCM transfer size, since it's the only model here sending uncompressed audio.

# Gemini is out

# RTF 1.02 means it synthesises slower than the audio plays, and 15.7 s on the long case is unusable. It also has the worst scaling — 2276 ms short to 15716 ms long, a 7× climb where MiniMax goes 704 → 1545.

# Suggested four
# model	median	p95	RTF	why
# minimax-speech	1021	1545	0.19	fastest and most consistent
# qwen-tts	1333	2040	0.21	close second, tight tail, different vendor
# kokoro	1302	5052	0.30	open-weight — the only self-hostable one, worth keeping for that reason alone
# deepgram-aura	2778	6520	0.50	purpose-built speech vendor, mid-tier reference point

# Dropping Gemini (RTF ≥ 1) and Orpheus (slowest of the rest, and now redundant since MiniMax and Qwen occupy the fast-commercial slot).

# Keeping Kokoro despite its tail is deliberate: it's your only open-weight TTS, which mirrors the open/proprietary split you built into the LLM lineup. Its variance is also interesting rather than disqualifying — you can attribute it to PCM transfer size, which is a finding rather than a mystery.

# One caveat: this is all latency. Your listening impression that MiniMax sounds better in the actual quiz matters as much, and that's the axis these numbers can't reach.