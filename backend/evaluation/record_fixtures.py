import os
import queue
import re
import shutil
import subprocess
import sys
import threading

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 48000
CHANNELS = 1
DTYPE = "int16"

OUTPUT_DIR = "fixtures"

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def record_until_enter() -> np.ndarray:
    """Record from the default microphone until Enter is pressed."""
    audio_queue: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[mic status: {status}]", file=sys.stderr)
        audio_queue.put(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=callback,
    )

    print("  recording... press Enter to stop.")
    stream.start()
    input()
    stream.stop()
    stream.close()

    chunks = []
    while not audio_queue.empty():
        chunks.append(audio_queue.get())

    if not chunks:
        return np.array([], dtype=DTYPE)
    return np.concatenate(chunks, axis=0)


def save_webm(audio: np.ndarray, path: str) -> bool:

    command = [
        "ffmpeg", "-y",
        "-f", "s16le",                  # input: raw 16-bit little-endian PCM
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-i", "pipe:0",
        "-c:a", "libopus",              # what the browser uses inside WebM
        "-b:a", "64k",
        path,
    ]
    result = subprocess.run(
        command,
        input=audio.tobytes(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        print("  ffmpeg failed:")
        print("  " + result.stderr.decode(errors="replace").strip()[-300:])
        return False
    return True


def save_wav(audio: np.ndarray, path: str) -> None:
    """Fallback when ffmpeg is unavailable."""
    from scipy.io import wavfile
    wavfile.write(path, SAMPLE_RATE, audio)


def safe_name(name: str) -> str:
    """Turn free text into a filename: lowercase, no spaces or odd characters."""
    name = name.strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_\-]", "", name)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    extension = "webm" if HAS_FFMPEG else "wav"

    print("=" * 56)
    print("  FIXTURE RECORDER")
    print(f"  {SAMPLE_RATE} Hz mono -> .{extension}  ->  {OUTPUT_DIR}/")
    if not HAS_FFMPEG:
        print("  WARNING: ffmpeg not found - saving .wav instead of .webm.")
        print("  The browser sends webm, so wav fixtures are less faithful.")
    print("=" * 56)
    print("\n  Enter = start, Enter = stop, then type a filename.")
    print("  Type q at the prompt to quit.\n")

    count = 0
    while True:
        command = input("> press Enter to record (q to quit): ").strip().lower()
        if command == "q":
            print(f"\n  Done. {count} file(s) saved in {OUTPUT_DIR}/\n")
            break

        audio = record_until_enter()
        duration = len(audio) / SAMPLE_RATE
        print(f"  captured {duration:.1f}s")

        if duration < 0.3:
            print("  too short - discarded.\n")
            continue

        # keep the recording in memory until a name is given, so a mistyped
        # name does not cost the take
        while True:
            raw_name = input("  filename (or Enter to discard): ").strip()
            if not raw_name:
                print("  discarded.\n")
                break

            name = safe_name(raw_name)
            if not name:
                print("  that name has no usable characters, try again.")
                continue

            path = os.path.join(OUTPUT_DIR, f"{name}.{extension}")
            if os.path.exists(path):
                overwrite = input(f"  {path} exists. Overwrite? (y/n): ").strip().lower()
                if overwrite != "y":
                    continue

            if HAS_FFMPEG:
                ok = save_webm(audio, path)
            else:
                save_wav(audio, path)
                ok = True

            if ok:
                size_kb = os.path.getsize(path) / 1024
                print(f"  saved {path}  ({size_kb:.0f} KB)\n")
                count += 1
            break


if __name__ == "__main__":
    main()