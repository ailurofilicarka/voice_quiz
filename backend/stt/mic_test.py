import sys
import time
import threading
import queue
import io

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

from whisper_groq import transcribe, MODEL

# AUDIO SETTINGS
SAMPLE_RATE = 16000     # 16 kHz is perfect for speech
CHANNELS = 1            # mono - speech doesn't need stereo
DTYPE = "int16"         # 16-bit audio - standard for WAV files

# Record from the default microphone until the user presses Enter.
# Returns the recorded audio as a 1D NumPy array of int16 samples.
#
# audio capture - background thread
# waiting for the Enter key - main thread
def record_until_enter() -> np.ndarray:
    # A thread-safe queue for audio chunks
    audio_queue: queue.Queue = queue.Queue()

    # flag to tell the background thread to stop
    stop_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        if status:
            # status warnings
            print(f"[mic status: {status}]", file=sys.stderr)
        audio_queue.put(indata.copy())

    # Start recording - runs in background thread automatically
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=audio_callback,
    )

    print("Recording... press Enter to stop.")
    stream.start()

    # block the main thread until enter is pressed (stop recording)
    input()
    stop_event.set()

    # stop and clean up
    stream.stop()
    stream.close()

    # collect all the chunks from the queue
    chunks = []
    while not audio_queue.empty():
        chunks.append(audio_queue.get())

    if not chunks:
        return np.array([], dtype=DTYPE)

    audio = np.concatenate(chunks, axis=0)
    return audio

# converts a NumPy audio array into raw WAV file bytes
# as Whisper's transcribe() function expects
def audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    wav_write(buffer, SAMPLE_RATE, audio)

    buffer.seek(0)
    return buffer.read()

# main loop - only for running in terminal
def main():
    print("STT — Live microphone test")
    print(f"Model: {MODEL}\n")

    while True:
        cmd = input("Press Enter to record (or 'q' to quit): ").strip().lower()
        if cmd == "q":
            break

        # Record from the mic
        audio = record_until_enter()

        duration_s = len(audio) / SAMPLE_RATE
        print(f"Stopped. Captured {duration_s:.1f} seconds of audio.")

        wav_bytes = audio_to_wav_bytes(audio)

        print("Transcribing...")
        t_start = time.time()
        try:
            # filename - because of the format
            transcript = transcribe(wav_bytes, filename="mic_recording.wav")
        except Exception as e:
            print(f"ERROR: {e}\n")
            continue
        latency_ms = int((time.time() - t_start) * 1000)

        # Show the result
        print("-" * 50)
        print(f"Transcript: {transcript}")
        print(f"Latency:    {latency_ms} ms")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()