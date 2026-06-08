import io
import time
import sys
from collections import deque
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

from whisper_groq import transcribe, MODEL

# AUDIO SETTINGS
SAMPLE_RATE = 16000     # 16 kHz is perfect for speech
CHANNELS = 1            # mono - speech doesn't need stereo
DTYPE = "int16"         # 16-bit audio - standard for WAV files

# VAD PARAMETERS
CHUNK_DURATION_MS = 30          # how big each audio chunk is
THRESHOLD_MULTIPLIER = 3.0      # speech = this number x background loudness
CALIBRATION_DURATION_MS = 500   # how long to measure background noise at startup
SILENCE_DURATION_MS = 1000      # how long the user is silent before we can say "speeking stopped"
MAX_RECORDING_MS = 15000        # safety upper limit for recording
PRE_SPEECH_BUFFER_MS = 300      # keep the last 300ms of audio in memory
SPEECH_START_CHUNKS = 3         # how many loud chunks in a row to detect speech

# Measure how loud audio chunk is by using Root Mean Square
def compute_rms(audio_chunk: np.ndarray) -> float:
    audio_float = audio_chunk.astype(np.float64)
    return float(np.sqrt(np.mean(audio_float ** 2)))

# Record background room noise to compute its average loudness
def calibrate_noise_floor() -> float:
    print(f"Calibrating background noise — stay quiet for {CALIBRATION_DURATION_MS}ms...")
    samples = int(SAMPLE_RATE * CALIBRATION_DURATION_MS / 1000)

    audio = sd.rec(samples, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE)
    sd.wait()

    audio = audio[:, 0]
    noise_rms = compute_rms(audio)
    print(f"Noise floor: {noise_rms:.1f} (threshold will be {noise_rms * THRESHOLD_MULTIPLIER:.1f})")
    return noise_rms

# main speech detection function
def record_vad(noise_floor: float) -> np.ndarray | None:
    
    threshold = noise_floor * THRESHOLD_MULTIPLIER
    samples_per_chunk = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

    chunks_for_silence = SILENCE_DURATION_MS // CHUNK_DURATION_MS
    chunks_for_max = MAX_RECORDING_MS // CHUNK_DURATION_MS
    chunks_for_pre_buffer = PRE_SPEECH_BUFFER_MS // CHUNK_DURATION_MS

    pre_buffer = deque(maxlen=chunks_for_pre_buffer)

    recording = []
    speech_started = False
    silent_chunks = 0
    loud_chunks = 0
    total_chunks = 0

    with sd.InputStream(
        samplerate=SAMPLE_RATE, 
        channels=CHANNELS, 
        dtype=DTYPE, 
        blocksize=samples_per_chunk,
    ) as stream: 
        
        print("Listening...")

        while True:
            chunk, overflowed = stream.read(samples_per_chunk)
            chunk = chunk[:, 0]

            rms = compute_rms(chunk)
            is_loud = rms > threshold
            total_chunks += 1

            if not speech_started:
                # WAITING FOR SPEECH
                pre_buffer.append(chunk)

                if is_loud:
                    loud_chunks += 1
                    if loud_chunks >= SPEECH_START_CHUNKS:
                        print("Speech detected! Recording...")
                        recording.extend(list(pre_buffer))
                        speech_started = True
                        silent_chunks = 0
                else:
                    loud_chunks = 0

                if total_chunks >= chunks_for_max:
                    print("Timeout - no speech detected")
                    return None
            
            else:
                # ACIVELY RECORDING
                recording.append(chunk)

                if is_loud:
                    silent_chunks = 0
                else:
                    silent_chunks += 1
                    if silent_chunks >= chunks_for_silence:
                        print("Silence detedted! Recording finished")
                        break

                if len(recording) >= chunks_for_max:
                    print("Max duration reached! Finishing")
                    break

    if not recording:
        return None
    return np.concatenate(recording, axis=0)

def audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    wav_write(buffer, SAMPLE_RATE, audio)
    buffer.seek(0)
    return buffer.read()

def main():
    print("STT — VAD test\n")

    noise_floor = calibrate_noise_floor()
 
    print("\nReady.")
 
    while True:
        cmd = input("Press Enter to listen ('q' = quit): ").strip().lower()
        if cmd == "q":
            break
 
        audio = record_vad(noise_floor)
 
        if audio is None:
            continue
 
        duration_s = len(audio) / SAMPLE_RATE
        print(f"Captured {duration_s:.1f} seconds of audio")
 
        # Convert and transcribe
        wav_bytes = audio_to_wav_bytes(audio)
 
        print("Transcribing...")
        t_start = time.time()
        try:
            transcript = transcribe(wav_bytes, filename="vad_recording.wav")
        except Exception as e:
            print(f"ERROR: {e}\n")
            continue
        latency_ms = int((time.time() - t_start) * 1000)
 
        print("-" * 50)
        print(f"Transcript: {transcript}")
        print(f"Latency:    {latency_ms} ms")
        print(f"Words:      {len(transcript.split())}")
        print("-" * 50 + "\n")
 
 
if __name__ == "__main__":
    main()