import os
import sys
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Configuration
MODEL = "canopylabs/orpheus-v1-english" 
VOICE = "autumn"      # hannah, austin, autumn, troy
RESPONSE_FORMAT = "wav"

# Client initialization
client = None       # will be created on first use

def get_client() -> Groq:
    
    global client
    
    if client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not found in your .env file.")
        client = Groq(api_key=api_key)

    return client

# Core function - takes text, returns audio bytes
def synthesize(text: str, voice: str = VOICE, model: str = MODEL) -> bytes:
    client = get_client()
    response = client.audio.speech.create(model=model, voice=voice, input=text, response_format=RESPONSE_FORMAT)
    raw_byes = response.read()
    return fix_wav_header(raw_byes)

def fix_wav_header(wav_bytes: bytes) -> bytes:
    import io
    import warnings
    from scipy.io import wavfile

    # the warning says the file size written in the header doesn't match the actual file size,
    # but the audio data itself is read correctly anyway
    with warnings.catch_warnings():
        # scilence the warning
        warnings.simplefilter("ignore")
        sample_rate, audio = wavfile.read(io.BytesIO(wav_bytes))

    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, audio)
    buffer.seek(0)
    return buffer.read()

# Wrapper - synthesizes and saves to wav file
def synthesize_to_file(text: str, file_path: str, voice: str = VOICE) -> None:
    audio_bytes = synthesize(text, voice=voice)
    with open(file_path, "wb") as f:
        f.write(audio_bytes)

# Terminal mode
def play_wav_bytes(wav_bytes: bytes) -> None:

    import io
    import sounddevice as sd
    from scipy.io import wavfile

    sample_rate, audio = wavfile.read(io.BytesIO(wav_bytes))

    sd.play(audio, samplerate=sample_rate)
    sd.wait()

def main():
    print("-" * 50)
    print("TTS - Terminal mode")
    print("-" * 50)

    counter = 1
    while True:
        text = input("Text to speak (or 'q' to quit): ").strip()
        if text.lower() == "q":
            break
        if not text:
            continue
 
        print("Synthesizing...")
        t_start = time.time()
        try:
            wav_bytes = synthesize(text)
        except Exception as e:
            print(f"ERROR: {e}\n")
            continue
        latency_ms = int((time.time() - t_start) * 1000)
 
        # # Save to file
        # out_path = f"tts_output_{counter}.wav"
        # with open(out_path, "wb") as f:
        #     f.write(wav_bytes)
 
        size_kb = len(wav_bytes) / 1024
        print(f"Generated {size_kb:.0f} KB in {latency_ms} ms")
 
        # Play it
        print("Playing...")
        try:
            play_wav_bytes(wav_bytes)
        except Exception as e:
            print(f"(Could not play audio: {e} — but the WAV file was saved.)")
 
        print("-" * 50 + "\n")
        counter += 1
 
 
if __name__ == "__main__":
    main()