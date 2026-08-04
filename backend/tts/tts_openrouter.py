import os
import io
import re
import time
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import sounddevice as sd
import soundfile as sf

load_dotenv()

BASE_URL = "https://openrouter.ai/api/v1"

TTS_MODELS = {
    "1": ("Kokoro 82M", "hexgrad/kokoro-82m", "af_bella", "pcm"),
    "2": ("Gemini 3.1 Flash TTS", "google/gemini-3.1-flash-tts-preview", "Charon", "pcm"),
    "3": ("Qwen Audio 3.0 TTS", "qwen/qwen-audio-3.0-tts-flash", "loongjohn", "mp3"),
    "4": ("Mistral: Voxtral Mini TTS", "mistralai/voxtral-mini-tts-2603", "en_paul_sad", "mp3"),
    "5": ("Deepgram: Aura-2", "deepgram/aura-2", "aura-2-thalia-en", "pcm"),
}

# default values
SAMPLE_RATE = 24000
MODEL = "hexgrad/kokoro-82m"
VOICE = "af_bella"
RESPONSE_FORMAT = "pcm"

# Client initialization
client = None

def get_client() -> OpenAI:

    global client

    if client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not found in your .env file.")
        client = OpenAI(base_url=BASE_URL, api_key=api_key)

    return client

# Core function - takes text, returns audio bytes
def synthesize(text: str, voice: str = VOICE, model: str = MODEL, response_format: str = RESPONSE_FORMAT) -> tuple[bytes, int]:

    raw = get_client().audio.speech.with_raw_response.create(
        model=model, voice=voice, input=text, response_format=response_format,
    )
    audio_bytes = raw.content

    sample_rate = 0
    if response_format == "pcm":
        content_type = raw.headers.get("content-type", "")
        match = re.search(r"rate=(\d+)", content_type)
        sample_rate = int(match.group(1)) if match else SAMPLE_RATE

    return audio_bytes, sample_rate

# Wrapper - synthesizes and saves to file
def synthesize_to_file(text: str, file_path: str, voice: str = VOICE, model: str = MODEL, response_format: str = RESPONSE_FORMAT) -> None:
    audio_bytes, _ = synthesize(text, voice=voice, model=model, response_format=response_format)
    with open(file_path, "wb") as f:
        f.write(audio_bytes)

def play_audio_bytes(audio_bytes: bytes, fmt: str = RESPONSE_FORMAT, sample_rate: int = 0) -> None:
    if fmt == "pcm":
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        rate = sample_rate or SAMPLE_RATE
    else:
        audio, rate = sf.read(io.BytesIO(audio_bytes))

    sd.play(audio, samplerate=rate)
    sd.wait()

def main():
    print("-" * 50)
    print("TTS - Terminal mode (OpenRouter)")
    print("-" * 50)

    # Model selection
    print("\nAvailable models:")
    for key, (name, slug, voice, fmt) in TTS_MODELS.items():
        print(f"   {key}. {name:20s} voice={voice:18s} format={fmt}")
    choice = input("Choose a model (press Enter for 1): ").strip() or "1"
    model_name, model, voice, fmt = TTS_MODELS.get(choice, TTS_MODELS["1"])

    print(f"\nModel: {model}  |  Voice: {voice}  |  Format: {fmt}")
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
            audio_bytes, sample_rate = synthesize(text, voice=voice, model=model, response_format=fmt)
        except Exception as e:
            print(f"ERROR: {e}\n")
            continue
        latency_ms = int((time.time() - t_start) * 1000)

        # # Save to file
        # out_path = f"tts_output_{counter}.{RESPONSE_FORMAT}"
        # with open(out_path, "wb") as f:
        #     f.write(audio_bytes)

        size_kb = len(audio_bytes) / 1024
        rate_txt = f" {sample_rate} Hz" if sample_rate else ""
        print(f"Generated {size_kb:.0f} KB{rate_txt} in {latency_ms} ms")

        # Play it
        print("Playing...")
        try:
            play_audio_bytes(audio_bytes, fmt, sample_rate=sample_rate)
        except Exception as e:
            print(f"(Could not play audio: {e})")

        print("-" * 50 + "\n")
        counter += 1

if __name__ == "__main__":
    main()