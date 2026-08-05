import os
import sys
import time
import io
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = "https://openrouter.ai/api/v1"

MODEL = "openai/whisper-1"
LANGUAGE = "en"

# name, slug, language
STT_MODELS = {
    "1": ("Whisper v1 (OpenAI)", "openai/whisper-1", "en"),
    "2": ("GPT-4o Mini Transcribe", "openai/gpt-4o-mini-transcribe", "en"),
    "3": ("Deepgram Nova-3", "deepgram/nova-3", "en"),
    "4": ("NVIDIA Parakeet TDT", "nvidia/parakeet-tdt-0.6b-v3", "en"),
    "5": ("Voxtral Mini Transcribe", "mistralai/voxtral-mini-transcribe", "en"),
    "6": ("Qwen3 ASR Flash", "qwen/qwen3-asr-flash-2026-02-10", "en"),
    "7": ("Google Chirp 3", "google/chirp-3", "en"),
}

client = None

def get_client() -> OpenAI:
    global client

    if client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not found in your .env file.")
        client = OpenAI(base_url=BASE_URL, api_key=api_key)

    return client

# Takes raw audio bytes  and returns the transcribed text
def transcribe(audio_bytes: bytes, filename: str = "audio.wav", model: str = MODEL, language: str | None = LANGUAGE) -> str:
    client = get_client()

    kwargs = {}
    if language:
        kwargs["language"] = language

    response = client.audio.transcriptions.create(
        file = (filename, io.BytesIO(audio_bytes)),
        model = model,
        response_format = "json",
        **kwargs,
    )

    return response.text.strip()

# Wrapper - reads a file from disk and transcribes it
def transcribe_file(file_path: str, model: str = MODEL, language: str | None = LANGUAGE) -> str:
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    filename = os.path.basename(file_path)
    return transcribe(audio_bytes, filename, model=model, language=language)

# terminal test
def main():
    if len(sys.argv) < 2:
        print("Usage: py stt_openrouter.py <path_to_audio_file>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.isfile(file_path):
        print(f"ERROR: file not found: {file_path}")
        sys.exit(1)

    # Model selection
    print("\nAvailable models:")
    for key, (name, slug, lang) in STT_MODELS.items():
        print(f"{key}. {name:24s} ({slug})")
    choice = input("Choose a model (press Enter for 1): ").strip() or "1"
    model_name, model, language = STT_MODELS.get(choice, STT_MODELS["1"])

    size_kb = os.path.getsize(file_path) / 1024

    print("\nSTT — Terminal test (OpenRouter)")
    print(f"Model: {model}")
    print(f"File:  {file_path} ({size_kb:.1f} KB)")
    print("\nTranscribing...\n")

    t_start = time.time()
    try:
        transcript = transcribe_file(file_path, model=model, language=language)
    except Exception as e:
        print(f"ERROR: Transcription failed: {e}")
        sys.exit(1)

    duration_ms = int((time.time() - t_start) * 1000)

    print("-" * 50)
    print(f"Transcript: {transcript}")
    print(f"Latency: {duration_ms} ms")
    print(f"Words:   {len(transcript.split())}")
    print("-" * 50)

if __name__ == "__main__":
    main()