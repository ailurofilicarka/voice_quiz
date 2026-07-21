##############################################
# Speech-to-Text using Whisper via Groq's API
##############################################
import os
import sys
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Configuration

# other options:
# whisper-large-v3 - most accurate, slower
# distil-whisper-large-v3-en - English-only, very fast
MODEL = "whisper-large-v3-turbo"

# whisper can also auto-detect
LANGUAGE = "en"

# Create the Groq client on first use
client = None

def get_client() -> Groq:
    global client

    if client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not found")
        client = Groq(api_key=api_key)

    return client

# Takes raw audio bytes (wav, mp3, m4a, ogg, flac)
# and returns the transcribed text
def transcribe(audio_bytes: bytes, filename: str = "audio.wav", model: str = MODEL) -> str:
    client = get_client()

    response = client.audio.transcriptions.create(
        file = (filename, audio_bytes),
        model = model,
        language = LANGUAGE,
        response_format = "json",
    )

    return response.text.strip()

# Wraper - reads a file from disk and transcribes it
def transcribe_file(file_path: str) -> str:
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    filename = os.path.basename(file_path)
    return transcribe(audio_bytes, filename)

# For testing in terminal
def main():
    if len(sys.argv) < 2:
        print("ERROR arg")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.isfile(file_path):
        print("ERROR path")
        sys.exit(1)

    print("STT — Terminal test")
    print("\nTranscribing... (this calls Groq API)\n")
 
    t_start = time.time()
    try:
        transcript = transcribe_file(file_path)
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