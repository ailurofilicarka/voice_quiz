from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import time
import io

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app = FastAPI(title="Voice Quiz API", version="0.1.0")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def no_cache_static(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
    return response

# Pydantic models
class ModelConfig(BaseModel):
    stt: str = "whisper-openai"     # sst model
    llm: str = "gpt4o-mini"         # llm
    tts: str = "openai-tts"         # tss model


class AnswerRequest(BaseModel):
    transcript: str               # what the user said (from STT)
    question: str                 # the question that was asked
    question_index: int           # which question number
    config: ModelConfig           # which models to use


class StartRequest(BaseModel):
    topic: str = "general knowledge"
    num_questions: int = 5
    config: ModelConfig = ModelConfig()


class QuizResponse(BaseModel):
    success: bool
    message: str                  # text to speak via TTS
    is_correct: bool = False      # did the user answer correctly?
    next_question: str = ""       # the next question to ask
    quiz_done: bool = False       # True when all questions are finished
    latency_ms: dict = {}         # timing data for the debug panel

# in-memory quiz state
quiz_sessions = {}   # session_id → quiz state dict

def get_placeholder_questions(topic: str) -> list[str]:
    return [
        "What is the capital of France?",
        "What is the largest planet in our solar system?",
        "Who wrote the play Romeo and Juliet?",
        "What is the chemical symbol for water?",
        "In what year did the First World War begin?",
    ] # hard coded questions for now

@app.get("/")
async def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")

# called when 'start quiz' pressed
@app.post("/api/start")
async def start_quiz(request: StartRequest):
    t_start = time.time()

    # create a simple session ID (timestamp-based for now)
    session_id = str(int(time.time()))

    # get questions
    questions = get_placeholder_questions(request.topic)

    # store session state in memory
    quiz_sessions[session_id] = {
        "questions":      questions,
        "current_index":  0,
        "correct_count":  0,
        "config":         request.config.model_dump(),
    }

    first_question = questions[0]

    # PLACEHOLDER latency (wiil be updated)
    latency_ms = {"llm": 0, "tts": 0, "total": int((time.time() - t_start) * 1000)}

    return {
        "success": True,
        "session_id": session_id,
        "message": f"Welcome! Here is your first question: {first_question}",
        "next_question": first_question,
        "quiz_done": False,
        "latency_ms": latency_ms,
    }

# called after user voice been recorded
@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...), stt_model: str = "whisper-openai"):
    t_start = time.time()

    # read audio bytes - will be updated
    audio_bytes = await audio.read()
    file_size_kb = round(len(audio_bytes) / 1024, 1)

    # fake transcript
    transcript = "Paris"    # hardcoded for now

    latency_ms = int((time.time() - t_start) * 1000)

    return {
        "success": True,
        "transcript": transcript,
        "stt_model": stt_model,
        "file_size_kb": file_size_kb,
        "latency_ms": latency_ms,
    }

@app.post("/api/evaluate")
async def evaluate_answer(request: AnswerRequest):
    t_start = time.time()

    session = quiz_sessions.get("default") # simple single-session for now

    # naive correctness check — just looks for keywords
    correct_answers = {
        "What is the capital of France?":                     ["paris"],
        "What is the largest planet in our solar system?":    ["jupiter"],
        "Who wrote the play Romeo and Juliet?":               ["shakespeare", "william"],
        "What is the chemical symbol for water?":             ["h2o"],
        "In what year did the First World War begin?":        ["1914"],
    }
    expected = correct_answers.get(request.question, [])
    is_correct = any(
        kw in request.transcript.lower() for kw in expected
    )

    # build feedback message
    if is_correct:
        feedback = f"Correct! Well done."
    else:
        hint = expected[0] if expected else "the correct answer"
        feedback = f"Not quite — the answer was {hint}."

    # decide next question
    next_questions = get_placeholder_questions("general knowledge")
    next_idx = request.question_index + 1
    quiz_done = next_idx >= len(next_questions)
    next_question = "" if quiz_done else next_questions[next_idx]

    if quiz_done:
        message = feedback + " That was the last question — quiz complete!"
    else:
        message = feedback + f" Next question: {next_question}"

    latency_ms = {
        "llm":   int((time.time() - t_start) * 1000),
        "total": int((time.time() - t_start) * 1000),
    }

    return QuizResponse(
        success=True,
        message=message,
        is_correct=is_correct,
        next_question=next_question,
        quiz_done=quiz_done,
        latency_ms=latency_ms,
    )

# TO DO
@app.post("/api/speak")
async def text_to_speech(text: str, tts_model: str = "openai-tts"):

    t_start = time.time()

    # placeholder: return a valid but silent WAV file (44 bytes — just the header)
    silent_wav = bytes([
        0x52, 0x49, 0x46, 0x46,  # "RIFF"
        0x24, 0x00, 0x00, 0x00,  # chunk size
        0x57, 0x41, 0x56, 0x45,  # "WAVE"
        0x66, 0x6D, 0x74, 0x20,  # "fmt "
        0x10, 0x00, 0x00, 0x00,  # subchunk1 size
        0x01, 0x00,              # PCM format
        0x01, 0x00,              # mono
        0x44, 0xAC, 0x00, 0x00,  # 44100 Hz sample rate
        0x88, 0x58, 0x01, 0x00,  # byte rate
        0x02, 0x00,              # block align
        0x10, 0x00,              # bits per sample
        0x64, 0x61, 0x74, 0x61,  # "data"
        0x00, 0x00, 0x00, 0x00,  # data size = 0 (silent)
    ])

    latency_ms = int((time.time() - t_start) * 1000)

    # placeholder
    return StreamingResponse(io.BytesIO(silent_wav), media_type="audio/wav", headers={"X-Latency-Ms": str(latency_ms), "X-TTS-Model":  tts_model,})

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "version": "0.1.0",
        "message": "Voice Quiz API is running (placeholder mode)",
    }

# models for debug pannel
@app.get("/api/models")
async def list_models():
    return {
        "stt": [
            {"id": "whisper-openai", "name": "Whisper (OpenAI)", "provider": "OpenAI"},
            {"id": "whisper-groq",   "name": "Whisper (Groq)",   "provider": "Groq"},
            {"id": "deepgram",       "name": "Deepgram Nova-2",  "provider": "Deepgram"},
        ],
        "llm": [
            {"id": "gpt4o-mini",    "name": "GPT-4o mini",      "provider": "OpenAI"},
            {"id": "claude-sonnet", "name": "Claude Sonnet",     "provider": "Anthropic"},
            {"id": "llama-groq",    "name": "Llama 3.3 (Groq)", "provider": "Groq"},
        ],
        "tts": [
            {"id": "openai-tts",  "name": "OpenAI TTS-1",   "provider": "OpenAI"},
            {"id": "elevenlabs",  "name": "ElevenLabs",      "provider": "ElevenLabs"},
            {"id": "kokoro",      "name": "Kokoro (local)",  "provider": "Local"},
        ],
    }