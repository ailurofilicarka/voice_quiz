from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import time
import io
import traceback

from llm.groq_client import evaluate_answer, generate_question
from stt.whisper_groq import transcribe

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
    stt: str = "whisper-groq"
    llm: str = "llama-groq"
    tts: str = "openai-tts"

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
current_session = {}   # session_id → quiz state dict

@app.get("/")
async def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")

# called when 'start quiz' pressed
@app.post("/api/start")
async def start_quiz(request: StartRequest):
    global current_session
    t_start = time.time()

    # generate first question with LLM
    try:
        t_llm_start = time.time()
        first_question = generate_question(
            topic=request.topic,
            question_number=1,
            previous_questions=[],
        )
        llm_latency_ms = int((time.time() - t_llm_start) * 1000)
    except Exception as e:
        print("\n=== START QUIZ ERROR ===")
        traceback.print_exc()
        print("==========================")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate first question: {str(e)}"
        )

    # initialize the session
    current_session = {
        "topic": request.topic,
        "num_questions": request.num_questions,
        "previous_questions": [first_question],
        "current_index": 0,
        "correct_count": 0,
        "config": request.config.model_dump(),
    }

    total_latency_ms = int((time.time() - t_start) * 1000)

    return {
        "success": True,
        "message": f"Welcome! Here is your first question: {first_question}",
        "next_question": first_question,
        "quiz_done": False,
        "latency_ms": {
            "llm": llm_latency_ms,
            "total": total_latency_ms
        },
    }

# called after user voice been recorded
@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...), stt_model: str = "whisper-groq"):
    t_start = time.time()

    # read audio bytes - will be updated
    audio_bytes = await audio.read()
    file_size_kb = round(len(audio_bytes) / 1024, 1)

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="No audio data received"
        )
    
    filename = audio.filename or "audio.wav"    # so whisper can detect the format from the extension

    # Call SST module
    # sst_model param is still only info
    try:
        t_sst_start = time.time()
        transcript = transcribe(audio_bytes, filename=filename)
        sst_latency_ms = int((time.time() - t_sst_start) * 1000)
    except Exception as e:
        print("\n=== SST ERROR ===")
        traceback.print_exc()
        print("====================\n")
        raise HTTPException(
            status_code=500,
            detail=f"SST failed: {str(e)}"
        )

    total_latency_ms = int((time.time() - t_start) * 1000)

    return {
        "success": True,
        "transcript": transcript,
        "stt_model": stt_model,
        "file_size_kb": file_size_kb,
        "latency_ms": {
            "sst": sst_latency_ms,
            "total": total_latency_ms,
        },
    }

@app.post("/api/evaluate")
async def evaluate_answer_endpoint(request: AnswerRequest):
    
    global current_session
    t_start = time.time()

    if not current_session:
        raise HTTPException(
            status_code=400,
            detail="No active quiz session."
        )

    # evaluate user's answer
    try:
        t_llm_start = time.time()
        is_correct, llm_feedback = evaluate_answer(request.question, request.transcript)
        eval_latency_ms = int((time.time() - t_llm_start) * 1000)
    except Exception as e:
        import traceback
        print("\n=== LLM EVALUATION ERROR ===")
        traceback.print_exc()
        print("============================\n")
        raise HTTPException(
            status_code=500,
            detail=f"LLM evaluation failed: {str(e)}"
        )
    
    current_session["current_index"] += 1
    if is_correct:
        current_session["correct_count"] += 1

    quiz_done = current_session["current_index"] >= current_session["num_questions"]

    next_question = ""
    gen_latency_ms = 0
    if not quiz_done:
        try:
            t_gen_start = time.time()
            next_question = generate_question(
                topic = current_session["topic"],
                question_number = current_session["current_index"] + 1,
                previous_questions = current_session["previous_questions"],
            )
            gen_latency_ms = int((time.time() - t_gen_start) * 1000)
            current_session["previous_questions"].append(next_question)
        except Exception as e:
            print("\n=== QUESTION GENERATION ERROR ===")
            traceback.print_exc()
            print("===================================")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate next question: {str(e)}"
            )
        
    if quiz_done:
        message = llm_feedback + " That was the last question - quiz complete!"
    else:
        message = llm_feedback

    total_latency_ms = int((time.time() - t_start) * 1000)

    return QuizResponse(
        success=True,
        message=message,
        is_correct=is_correct,
        next_question=next_question,
        quiz_done=quiz_done,
        latency_ms={
            "llm": eval_latency_ms + gen_latency_ms,
            "llm_eval": eval_latency_ms,
            "llm_gen": gen_latency_ms,
            "total": total_latency_ms,
        },
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