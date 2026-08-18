from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import time
import io
import traceback
import wave

from llm import groq_client, openrouter_client
from tts import tts_groq, tts_openrouter
from stt import whisper_groq, stt_openrouter
from llm.hosts import GREETINGS

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
    tts: str = "orpheus"

class AnswerRequest(BaseModel):
    transcript: str               # what the user said (from STT)
    question: str                 # the question that was asked
    question_index: int           # which question number
    config: ModelConfig           # which models to use

class SpeakRequest(BaseModel):
    text: str                     # the text to synthesize
    tts_model: str = "orpheus"    # info for now

class StartRequest(BaseModel):
    topic: str = "general knowledge"
    num_questions: int = 5
    personality: str = "classic"
    config: ModelConfig = ModelConfig()

class QuizResponse(BaseModel):
    success: bool
    message: str                  # text to speak via TTS
    is_correct: bool = False      # did the user answer correctly?
    next_question: str = ""       # the next question to ask
    quiz_done: bool = False       # True when all questions are finished
    latency_ms: dict = {}         # timing data for the debug panel

# MODEL REGISTRY
LLM_MODELS = {
    # "llama-groq": {"provider": "groq", "model": "llama-3.3-70b-versatile"}, NEMA VISE
    # "llama-8b-groq": "llama-3.1-8b-instant",
    "qwen-openrouter": {"provider": "openrouter", "model": "qwen/qwen3.6-27b"},
    "mistral-openrouter": {"provider": "openrouter", "model": "mistralai/mistral-medium-3-5"},
    "gpt-openrouter":  {"provider": "openrouter", "model": "openai/gpt-5.4-mini"},
    "gemini-openrouter": {"provider": "openrouter", "model": "google/gemini-3.7-flash"},
    "nemotron-openrouter": {"provider": "openrouter", "model": "nvidia/nemotron-3.5-lightning:free"}
}

LLM_CLIENTS = {
    "groq": groq_client,
    "openrouter": openrouter_client,
}

LLM_TIMEOUT_S = {
    "groq": 20,
    "openrouter": 25,
}

STT_MODELS = {
    "whisper-large-v3": {"provider": "openrouter", "model": "openai/whisper-large-v3-turbo", "language": "en"},
    "gpt4o-transcribe": {"provider": "openrouter", "model": "openai/gpt-4o-mini-transcribe", "language": "en"},
    "nova-3": {"provider": "openrouter", "model": "deepgram/nova-3", "language": "en"},
    "voxtral-stt": {"provider": "openrouter", "model": "mistralai/voxtral-mini-transcribe", "language": "en"},
}

STT_CLIENTS = {
    "groq": whisper_groq,
    "openrouter": stt_openrouter,
}

TTS_MODELS = {
    # "orpheus": {"provider": "groq", "model": "canopylabs/orpheus-v1-english", "voice": "troy", "format": "wav"},
    "kokoro": {"provider": "openrouter", "model": "hexgrad/kokoro-82m", "voice": "af_bella", "format": "pcm"},
    # "gemini-tts": {"provider": "openrouter", "model": "google/gemini-3.1-flash-tts-preview", "voice": "Charon", "format": "pcm"},
    "qwen-tts": {"provider": "openrouter", "model": "qwen/qwen-audio-3.0-tts-flash", "voice": "loongjohn", "format": "mp3"},
    "deepgram-aura": {"provider": "openrouter", "model": "deepgram/aura-2", "voice": "aura-2-thalia-en", "format": "pcm"},
    "minimax-speech": {"provider": "openrouter", "model": "minimax/speech-2.8-turbo", "voice": "Friendly_Person", "format": "mp3"},
}

TTS_CLIENTS = {
    "groq": tts_groq,
    "openrouter": tts_openrouter,
}

PCM_SAMPLE_RATE = 24000

def resolve_model(registry: dict, model_id: str, component: str) -> str | dict:
    if model_id not in registry:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown {component} model: '{model_id}'. "
                   f"Available: {list(registry.keys())}"
        )
    return registry[model_id]

def resolve_llm(model_id: str):

    entry = LLM_MODELS.get(model_id)
    if entry is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown LLM model: '{model_id}'. Available: {list(LLM_MODELS.keys())}"
        )
    provider = entry["provider"]
    client_module = LLM_CLIENTS[provider]
    return client_module, entry["model"]

# NORMALIZATION LAYER

def call_generate_question(client_module, model_name: str,
                           topic: str, question_number: int, previous_questions: list[str]):
    t0 = time.time()
    result = client_module.generate_question(
        topic=topic, question_number=question_number,
        previous_questions=previous_questions, model=model_name,
    )
    if isinstance(result, tuple):
        question, _reasoning, client_latency_ms = result
        return question, client_latency_ms
    return result, (time.time() - t0) * 1000
 
 
def call_evaluate_answer(client_module, model_name: str, question: str, transcript: str,
                         host: str, question_number: int, total_questions: int, score: int, streak: int):
    t0 = time.time()
    result = client_module.evaluate_answer(question, transcript, model=model_name, host=host,
        question_number=question_number, total_questions=total_questions, score=score, streak=streak,)
    if len(result) == 4:
        is_correct, speech, _reasoning, client_latency_ms = result
        return is_correct, speech, client_latency_ms
    is_correct, speech = result
    return is_correct, speech, (time.time() - t0) * 1000

# pcm is only format some models support
def pcm_to_wav(pcm_bytes: bytes, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(pcm_bytes)
    buffer.seek(0)
    return buffer.read()

# call the right TTS provider
def call_synthesize(cfg: dict, text: str) -> tuple[bytes, str]:

    client_module = TTS_CLIENTS[cfg["provider"]]
    fmt = cfg.get("format", "wav")

    if cfg["provider"] == "openrouter":
        audio_bytes, sample_rate = client_module.synthesize(text, voice=cfg["voice"], model=cfg["model"], response_format=fmt)
    else:
        audio_bytes = client_module.synthesize(text, voice=cfg["voice"], model=cfg["model"])
        sample_rate = 0

    if fmt == "pcm":
        audio_bytes = pcm_to_wav(audio_bytes, sample_rate or PCM_SAMPLE_RATE)
        return audio_bytes, "audio/wav"
    if fmt == "mp3":
        return audio_bytes, "audio/mpeg"
    return audio_bytes, "audio/wav"

# call the right STT provider
def call_transcribe(cfg: dict, audio_bytes: bytes, filename: str) -> str:
    client_module = STT_CLIENTS[cfg["provider"]]

    if cfg["provider"] == "openrouter":
        return client_module.transcribe(audio_bytes, filename=filename, model=cfg["model"], language=cfg.get("language"))
    return client_module.transcribe(audio_bytes, filename=filename, model=cfg["model"])


# SESSION STATE

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

    llm_client, llm_model = resolve_llm(request.config.llm)

    greeting = GREETINGS.get(request.personality, GREETINGS["classic"])

    # generate first question with LLM
    try:
        first_question, llm_latency_ms = call_generate_question(
            llm_client,
            llm_model,
            topic=request.topic,
            question_number=1,
            previous_questions=[],
        )
    except Exception as e:
        print("\n=== START QUIZ ERROR ===")
        traceback.print_exc()
        print("==========================")
        raise HTTPException(status_code=500, detail=f"Failed to generate first question: {str(e)}")

    # initialize the session
    current_session = {
        "topic": request.topic,
        "personality": request.personality,
        "num_questions": request.num_questions,
        "previous_questions": [first_question],
        "current_index": 0,
        "correct_count": 0,
        "streak": 0,
        "config": request.config.model_dump(),
    }

    total_latency_ms = int((time.time() - t_start) * 1000)

    return {
        "success": True,
        "message": f"{greeting} {first_question}",
        "next_question": first_question,
        "quiz_done": False,
        "latency_ms": {
            "llm": int(llm_latency_ms),
            "total": total_latency_ms
        },
    }

# called after user voice been recorded
@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...), stt_model: str = "whisper-groq"):
    t_start = time.time()

    # read audio bytes
    audio_bytes = await audio.read()
    file_size_kb = round(len(audio_bytes) / 1024, 1)

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio data received")
    
    filename = audio.filename or "audio.wav"    # so whisper can detect the format from the extension

    stt_model_name = resolve_model(STT_MODELS, stt_model, "STT")

    # Call STT module
    # stt_model param is still only info
    try:
        t_stt_start = time.time()
        transcript = call_transcribe(stt_model_name, audio_bytes, filename)
        stt_latency_ms = int((time.time() - t_stt_start) * 1000)
    except Exception as e:
        print("\n=== STT ERROR ===")
        traceback.print_exc()
        print("====================\n")
        raise HTTPException(status_code=500, detail=f"STT failed: {str(e)}")

    total_latency_ms = int((time.time() - t_start) * 1000)

    return {
        "success": True,
        "transcript": transcript,
        "stt_model": stt_model,
        "file_size_kb": file_size_kb,
        "latency_ms": {
            "stt": stt_latency_ms,
            "total": total_latency_ms,
        },
    }

@app.post("/api/evaluate")
async def evaluate_answer_endpoint(request: AnswerRequest):
    
    global current_session
    t_start = time.time()

    if not current_session:
        raise HTTPException(status_code=400, detail="No active quiz session.")

    llm_client, llm_model = resolve_llm(request.config.llm)

    # evaluate user's answer
    try:
        is_correct, llm_feedback, eval_latency_ms = call_evaluate_answer(
            llm_client,
            llm_model,
            question=request.question,
            transcript=request.transcript,
            host=current_session.get("personality", "classic"),
            question_number=current_session["current_index"] + 1,
            total_questions=current_session["num_questions"],
            score=current_session["correct_count"],
            streak=current_session["streak"],
            )
    except Exception as e:
        print("\n=== LLM EVALUATION ERROR ===")
        traceback.print_exc()
        print("============================\n")
        raise HTTPException(status_code=500, detail=f"LLM evaluation failed: {str(e)}")
    
    current_session["current_index"] += 1
    if is_correct:
        current_session["correct_count"] += 1
        current_session["streak"] += 1
    else:
        current_session["streak"] = 0

    quiz_done = current_session["current_index"] >= current_session["num_questions"]

    next_question = ""
    gen_latency_ms = 0
    if not quiz_done:
        try:
            next_question, gen_latency_ms = call_generate_question(
                llm_client,
                llm_model,
                topic = current_session["topic"],
                question_number = current_session["current_index"] + 1,
                previous_questions = current_session["previous_questions"],
            )
            current_session["previous_questions"].append(next_question)
        except Exception as e:
            print("\n=== QUESTION GENERATION ERROR ===")
            traceback.print_exc()
            print("===================================")
            raise HTTPException(status_code=500, detail=f"Failed to generate next question: {str(e)}")
        
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
            "llm": int(eval_latency_ms + gen_latency_ms),
            "llm_eval": int(eval_latency_ms),
            "llm_gen": int(gen_latency_ms),
            "total": total_latency_ms,
        },
    )

@app.post("/api/speak")
async def text_to_speech(request: SpeakRequest):

    t_start = time.time()

    # guard against empty text
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided.")
    
    tts_config = resolve_model(TTS_MODELS, request.tts_model, "TTS")

    # call TTS module
    try:
        tts_start = time.time()
        audio_bytes, media_type = call_synthesize(tts_config, request.text)
        tts_latency = int((time.time() - tts_start) * 1000)

        # # TEMPORARY DEBUG: save every TTS output
        # import re
        # import os
        # from datetime import datetime
        # os.makedirs("../results/tts_debug", exist_ok=True)
        # stamp = datetime.now().strftime("%H%M%S")
        # slug = re.sub(r"[^a-zA-Z0-9]+", "_", request.text[:40]).strip("_")
        # with open(f"../results/tts_debug/{stamp}_{slug}.wav", "wb") as f:
        #     f.write(audio_bytes)

    except Exception as e:
        print("\n=== TTS ERROR ===")
        traceback.print_exc()
        print("===================\n")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")
    
    total_latency = int((time.time() - t_start) * 1000)

    # stream audio bytes back to the browser
    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type=media_type,
        headers={
            "X-TTS-Latency-Ms": str(tts_latency),
            "X-Total-Latency-Ms": str(total_latency),
        },
    )

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "version": "0.1.0",
        "message": "Voice Quiz API is running",
    }

# models for debug pannel
@app.get("/api/models")
async def list_models():
    return {
        "stt": [
            {"id": "whisper-large-v3", "name": "Whisper Large V3 Turbo", "provider": "OpenRouter"},
            {"id": "gpt4o-transcribe", "name": "GPT-4o Mini Transcribe","provider": "OpenRouter"},
            {"id": "nova-3", "name": "Deepgram Nova-3", "provider": "OpenRouter"},
            {"id": "voxtral-stt", "name": "Voxtral Transcribe", "provider": "OpenRouter"},
        ],
        "llm": [
            # {"id": "llama-groq", "name": "LLama 3.3 70B", "provider": "Groq"},
            {"id": "qwen-openrouter", "name": "Qwen 3.6 27B", "provider": "OpenRouter"},
            {"id": "mistral-openrouter", "name": "Mistral Medium 3.5", "provider": "OpenRouter"},
            {"id": "gpt-openrouter", "name": "GPT-5.4 Mini", "provider": "OpenRouter"},
            {"id": "gemini-openrouter", "name": "Google Gemini 3.7 Flash", "provider": "OpenRouter"},
            {"id": "nemotron-openrouter", "name": "Nvidia Nemotron 3.5 Lightning", "provider": "OpenRouter"},
        ],
        "tts": [
            # {"id": "orpheus", "name": "Orpheus (English)", "provider": "Groq"},
            {"id": "kokoro", "name": "Kokoro 82M", "provider": "OpenRouter"},
            # {"id": "gemini-tts", "name": "Gemini 3.1 Flash TTS", "provider": "OpenRouter"},
            {"id": "qwen-tts", "name": "Qwen Audio 3.0 TTS", "provider": "OpenRouter"},
            {"id": "deepgram-aura", "name": "Deepgram Aura-2", "provider": "OpenRouter"},
            {"id": "minimax-speech", "name": "MiniMax: Speech 2.8", "provider": "OpenRouter"}
        ],
    }