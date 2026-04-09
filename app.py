import os
import re
import random
import tempfile
import time
from typing import Optional

import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from resemblyzer import VoiceEncoder, preprocess_wav
import uvicorn

# =========================
# APP SETUP
# =========================

app = FastAPI(title="Voice Biometrics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# CONFIG
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DIR = os.path.join(BASE_DIR, "data", "users")

VOICE_THRESHOLD = 0.72
PHRASE_THRESHOLD = 0.75
NUM_SAMPLES = 5
CHALLENGE_EXPIRY_SECONDS = 120  # 2 min

CHALLENGE_PHRASES = [
    "my voice is my password",
    "unlock the system now",
    "secure access granted",
    "voice authentication enabled",
    "identity verified successfully",
    "open sesame right now",
    "grant me access today",
    "this is my voice key",
]

os.makedirs(USER_DIR, exist_ok=True)

# Stores: username -> {"phrase": "...", "created_at": ...}
active_challenges = {}

# =========================
# LOAD MODELS ONCE
# =========================

print("⏳ Loading Resemblyzer...")
encoder = VoiceEncoder()
print("✅ Resemblyzer ready.")

print("⏳ Loading Faster-Whisper...")
whisper = WhisperModel("base", device="cpu", compute_type="int8")
print("✅ Faster-Whisper ready.")

# =========================
# HELPERS
# =========================

def sanitize_username(username: str) -> str:
    username = username.strip().lower()
    username = re.sub(r"[^a-zA-Z0-9_-]", "", username)
    return username

def get_user_file_path(username: str) -> str:
    return os.path.join(USER_DIR, f"{username}.npy")

def get_temp_embedding_path(username: str, sample_index: int) -> str:
    return os.path.join(USER_DIR, f"{username}_tmp_{sample_index}.npy")

def extract_embedding(audio_path: str) -> np.ndarray:
    wav = preprocess_wav(audio_path)
    emb = encoder.embed_utterance(wav)
    emb = emb / (norm(emb) + 1e-10)
    return emb.astype(np.float32)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten()
    b = b.flatten()
    denom = (norm(a) * norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)

def load_embeddings(username: str) -> Optional[np.ndarray]:
    path = get_user_file_path(username)
    if not os.path.exists(path):
        return None
    emb = np.load(path, allow_pickle=False)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    return emb

def transcribe(audio_path: str) -> str:
    segments, _ = whisper.transcribe(
        audio_path,
        language="en",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
    )
    text = " ".join(segment.text for segment in segments).strip().lower()
    for ch in ",.!?":
        text = text.replace(ch, "")
    return text

def phrase_matches(spoken: str, expected: str) -> tuple[bool, float]:
    spoken_words = set(spoken.split())
    expected_words = set(expected.split())

    if not expected_words:
        return False, 0.0

    ratio = len(spoken_words & expected_words) / len(expected_words)
    ratio = round(ratio, 3)
    return ratio >= PHRASE_THRESHOLD, ratio

async def save_audio_temp(file: UploadFile) -> str:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    suffix = ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(content)
        tmp.flush()
    finally:
        tmp.close()

    return tmp.name

def cleanup_temp_enrollment_files(username: str):
    for i in range(NUM_SAMPLES):
        temp_path = get_temp_embedding_path(username, i)
        if os.path.exists(temp_path):
            os.remove(temp_path)

def challenge_is_expired(created_at: float) -> bool:
    return (time.time() - created_at) > CHALLENGE_EXPIRY_SECONDS

# =========================
# ROUTES
# =========================

@app.get("/")
def root():
    return {
        "message": "Voice Biometrics API running ✅",
        "voice_threshold": VOICE_THRESHOLD,
        "phrase_threshold": PHRASE_THRESHOLD,
        "num_samples_required": NUM_SAMPLES,
    }

@app.get("/users")
def list_users():
    files = os.listdir(USER_DIR)
    users = sorted([f.replace(".npy", "") for f in files if f.endswith(".npy") and "_tmp_" not in f])
    return {"users": users, "count": len(users)}

@app.post("/enroll")
async def enroll(
    username: str = Form(...),
    sample_index: int = Form(...),
    audio: UploadFile = File(...)
):
    username = sanitize_username(username)

    if not username:
        raise HTTPException(status_code=400, detail="Invalid username.")
    if sample_index < 0 or sample_index >= NUM_SAMPLES:
        raise HTTPException(status_code=400, detail=f"sample_index must be between 0 and {NUM_SAMPLES - 1}.")

    tmp_audio_path = await save_audio_temp(audio)

    try:
        embedding = extract_embedding(tmp_audio_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {str(e)}")
    finally:
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)

    # Save this sample temporarily
