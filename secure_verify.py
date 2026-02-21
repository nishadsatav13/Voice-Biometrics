import os
import random
import numpy as np
from numpy.linalg import norm
from faster_whisper import WhisperModel

from extract_embedding_resemblyzer import extract_embedding
from record_audio import record_audio


# =========================
# CONFIG
# =========================

USER_DIR = "data/users"

VOICE_THRESHOLD = 0.72           # cosine similarity threshold for speaker match
PHRASE_MATCH_THRESHOLD = 0.75    # word overlap ratio to allow minor recognition errors

MAX_ATTEMPTS = 3                 # max failed attempts before lockout

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

# Whisper model size — "base" is best balance of speed and accuracy for short phrases
# Options: "tiny", "base", "small", "medium" (larger = more accurate but slower)
WHISPER_MODEL_SIZE = "base"

# Run on CPU. Change to "cuda" if you have an NVIDIA GPU for faster inference.
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE = "int8"   # int8 is fastest on CPU, no quality loss for short phrases


# =========================
# Load Whisper model once at startup
# (auto-downloads on first run, ~150MB for base)
# =========================

print("⏳ Loading Whisper model...")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
print("✅ Whisper model ready.\n")


# =========================
# HELPER: Cosine Similarity
# =========================

def cosine_similarity(a, b):
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    denom = norm(a) * norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# =========================
# HELPER: Load User Embeddings
# =========================

def load_user_embeddings(username):
    path = os.path.join(USER_DIR, f"{username}.npy")
    if not os.path.exists(path):
        return None
    emb = np.load(path, allow_pickle=True)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    return emb


# =========================
# HELPER: Transcribe Audio with Faster-Whisper
# =========================

def transcribe_audio(audio_file):
    """
    Transcribes audio using Faster-Whisper.
    Returns recognized text as a lowercase string.
    """
    segments, info = whisper_model.transcribe(
        audio_file,
        language="en",           # force English for faster, more accurate results
        beam_size=5,             # higher beam = more accurate, slightly slower
        vad_filter=True,         # filters out silence automatically
        vad_parameters=dict(min_silence_duration_ms=300)
    )

    # Collect all segments into one string
    recognized = " ".join(segment.text for segment in segments).strip().lower()

    # Clean up punctuation Whisper sometimes adds
    recognized = recognized.replace(",", "").replace(".", "").replace("!", "").replace("?", "")

    return recognized


# =========================
# HELPER: Phrase Match Check
# =========================

def phrase_matches(spoken_text, expected_phrase):
    """
    Checks word overlap between spoken text and expected phrase.
    Allows for minor transcription errors without failing the user.
    """
    spoken_words = set(spoken_text.lower().split())
    expected_words = set(expected_phrase.lower().split())

    if not expected_words:
        return False

    overlap = spoken_words & expected_words
    ratio = len(overlap) / len(expected_words)

    print(f"\n📝 Expected   : {expected_phrase}")
    print(f"🗣️  Recognized : {spoken_text if spoken_text else '(nothing detected)'}")
    print(f"🔤 Word match  : {len(overlap)}/{len(expected_words)} ({ratio:.0%})")

    return ratio >= PHRASE_MATCH_THRESHOLD


# =========================
# MAIN: Verify User
# =========================

def verify_user(username):

    # ── 1. Load enrolled embeddings ─────────────────────────────────────────
    stored_embeddings = load_user_embeddings(username)
    if stored_embeddings is None:
        print(f"\n❌ User '{username}' is not enrolled. Please enroll first.")
        return False

    print(f"\n🔐 Starting challenge-response verification for: {username}")

    attempt = 0

    while attempt < MAX_ATTEMPTS:
        attempt += 1
        print(f"\n── Attempt {attempt}/{MAX_ATTEMPTS} ──────────────────────────────────────")

        # ── 2. Generate a fresh random challenge phrase ──────────────────────
        challenge_phrase = random.choice(CHALLENGE_PHRASES)
        print("\n🎯 Speak the following phrase clearly:")
        print(f'   ➡️  "{challenge_phrase}"')

        input("\nPress ENTER when ready and speak...")

        audio_file = f"{username}_verify_{attempt}.wav"
        record_audio(audio_file)

        # ── 3. Speaker Verification (voice match) ────────────────────────────
        print("\n🔍 Verifying speaker identity...")
        test_embedding = extract_embedding(audio_file)

        scores = [cosine_similarity(emb, test_embedding) for emb in stored_embeddings]
        best_score = max(scores)
        avg_score = sum(scores) / len(scores)

        print(f"   Best similarity : {best_score:.3f}  (threshold: {VOICE_THRESHOLD})")
        print(f"   Avg  similarity : {avg_score:.3f}")

        voice_passed = best_score >= VOICE_THRESHOLD

        # ── 4. Phrase Verification (Faster-Whisper transcription) ────────────
        print("\n🔍 Verifying spoken phrase...")
        recognized_text = transcribe_audio(audio_file)

        phrase_passed = phrase_matches(recognized_text, challenge_phrase)

        # ── 5. Final Decision ────────────────────────────────────────────────
        print("\n── Result ───────────────────────────────────────────────────────")
        print(f"   Voice match  : {'✅ PASS' if voice_passed  else '❌ FAIL'}")
        print(f"   Phrase match : {'✅ PASS' if phrase_passed else '❌ FAIL'}")

        if voice_passed and phrase_passed:
            print("\n✅ ACCESS GRANTED — Identity and phrase verified.")
            if os.path.exists(audio_file):
                os.remove(audio_file)
            return True

        elif not voice_passed and not phrase_passed:
            print("\n❌ Both voice and phrase failed.")
        elif not voice_passed:
            print("\n❌ Voice does not match enrolled user.")
        elif not phrase_passed:
            print("\n❌ Spoken phrase did not match the challenge.")

        # Clean up failed attempt audio
        if os.path.exists(audio_file):
            os.remove(audio_file)

        if attempt < MAX_ATTEMPTS:
            print(f"\n🔄 {MAX_ATTEMPTS - attempt} attempt(s) remaining.")

    # ── Lockout ──────────────────────────────────────────────────────────────
    print("\n🔒 ACCESS DENIED — Maximum attempts reached.")
    print("   Too many failed verification attempts.")
    return False


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    username = input("Enter username: ").strip()
    verify_user(username)