import os
import random
import tempfile
import re
import numpy as np
from numpy.linalg import norm
import streamlit as st
from resemblyzer import VoiceEncoder, preprocess_wav
from faster_whisper import WhisperModel

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VoiceAuth — Biometric Authentication",
    page_icon="🎙️",
    layout="centered"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .main { background-color: #0a0e1a; }

    .title-text {
        font-family: 'Syne', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 30%, #4f9eff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle-text {
        text-align: center;
        color: rgba(200,210,255,0.6);
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }
    .result-granted {
        background: rgba(0,212,170,0.1);
        border: 1px solid rgba(0,212,170,0.4);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        font-family: 'Syne', sans-serif;
        font-size: 1.4rem;
        font-weight: 800;
        color: #00d4aa;
    }
    .result-denied {
        background: rgba(255,79,110,0.1);
        border: 1px solid rgba(255,79,110,0.4);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        font-family: 'Syne', sans-serif;
        font-size: 1.4rem;
        font-weight: 800;
        color: #ff4f6e;
    }
    .challenge-box {
        background: rgba(79,158,255,0.08);
        border: 1px solid rgba(79,158,255,0.3);
        border-radius: 14px;
        padding: 18px 24px;
        text-align: center;
        margin: 1rem 0;
    }
    .challenge-label {
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #4f9eff;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .challenge-phrase {
        font-family: 'Syne', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: white;
    }
    .upload-box {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 12px 16px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DIR = os.path.join(BASE_DIR, "data", "users")
VOICE_THRESHOLD = 0.78
PHRASE_THRESHOLD = 0.75
NUM_SAMPLES = 5

CHALLENGE_PHRASES = [
    "my voice is my password",
    "unlock the system now",
    "voice authentication enabled",
    "identity verified successfully",
    "this is my voice key",
    "secure access granted",
    "grant me access today",
    "open sesame right now",
]

os.makedirs(USER_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD MODELS (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_encoder():
    return VoiceEncoder()

@st.cache_resource
def load_whisper():
    return WhisperModel("base", device="cpu", compute_type="int8")

encoder = load_encoder()
whisper = load_whisper()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def sanitize_username(username: str) -> str:
    username = username.strip().lower()
    username = re.sub(r"[^a-zA-Z0-9_-]", "", username)
    return username

def extract_embedding(audio_bytes: bytes) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        wav = preprocess_wav(tmp_path)
        emb = encoder.embed_utterance(wav)
        n = norm(emb)
        return emb / n if n != 0 else emb
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def cosine_similarity(a, b):
    a, b = np.array(a).flatten(), np.array(b).flatten()
    d = norm(a) * norm(b)
    return float(np.dot(a, b) / d) if d != 0 else 0.0

def load_embeddings(username: str):
    path = os.path.join(USER_DIR, f"{username}.npy")
    if not os.path.exists(path):
        return None
    emb = np.load(path, allow_pickle=False)
    return emb.reshape(1, -1) if emb.ndim == 1 else emb

def transcribe(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        segments, _ = whisper.transcribe(
            tmp_path,
            language="en",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300)
        )
        text = " ".join(s.text for s in segments).strip().lower()
        for ch in ",.!?":
            text = text.replace(ch, "")
        return text
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def phrase_match_ratio(spoken: str, expected: str) -> float:
    sw = set(spoken.split())
    ew = set(expected.split())
    if not ew:
        return 0.0
    return len(sw & ew) / len(ew)

def list_users():
    return sorted([
        f.replace(".npy", "")
        for f in os.listdir(USER_DIR)
        if f.endswith(".npy")
    ])

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="title-text">🎙 VoiceAuth</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Biometric Voice Authentication System</div>', unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "enroll_embeddings" not in st.session_state:
    st.session_state.enroll_embeddings = []

if "enroll_username" not in st.session_state:
    st.session_state.enroll_username = ""

if "challenge_phrase" not in st.session_state:
    st.session_state.challenge_phrase = None

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_enroll, tab_login = st.tabs(["🧬 Enroll", "🔐 Login"])

# ═══════════════════════════════════════════════
# ENROLL TAB
# ═══════════════════════════════════════════════
with tab_enroll:
    st.subheader("Create Voice Profile")
    st.caption("Upload 5 voice samples (.wav) to register your biometric identity.")

    username_input = st.text_input(
        "Username",
        placeholder="Enter your username",
        key="enroll_user"
    )

    username = sanitize_username(username_input)

    if username_input and not username:
        st.warning("⚠️ Username can only contain letters, numbers, underscores, and hyphens.")

    if username:
        if st.session_state.enroll_username != username:
            st.session_state.enroll_embeddings = []
            st.session_state.enroll_username = username

        done = len(st.session_state.enroll_embeddings)
        total = NUM_SAMPLES

        st.progress(done / total, text=f"Samples uploaded: {done} / {total}")

        dots = ""
        for i in range(total):
            if i < done:
                dots += "🟢 "
            elif i == done:
                dots += "🔵 "
            else:
                dots += "⚪ "
        st.markdown(f"**{dots}**")

        existing_user_path = os.path.join(USER_DIR, f"{username}.npy")
        already_enrolled = os.path.exists(existing_user_path)

        if already_enrolled and done == 0:
            st.success(f"✅ **{username}** is already enrolled.")
            if st.button("🔄 Re-enroll (overwrite)"):
                try:
                    os.remove(existing_user_path)
                except:
                    pass
                st.session_state.enroll_embeddings = []
                st.rerun()

        else:
            if done < total:
                st.info(f"📁 Upload sample **{done + 1} of {total}** (WAV format preferred, 3–4 sec speech)")
                audio_file = st.file_uploader(
                    f"Upload Sample {done + 1}",
                    type=["wav", "mp3", "m4a"],
                    key=f"enroll_audio_{done}"
                )

                if audio_file is not None:
                    audio_bytes = audio_file.read()

                    with st.spinner("Processing voice sample..."):
                        try:
                            emb = extract_embedding(audio_bytes)
                            st.session_state.enroll_embeddings.append(emb)
                            st.success(f"✅ Sample {done + 1} uploaded successfully!")

                            if len(st.session_state.enroll_embeddings) == total:
                                all_emb = np.array(st.session_state.enroll_embeddings, dtype=np.float32)
                                save_path = os.path.join(USER_DIR, f"{username}.npy")
                                np.save(save_path, all_emb)

                                st.balloons()
                                st.success(f"🎉 **{username}** enrolled successfully with {total} voice samples!")
                                st.session_state.enroll_embeddings = []
                                st.session_state.enroll_username = ""
                            else:
                                st.rerun()

                        except Exception as e:
                            st.error(f"Error processing audio: {e}")
            else:
                st.success(f"🎉 Enrollment complete for **{username}**")

    users = list_users()
    if users:
        st.divider()
        st.caption(f"**Enrolled users:** {', '.join(users)}")

# ═══════════════════════════════════════════════
# LOGIN TAB
# ═══════════════════════════════════════════════
with tab_login:
    st.subheader("Voice Verification")
    st.caption("Upload a voice sample speaking the challenge phrase.")

    login_input = st.text_input(
        "Username",
        placeholder="Enter your username",
        key="login_user"
    )

    login_user = sanitize_username(login_input)

    if login_input and not login_user:
        st.warning("⚠️ Invalid username format.")

    if login_user:
        stored = load_embeddings(login_user)

        if stored is None:
            st.warning(f"⚠️ User **{login_user}** is not enrolled. Please enroll first.")
        else:
            st.success(f"✅ User found — {stored.shape[0]} enrollment samples loaded.")

            if st.session_state.challenge_phrase is None:
                st.session_state.challenge_phrase = random.choice(CHALLENGE_PHRASES)

            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"""
                <div class="challenge-box">
                    <div class="challenge-label">🔐 Speak this phrase</div>
                    <div class="challenge-phrase">"{st.session_state.challenge_phrase}"</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.write("")
                st.write("")
                if st.button("🎲 New Phrase"):
                    st.session_state.challenge_phrase = random.choice(CHALLENGE_PHRASES)
                    st.rerun()

            st.info("📁 Upload your verification audio speaking the phrase above")

            verify_audio = st.file_uploader(
                "Upload Verification Audio",
                type=["wav", "mp3", "m4a"],
                key="verify_audio"
            )

            if verify_audio is not None:
                audio_bytes = verify_audio.read()

                with st.spinner("🔍 Verifying identity..."):
                    try:
                        # Voice check
                        test_emb = extract_embedding(audio_bytes)
                        scores = [cosine_similarity(e, test_emb) for e in stored]
                        best_score = max(scores)
                        avg_score = round(sum(scores) / len(scores), 3)
                        voice_passed = best_score >= VOICE_THRESHOLD

                        # Phrase check
                        spoken_text = transcribe(audio_bytes)
                        ratio = phrase_match_ratio(spoken_text, st.session_state.challenge_phrase)
                        phrase_passed = ratio >= PHRASE_THRESHOLD

                        access_granted = voice_passed and phrase_passed

                        st.divider()

                        if access_granted:
                            st.markdown('<div class="result-granted">✅ ACCESS GRANTED</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="result-denied">❌ ACCESS DENIED</div>', unsafe_allow_html=True)

                        st.write("")

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Voice Score", f"{best_score:.3f}", f"threshold {VOICE_THRESHOLD}")
                        c2.metric("Voice Match", "✅ PASS" if voice_passed else "❌ FAIL")
                        c3.metric("Phrase Match", "✅ PASS" if phrase_passed else "❌ FAIL")
                        c4.metric("Word Overlap", f"{ratio*100:.0f}%")

                        st.caption(f"**You said:** *\"{spoken_text if spoken_text else 'nothing detected'}\"*")
                        st.caption(f"**Expected:** *\"{st.session_state.challenge_phrase}\"*")
                        st.caption(f"**Average similarity across stored samples:** {avg_score}")

                        st.session_state.challenge_phrase = random.choice(CHALLENGE_PHRASES)

                    except Exception as e:
                        st.error(f"Verification error: {e}")
