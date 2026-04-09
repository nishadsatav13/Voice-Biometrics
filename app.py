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
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: radial-gradient(ellipse at top, #1a0000 0%, #0a0a0a 60%, #0d0000 100%) !important;
}

.title-wrap {
    text-align: center;
    margin-bottom: 4px;
}
.title-text {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 5%, #e8003d 50%, #d4af37 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle-text {
    text-align: center;
    color: rgba(255,220,210,0.65);
    font-size: 0.9rem;
    letter-spacing: 0.06em;
    margin-bottom: 1.6rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(212,175,55,0.15);
    padding: 5px;
    border-radius: 14px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    color: rgba(255,220,210,0.7) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #c0001f, #8b0000) !important;
    color: white !important;
    box-shadow: 0 4px 16px rgba(192,0,31,0.4);
}

.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(212,175,55,0.18);
    border-radius: 20px;
    padding: 24px 28px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
    margin-bottom: 16px;
}

.tips-box {
    background: rgba(192,0,31,0.1);
    border-left: 4px solid #e8003d;
    border-radius: 0 12px 12px 0;
    padding: 12px 16px;
    margin: 12px 0;
    color: rgba(255,235,225,0.9);
    font-size: 0.88rem;
    line-height: 1.7;
}

.dots-row {
    text-align: center;
    font-size: 1.1rem;
    margin: 8px 0;
}

.challenge-box {
    background: rgba(192,0,31,0.09);
    border: 1px solid rgba(212,175,55,0.3);
    border-radius: 16px;
    padding: 18px 24px;
    text-align: center;
    margin: 12px 0;
}
.challenge-label {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #d4af37;
    font-weight: 700;
    margin-bottom: 8px;
}
.challenge-phrase {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
}

.result-granted {
    background: rgba(0,200,130,0.09);
    border: 2px solid rgba(0,200,130,0.4);
    border-radius: 16px;
    padding: 22px;
    text-align: center;
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #00c882;
    box-shadow: 0 0 30px rgba(0,200,130,0.15);
}
.result-denied {
    background: rgba(220,0,40,0.09);
    border: 2px solid rgba(220,0,40,0.4);
    border-radius: 16px;
    padding: 22px;
    text-align: center;
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #ff3355;
    box-shadow: 0 0 30px rgba(220,0,40,0.15);
}

.section-head {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #d4af37;
    margin-bottom: 4px;
    letter-spacing: 0.04em;
}

.stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(212,175,55,0.25) !important;
    border-radius: 12px !important;
    color: white !important;
}
.stTextInput > div > div > input:focus {
    border-color: #e8003d !important;
    box-shadow: 0 0 0 3px rgba(232,0,61,0.2) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #c0001f, #8b0000) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 16px rgba(192,0,31,0.3) !important;
}

[data-testid="stMetricLabel"] {
    color: rgba(255,220,210,0.7) !important;
    font-size: 0.8rem !important;
}
[data-testid="stMetricValue"] {
    color: white !important;
    font-family: 'Syne', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DIR = os.path.join(BASE_DIR, "data", "users")

VOICE_THRESHOLD = 0.75
PHRASE_THRESHOLD = 0.50
NUM_SAMPLES = 5

CHALLENGE_PHRASES = [
    "open the voice lock",
    "grant access now",
    "verify my identity",
    "let me log in",
    "my voice is unique",
    "this is my access key",
]

os.makedirs(USER_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading voice encoder...")
def load_encoder():
    return VoiceEncoder()

@st.cache_resource(show_spinner="Loading speech model...")
def load_whisper():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

encoder = load_encoder()
whisper = load_whisper()

# ─────────────────────────────────────────────
# COMPATIBLE AUDIO INPUT
# ─────────────────────────────────────────────
def get_audio_input(label, key):
    if hasattr(st, "audio_input"):
        return st.audio_input(label, key=key)
    elif hasattr(st, "experimental_audio_input"):
        return st.experimental_audio_input(label, key=key)
    else:
        st.error("This Streamlit version does not support microphone recording.")
        return None

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "", name.strip().lower())

def extract_embedding(audio_bytes: bytes) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        wav = preprocess_wav(tmp)
        emb = encoder.embed_utterance(wav)
        n = norm(emb)
        return emb / n if n != 0 else emb
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def cosine_sim(a, b):
    a, b = np.array(a).flatten(), np.array(b).flatten()
    d = norm(a) * norm(b)
    return float(np.dot(a, b) / d) if d != 0 else 0.0

def load_embeddings(username: str):
    path = os.path.join(USER_DIR, f"{username}.npy")
    if not os.path.exists(path):
        return None
    emb = np.load(path, allow_pickle=False)
    return emb.reshape(1, -1) if emb.ndim == 1 else emb

def transcribe_audio(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        segments, _ = whisper.transcribe(
            tmp,
            language="en",
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=200)
        )
        text = " ".join(s.text for s in segments).strip().lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def phrase_ratio(spoken: str, expected: str) -> float:
    sw = set(spoken.split())
    ew = set(re.sub(r"[^a-z0-9\s]", "", expected.lower()).split())
    return len(sw & ew) / len(ew) if ew else 0.0

def list_users():
    return sorted([f.replace(".npy","") for f in os.listdir(USER_DIR) if f.endswith(".npy")])

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, val in {
    "enroll_embeddings": [],
    "enroll_username": "",
    "challenge_phrase": None,
    "last_enroll_hash": None,
    "last_verify_hash": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="title-wrap"><span class="title-text">🎙 VoiceAuth</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">BIOMETRIC VOICE AUTHENTICATION SYSTEM</div>', unsafe_allow_html=True)
st.divider()

tab_enroll, tab_login = st.tabs(["🧬 Enroll", "🔐 Login"])

# ═══════════════════════════════════════════
# ENROLL TAB
# ═══════════════════════════════════════════
with tab_enroll:
    st.markdown('<div class="section-head">Create Voice Profile</div>', unsafe_allow_html=True)
    st.caption("Record 5 voice samples to register your biometric identity.")

    st.markdown("""
    <div class="tips-box">
        <b>Recording Tips:</b><br>
        • Speak clearly for 2–4 seconds<br>
        • Keep same mic distance<br>
        • Use a quiet room<br>
        • Pause a second after speaking
    </div>
    """, unsafe_allow_html=True)

    username_raw = st.text_input("Username", placeholder="e.g. nishad", key="enroll_user_input")
    username = sanitize(username_raw)

    if username_raw and not username:
        st.warning("⚠️ Username can only contain letters, numbers, _ and -")

    if username:
        if st.session_state.enroll_username != username:
            st.session_state.enroll_embeddings = []
            st.session_state.enroll_username = username
            st.session_state.last_enroll_hash = None

        done = len(st.session_state.enroll_embeddings)
        total = NUM_SAMPLES

        # FIXED PLACEHOLDER PROGRESS BLOCK
        progress_placeholder = st.empty()
        with progress_placeholder.container():
            if done >= total:
                st.progress(1.0, text=f"Voice samples: {total} / {total}")
                dots = "🟢 🟢 🟢 🟢 🟢"
            else:
                st.progress(done / total, text=f"Voice samples: {done} / {total}")
                dots = "".join(["🟢 " if i < done else ("🔴 " if i == done else "⚪ ") for i in range(total)])

            st.markdown(f'<div class="dots-row">{dots}</div>', unsafe_allow_html=True)

        enroll_path = os.path.join(USER_DIR, f"{username}.npy")
        already_done = os.path.exists(enroll_path) and done == 0

        if already_done:
            st.success(f"✅ **{username}** is already enrolled and ready to login!")
            if st.button("🔄 Re-enroll (overwrite existing)", key="reenroll"):
                try:
                    os.remove(enroll_path)
                except:
                    pass
                st.session_state.enroll_embeddings = []
                st.session_state.last_enroll_hash = None
                st.rerun()

        elif done < total:
            st.markdown(f"""
            <div class="glass-card">
                <div style="text-align:center; margin-bottom:8px;">
                    <span style="font-family:Syne; font-size:1.15rem; font-weight:700; color:#d4af37;">
                        Sample {done + 1} of {total}
                    </span><br>
                    <span style="color:rgba(255,220,210,0.7); font-size:0.9rem;">
                        Tap below and record your voice
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            audio = get_audio_input(f"🎤 Record Sample {done + 1}", key=f"enroll_audio_{done}")

            if audio is not None:
                audio_bytes = audio.read()
                current_hash = hash(audio_bytes[:500])

                if st.session_state.last_enroll_hash != current_hash:
                    with st.spinner(f"Processing sample {done + 1}..."):
                        try:
                            emb = extract_embedding(audio_bytes)
                            st.session_state.enroll_embeddings.append(emb)
                            st.session_state.last_enroll_hash = current_hash

                            current_done = len(st.session_state.enroll_embeddings)

                            if current_done >= total:
                                all_emb = np.array(st.session_state.enroll_embeddings[:total], dtype=np.float32)
                                np.save(enroll_path, all_emb)

                                # FIXED: replace old 4/5 block instead of stacking
                                progress_placeholder.empty()
                                with progress_placeholder.container():
                                    st.progress(1.0, text=f"Voice samples: {total} / {total}")
                                    st.markdown('<div class="dots-row">🟢 🟢 🟢 🟢 🟢</div>', unsafe_allow_html=True)

                                st.balloons()
                                st.success(f"🎉 **{username}** enrolled successfully with {total} samples!")

                                st.session_state.enroll_embeddings = []
                                st.session_state.enroll_username = ""
                                st.session_state.last_enroll_hash = None
                            else:
                                st.success(f"✅ Sample {current_done} saved! {total - current_done} more to go.")
                                st.rerun()

                        except Exception as e:
                            st.error(f"Error: {e}")

    users = list_users()
    if users:
        st.divider()
        st.caption(f"**Enrolled users ({len(users)}):** {', '.join(u.capitalize() for u in users)}")

# ═══════════════════════════════════════════
# LOGIN TAB
# ═══════════════════════════════════════════
with tab_login:
    st.markdown('<div class="section-head">Voice Verification</div>', unsafe_allow_html=True)
    st.caption("Speak the challenge phrase to authenticate your identity.")

    login_raw = st.text_input("Username", placeholder="Enter your username", key="login_user_input")
    login_user = sanitize(login_raw)

    if login_raw and not login_user:
        st.warning("⚠️ Invalid username.")

    if login_user:
        stored = load_embeddings(login_user)

        if stored is None:
            st.warning(f"⚠️ **{login_user}** is not enrolled. Go to Enroll tab first.")
        else:
            st.success(f"✅ User found — {stored.shape[0]} voice samples loaded.")

            if st.session_state.challenge_phrase is None:
                st.session_state.challenge_phrase = random.choice(CHALLENGE_PHRASES)

            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"""
                <div class="challenge-box">
                    <div class="challenge-label">Speak This Phrase Clearly</div>
                    <div class="challenge-phrase">"{st.session_state.challenge_phrase}"</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.write("")
                st.write("")
                if st.button("🎲", key="new_phrase", help="Get new phrase"):
                    st.session_state.challenge_phrase = random.choice(CHALLENGE_PHRASES)
                    st.session_state.last_verify_hash = None
                    st.rerun()

            st.markdown("""
            <div class="glass-card">
                <div style="text-align:center;">
                    <span style="font-family:Syne; font-size:1.1rem; font-weight:700; color:#d4af37;">
                        Verification Audio
                    </span><br>
                    <span style="color:rgba(255,220,210,0.7); font-size:0.88rem;">
                        Tap below and say the phrase above
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            verify_audio = get_audio_input("🎤 Record Verification", key="verify_audio_input")

            if verify_audio is not None:
                audio_bytes = verify_audio.read()
                current_hash = hash(audio_bytes[:500])

                if st.session_state.last_verify_hash != current_hash:
                    with st.spinner("🔍 Analysing voice..."):
                        try:
                            test_emb = extract_embedding(audio_bytes)
                            scores = [cosine_sim(e, test_emb) for e in stored]
                            best_score = max(scores)
                            avg_score = round(sum(scores) / len(scores), 3)

                            # BETTER DEMO-SAFE VOICE PASS LOGIC
                            voice_passed = (best_score >= VOICE_THRESHOLD) or (avg_score >= 0.72)

                            spoken_text = transcribe_audio(audio_bytes)
                            ratio = phrase_ratio(spoken_text, st.session_state.challenge_phrase)
                            phrase_passed = ratio >= PHRASE_THRESHOLD

                            access_granted = voice_passed and phrase_passed
                            st.session_state.last_verify_hash = current_hash

                            st.divider()

                            if access_granted:
                                st.markdown('<div class="result-granted">✅ ACCESS GRANTED</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="result-denied">❌ ACCESS DENIED</div>', unsafe_allow_html=True)

                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Best Score", f"{best_score:.3f}", f"threshold {VOICE_THRESHOLD}")
                            c2.metric("Avg Score", f"{avg_score:.3f}")
                            c3.metric("Voice Match", "✅ PASS" if voice_passed else "❌ FAIL")
                            c4.metric("Phrase Match", "✅ PASS" if phrase_passed else "❌ FAIL")

                            st.caption(f'**You said:** *"{spoken_text if spoken_text else "nothing detected"}"*')
                            st.caption(f'**Expected:** *"{st.session_state.challenge_phrase}"*')
                            st.caption(f'**Word overlap:** {ratio*100:.0f}%')

                            if not access_granted:
                                if not voice_passed and not phrase_passed:
                                    st.warning("⚠️ Both voice and phrase failed. Speak clearly and say the exact phrase.")
                                elif not voice_passed:
                                    st.warning(f"⚠️ Voice score {best_score:.3f} is below threshold {VOICE_THRESHOLD}. Try again in a quieter place.")
                                elif not phrase_passed:
                                    st.warning(f"⚠️ Only {ratio*100:.0f}% of words matched. Say the exact phrase shown.")

                            st.session_state.challenge_phrase = random.choice(CHALLENGE_PHRASES)

                        except Exception as e:
                            st.error(f"Verification error: {e}")
