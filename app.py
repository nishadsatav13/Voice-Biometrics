import os
import random
import tempfile
import re
import numpy as np
from numpy.linalg import norm
import streamlit as st
from resemblyzer import VoiceEncoder, preprocess_wav
from faster_whisper import WhisperModel
from audio_recorder_streamlit import audio_recorder

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VoiceAuth — Biometric Authentication",
    page_icon="🎙️",
    layout="centered"
)

# ─────────────────────────────────────────────
# CUSTOM CSS (red-black-gold theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main {
        background: linear-gradient(180deg, #090909 0%, #120707 100%);
    }

    .title-text {
        font-family: 'Syne', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 10%, #ff3b3b 55%, #d4af37 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }

    .subtitle-text {
        text-align: center;
        color: rgba(255,235,220,0.72);
        font-size: 0.95rem;
        margin-bottom: 1.8rem;
    }

    .tips-box {
        background: rgba(255, 59, 59, 0.08);
        border-left: 4px solid #ff3b3b;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 14px 0;
        color: rgba(255,240,235,0.92);
    }

    .record-box {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(212,175,55,0.20);
        border-radius: 18px;
        padding: 20px;
        margin: 16px 0;
        text-align: center;
        box-shadow: 0 0 24px rgba(255,59,59,0.08);
    }

    .record-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
    }

    .record-sub {
        color: rgba(255,235,220,0.75);
        font-size: 0.95rem;
        margin-bottom: 12px;
    }

    .challenge-box {
        background: rgba(255, 59, 59, 0.08);
        border: 1px solid rgba(212,175,55,0.28);
        border-radius: 16px;
        padding: 18px 24px;
        text-align: center;
        margin: 1rem 0;
    }

    .challenge-label {
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #d4af37;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .challenge-phrase {
        font-family: 'Syne', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        color: white;
    }

    .result-granted {
        background: rgba(0,212,170,0.08);
        border: 1px solid rgba(0,212,170,0.35);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        font-family: 'Syne', sans-serif;
        font-size: 1.45rem;
        font-weight: 800;
        color: #00d4aa;
    }

    .result-denied {
        background: rgba(255,79,110,0.09);
        border: 1px solid rgba(255,79,110,0.35);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        font-family: 'Syne', sans-serif;
        font-size: 1.45rem;
        font-weight: 800;
        color: #ff5c74;
    }

    .wave {
        display: flex;
        justify-content: center;
        align-items: end;
        gap: 6px;
        height: 40px;
        margin-top: 10px;
        margin-bottom: 6px;
    }

    .bar {
        width: 6px;
        border-radius: 10px;
        background: linear-gradient(180deg, #ff3b3b, #d4af37);
        animation: pulse 1s infinite ease-in-out;
    }

    .bar:nth-child(1) { height: 18px; animation-delay: 0s; }
    .bar:nth-child(2) { height: 28px; animation-delay: 0.15s; }
    .bar:nth-child(3) { height: 36px; animation-delay: 0.3s; }
    .bar:nth-child(4) { height: 24px; animation-delay: 0.45s; }
    .bar:nth-child(5) { height: 32px; animation-delay: 0.6s; }

    @keyframes pulse {
        0%, 100% { transform: scaleY(0.6); opacity: 0.6; }
        50% { transform: scaleY(1.15); opacity: 1; }
    }

    .record-label {
        display: inline-block;
        background: linear-gradient(135deg, #ff3b3b, #a30000);
        color: white;
        padding: 10px 18px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 8px;
        box-shadow: 0 6px 20px rgba(255,59,59,0.25);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DIR = os.path.join(BASE_DIR, "data", "users")

BEST_SCORE_THRESHOLD = 0.81
AVG_SCORE_THRESHOLD = 0.75
PHRASE_THRESHOLD = 0.80
NUM_SAMPLES = 5
MIN_TRANSCRIBED_WORDS = 2

CHALLENGE_PHRASES = [
    "hello my name is secure",
    "today is a good day",
    "open the voice lock",
    "this is my access key",
    "let me log in",
    "verify my identity",
    "my voice is unique",
    "grant access now",
]

os.makedirs(USER_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD MODELS
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
            vad_parameters=dict(min_silence_duration_ms=250)
        )
        text = " ".join(s.text for s in segments).strip().lower()
        for ch in ",.!?":
            text = text.replace(ch, "")
        return text
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def phrase_match_ratio(spoken: str, expected: str) -> float:
    spoken_words = set(spoken.split())
    expected_words = set(expected.split())
    if not expected_words:
        return 0.0
    return len(spoken_words & expected_words) / len(expected_words)

def list_users():
    return sorted([
        f.replace(".npy", "")
        for f in os.listdir(USER_DIR)
        if f.endswith(".npy")
    ])

def is_good_transcription(text: str) -> bool:
    return len(text.split()) >= MIN_TRANSCRIBED_WORDS

def voice_decision(best_score: float, avg_score: float) -> bool:
    return best_score >= BEST_SCORE_THRESHOLD and avg_score >= AVG_SCORE_THRESHOLD

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "enroll_embeddings" not in st.session_state:
    st.session_state.enroll_embeddings = []

if "enroll_username" not in st.session_state:
    st.session_state.enroll_username = ""

if "challenge_phrase" not in st.session_state:
    st.session_state.challenge_phrase = None

if "last_enroll_audio_id" not in st.session_state:
    st.session_state.last_enroll_audio_id = None

if "last_verify_audio_id" not in st.session_state:
    st.session_state.last_verify_audio_id = None

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="title-text">🎙 VoiceAuth</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Biometric Voice Authentication System</div>', unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_enroll, tab_login = st.tabs(["🧬 Enroll", "🔐 Login"])

# ═══════════════════════════════════════════════
# ENROLL TAB
# ═══════════════════════════════════════════════
with tab_enroll:
    st.subheader("Create Voice Profile")
    st.caption("Record 5 short voice samples to register your identity.")

    st.markdown("""
    <div class="tips-box">
        <b>Best recording tips:</b><br>
        • Speak clearly for 3–4 seconds<br>
        • After clicking record, speak naturally and stop<br>
        • Recording auto-finishes after short speech / silence<br>
        • Use the same room and device for best results
    </div>
    """, unsafe_allow_html=True)

    username_input = st.text_input("Username", placeholder="Enter your username", key="enroll_user")
    username = sanitize_username(username_input)

    if username_input and not username:
        st.warning("⚠️ Username can only contain letters, numbers, underscores, and hyphens.")

    if username:
        if st.session_state.enroll_username != username:
            st.session_state.enroll_embeddings = []
            st.session_state.enroll_username = username
            st.session_state.last_enroll_audio_id = None

        done = len(st.session_state.enroll_embeddings)
        total = NUM_SAMPLES

        st.progress(done / total, text=f"Samples recorded: {done} / {total}")

        dots = ""
        for i in range(total):
            if i < done:
                dots += "🟢 "
            elif i == done:
                dots += "🔴 "
            else:
                dots += "⚪ "
        st.markdown(f"**{dots}**")

        existing_user_path = os.path.join(USER_DIR, f"{username}.npy")
        already_enrolled = os.path.exists(existing_user_path)

        if already_enrolled and done == 0:
            st.success(f"✅ **{username}** is already enrolled.")
            if st.button("🔄 Re-enroll (overwrite)", key="reenroll_btn"):
                try:
                    os.remove(existing_user_path)
                except:
                    pass
                st.session_state.enroll_embeddings = []
                st.session_state.last_enroll_audio_id = None
                st.rerun()

        else:
            if done < total:
                st.markdown(f"""
                <div class="record-box">
                    <div class="record-label">RECORD</div>
                    <div class="record-title">Sample {done + 1} of {total}</div>
                    <div class="record-sub">Tap record, speak for about 3–4 seconds, then pause.</div>
                    <div class="wave">
                        <div class="bar"></div>
                        <div class="bar"></div>
                        <div class="bar"></div>
                        <div class="bar"></div>
                        <div class="bar"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                audio_bytes = audio_recorder(
                    text="Record",
                    recording_color="#ff3b3b",
                    neutral_color="#d4af37",
                    icon_name="microphone",
                    icon_size="2x",
                    pause_threshold=1.2,
                    sample_rate=41000,
                    key=f"enroll_audio_{done}"
                )

                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
                    current_audio_id = hash(audio_bytes)

                    if st.session_state.last_enroll_audio_id != current_audio_id:
                        with st.spinner("Processing voice sample..."):
                            try:
                                emb = extract_embedding(audio_bytes)
                                st.session_state.enroll_embeddings.append(emb)
                                st.session_state.last_enroll_audio_id = current_audio_id

                                st.success(f"✅ Sample {done + 1} recorded successfully!")

                                if len(st.session_state.enroll_embeddings) == total:
                                    all_emb = np.array(st.session_state.enroll_embeddings, dtype=np.float32)
                                    save_path = os.path.join(USER_DIR, f"{username}.npy")
                                    np.save(save_path, all_emb)

                                    st.balloons()
                                    st.success(f"🎉 **{username}** enrolled successfully with {total} voice samples!")

                                    st.session_state.enroll_embeddings = []
                                    st.session_state.enroll_username = ""
                                    st.session_state.last_enroll_audio_id = None
                                else:
                                    st.rerun()

                            except Exception as e:
                                st.error(f"Error processing audio: {e}")

    users = list_users()
    if users:
        st.divider()
        st.caption(f"**Enrolled users:** {', '.join(users)}")

# ═══════════════════════════════════════════════
# LOGIN TAB
# ═══════════════════════════════════════════════
with tab_login:
    st.subheader("Voice Verification")
    st.caption("Speak the phrase shown below to verify your identity.")

    login_input = st.text_input("Username", placeholder="Enter your username", key="login_user")
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
                st.markdown(f'''
                <div class="challenge-box">
                    <div class="challenge-label">Speak this phrase</div>
                    <div class="challenge-phrase">"{st.session_state.challenge_phrase}"</div>
                </div>
                ''', unsafe_allow_html=True)

            with col2:
                st.write("")
                st.write("")
                if st.button("🎲 New Phrase", key="new_phrase_btn"):
                    st.session_state.challenge_phrase = random.choice(CHALLENGE_PHRASES)
                    st.session_state.last_verify_audio_id = None
                    st.rerun()

            st.markdown("""
            <div class="record-box">
                <div class="record-label">RECORD</div>
                <div class="record-title">Verification Audio</div>
                <div class="record-sub">Tap record and speak the phrase clearly in one go.</div>
                <div class="wave">
                    <div class="bar"></div>
                    <div class="bar"></div>
                    <div class="bar"></div>
                    <div class="bar"></div>
                    <div class="bar"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            verify_audio_bytes = audio_recorder(
                text="Record",
                recording_color="#ff3b3b",
                neutral_color="#d4af37",
                icon_name="microphone",
                icon_size="2x",
                pause_threshold=1.2,
                sample_rate=41000,
                key="verify_audio"
            )

            if verify_audio_bytes:
                st.audio(verify_audio_bytes, format="audio/wav")
                current_verify_audio_id = hash(verify_audio_bytes)

                if st.session_state.last_verify_audio_id != current_verify_audio_id:
                    with st.spinner("🔍 Verifying identity..."):
                        try:
                            test_emb = extract_embedding(verify_audio_bytes)
                            scores = [cosine_similarity(e, test_emb) for e in stored]
                            best_score = max(scores)
                            avg_score = round(sum(scores) / len(scores), 3)

                            voice_passed = voice_decision(best_score, avg_score)

                            spoken_text = transcribe(verify_audio_bytes)
                            ratio = phrase_match_ratio(spoken_text, st.session_state.challenge_phrase)
                            phrase_passed = ratio >= PHRASE_THRESHOLD
                            enough_speech = is_good_transcription(spoken_text)

                            access_granted = voice_passed and phrase_passed and enough_speech
                            st.session_state.last_verify_audio_id = current_verify_audio_id

                            st.divider()

                            if access_granted:
                                st.markdown('<div class="result-granted">✅ ACCESS GRANTED</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="result-denied">❌ ACCESS DENIED</div>', unsafe_allow_html=True)

                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Best Voice", f"{best_score:.3f}", f"≥ {BEST_SCORE_THRESHOLD}")
                            c2.metric("Avg Voice", f"{avg_score:.3f}", f"≥ {AVG_SCORE_THRESHOLD}")
                            c3.metric("Phrase Match", "✅ PASS" if phrase_passed else "❌ FAIL")
                            c4.metric("Word Overlap", f"{ratio*100:.0f}%")

                            st.caption(f'**You said:** *"{spoken_text if spoken_text else "nothing detected"}"*')
                            st.caption(f'**Expected:** *"{st.session_state.challenge_phrase}"*')

                            if not enough_speech:
                                st.warning("⚠️ Very little speech detected. Speak clearly for 3–4 seconds.")
                            elif not voice_passed and phrase_passed:
                                st.warning("⚠️ Phrase matched, but voice similarity was not strong enough.")
                            elif voice_passed and not phrase_passed:
                                st.warning("⚠️ Voice matched, but phrase did not match enough words.")
                            elif not access_granted:
                                st.warning("⚠️ Authentication failed. Try again clearly in a quieter environment.")

                            st.session_state.challenge_phrase = random.choice(CHALLENGE_PHRASES)

                        except Exception as e:
                            st.error(f"Verification error: {e}")
