# Voice Biometric Authentication System

🔗 **Live Demo:** https://voice-biometrics-nsxbb8paq7lmingaewqenp.streamlit.app/

## About
A passwordless voice biometric authentication system using:
- Resemblyzer — 256-dim speaker embeddings via metric learning
- Faster-Whisper — challenge-response speech verification
- Cosine similarity — voice matching (threshold: 0.82)
- Streamlit — deployed frontend

## How it works
1. **Enroll** — Record 5 voice samples to create your voice profile
2. **Login** — Speak a random challenge phrase to authenticate
3. Both voice identity AND phrase must match to grant access