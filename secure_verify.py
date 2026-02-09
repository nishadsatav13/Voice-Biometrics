import os
import numpy as np
from numpy.linalg import norm

from extract_embedding import extract_embedding
from record_audio import record_audio
from speech_to_text import recognize_speech
from challenge import get_random_phrase

USER_DIR = "data/users"
THRESHOLD = 0.75

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def secure_verify(username):
    user_file = os.path.join(USER_DIR, f"{username}.npy")

    if not os.path.exists(user_file):
        print("❌ User not found")
        return

    phrase = get_random_phrase()
    print(f"\n🔐 Speak this phrase clearly:\n➡️  '{phrase}'")

    record_audio("secure_login.wav")

    # WHO check
    test_emb = extract_embedding("secure_login.wav")
    stored_emb = np.load(user_file)
    score = cosine_similarity(stored_emb, test_emb)

    # WHAT check
    spoken_text = recognize_speech()
    print("📝 Recognized text:", spoken_text)

    if phrase.lower() in spoken_text.lower() and score >= THRESHOLD:
        print(f"✅ ACCESS GRANTED (score={score:.2f})")
    else:
        print(f"❌ ACCESS DENIED (score={score:.2f})")

if __name__ == "__main__":
    user = input("Enter username: ").strip()
    secure_verify(user)
