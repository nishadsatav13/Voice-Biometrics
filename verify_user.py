import os
import numpy as np
from extract_embedding import extract_embedding
from record_audio import record_audio
from numpy.linalg import norm

USER_DIR = "data/users"
THRESHOLD = 0.75   # biometric decision threshold

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def verify_user(username):
    user_file = os.path.join(USER_DIR, f"{username}.npy")

    if not os.path.exists(user_file):
        print("❌ User not found")
        return

    print(f"\n🔐 Verifying user: {username}")

    audio_file = f"{username}_login.wav"
    record_audio(audio_file)

    test_embedding = extract_embedding(audio_file)
    stored_embedding = np.load(user_file)

    score = cosine_similarity(stored_embedding, test_embedding)

    print(f"🔢 Similarity score: {score:.3f}")

    if score >= THRESHOLD:
        print("✅ ACCESS GRANTED")
    else:
        print("❌ ACCESS DENIED")

if __name__ == "__main__":
    username = input("Enter username to login: ").strip()
    verify_user(username)
