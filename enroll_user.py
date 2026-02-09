import os
import numpy as np
from extract_embedding import extract_embedding
from record_audio import record_audio

# Folder to store user embeddings
USER_DIR = "data/users"
os.makedirs(USER_DIR, exist_ok=True)

def enroll_user(username):
    print(f"\n👤 Enrolling user: {username}")

    audio_file = f"{username}_enroll.wav"
    record_audio(audio_file)

    embedding = extract_embedding(audio_file)

    save_path = os.path.join(USER_DIR, f"{username}.npy")
    np.save(save_path, embedding)

    print(f"✅ User '{username}' enrolled successfully")
    print(f"📁 Embedding saved at: {save_path}")

if __name__ == "__main__":
    username = input("Enter username to enroll: ").strip()
    enroll_user(username)
