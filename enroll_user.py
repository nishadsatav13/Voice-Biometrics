import os
import numpy as np

from extract_embedding_resemblyzer import extract_embedding
from record_audio import record_audio


# =========================
# CONFIG
# =========================

USER_DIR = "data/users"

# Number of enrollment samples
NUM_SAMPLES = 5


# =========================
# Enroll user
# =========================

def enroll_user(username):

    os.makedirs(USER_DIR, exist_ok=True)

    print(f"\n👤 Enrolling user: {username}")
    print(f"Please record {NUM_SAMPLES} voice samples\n")

    embeddings = []

    for i in range(NUM_SAMPLES):

        print(f"\nSample {i+1}/{NUM_SAMPLES}")

        input("Press ENTER to record...")

        audio_file = f"{username}_enroll_{i}.wav"

        # Record voice
        record_audio(audio_file)

        # Extract embedding
        embedding = extract_embedding(audio_file)

        embeddings.append(embedding)

        print("✅ Sample recorded")

    # Convert to numpy array
    embeddings = np.array(embeddings)

    # Save embeddings
    save_path = os.path.join(USER_DIR, f"{username}.npy")

    np.save(save_path, embeddings)

    print("\n============================")
    print("✅ Enrollment complete")
    print(f"User: {username}")
    print(f"Samples stored: {len(embeddings)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved at: {save_path}")
    print("============================\n")


# =========================
# Main
# =========================

if __name__ == "__main__":

    username = input("Enter username to enroll: ").strip()

    enroll_user(username)