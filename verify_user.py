import os
import numpy as np
from numpy.linalg import norm

from extract_embedding_resemblyzer import extract_embedding
from record_audio import record_audio


USER_DIR = "data/users"

THRESHOLD = 0.75


# cosine similarity
def cosine_similarity(a, b):

    a = np.array(a).flatten()
    b = np.array(b).flatten()

    denom = norm(a) * norm(b)

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


# verification
def verify_user(username):

    user_file = os.path.join(USER_DIR, f"{username}.npy")

    if not os.path.exists(user_file):

        print("❌ User not found")
        return


    print(f"\n🔐 Verifying user: {username}")


    audio_file = f"{username}_login.wav"

    input("\nPress ENTER to record login voice...")

    record_audio(audio_file)


    test_embedding = extract_embedding(audio_file)

    stored_embeddings = np.load(user_file)


    print("\nStored samples:", stored_embeddings.shape[0])


    scores = []

    for emb in stored_embeddings:

        score = cosine_similarity(emb, test_embedding)

        scores.append(score)


    best_score = max(scores)

    avg_score = sum(scores) / len(scores)


    print("\nSimilarity scores:", [round(s,3) for s in scores])

    print(f"\nBest score: {best_score:.3f}")
    print(f"Average score: {avg_score:.3f}")


    if best_score >= THRESHOLD:

        print("\n✅ ACCESS GRANTED")

    else:

        print("\n❌ ACCESS DENIED")


if __name__ == "__main__":

    username = input("Enter username: ").strip()

    verify_user(username)