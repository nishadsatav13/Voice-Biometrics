import os
import numpy as np
from numpy.linalg import norm

USER_DIR = "data/users"


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def calibrate():
    users = [
        f for f in os.listdir(USER_DIR)
        if f.endswith(".npy")
    ]

    genuine_scores = []
    impostor_scores = []

    embeddings = {
        user: np.load(os.path.join(USER_DIR, user))
        for user in users
    }

    for user, emb_bank in embeddings.items():
        for i in range(len(emb_bank)):
            for j in range(i + 1, len(emb_bank)):
                genuine_scores.append(
                    cosine_similarity(emb_bank[i], emb_bank[j])
                )

    user_list = list(embeddings.keys())

    for i in range(len(user_list)):
        for j in range(i + 1, len(user_list)):
            emb_a = embeddings[user_list[i]][0]
            emb_b = embeddings[user_list[j]][0]

            impostor_scores.append(
                cosine_similarity(emb_a, emb_b)
            )

    print("\n📊 Calibration Results")
    print("-----------------------")
    print(f"Genuine avg: {np.mean(genuine_scores):.3f}")
    print(f"Impostor avg: {np.mean(impostor_scores):.3f}")

    suggested = (np.mean(genuine_scores) +
                 np.mean(impostor_scores)) / 2

    print(f"\n🎯 Suggested threshold: {suggested:.3f}")


if __name__ == "__main__":
    calibrate()
