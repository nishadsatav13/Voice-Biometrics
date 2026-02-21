from extract_embedding import extract_embedding
from record_audio import record_audio
import numpy as np

print("\n🎤 Recording sample 1...")
record_audio("sample1.wav")

input("\nPress ENTER to record sample 2...")

print("\n🎤 Recording sample 2...")
record_audio("sample2.wav")

emb1 = extract_embedding("sample1.wav")
emb2 = extract_embedding("sample2.wav")

diff = np.linalg.norm(emb1 - emb2)

print("\nEmbedding difference:", diff)

print("\nFirst 5 values sample 1:", emb1[:5])
print("First 5 values sample 2:", emb2[:5])
