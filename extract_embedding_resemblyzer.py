from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np

encoder = VoiceEncoder()

def extract_embedding(audio_path):

    wav = preprocess_wav(audio_path)

    embedding = encoder.embed_utterance(wav)

    # normalize embedding
    embedding = embedding / np.linalg.norm(embedding)

    return embedding


if __name__ == "__main__":

    emb = extract_embedding("testvoice.wav")

    print("Embedding shape:", emb.shape)