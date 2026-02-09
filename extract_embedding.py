import librosa
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("embedding_model.h5")

def extract_embedding(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=39)
    mfcc = mfcc.T

    if mfcc.shape[0] < 712:
        pad = np.zeros((712 - mfcc.shape[0], 39))
        mfcc = np.vstack((mfcc, pad))
    else:
        mfcc = mfcc[:712, :]

    mfcc = np.expand_dims(mfcc, axis=0)
    embedding = model.predict(mfcc, verbose=0)

    return embedding[0]

if __name__ == "__main__":
    emb = extract_embedding("testvoice.wav")
    print("Embedding shape:", emb.shape)
