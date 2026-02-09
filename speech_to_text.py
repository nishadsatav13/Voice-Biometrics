import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer

MODEL_PATH = "vosk-model-small-en-us-0.15"

FS = 16000
DURATION = 4

model = Model(MODEL_PATH)

def recognize_speech():
    rec = KaldiRecognizer(model, FS)

    print("🎤 Speak the phrase...")
    audio = sd.rec(int(DURATION * FS), samplerate=FS, channels=1)
    sd.wait()

    rec.AcceptWaveform(audio.tobytes())
    result = json.loads(rec.Result())
    return result.get("text", "")

if __name__ == "__main__":
    print(recognize_speech())
