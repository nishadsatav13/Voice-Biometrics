import json
import sounddevice as sd
import soundfile as sf
from vosk import Model, KaldiRecognizer


MODEL_PATH = "vosk_model/vosk_model"

FS = 16000
DURATION = 4

model = Model(MODEL_PATH)


def recognize_speech(audio_file=None):

    rec = KaldiRecognizer(model, FS)
    rec.SetWords(True)

    if audio_file is not None:

        data, samplerate = sf.read(audio_file)

        if data.dtype != 'int16':
            data = (data * 32767).astype('int16')

        rec.AcceptWaveform(data.tobytes())

    else:

        print("🎤 Speak now...")
        audio = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='int16')
        sd.wait()

        rec.AcceptWaveform(audio.tobytes())

    result = json.loads(rec.Result())

    text = result.get("text", "")

    print("Recognized text:", text)

    return text


if __name__ == "__main__":
    print(recognize_speech())
