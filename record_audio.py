import sounddevice as sd
from scipy.io.wavfile import write

FS = 16000        # Sampling rate (model ke hisaab se)
DURATION = 4      # seconds

def record_audio(filename="live.wav"):
    print("🎤 Speak now...")
    audio = sd.rec(int(DURATION * FS), samplerate=FS, channels=1)
    sd.wait()
    write(filename, FS, audio)
    print(f"✅ Audio saved as {filename}")

if __name__ == "__main__":
    record_audio()
