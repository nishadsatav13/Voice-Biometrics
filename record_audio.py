import sounddevice as sd
import soundfile as sf
import numpy as np
import time

def record_audio(filename, duration=4, fs=16000):
    print("\n🎤 Get ready...")
    time.sleep(2)

    print("🔴 Recording NOW — speak!")
    
    # Reset audio device properly
    sd.stop()
    sd.default.samplerate = fs
    sd.default.channels = 1

    audio = sd.rec(int(duration * fs), dtype='float32')
    sd.wait()

    # Ensure audio is not empty
    if np.max(np.abs(audio)) < 1e-4:
        print("⚠️ Warning: very low audio detected")

    sf.write(filename, audio, fs)

    print(f"✅ Saved: {filename}\n")
