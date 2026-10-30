import os
import soundfile as sf
import librosa
import numpy as np

print("Checking audio.wav:")
if os.path.exists("audio.wav"):
    y, sr = librosa.load("audio.wav", sr=16000)
    dur = len(y) / sr
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    f0_clean = f0[~np.isnan(f0)] if f0 is not None else []
    mean_f0 = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0
    print(f"audio.wav: duration={dur:.2f}s, mean pitch={mean_f0:.1f}Hz")

print("\nChecking sub_ori.srt vs sub_tra.srt:")
if os.path.exists("sub_ori.srt"):
    with open("sub_ori.srt", "r", encoding="utf-8") as f:
        print("--- sub_ori.srt ---")
        print(f.read())

if os.path.exists("sub_tra.srt"):
    with open("sub_tra.srt", "r", encoding="utf-8") as f:
        print("--- sub_tra.srt ---")
        print(f.read())
