import librosa
import numpy as np
import time

t0 = time.time()
y, sr = librosa.load('audio.wav', sr=16000, duration=8.0)
f0 = librosa.yin(y, fmin=75, fmax=350, sr=sr)
f0_clean = f0[(f0 > 75) & (f0 < 350)]
mean_pitch = float(np.mean(f0_clean))
print(f"Pitch: {mean_pitch:.1f} Hz, Elapsed: {time.time()-t0:.2f}s")
