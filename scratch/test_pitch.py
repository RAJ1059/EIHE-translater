import asyncio
import edge_tts
import librosa
import numpy as np
import os

text = "Hello, I'm Dr. Ise Alvarez, a physician specializing in aesthetic medicine and pulmonology."
os.makedirs("scratch", exist_ok=True)

async def test(pitch_val):
    comm = edge_tts.Communicate(text, "en-US-AvaMultilingualNeural", pitch=pitch_val)
    f = f"scratch/test_pitch_{pitch_val}.mp3"
    await comm.save(f)
    y, sr = librosa.load(f, sr=None)
    f0, vf, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    f0_v = f0[vf] if vf is not None else np.array([])
    return round(float(np.mean(f0_v)), 1)

async def main():
    for p in ["+0Hz", "+25Hz", "+35Hz", "+45Hz", "+55Hz", "+60Hz"]:
        mean_f0 = await test(p)
        print(f"Pitch {p:6} -> {mean_f0} Hz")

if __name__ == "__main__":
    asyncio.run(main())
