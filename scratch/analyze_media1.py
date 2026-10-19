import os
import json
import librosa
import numpy as np
import soundfile as sf

def analyze_audio(file_path):
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    try:
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        rms = float(np.sqrt(np.mean(y**2)))
        rms_db = float(20 * np.log10(rms + 1e-9))
        
        # Pitch estimation using pyin
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
        )
        f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
        
        if len(f0_voiced) > 0:
            pitch_mean = float(np.mean(f0_voiced))
            pitch_std = float(np.std(f0_voiced))
            pitch_median = float(np.median(f0_voiced))
            pitch_min = float(np.min(f0_voiced))
            pitch_max = float(np.max(f0_voiced))
        else:
            pitch_mean = pitch_std = pitch_median = pitch_min = pitch_max = 0.0
            
        # Spectral centroid (brightness / vocal timbre indicator)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = float(np.mean(cent))
        
        return {
            "path": file_path,
            "duration": round(duration, 3),
            "sample_rate": sr,
            "rms_db": round(rms_db, 2),
            "pitch_mean_hz": round(pitch_mean, 1),
            "pitch_std_hz": round(pitch_std, 1),
            "pitch_median_hz": round(pitch_median, 1),
            "pitch_range_hz": [round(pitch_min, 1), round(pitch_max, 1)],
            "spectral_centroid_hz": round(spectral_centroid_mean, 1)
        }
    except Exception as e:
        return {"error": str(e)}

files_to_check = {
    "original_full_audio": "audio.wav",
    "original_segment_1": r"processed\SPEAKER_00audio_ynwhJ4891WJWZXar\1.201.wav",
    "edge_tts_raw_segment_1": r"audio\1.201.mp3",
    "edge_tts_accelerated_segment_1": r"audio2\audio\1.201.ogg",
    "edge_tts_raw_segment_2": r"audio\11.704.mp3",
    "edge_tts_accelerated_segment_2": r"audio2\audio\11.704.ogg",
    "final_dub_solo": "audio_dub_solo.ogg",
    "final_mix": "audio_mix.mp3"
}

# Also check any file in processed tts folder
processed_tts_dir = r"processed\ttsSPEAKER_00audio_ynwhJ4891WJWZXar"
if os.path.exists(processed_tts_dir):
    for f in os.listdir(processed_tts_dir):
        if f.endswith(".wav"):
            files_to_check[f"openvoice_converted_{f}"] = os.path.join(processed_tts_dir, f)

results = {}
for name, path in files_to_check.items():
    results[name] = analyze_audio(path)

print(json.dumps(results, indent=2))
