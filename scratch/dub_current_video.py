import os
import sys
sys.path.insert(0, os.path.abspath("."))
import re
import asyncio
import librosa
import soundfile as sf
from pydub import AudioSegment
import numpy as np
import edge_tts
from soni_translate.audio_segments import Mixer

def parse_srt(srt_path):
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    
    entries = []
    pattern = re.compile(r"(\d+)\r?\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\r?\n(.*?)(?=\n\n|\n*$)", re.DOTALL)
    for match in pattern.finditer(content):
        idx = int(match.group(1))
        start_str = match.group(2)
        end_str = match.group(3)
        text = match.group(4).strip().replace("\n", " ")
        
        def to_sec(s):
            h, m, sec_ms = s.split(":")
            sec, ms = sec_ms.split(",")
            return int(h)*3600 + int(m)*60 + int(sec) + int(ms)/1000.0
            
        entries.append({
            "idx": idx,
            "start": to_sec(start_str),
            "end": to_sec(end_str),
            "text": text,
        })
    return entries

async def generate_tts(text, output_file, voice="en-US-AndrewMultilingualNeural", pitch="-15Hz", retries=4):
    for attempt in range(retries):
        try:
            communicate = edge_tts.Communicate(text, voice, pitch=pitch)
            await communicate.save(output_file)
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                return
        except Exception as e:
            print(f"Edge TTS attempt {attempt+1} failed: {e}. Retrying...")
            if attempt == retries - 1:
                raise e
            await asyncio.sleep(1.5 * (attempt + 1))

def main():
    srt_path = "sub_tra.srt"
    segments = parse_srt(srt_path)
    print(f"Loaded {len(segments)} segments from {srt_path}")
    
    os.makedirs("scratch/temp_tts", exist_ok=True)
    os.makedirs("scratch/temp_acc", exist_ok=True)
    
    processed_audio_files = []
    
    for seg in segments:
        idx = seg["idx"]
        text = seg["text"]
        start = seg["start"]
        end = seg["end"]
        duration_true = end - start
        
        raw_tts_file = f"scratch/temp_tts/{idx}.mp3"
        acc_tts_file = f"scratch/temp_acc/{idx}.wav"
        
        print(f"\n[Segment {idx}] [{start:.3f} --> {end:.3f}] (duration: {duration_true:.3f}s)")
        print(f"Text: {text}")
        
        # 1. Generate Edge TTS with calibrated pitch (-15Hz for male Dr. Pablo ~126Hz)
        asyncio.run(generate_tts(text, raw_tts_file, voice="en-US-AndrewMultilingualNeural", pitch="-15Hz"))
        
        # Convert to wav and check duration
        os.system(f'ffmpeg -y -loglevel panic -i {raw_tts_file} -acodec pcm_s16le -ar 44100 -ac 2 {acc_tts_file}')
        tts_dur = librosa.get_duration(filename=acc_tts_file)
        print(f"TTS Raw Duration: {tts_dur:.3f}s, Target Window: {duration_true:.3f}s")
        
        # 2. Acceleration rate regulation: strictly respect pauses!
        # If TTS is longer than duration_true, speed up to fit inside duration_true
        if tts_dur > duration_true:
            speed = tts_dur / duration_true
            speed = min(speed, 2.5)
            print(f"Speeding up by {speed:.3f}x to fit inside speech window")
            filters = []
            s = speed
            while s > 2.0:
                filters.append("atempo=2.0")
                s /= 2.0
            while s < 0.5:
                filters.append("atempo=0.5")
                s /= 0.5
            filters.append(f"atempo={s:.3f}")
            filter_str = ",".join(filters)
            tmp_acc = f"scratch/temp_acc/{idx}_acc.wav"
            os.system(f'ffmpeg -y -loglevel panic -i {acc_tts_file} -filter:a "{filter_str}" {tmp_acc}')
            os.replace(tmp_acc, acc_tts_file)
            tts_dur = librosa.get_duration(filename=acc_tts_file)
            print(f"New Accelerated Duration: {tts_dur:.3f}s")
        else:
            print("TTS naturally fits inside window. Silence/pause preserved!")
            
        processed_audio_files.append((start, end, acc_tts_file))

    # 3. Create composite dub audio
    total_duration = segments[-1]["end"] + 2.0
    print(f"\nBuilding composite dub audio (total duration: {total_duration:.2f}s)...")
    base_audio = AudioSegment.silent(duration=int(total_duration * 1000), frame_rate=44100)
    mixer = Mixer()
    mixer.overlay(base_audio)
    
    for start, end, fpath in processed_audio_files:
        audio_seg = AudioSegment.from_file(fpath)
        start_ms = int(start * 1000)
        mixer.overlay(audio_seg, position=start_ms)
        
    final_dub = mixer.to_audio_segment()
    dub_wav = "scratch/dub_voice.wav"
    final_dub.export(dub_wav, format="wav")
    print(f"Dub voice audio written to {dub_wav}")
    
    # 4. Mix with original audio (duck original to volume 0.18, translated to 1.80)
    print("Mixing dub with background audio...")
    mixed_audio = "scratch/mixed_audio.mp3"
    mix_cmd = (
        f'ffmpeg -y -i audio.wav -i {dub_wav} '
        f'-filter_complex "[0:0]volume=0.18[a];[1:0]volume=1.80[b];[a][b]amix=inputs=2:duration=longest" '
        f'-c:a libmp3lame {mixed_audio}'
    )
    os.system(mix_cmd)
    
    # 5. Mux with Video.mp4 into outputs/Media8__en.mp4 and video_dub.mp4
    print("Muxing final video...")
    for out_path in ["outputs/Media8__en.mp4", "video_dub.mp4"]:
        mux_cmd = (
            f'ffmpeg -y -i Video.mp4 -i {mixed_audio} '
            f'-c:v copy -c:a aac -b:a 192k -map 0:v -map 1:a -shortest {out_path}'
        )
        os.system(mux_cmd)
        print(f"Created: {out_path} ({os.path.getsize(out_path)} bytes)")

if __name__ == "__main__":
    main()
