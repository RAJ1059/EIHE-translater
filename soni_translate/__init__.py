"""
SoniTranslate package initialization.
Ensures torchaudio, torch.load, and pyannote environment compatibility.
"""
import sys

try:
    import torchaudio
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *args, **kwargs: None
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    
    import types
    from dataclasses import dataclass
    if "torchaudio.backend" not in sys.modules:
        backend = types.ModuleType("torchaudio.backend")
        common = types.ModuleType("torchaudio.backend.common")
        @dataclass
        class AudioMetaData:
            sample_rate: int = 16000
            num_frames: int = 0
            num_channels: int = 1
            bits_per_sample: int = 16
            encoding: str = "PCM_S"
        common.AudioMetaData = AudioMetaData
        backend.common = common
        sys.modules["torchaudio.backend"] = backend
        sys.modules["torchaudio.backend.common"] = common
except Exception:
    pass

try:
    import torch
    # Ensure PyTorch 2.6+ unpickler always loads models with weights_only=False
    _orig_torch_load = torch.load

    def _safe_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)

    torch.load = _safe_torch_load
except Exception:
    pass
