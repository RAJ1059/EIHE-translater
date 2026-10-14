import logging
import sys
import warnings
import os


def setup_logger(name_log):
    lg = logging.getLogger(name_log)
    lg.setLevel(logging.INFO)

    _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
    _default_handler.flush = sys.stderr.flush
    lg.addHandler(_default_handler)

    lg.propagate = False

    for handler in lg.handlers:
        formatter = logging.Formatter("[%(levelname)s] >> %(message)s")
        handler.setFormatter(formatter)

    return lg


logger = setup_logger("sonitranslate")
logger.setLevel(logging.INFO)


def set_logging_level(verbosity_level):
    logging_level_mapping = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    logger.setLevel(logging_level_mapping.get(verbosity_level, logging.INFO))


def configure_logging_libs(debug=False):
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings(
        action="ignore", category=UserWarning, module="pyannote"
    )
    modules = [
        "numba", "httpx", "markdown_it", "speechbrain", "fairseq", "pyannote",
        "faiss",
        "pytorch_lightning.utilities.migration.utils",
        "pytorch_lightning.utilities.migration",
        "pytorch_lightning",
        "lightning",
        "lightning.pytorch.utilities.migration.utils",
    ]
    try:
        for module in modules:
            level = logging.ERROR if module in ["speechbrain", "fairseq"] else logging.WARNING
            logging.getLogger(module).setLevel(level)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3" if not debug else "1"

        # Silence Windows asyncio WinError 10054 connection reset noise
        if sys.platform == "win32":
            try:
                from asyncio.proactor_events import _ProactorBasePipeTransport
                _orig_ccl = _ProactorBasePipeTransport._call_connection_lost

                def _safe_ccl(self, exc):
                    try:
                        return _orig_ccl(self, exc)
                    except (ConnectionResetError, OSError):
                        pass

                _ProactorBasePipeTransport._call_connection_lost = _safe_ccl
            except Exception:
                pass

        # Compatibility shims for torchaudio / pyannote
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

        # Compatibility patch for PyTorch 2.6+ weights_only loading in pyannote / whisperx
        try:
            import torch
            _orig_torch_load = torch.load

            def _safe_torch_load(*args, **kwargs):
                kwargs["weights_only"] = False
                return _orig_torch_load(*args, **kwargs)

            torch.load = _safe_torch_load
        except Exception:
            pass

        # fix verbose pyannote audio
        def fix_verbose_pyannote(*args, what=""):
            pass
        import pyannote.audio.core.model  # noqa
        pyannote.audio.core.model.check_version = fix_verbose_pyannote
    except Exception as error:
        logger.error(str(error))
