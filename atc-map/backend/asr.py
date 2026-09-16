"""
Speech-to-text (faster-whisper)
────────────────────────────────
Transcribes one audio segment (a single ATC/pilot transmission) at a time,
with word-level timestamps offset to be absolute within the recording.

Ported from audio_proto/vad_server.py steps 1-3.
"""

import os
import shutil
import threading
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

TARGET_SR = 16000
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small.en")

# Seeds Whisper's decoder with domain vocabulary (kept under its 224-token cap)
AVIATION_PROMPT = (
    "Air traffic control radio at Kennedy airport. Kennedy Ground, Kennedy Tower, "
    "Delta 795 heavy, runway 31L, runway 22R, exit right at Zulu Alpha, "
    "taxi via Alpha, Bravo, Charlie, Echo, Foxtrot, Golf, Hotel, India, Juliet, "
    "Kilo, Lima, Mike, November, Oscar, Papa, Quebec, Romeo, Sierra, Tango, "
    "Uniform, Victor, Whiskey, X-ray, Yankee, Zulu, hold short of, cleared to land, "
    "cleared for takeoff, contact ground point niner, readback correct, "
    "wind one zero zero at two zero, gust two eight."
)

_whisper_model = None
_whisper_lock = threading.Lock()


def _preload_cuda_libs() -> None:
    """ctranslate2 dlopens cuBLAS/cuDNN at runtime; the pip-installed copies are
    not on the loader search path, so load them into the process by full path."""
    import ctypes
    import nvidia  # namespace package: iterate __path__, __file__ is None

    for root in nvidia.__path__:
        for pattern in ("cublas/lib/libcublas*.so.*", "cudnn/lib/libcudnn*.so.*"):
            for lib in sorted(Path(root).glob(pattern)):
                try:
                    ctypes.CDLL(str(lib))
                except OSError:
                    pass


def get_whisper() -> WhisperModel:
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            if shutil.which("nvidia-smi"):
                try:
                    _preload_cuda_libs()
                    model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
                    # cuDNN kernels only load on first inference — fail here, not mid-job
                    next(model.transcribe(np.zeros(TARGET_SR, dtype=np.float32))[0], None)
                    _whisper_model = model
                    print(f"whisper: {WHISPER_MODEL} on cuda (float16)")
                    return _whisper_model
                except Exception as e:
                    print(f"whisper: GPU init failed ({e}); falling back to CPU")
            _whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
            print(f"whisper: {WHISPER_MODEL} on cpu (int8)")
        return _whisper_model


def transcribe_segment(chunk: np.ndarray, start_s: float) -> dict:
    """Transcribe one segment. `start_s` offsets word timestamps to be
    absolute within the original recording. Returns {transcript, words}."""
    whisper = get_whisper()  # may download the model on first run
    pieces, _ = whisper.transcribe(
        chunk,
        language="en",
        beam_size=5,
        word_timestamps=True,
        initial_prompt=AVIATION_PROMPT,
        condition_on_previous_text=False,
    )
    texts, words = [], []
    for piece in pieces:
        texts.append(piece.text.strip())
        for w in piece.words or []:
            words.append({"w": w.word.strip(), "t": round(float(start_s + w.start), 2)})
    return {"transcript": " ".join(t for t in texts if t), "words": words}
