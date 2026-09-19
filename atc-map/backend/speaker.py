"""
Speaker ID (ATC vs. PILOT)
────────────────────────────
Cosine similarity of a resemblyzer voice embedding against the precomputed
`atc_fingerprint.npy` reference. Segments whose similarity falls in the
ambiguity band around the threshold fall back to LLM text classification
(ATC issues instructions, pilots read back ending with their callsign).

Ported from audio_proto/vad_server.py step 3.
"""

import os
import threading
from pathlib import Path

import numpy as np
from resemblyzer import VoiceEncoder

# similarity >= this -> "ATC", else "PILOT". Calibrated per recording via the
# prototype's threshold slider; this is just the server-side default.
ATC_SIM_THRESHOLD = 0.75
ATC_AMBIGUITY_BAND = 0.05  # similarity within +-this of the threshold -> LLM fallback
ATC_FINGERPRINT_PATH = os.environ.get(
    "ATC_FINGERPRINT_PATH", str(Path(__file__).resolve().parent.parent.parent / "atc_fingerprint.npy")
)

_speaker_encoder = None
_speaker_lock = threading.Lock()
_atc_fingerprint = None


def get_speaker_encoder() -> VoiceEncoder:
    global _speaker_encoder
    with _speaker_lock:
        if _speaker_encoder is None:
            _speaker_encoder = VoiceEncoder()
        return _speaker_encoder


def get_atc_fingerprint() -> np.ndarray:
    global _atc_fingerprint
    if _atc_fingerprint is None:
        path = Path(ATC_FINGERPRINT_PATH)
        if not path.exists():
            raise FileNotFoundError(f"ATC fingerprint not found at {path}")
        _atc_fingerprint = np.load(path)
    return _atc_fingerprint


def build_fingerprint(chunks: list[np.ndarray]) -> np.ndarray:
    """Build one normalized voice fingerprint from known samples of a speaker."""
    if not chunks:
        raise ValueError("at least one reference sample is required")
    encoder = get_speaker_encoder()
    embeddings = [encoder.embed_utterance(chunk) for chunk in chunks if chunk.size]
    if not embeddings:
        raise ValueError("reference samples are empty")
    fingerprint = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(fingerprint)
    if norm == 0:
        raise ValueError("could not build a speaker fingerprint")
    return fingerprint / norm


def identify_speaker(
    chunk: np.ndarray,
    transcript: str = "",
    fingerprint: np.ndarray | None = None,
) -> dict:
    """Embed one segment and classify it ATC/PILOT. When the similarity score
    lands in the ambiguity band around the threshold and a transcript is
    available, breaks the tie with LLM text classification instead of the
    raw cutoff. ``fingerprint`` can provide a recording-specific ATC reference;
    omitting it preserves the server's configured reference. Returns
    {speaker, similarity}."""
    encoder = get_speaker_encoder()
    if fingerprint is None:
        fingerprint = get_atc_fingerprint()
    embedding = encoder.embed_utterance(chunk)
    similarity = round(float(np.dot(embedding, fingerprint)), 3)

    if transcript and abs(similarity - ATC_SIM_THRESHOLD) <= ATC_AMBIGUITY_BAND:
        from llm import classify_speaker
        try:
            speaker = classify_speaker(transcript)
        except Exception:
            speaker = "ATC" if similarity >= ATC_SIM_THRESHOLD else "PILOT"
    else:
        speaker = "ATC" if similarity >= ATC_SIM_THRESHOLD else "PILOT"

    return {"speaker": speaker, "similarity": similarity}
