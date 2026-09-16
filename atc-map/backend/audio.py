"""
Audio upload, decoding, and VAD segmentation
──────────────────────────────────────────────
Upload endpoint accepts an mp3, decodes it to 16 kHz mono PCM, splits it into
transmissions with silero-VAD, and hands each segment off to asr.py and
speaker.py in a background thread so the upload returns immediately.

Ported from audio_proto/vad_server.py steps 1-3.
"""

import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
from fastapi import APIRouter, HTTPException, UploadFile
from silero_vad import get_speech_timestamps, load_silero_vad

from state import audio_jobs
import asr
import speaker

# ---- VAD tunables -----------------------------------------------------------
VAD_THRESHOLD = 0.5   # speech probability threshold
MIN_SILENCE_S = 0.1   # speech regions closer than this are merged
MIN_SPEECH_S = 0.3    # regions shorter than this are dropped (squelch clicks)
SPEECH_PAD_S = 0.1    # padding added to each side of a detected region
# -----------------------------------------------------------------------------

TARGET_SR = 16000

router = APIRouter()

_vad_model = None
# audio for in-flight jobs, keyed by job_id — kept out of state.audio_jobs so
# the job dict itself stays small/JSON-serializable
_job_audio: dict[str, np.ndarray] = {}


def decode_to_16k_mono(path: str) -> np.ndarray:
    """Decode any audio file to 16 kHz mono float32 PCM."""
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-v", "error", "-i", path,
            "-f", "f32le", "-ac", "1", "-ar", str(TARGET_SR), "-",
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise ValueError(f"ffmpeg failed: {proc.stderr.decode(errors='replace')[:500]}")
        return np.frombuffer(proc.stdout, dtype=np.float32).copy()
    # No ffmpeg on PATH: soundfile's bundled libsndfile decodes mp3/wav/flac/ogg
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = soxr.resample(audio, sr, TARGET_SR)
    return np.ascontiguousarray(audio, dtype=np.float32)


def vad_split(audio: np.ndarray) -> list[dict]:
    global _vad_model
    if _vad_model is None:
        _vad_model = load_silero_vad()
    regions = get_speech_timestamps(
        torch.from_numpy(audio),
        _vad_model,
        sampling_rate=TARGET_SR,
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=int(MIN_SPEECH_S * 1000),
        min_silence_duration_ms=int(MIN_SILENCE_S * 1000),
        speech_pad_ms=int(SPEECH_PAD_S * 1000),
    )
    segments = []
    for i, r in enumerate(regions, start=1):
        start_s = round(r["start"] / TARGET_SR, 2)
        end_s = round(r["end"] / TARGET_SR, 2)
        segments.append({
            "segment": i,
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": round(end_s - start_s, 2),
        })
    return segments


def transcribe_job(job_id: str) -> None:
    """Worker thread: fill in transcript, words, and ATC/PILOT speaker ID for
    each segment of a job."""
    job = audio_jobs[job_id]
    audio = _job_audio[job_id]
    try:
        job["status"] = "transcribing"
        for seg in job["segments"]:
            chunk = audio[int(seg["start_s"] * TARGET_SR):int(seg["end_s"] * TARGET_SR)]

            asr_result = asr.transcribe_segment(chunk, seg["start_s"])
            seg["transcript"] = asr_result["transcript"]
            seg["words"] = asr_result["words"]

            speaker_result = speaker.identify_speaker(chunk, seg["transcript"])
            seg["speaker"] = speaker_result["speaker"]
            seg["similarity"] = speaker_result["similarity"]

            job["done"] += 1
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        _job_audio.pop(job_id, None)


@router.post("/upload-audio")
async def upload_audio(file: UploadFile):
    suffix = Path(file.filename or "upload.mp3").suffix or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        try:
            audio = decode_to_16k_mono(tmp.name)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"could not decode audio: {e}")
    if audio.size == 0:
        raise HTTPException(status_code=400, detail="decoded audio is empty")

    segments = vad_split(audio)
    for seg in segments:
        seg["transcript"] = None  # filled in by the job as it progresses
        seg["words"] = None
        seg["speaker"] = None
        seg["similarity"] = None

    job_id = uuid.uuid4().hex[:12]
    audio_jobs[job_id] = {
        "status": "loading_model",
        "done": 0,
        "total": len(segments),
        "error": None,
        "segments": segments,
    }
    _job_audio[job_id] = audio
    threading.Thread(target=transcribe_job, args=(job_id,), daemon=True).start()

    return {
        "job_id": job_id,
        "duration_s": round(audio.size / TARGET_SR, 2),
        "tunables": {
            "vad_threshold": VAD_THRESHOLD,
            "min_silence_s": MIN_SILENCE_S,
            "min_speech_s": MIN_SPEECH_S,
            "speech_pad_s": SPEECH_PAD_S,
            "atc_sim_threshold": speaker.ATC_SIM_THRESHOLD,
        },
        "segments": segments,
    }


@router.get("/audio-job/{job_id}")
def audio_job(job_id: str):
    job = audio_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return {
        "status": job["status"],
        "progress": {"done": job["done"], "total": job["total"]},
        "error": job["error"],
        "segments": job["segments"],
    }
