# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastapi>=0.104.0",
#     "uvicorn>=0.24.0",
#     "python-multipart>=0.0.6",
#     "silero-vad>=5.1",
#     "faster-whisper>=1.0",
#     "soundfile>=0.12",
#     "soxr>=0.3",
#     "numpy>=1.24.0",
#     "nvidia-cublas-cu12",
#     "nvidia-cudnn-cu12>=9,<10",
#     "resemblyzer>=0.1.4",
#     "setuptools<81",
#     "torch",
#     "torchaudio",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchaudio = { index = "pytorch-cpu" }
# ///
"""Audio prototype, steps 1-3: upload an mp3, VAD-split it into transmissions,
listen to each cut, and watch per-row transcripts + ATC/PILOT speaker labels
fill in as faster-whisper and resemblyzer work through the segments in the
background. Run with:

    uv run audio_proto/vad_server.py

then open http://localhost:8100
"""

import os
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
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
from resemblyzer import VoiceEncoder
from silero_vad import get_speech_timestamps, load_silero_vad

# ---- VAD tunables -----------------------------------------------------------
VAD_THRESHOLD = 0.5   # speech probability threshold
MIN_SILENCE_S = 0.1   # speech regions closer than this are merged
MIN_SPEECH_S = 0.3    # regions shorter than this are dropped (squelch clicks)
SPEECH_PAD_S = 0.1    # padding added to each side of a detected region
# -----------------------------------------------------------------------------

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

# ---- Speaker-ID tunables -----------------------------------------------------
# similarity >= this -> "ATC", else "PILOT". Start ~0.9 per the design doc, but
# calibrate with the checkpoint slider in the page — short/noisy segments pull
# the true-ATC similarity down, so the right cut varies per recording.
ATC_SIM_THRESHOLD = 0.75
ATC_FINGERPRINT_PATH = os.environ.get(
    "ATC_FINGERPRINT_PATH", str(Path(__file__).resolve().parent.parent / "atc_fingerprint.npy")
)
# -----------------------------------------------------------------------------

TARGET_SR = 16000
PORT = 8100

app = FastAPI(title="ATC audio prototype — steps 1-3: VAD split + transcription + speaker ID")

_vad_model = None
_whisper_model = None
_whisper_lock = threading.Lock()
_speaker_encoder = None
_speaker_lock = threading.Lock()
_atc_fingerprint = None

# job_id -> {status, done, total, error, segments}; in-memory only, fine for a prototype
JOBS: dict[str, dict] = {}


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


def transcribe_job(job_id: str, audio: np.ndarray) -> None:
    """Worker thread: fill in transcript, words, and ATC/PILOT speaker ID for
    each segment of a job."""
    job = JOBS[job_id]
    try:
        whisper = get_whisper()  # may download the model on first run
        encoder = get_speaker_encoder()
        fingerprint = get_atc_fingerprint()
        job["status"] = "transcribing"
        for seg in job["segments"]:
            chunk = audio[int(seg["start_s"] * TARGET_SR):int(seg["end_s"] * TARGET_SR)]

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
                    words.append({"w": w.word.strip(), "t": round(float(seg["start_s"] + w.start), 2)})
            seg["transcript"] = " ".join(t for t in texts if t)
            seg["words"] = words

            embedding = encoder.embed_utterance(chunk)
            similarity = round(float(np.dot(embedding, fingerprint)), 3)
            seg["similarity"] = similarity
            seg["speaker"] = "ATC" if similarity >= ATC_SIM_THRESHOLD else "PILOT"

            job["done"] += 1
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


@app.post("/upload-audio")
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
    JOBS[job_id] = {
        "status": "loading_model",
        "done": 0,
        "total": len(segments),
        "error": None,
        "segments": segments,
    }
    threading.Thread(target=transcribe_job, args=(job_id, audio), daemon=True).start()

    return {
        "job_id": job_id,
        "duration_s": round(audio.size / TARGET_SR, 2),
        "tunables": {
            "vad_threshold": VAD_THRESHOLD,
            "min_silence_s": MIN_SILENCE_S,
            "min_speech_s": MIN_SPEECH_S,
            "speech_pad_s": SPEECH_PAD_S,
            "atc_sim_threshold": ATC_SIM_THRESHOLD,
        },
        "segments": segments,
    }


@app.get("/audio-job/{job_id}")
def audio_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return {
        "status": job["status"],
        "progress": {"done": job["done"], "total": job["total"]},
        "error": job["error"],
        "segments": job["segments"],
    }


PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>ATC audio — VAD + transcription + speaker ID prototype</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: system-ui, sans-serif; background: #101418; color: #dde3ea;
         max-width: 860px; margin: 2rem auto; padding: 0 1rem; }
  h1 { font-size: 1.2rem; font-weight: 600; }
  .muted { color: #8b95a1; font-size: 0.85rem; }
  #controls { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; margin: 1rem 0; }
  audio { width: 100%; margin: 0.5rem 0 1rem; }
  table { border-collapse: collapse; width: 100%; }
  th, td { padding: 0.45rem 0.7rem; text-align: left; border-bottom: 1px solid #232a32; }
  th { color: #8b95a1; font-weight: 500; font-size: 0.8rem; text-transform: uppercase; }
  td.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
  td.txt { width: 50%; }
  td.txt.pending { color: #566270; }
  td.spk { white-space: nowrap; }
  td.spk.pending { color: #566270; }
  .chip { padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: 600; }
  .chip.atc { background: #123354; color: #7ab8ff; }
  .chip.pilot { background: #4a2c12; color: #ffb37a; }
  .chip .sim { opacity: 0.65; font-weight: 400; }
  tr.playing { background: #1a2833; }
  button.play { background: #1f6feb; color: white; border: 0; border-radius: 5px;
                padding: 0.3rem 0.8rem; cursor: pointer; font-size: 0.9rem; }
  button.play:hover { background: #2b7bff; }
  #status { margin: 1rem 0; }
  #status.err { color: #ff7b72; }
</style>
</head>
<body>
<h1>VAD split + transcription + speaker ID prototype</h1>
<p class="muted">Upload an ATC recording &rarr; it is cut into transmissions,
transcribed, and each row gets an ATC/PILOT chip from cosine similarity
against <code>atc_fingerprint.npy</code>. Click &#9654; to check a row against
what you hear; drag the threshold slider until the chips match your ear &mdash;
it relabels instantly from the similarity scores already fetched, no
re-upload needed.</p>

<div id="controls">
  <input type="file" id="file" accept=".mp3,audio/*">
  <label class="muted" style="margin-left: auto; display: flex; align-items: center; gap: 0.5rem;">
    ATC threshold: <span id="threshVal">0.75</span>
    <input type="range" id="thresh" min="0" max="1" step="0.01" value="0.75">
  </label>
</div>
<audio id="player" controls></audio>
<div id="status" class="muted"></div>
<table id="tbl" hidden>
  <thead><tr><th></th><th>#</th><th>start &rarr; end</th><th>duration</th><th>speaker</th><th>transcript</th></tr></thead>
  <tbody id="rows"></tbody>
</table>

<script>
const player = document.getElementById('player');
const status = document.getElementById('status');
const tbl = document.getElementById('tbl');
const rows = document.getElementById('rows');
const threshSlider = document.getElementById('thresh');
const threshVal = document.getElementById('threshVal');

let segments = [];
let stopAt = null;      // pause when playback passes this time
let playingRow = null;  // index into segments, or null

function renderChip(td, seg) {
  if (seg.similarity === null || seg.similarity === undefined) return;
  const label = seg.similarity >= parseFloat(threshSlider.value) ? 'ATC' : 'PILOT';
  td.className = 'spk';
  td.innerHTML = '<span class="chip ' + label.toLowerCase() + '">' + label
    + ' <span class="sim">' + seg.similarity.toFixed(2) + '</span></span>';
}

threshSlider.addEventListener('input', () => {
  threshVal.textContent = threshSlider.value;
  segments.forEach((seg, i) => {
    const td = rows.children[i]?.querySelector('.spk');
    if (td) renderChip(td, seg);
  });
});

function fmt(t) {
  const m = Math.floor(t / 60), s = (t % 60).toFixed(2).padStart(5, '0');
  return m + ':' + s;
}

function setStatus(msg, err = false) {
  status.textContent = msg;
  status.className = err ? 'err' : 'muted';
}

function refreshHighlight() {
  [...rows.children].forEach((tr, i) => {
    tr.classList.toggle('playing', i === playingRow);
    tr.querySelector('button').textContent = i === playingRow ? '\\u23F8' : '\\u25B6';
  });
}

function playSegment(i) {
  if (playingRow === i && !player.paused) { player.pause(); return; }
  const seg = segments[i];
  stopAt = seg.end_s;
  playingRow = i;
  player.currentTime = seg.start_s;
  player.play();
  refreshHighlight();
}

// stop precisely at segment end (timeupdate alone is too coarse, ~250 ms)
(function tick() {
  if (stopAt !== null && !player.paused && player.currentTime >= stopAt) {
    player.pause();
  }
  requestAnimationFrame(tick);
})();

player.addEventListener('pause', () => {
  stopAt = null; playingRow = null; refreshHighlight();
});
// user grabbed the seek bar: cancel any pending segment stop
player.addEventListener('seeking', () => {
  if (playingRow !== null && Math.abs(player.currentTime - segments[playingRow].start_s) > 0.05) {
    stopAt = null; playingRow = null; refreshHighlight();
  }
});

let baseStatus = '';
let uploadGen = 0;  // bumped per upload so a stale poll loop stops itself

function fillTranscripts(segs) {
  segs.forEach((seg, i) => {
    const txtTd = rows.children[i]?.querySelector('.txt');
    if (!txtTd || seg.transcript === null || !txtTd.classList.contains('pending')) return;
    txtTd.textContent = seg.transcript || '(unintelligible)';
    txtTd.classList.remove('pending');
    segments[i] = seg;
    const spkTd = rows.children[i]?.querySelector('.spk');
    if (spkTd) renderChip(spkTd, seg);
  });
}

async function pollJob(jobId, gen) {
  while (gen === uploadGen) {
    await new Promise(r => setTimeout(r, 1000));
    let job;
    try {
      const res = await fetch('/audio-job/' + jobId);
      if (!res.ok) throw new Error('HTTP ' + res.status);
      job = await res.json();
    } catch (err) {
      setStatus('lost transcription job: ' + err.message, true);
      return;
    }
    if (gen !== uploadGen) return;
    fillTranscripts(job.segments);
    if (job.status === 'done') { setStatus(baseStatus); return; }
    if (job.status === 'error') { setStatus('transcription failed: ' + job.error, true); return; }
    setStatus(baseStatus + (job.status === 'loading_model'
      ? ' \\u2014 loading whisper model\\u2026'
      : ' \\u2014 transcribing ' + job.progress.done + '/' + job.progress.total + '\\u2026'));
  }
}

document.getElementById('file').addEventListener('change', async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const gen = ++uploadGen;
  player.src = URL.createObjectURL(f);
  tbl.hidden = true;
  rows.innerHTML = '';
  setStatus('analyzing\\u2026');
  const form = new FormData();
  form.append('file', f);
  try {
    const res = await fetch('/upload-audio', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    segments = data.segments;
    baseStatus = segments.length + ' transmissions in ' + fmt(data.duration_s)
      + '  (threshold ' + data.tunables.vad_threshold
      + ', min silence ' + data.tunables.min_silence_s + 's'
      + ', min speech ' + data.tunables.min_speech_s + 's)';
    setStatus(baseStatus);
    threshSlider.value = data.tunables.atc_sim_threshold;
    threshVal.textContent = data.tunables.atc_sim_threshold;
    for (const [i, seg] of segments.entries()) {
      const tr = document.createElement('tr');
      tr.innerHTML = '<td><button class="play">\\u25B6</button></td>'
        + '<td class="num">' + seg.segment + '</td>'
        + '<td class="num">' + fmt(seg.start_s) + ' \\u2192 ' + fmt(seg.end_s) + '</td>'
        + '<td class="num">' + seg.duration_s.toFixed(2) + 's</td>'
        + '<td class="spk pending">\\u2026</td>'
        + '<td class="txt pending">\\u2026</td>';
      tr.querySelector('button').addEventListener('click', () => playSegment(i));
      rows.appendChild(tr);
    }
    tbl.hidden = false;
    pollJob(data.job_id, gen);
  } catch (err) {
    setStatus('upload failed: ' + err.message, true);
  }
});
</script>
</body>
</html>"""


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(PAGE)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
