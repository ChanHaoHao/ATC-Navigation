# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastapi>=0.104.0",
#     "uvicorn>=0.24.0",
#     "python-multipart>=0.0.6",
#     "silero-vad>=5.1",
#     "soundfile>=0.12",
#     "soxr>=0.3",
#     "numpy>=1.24.0",
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
"""Step-1 audio prototype: upload an mp3, VAD-split it into transmissions,
listen to each cut. Run with:

    uv run audio_proto/vad_server.py

then open http://localhost:8100
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from silero_vad import get_speech_timestamps, load_silero_vad

# ---- VAD tunables -----------------------------------------------------------
VAD_THRESHOLD = 0.5   # speech probability threshold
MIN_SILENCE_S = 0.1   # speech regions closer than this are merged
MIN_SPEECH_S = 0.3    # regions shorter than this are dropped (squelch clicks)
SPEECH_PAD_S = 0.1    # padding added to each side of a detected region
# -----------------------------------------------------------------------------

TARGET_SR = 16000
PORT = 8100

app = FastAPI(title="ATC audio prototype — step 1: VAD split")

_vad_model = None


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
    return {
        "duration_s": round(audio.size / TARGET_SR, 2),
        "tunables": {
            "vad_threshold": VAD_THRESHOLD,
            "min_silence_s": MIN_SILENCE_S,
            "min_speech_s": MIN_SPEECH_S,
            "speech_pad_s": SPEECH_PAD_S,
        },
        "segments": vad_split(audio),
    }


PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>ATC audio — VAD split prototype</title>
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
  td.num { font-variant-numeric: tabular-nums; }
  tr.playing { background: #1a2833; }
  button.play { background: #1f6feb; color: white; border: 0; border-radius: 5px;
                padding: 0.3rem 0.8rem; cursor: pointer; font-size: 0.9rem; }
  button.play:hover { background: #2b7bff; }
  #status { margin: 1rem 0; }
  #status.err { color: #ff7b72; }
</style>
</head>
<body>
<h1>VAD split prototype</h1>
<p class="muted">Upload an ATC recording &rarr; it is cut into transmissions.
Click &#9654; on a row to hear exactly what the VAD cut. Tune the constants at
the top of <code>vad_server.py</code> if cuts merge or split transmissions.</p>

<div id="controls">
  <input type="file" id="file" accept=".mp3,audio/*">
</div>
<audio id="player" controls></audio>
<div id="status" class="muted"></div>
<table id="tbl" hidden>
  <thead><tr><th></th><th>#</th><th>start &rarr; end</th><th>duration</th></tr></thead>
  <tbody id="rows"></tbody>
</table>

<script>
const player = document.getElementById('player');
const status = document.getElementById('status');
const tbl = document.getElementById('tbl');
const rows = document.getElementById('rows');

let segments = [];
let stopAt = null;      // pause when playback passes this time
let playingRow = null;  // index into segments, or null

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

document.getElementById('file').addEventListener('change', async (e) => {
  const f = e.target.files[0];
  if (!f) return;
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
    setStatus(segments.length + ' transmissions in ' + fmt(data.duration_s)
      + '  (threshold ' + data.tunables.vad_threshold
      + ', min silence ' + data.tunables.min_silence_s + 's'
      + ', min speech ' + data.tunables.min_speech_s + 's)');
    for (const [i, seg] of segments.entries()) {
      const tr = document.createElement('tr');
      tr.innerHTML = '<td><button class="play">\\u25B6</button></td>'
        + '<td class="num">' + seg.segment + '</td>'
        + '<td class="num">' + fmt(seg.start_s) + ' \\u2192 ' + fmt(seg.end_s) + '</td>'
        + '<td class="num">' + seg.duration_s.toFixed(2) + 's</td>';
      tr.querySelector('button').addEventListener('click', () => playSegment(i));
      rows.appendChild(tr);
    }
    tbl.hidden = false;
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
