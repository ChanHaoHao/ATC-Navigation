# Audio Transcription Pipeline — Implementation Plan

Upload an ATC recording (mp3), press **play**, and watch the transcript appear
sentence-by-sentence in sync with the audio — with each transmission labeled
ATC or Pilot, and ATC taxi instructions automatically driving the existing map
highlighting pipeline.

---

## Development strategy: prototype first, integrate last

The pipeline is built as a **standalone prototype** — a single Python script
serving its own simple web page — completely separate from the existing
React frontend and `backend/server.py`. Each step adds one pipeline stage and
is verified by eye/ear in the prototype page before moving on. Only when the
whole pipeline works end-to-end does it get merged into the main app (final
step).

```
audio_proto/
└── vad_server.py     # one FastAPI script: serves the test page + all endpoints
```

Run with `python audio_proto/vad_server.py` → open `http://localhost:8100`.

---

## Core design decisions

| Decision | Choice | Why |
|---|---|---|
| Segmentation unit | **Transmission** (one push-to-talk burst ≈ one "sentence") | ATC radio is push-to-talk: transmissions are separated by squelch/silence, so a VAD finds them reliably. Word boundaries can't be detected acoustically. |
| When to transcribe | **On upload, not during playback** | Whisper on a whole recording takes seconds–minutes; doing it up front makes playback a pure frontend timing problem. The user experience is identical to live transcription. |
| Reveal granularity | **Per transmission** at its start time | Simple and robust. Word-level timestamps are still captured (Whisper gives them for free) so a word-by-word "karaoke" reveal can be layered on later without re-architecting. |
| Speaker attribution | **Per transmission**: cosine-sim against the existing `atc_fingerprint.npy`, LLM text classification as fallback | Each transmission has exactly one speaker, and a reference embedding of the controller's voice already exists — a threshold on similarity beats clustering when you have a known target. |
| Transport | **Plain HTTP** (upload → poll job status) | No WebSockets needed since transcription happens before playback. Keeps the FastAPI backend simple. |
| Row playback in prototype | Play `[start_s, end_s]` of the **original file** in the browser | No per-segment audio files to slice or serve — one `<audio>` element seeks to `start_s` and pauses at `end_s`. |

Resulting pipeline:

```
mp3 upload
   → decode to 16 kHz mono PCM               (ffmpeg)
   → VAD: split into transmissions           (silero-vad)
   → per transmission: transcribe            (faster-whisper, word timestamps on)
   → per transmission: speaker ID            (resemblyzer embedding · atc_fingerprint.npy → ATC/PILOT)
   → JSON: [{start, end, speaker, transcript, words[]}]
   → frontend: <audio> playback + timed transcript reveal
   → ATC transmissions → existing /parse → map lights up
   → pilot transmissions → existing /check-readback
```

Everything downstream of the transcript (parse → resolve → colored segments →
per-callsign state) already exists and is reused unchanged.

**Segment shape** (the data contract for every step, fields added as steps land):

```json
{
  "segment": 3,
  "start_s": 41.2,
  "end_s": 47.9,
  "duration_s": 6.7,
  "speaker": "ATC",
  "similarity": 0.93,
  "transcript": "Delta 795 heavy, exit via Zulu Alpha at Foxtrot, join Alpha.",
  "words": [{ "w": "Delta", "t": 41.3 }, { "w": "795", "t": 41.7 }]
}
```

This is a superset of the existing CSV row shape
(`segment, start_s, speaker, similarity, transcript`) — deliberate, so the
main frontend's SCRIPT tab can eventually consume either source through one
code path.

---

## Step 1 — Prototype: VAD split + per-row playback

**Goal:** upload an mp3, see it split into transmissions, listen to each one.

`audio_proto/vad_server.py` (FastAPI, port 8100):

1. `GET /` — serves the test page (single inline HTML string, no build step).
2. `POST /upload-audio` — accepts the mp3, decodes to 16 kHz mono
   (ffmpeg), runs silero-VAD, returns segments JSON:
   `[{segment, start_s, end_s, duration_s}]`.
   - Merge speech regions separated by < 0.4 s of silence; drop regions
     shorter than ~0.3 s (squelch clicks).
   - Tunables as constants at the top of the file: `MIN_SILENCE_S`,
     `MIN_SPEECH_S`, `VAD_THRESHOLD`.
3. Test page:
   - file input → POSTs the mp3, keeps a browser object URL of it
   - renders one **row per segment**: `#`, `start → end`, duration, and a
     **▶ play button** that seeks the hidden `<audio>` element to `start_s`
     and pauses at `end_s` — so each row plays exactly what the VAD cut
   - a master player for the whole file, for comparing against the cuts

New deps: `fastapi uvicorn silero-vad soundfile numpy` + `ffmpeg` on PATH.

**Checkpoint (user):** upload a real recording, click through the rows —
does each row contain exactly one transmission? Tune the VAD constants here
before anything else is built on top.

## Step 2 — Prototype: per-row transcription

**Goal:** each row shows its transcript.

1. Load faster-whisper lazily at first use; model via env var `WHISPER_MODEL`
   (start `small.en`, try `medium.en` if accuracy disappoints).
2. Transcribe each VAD segment with `word_timestamps=True`, `initial_prompt`
   seeded with aviation vocabulary (phonetic alphabet, "runway", "cleared",
   callsign airlines…). Offset word times by segment start so they're absolute.
3. Since Whisper is slow, make it a job: `POST /upload-audio` returns
   `{job_id}` immediately with VAD results; transcription fills in per
   segment in the background; `GET /audio-job/{job_id}` reports
   `{status, progress, segments}` and the page polls it — rows appear
   instantly, transcripts pop in one by one.

**Checkpoint (user):** row text matches what you hear when you click ▶.
Tune model size / initial prompt now.

## Step 3 — Prototype: per-row speaker ID

**Goal:** each row gets an ATC / PILOT chip and a similarity score.

A precomputed ATC voice fingerprint already exists at `atc_fingerprint.npy`
(repo root — 256-dim L2-normalized resemblyzer d-vector), so no clustering:

1. Load the fingerprint once; path via env var `ATC_FINGERPRINT_PATH`
   (default: repo root).
2. Embed each transmission with resemblyzer's `VoiceEncoder` (same model that
   produced the fingerprint, so embeddings are comparable).
3. Cosine similarity against the fingerprint — both vectors are unit-norm, so
   it's a dot product. Store in `similarity`.
4. Label: `similarity >= ATC_SIM_THRESHOLD` → `"ATC"`, else `"PILOT"`.
   Threshold as a tunable constant (start ~0.9, calibrate in the checkpoint).
5. Fallback / tiebreak for segments in an ambiguity band around the threshold
   (e.g. ±0.05): LLM text classification — ATC issues instructions, pilots
   read back ending with their callsign. (Can defer to integration step since
   it needs the HF client.)

**Checkpoint (user):** labels match your ear per row; pick the threshold that
separates the classes (existing labeled CSVs are ground truth — their
`similarity` column came from this same fingerprint approach).

## Step 4 — Integrate into the main app

Only after the prototype pipeline is trusted:

**Backend** — move the prototype stages into real modules, wired into
`backend/server.py`:

| File | Content |
|---|---|
| `backend/audio.py` | upload endpoint, ffmpeg decode, VAD segmentation, job management |
| `backend/asr.py` | faster-whisper transcription with word timestamps |
| `backend/speaker.py` | embedding vs `atc_fingerprint.npy` cosine-sim, LLM text fallback |
| `backend/llm.py` | add `classify_speaker()` |
| `backend/requirements.txt` | add the new deps |

**Frontend** (`src/App.jsx`, following existing patterns):

1. **Upload:** "Load MP3" button next to "Load CSV Transcript" (same style,
   `accept=".mp3,audio/*"`). POST → poll job → feed segments into `csvRows`
   (same state, same shape) and switch to the SCRIPT tab. Keep an object URL
   of the mp3 for playback.
2. **Player:** a fixed mini-bar: play/pause, seek bar, elapsed/total time.
3. **Timed reveal:** on `timeupdate`, segments with `start_s <= currentTime`
   are shown; the current one gets the existing `isCurrent` highlight and
   auto-scrolls. Seeking backward re-hides — purely derived from
   `currentTime`, no extra state machine.
4. **Drive the map:** when an ATC segment is revealed, auto-call the existing
   `sendCsvRow(row)` (guarded by a `sentSegments` set so seeking doesn't
   double-send); PILOT segments auto-fire `/check-readback`. An
   **AUTO / MANUAL toggle** keeps today's click-SEND flow, with editable row
   text in manual mode for imperfect transcripts.
5. "Reset playback" clears map segments + backend aircraft state
   (`DELETE /aircraft-state/{callsign}`) so a recording replays cleanly.
6. Optional polish: word-by-word karaoke bolding inside the current segment
   using the `words[]` timestamps we already collected.

**Checkpoint:** the JFK demo recording end-to-end — press play, watch the
Delta 795 route resolve on the map in sync with the audio.

## Step 5 — Hardening & polish

- Error surfaces: ffmpeg missing, model download on first run (show
  "downloading model" in job status), empty/undecodable mp3.
- Long recordings: cap upload size; stream partial transcripts into the job
  result so rows fill in before the job finishes.
- Cache job results keyed by file hash (pattern exists: `backend/cache/`).
- README: audio workflow section, document `WHISPER_MODEL` /
  `ATC_FINGERPRINT_PATH` env vars and the ffmpeg requirement.

---

## Risks / open questions

- **ASR accuracy on radio audio** is the big one — ATC speech is fast, clipped,
  and noisy. Mitigations already in the plan: domain `initial_prompt`, model
  size knob, manual mode with editable rows. If `medium.en` still struggles,
  try a fine-tuned ATC Whisper checkpoint from HF (e.g. ATCO2-trained models).
- **Speaker ID on noisy audio** may need the text fallback more than expected;
  the step-3 interface allows swapping strategies without touching callers.
  Also note the fingerprint is per-controller: a recording from a different
  facility (or a controller shift change mid-recording) won't match it — the
  ambiguity-band LLM fallback is the safety net there.
- **First-run model downloads** (Whisper ~0.5–1.5 GB, silero, resemblyzer) —
  document it; consider a `/warmup` endpoint or startup preload.
- **torch dependency weight** (silero + resemblyzer pull it in). Acceptable for
  a local tool; if it bothers you, `webrtcvad` (no torch) is a lighter but less
  robust VAD swap.
