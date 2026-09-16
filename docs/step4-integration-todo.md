# Step 4 — Integrate Audio Pipeline Into Main App: TODO

Breaks [Step 4 of the audio pipeline plan](audio-pipeline-plan.md#step-4--integrate-into-the-main-app)
into two checkpointable halves, backend first. Each box should be verifiable
on its own before moving to the next (same spirit as prototype steps 1-3).

Source to port from: `audio_proto/vad_server.py` (steps 1-3, already
checkpointed by ear/eye in the standalone prototype).

---

## 4a — Backend: move the pipeline into `backend/` ✅ done

- [x] `backend/audio.py`
  - [x] `decode_to_16k_mono()` — ffmpeg subprocess decode, ported as-is
  - [x] `vad_split()` — silero-VAD segmentation, ported as-is (constants stay
        module-level tunables: `VAD_THRESHOLD`, `MIN_SILENCE_S`, `MIN_SPEECH_S`,
        `SPEECH_PAD_S`)
  - [x] `POST /upload-audio` — decode + VAD, create job, spawn background thread
  - [x] `GET /audio-job/{job_id}` — poll status/progress/segments
  - [x] Job store: in-memory dict, same pattern as `aircraft_state` in `state.py`
        (`audio_jobs: dict[str, dict]` lives there; the raw decoded-audio
        buffer per in-flight job stays in a module-local dict in `audio.py`
        so the job dict itself stays small/JSON-serializable)
- [x] `backend/asr.py`
  - [x] `get_whisper()` — lazy-load faster-whisper, GPU-with-CPU-fallback logic
        ported as-is (including the cuBLAS/cuDNN preload workaround)
  - [x] `transcribe_segment()` — one segment in, `{transcript, words[]}` out;
        `AVIATION_PROMPT` stays a module constant
  - [x] `WHISPER_MODEL` env var, default `small.en`
- [x] `backend/speaker.py`
  - [x] `get_speaker_encoder()`, `get_atc_fingerprint()` — ported as-is
  - [x] `identify_speaker()` — embedding + cosine similarity vs. fingerprint,
        `ATC_SIM_THRESHOLD` module constant, `ATC_FINGERPRINT_PATH` env var
  - [x] Ambiguity-band fallback: calls `llm.classify_speaker()` when similarity
        falls within `±0.05` of the threshold
- [x] `backend/llm.py`
  - [x] Added `classify_speaker(transcript) -> "ATC" | "PILOT"`
- [x] `backend/server.py`
  - [x] Imports and wires `audio.py`'s router in via `app.include_router(...)`
  - [x] Background job worker (`audio.transcribe_job`) calls
        `asr.transcribe_segment` + `speaker.identify_speaker` per segment
- [x] `backend/requirements.txt` — added `silero-vad`, `faster-whisper`,
      `soundfile`, `soxr`, `resemblyzer`, `torch`, `torchaudio`,
      `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` (linux only); noted `ffmpeg`
      must be on PATH
- [x] Verified: installed into `backend/.venv` via `uv pip install` (the venv
      is uv-managed, no `pip` binary — use `uv pip install -p .venv/bin/python
      -r requirements.txt`), then curl-tested on a scratch port:
      ```
      curl -s -F "file=@audio/delta795_exit_runway.mp3" http://localhost:8000/upload-audio
      curl -s http://localhost:8000/audio-job/<job_id>
      ```
      Confirmed: transcript and ATC/PILOT speaker labels for
      `delta795_exit_runway.mp3` match what the prototype produced.

**Checkpoint reached 2026-08-04:** backend alone reproduces the prototype's
output for a known recording, no frontend involved yet.

---

## 4b — Frontend: player + auto-drive the map ✅ done

- [x] "Load MP3 Recording" button in the SCRIPT tab's controls bar in
      `src/App.jsx` (`accept=".mp3,audio/*"`, same visual style as the
      debug-only "Load CSV Transcript" button)
- [x] POST to `/upload-audio`, poll `/audio-job/{id}`, map returned segments
      into the existing `csvRows` shape (normalizes the prototype's `"PILOT"`
      to the existing `"Pilot"` casing via `normalizeSpeaker()`), switches to
      SCRIPT tab reusing existing row rendering
- [x] Object URL of the uploaded mp3 kept in `audioSrc` for playback
- [x] Mini player bar: play/pause, seek bar, elapsed/total time
      (`fmtClock()`)
- [x] Timed reveal: rows with `start_s <= currentTime` render normally,
      earlier rows render collapsed/dimmed; current row gets `isCurrent`
      highlight + auto-scroll (`rowRefs` + `scrollIntoView`); seeking
      backward re-hides — purely derived from `currentTime`, no extra state
      machine
- [x] Auto-drive: ATC segments call existing `sendCsvRow(row)`, PILOT
      segments call existing `sendPilotRow(row)`; guarded by a
      `sentSegmentsRef` set so seeking/re-renders don't double-send
- [x] AUTO / MANUAL toggle — MANUAL keeps the click-SEND flow (rows still
      editable-by-resend since SEND is always available once transcribed)
- [x] "Reset playback" — clears map segments + calls
      `DELETE /aircraft-state/{callsign}` for all tracked callsigns so a
      recording can replay cleanly

**Checkpoint reached 2026-08-05 (partial — see note):** Verified with a
headless-Chromium (Playwright) run driving the real dev servers: GeoJSON
load → SCRIPT tab → mp3 upload → 3 VAD segments appear → "3 transmissions
transcribed" → play → row 1 reveals in sync with playback → AUTO mode
dispatches `POST /parse` with the ATC row's transcript at the moment the
playhead reaches it (confirmed via network listener). `npm run build`
passes with no new console errors.
**Not verified:** the final "map lights up" leg, because this sandbox has
no outbound network route to huggingface.co (a bare `curl` to the HF API
hangs the same way `/parse` does) — `/parse` is pre-existing, LLM-dependent
code untouched by this change. Re-run the same manual flow with a real
`HF_TOKEN` to see Delta 795's route resolve on the map in sync with the
audio.

---

## Deferred (Step 5 — hardening, not part of this pass)

- Error surfaces (ffmpeg missing, first-run model download, bad mp3)
- Upload size cap, streaming partial results for long recordings
- Cache job results by file hash (pattern exists: `backend/cache/`)
- README: document `WHISPER_MODEL` / `ATC_FINGERPRINT_PATH` and the ffmpeg
  requirement in the main (non-prototype) README section
- Word-by-word karaoke reveal using `words[]` (data already collected, purely
  additive UI polish)
