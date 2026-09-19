# MP3 channel cutter

Run from the repository root:

```bash
python3 guis/parser/stereo_cutter.py
```

Requires Python 3 with `numpy` and Tkinter, plus `ffmpeg` and `ffprobe` on `PATH`. Audio preview uses `aplay` (ALSA); editing and export still work without it.

Open a mono or stereo MP3. Mono files show one waveform; stereo files show separate left and right waveforms. Each visible channel has its own **Play**, **Stop**, **−5s**, and **+5s** controls. Play starts at that channel's current position; Stop keeps the position so Play can resume there. The skip buttons move the position by five seconds and keep playing if the channel is already playing. The red line and timestamp show the current position. Only one channel plays at a time.

Drag across a waveform or enter start and end times in seconds, then choose **Add cut**. Repeat to queue multiple clips from either channel. **Play range** previews the chosen range; **Full channel** selects it for cutting. Select a row to play or remove it. **Export all cuts as MP3** saves each queued clip as a separate mono MP3 in the chosen folder; the source file is never changed.

## ATC transcription GUI

This GUI uses the existing backend VAD, faster-whisper, and resemblyzer pipeline. Run it with the backend environment so those dependencies are available:

```bash
atc-map/backend/.venv/bin/python guis/parser/transcription_gui.py
```

1. Open a mono or stereo MP3. For stereo audio, choose **Left** or **Right**.
   Use **Play**, **Stop**, **-5s**, and **+5s** to navigate the selected channel; the timestamp and red playhead track the current position.
2. Either choose **Load ATC fingerprint...** to reuse a saved `atc_fingerprint.npy`, or drag over a clear controller transmission and choose **Add as ATC sample**. Add at least two samples when building a new fingerprint. Green ranges are the samples used to build it.
3. Choose **Split and transcribe selected channel**. The pipeline works in a temporary folder while it splits the selected channel with Silero VAD, transcribes each transmission with faster-whisper, and labels it ATC or PILOT by voice similarity to the marked samples.
4. After reviewing and editing the results, choose **Export results** and select the destination. The exported folder contains `atc_fingerprint.npy`, `transcription.csv`, and a `segments/` folder with every split transmission as a mono MP3. The CSV includes timestamps, speaker labels, similarity scores, transcripts, word timestamps, and the relative path to each MP3.
5. Select one result to edit its speaker or transcript. The speaker field accepts ATC, PILOT, UNKNOWN, or a custom name.
6. Selecting a result zooms the top waveform to that transmission. Click the zoomed waveform to place the red split marker, enter an absolute timestamp, or use the playback position, then choose **Split selected row**. Use **Show full audio** to return to the complete waveform.
7. Select two adjacent results and choose **Merge selected rows** to create one recording. Merge and split operations update both `transcription.csv` and the MP3 files in `segments/`.
8. Double-click a result to play it.

The first transcription can take several minutes if the Whisper model must be downloaded. The model is selected with the existing `WHISPER_MODEL` environment variable.
