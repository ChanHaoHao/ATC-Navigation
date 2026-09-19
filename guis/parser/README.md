# MP3 channel cutter

Run from the repository root:

```bash
python3 guis/parser/stereo_cutter.py
```

Requires Python 3 with `numpy` and Tkinter, plus `ffmpeg` and `ffprobe` on `PATH`. Audio preview uses `aplay` (ALSA); editing and export still work without it.

Open a mono or stereo MP3. Mono files show one waveform; stereo files show separate left and right waveforms. Each visible channel has its own **Play**, **Stop**, **−5s**, and **+5s** controls. Play starts at that channel's current position; Stop keeps the position so Play can resume there. The skip buttons move the position by five seconds and keep playing if the channel is already playing. The red line and timestamp show the current position. Only one channel plays at a time.

Drag across a waveform or enter start and end times in seconds, then choose **Add cut**. Repeat to queue multiple clips from either channel. **Play range** previews the chosen range; **Full channel** selects it for cutting. Select a row to play or remove it. **Export all cuts as MP3** saves each queued clip as a separate mono MP3 in the chosen folder; the source file is never changed.
