#!/usr/bin/env python3
"""Small desktop editor for cutting mono or stereo MP3 audio."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np


WAVEFORM_RATE = 800


@dataclass(frozen=True)
class Cut:
    channel: int
    start: float
    end: float


def probe_audio(path: Path) -> tuple[float, int]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=channels,duration:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise ValueError("This file has no audio stream.")
    channels = int(streams[0].get("channels", 0))
    duration = float(streams[0].get("duration") or data.get("format", {}).get("duration") or 0)
    if channels not in (1, 2):
        raise ValueError(f"Only mono or stereo audio is supported; this file has {channels} channel(s).")
    if duration <= 0:
        raise ValueError("Could not determine the audio duration.")
    return duration, channels


def load_waveform(path: Path, channels: int) -> np.ndarray:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn",
         "-ac", str(channels), "-ar", str(WAVEFORM_RATE), "-f", "f32le", "pipe:1"],
        capture_output=True, check=True,
    )
    samples = np.frombuffer(result.stdout, dtype="<f4")
    return samples[: len(samples) // channels * channels].reshape(-1, channels).copy()


def export_cut(source: Path, destination: Path, cut: Cut) -> None:
    # Decode before cutting so cuts land at the requested time rather than MP3 frame/keyframe boundaries.
    pan = f"pan=mono|c0=c{cut.channel}"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
         "-map", "0:a:0", "-vn", "-ss", f"{cut.start:.6f}",
         "-t", f"{cut.end - cut.start:.6f}", "-af", pan,
         "-codec:a", "libmp3lame", "-q:a", "2", str(destination)],
        capture_output=True, text=True, check=True,
    )


def format_time(seconds: float) -> str:
    minutes, secs = divmod(max(0, seconds), 60)
    return f"{int(minutes):02d}:{secs:06.3f}"


class ChannelView(ttk.LabelFrame):
    def __init__(self, master: tk.Misc, app: StereoCutter, channel: int):
        super().__init__(master, text=("Left" if channel == 0 else "Right") + " channel", padding=8)
        self.app = app
        self.channel = channel
        self.selection: tuple[float, float] | None = None
        self.drag_start: float | None = None
        self.playhead_position: float | None = None
        self.canvas = tk.Canvas(self, height=150, bg="#172033", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="x", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.begin_selection)
        self.canvas.bind("<B1-Motion>", self.move_selection)
        self.canvas.bind("<ButtonRelease-1>", self.end_selection)
        self.canvas.bind("<Configure>", lambda _event: self.draw())

        controls = ttk.Frame(self)
        controls.pack(fill="x", pady=(7, 0))
        self.start_var = tk.StringVar(value="0.000")
        self.end_var = tk.StringVar(value="0.000")
        ttk.Label(controls, text="Start (s)").pack(side="left")
        ttk.Entry(controls, textvariable=self.start_var, width=10).pack(side="left", padx=(4, 12))
        ttk.Label(controls, text="End (s)").pack(side="left")
        ttk.Entry(controls, textvariable=self.end_var, width=10).pack(side="left", padx=(4, 12))
        ttk.Button(controls, text="Select range", command=self.select_from_fields).pack(side="left", padx=3)
        ttk.Button(controls, text="Add cut", command=self.add_cut).pack(side="left", padx=3)
        playback_controls = ttk.Frame(self)
        playback_controls.pack(fill="x", pady=(5, 0))
        ttk.Button(playback_controls, text="Play", command=lambda: self.app.play(self.channel)).pack(side="left", padx=3)
        ttk.Button(playback_controls, text="Stop", command=lambda: self.app.stop_channel(self.channel)).pack(side="left", padx=3)
        ttk.Button(playback_controls, text="−5s", command=lambda: self.app.seek_channel(self.channel, -5)).pack(side="left", padx=3)
        ttk.Button(playback_controls, text="+5s", command=lambda: self.app.seek_channel(self.channel, 5)).pack(side="left", padx=3)
        ttk.Button(playback_controls, text="Play range", command=self.play_range).pack(side="left", padx=3)
        ttk.Button(playback_controls, text="Full channel", command=self.select_all).pack(side="left", padx=3)
        self.position_var = tk.StringVar(value="Now: --:--.---")
        ttk.Label(playback_controls, textvariable=self.position_var).pack(side="left", padx=15)

    def seconds_at(self, x: float) -> float:
        width = max(1, self.canvas.winfo_width() - 1)
        return max(0.0, min(self.app.duration, x / width * self.app.duration))

    def begin_selection(self, event: tk.Event) -> None:
        if not self.app.source:
            return
        self.drag_start = self.seconds_at(event.x)
        self.selection = (self.drag_start, self.drag_start)
        self.draw()

    def move_selection(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        end = self.seconds_at(event.x)
        self.selection = tuple(sorted((self.drag_start, end)))
        self.draw()

    def end_selection(self, event: tk.Event) -> None:
        self.move_selection(event)
        self.drag_start = None
        if self.selection:
            self.start_var.set(f"{self.selection[0]:.3f}")
            self.end_var.set(f"{self.selection[1]:.3f}")

    def get_range(self) -> tuple[float, float] | None:
        if not self.app.source:
            return None
        try:
            start, end = float(self.start_var.get()), float(self.end_var.get())
        except ValueError:
            messagebox.showerror("Invalid time", "Enter start and end times in seconds.")
            return None
        if not (0 <= start < end <= self.app.duration + 0.001):
            messagebox.showerror("Invalid range", f"Use 0 ≤ start < end ≤ {self.app.duration:.3f} seconds.")
            return None
        return start, min(end, self.app.duration)

    def select_from_fields(self) -> None:
        self.selection = self.get_range()
        self.draw()

    def select_all(self) -> None:
        if self.app.source:
            self.selection = (0.0, self.app.duration)
            self.start_var.set("0.000")
            self.end_var.set(f"{self.app.duration:.3f}")
            self.draw()

    def play_range(self) -> None:
        selected = self.get_range()
        if selected:
            self.app.play(self.channel, selected)

    def add_cut(self) -> None:
        selected = self.get_range()
        if selected:
            self.selection = selected
            self.draw()
            self.app.add_cut(Cut(self.channel, *selected))

    def draw(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 2 or height < 2:
            return
        mid = height / 2
        canvas.create_line(0, mid, width, mid, fill="#41516a")
        if self.app.waveform is None:
            canvas.create_text(width / 2, mid, text="Load an MP3 to see the waveform", fill="#cbd5e1")
            return
        samples = self.app.waveform[:, self.channel]
        if len(samples):
            # Min/max per pixel preserves visible spikes even for long recordings.
            edges = np.linspace(0, len(samples), width + 1, dtype=int)
            for x in range(width):
                chunk = samples[edges[x]:edges[x + 1]]
                if len(chunk):
                    lo, hi = float(chunk.min()), float(chunk.max())
                    canvas.create_line(x, mid - hi * (mid - 12), x, mid - lo * (mid - 12), fill="#5dd6ca")
        if self.selection and self.app.duration:
            x1 = self.selection[0] / self.app.duration * width
            x2 = self.selection[1] / self.app.duration * width
            canvas.create_rectangle(x1, 0, x2, height, outline="#ffd166", fill="#ffd166", stipple="gray25")
        canvas.create_text(5, height - 5, anchor="sw", text="0:00", fill="#cbd5e1")
        canvas.create_text(width - 5, height - 5, anchor="se", text=format_time(self.app.duration), fill="#cbd5e1")
        self.update_playhead(self.playhead_position)

    def update_playhead(self, position: float | None) -> None:
        self.playhead_position = position
        self.canvas.delete("playhead")
        if position is None or not self.app.duration:
            return
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        x = max(0, min(width - 1, position / self.app.duration * width))
        self.canvas.create_line(x, 0, x, height, fill="#ff6b6b", width=2, tags="playhead")


class StereoCutter(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("MP3 Channel Cutter")
        self.geometry("1050x680")
        self.minsize(760, 570)
        self.source: Path | None = None
        self.duration = 0.0
        self.channel_count = 2
        self.waveform: np.ndarray | None = None
        self.cuts: list[Cut] = []
        self.player: subprocess.Popen | None = None
        self.decoder: subprocess.Popen | None = None
        self.playback_channel: int | None = None
        self.playback_start = 0.0
        self.playback_end = 0.0
        self.playback_started_at = 0.0
        self.playback_timer: str | None = None
        self.busy = False

        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)
        top = ttk.Frame(root)
        top.pack(fill="x")
        ttk.Button(top, text="Open MP3…", command=self.open_file).pack(side="left")
        self.file_label = ttk.Label(top, text="No file loaded")
        self.file_label.pack(side="left", padx=12)
        self.views = [ChannelView(root, self, i) for i in range(2)]
        for view in self.views:
            view.pack(fill="x", pady=(12, 0))
        self.instructions = ttk.Label(root, text="Drag on a waveform or enter exact times. Add as many cuts as needed.")
        self.instructions.pack(anchor="w", pady=(12, 4))

        columns = ("channel", "start", "end", "length")
        self.table = ttk.Treeview(root, columns=columns, show="headings", height=6, selectmode="extended")
        for name, label, width in zip(columns, ("Channel", "Start", "End", "Length"), (110, 140, 140, 140)):
            self.table.heading(name, text=label)
            self.table.column(name, width=width, anchor="center")
        self.table.pack(fill="both", expand=True)
        bottom = ttk.Frame(root)
        bottom.pack(fill="x", pady=(8, 0))
        ttk.Button(bottom, text="Play selected cut", command=self.play_selected).pack(side="left")
        ttk.Button(bottom, text="Remove selected", command=self.remove_selected).pack(side="left", padx=8)
        ttk.Button(bottom, text="Export all cuts as MP3…", command=self.export_all).pack(side="right")
        self.status = tk.StringVar(value="Ready")
        ttk.Label(root, textvariable=self.status).pack(anchor="w", pady=(8, 0))
        self.protocol("WM_DELETE_WINDOW", self.close)

    def run_worker(self, job, done) -> None:
        self.busy = True

        def work():
            try:
                result = job()
            except Exception as exc:
                self.after(0, lambda error=exc: self.worker_error(error))
            else:
                self.after(0, lambda: self.worker_done(done, result))

        threading.Thread(target=work, daemon=True).start()

    def worker_error(self, error: Exception) -> None:
        self.busy = False
        detail = error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) and isinstance(error.stderr, str) else str(error)
        self.status.set("Operation failed")
        messagebox.showerror("Audio error", detail or "The audio command failed.")

    def worker_done(self, callback, result) -> None:
        self.busy = False
        callback(result)

    def open_file(self) -> None:
        if self.busy:
            return
        name = filedialog.askopenfilename(filetypes=[("MP3 audio", "*.mp3"), ("All files", "*")])
        if not name:
            return
        path = Path(name)
        self.stop_playback()
        self.status.set("Reading audio…")

        def load():
            duration, channels = probe_audio(path)
            return path, duration, channels, load_waveform(path, channels)

        def done(result):
            self.source, self.duration, self.channel_count, self.waveform = result
            self.cuts.clear()
            self.refresh_cuts()
            layout = "mono" if self.channel_count == 1 else "stereo"
            self.file_label.config(text=f"{self.source.name}  •  {layout}  •  {format_time(self.duration)}")
            self.views[0].configure(text="Mono channel" if self.channel_count == 1 else "Left channel")
            if self.channel_count == 1:
                self.views[1].pack_forget()
            else:
                self.views[1].pack(fill="x", pady=(12, 0), before=self.instructions)
            for view in self.views:
                view.selection = None
                view.start_var.set("0.000")
                view.end_var.set(f"{self.duration:.3f}")
                view.position_var.set("Now: --:--.---")
                view.update_playhead(None)
                if view.channel < self.channel_count:
                    view.draw()
            self.status.set("Loaded. Select a range on either channel and add a cut.")

        self.run_worker(load, done)

    def add_cut(self, cut: Cut) -> None:
        self.cuts.append(cut)
        self.refresh_cuts()
        self.status.set(f"{len(self.cuts)} cut(s) queued")

    def channel_name(self, channel: int) -> str:
        return "Mono" if self.channel_count == 1 else ("Left" if channel == 0 else "Right")

    def refresh_cuts(self) -> None:
        self.table.delete(*self.table.get_children())
        for i, cut in enumerate(self.cuts):
            self.table.insert("", "end", iid=str(i), values=(self.channel_name(cut.channel),
                              format_time(cut.start), format_time(cut.end), format_time(cut.end - cut.start)))

    def remove_selected(self) -> None:
        for i in sorted((int(item) for item in self.table.selection()), reverse=True):
            del self.cuts[i]
        self.refresh_cuts()

    def play_selected(self) -> None:
        selected = self.table.selection()
        if selected:
            cut = self.cuts[int(selected[0])]
            self.play(cut.channel, (cut.start, cut.end))

    def stop_playback(self) -> None:
        if self.playback_timer is not None:
            self.after_cancel(self.playback_timer)
            self.playback_timer = None
        if self.player and self.player.poll() is None:
            self.player.terminate()
        if self.decoder and self.decoder.poll() is None:
            self.decoder.terminate()
        self.player = None
        self.decoder = None
        if self.playback_channel is not None:
            view = self.views[self.playback_channel]
            position = min(self.playback_end, self.playback_start + time.monotonic() - self.playback_started_at)
            view.position_var.set(f"Stopped: {format_time(position)}")
            view.update_playhead(position)
        self.playback_channel = None

    def stop_channel(self, channel: int) -> None:
        if self.playback_channel == channel:
            self.stop_playback()
            self.status.set(f"Stopped {self.channel_name(channel).lower()} channel")

    def seek_channel(self, channel: int, seconds: float) -> None:
        if not self.source:
            return
        view = self.views[channel]
        if self.playback_channel == channel:
            current = min(self.playback_end, self.playback_start + time.monotonic() - self.playback_started_at)
        else:
            current = view.playhead_position or 0.0
        target = max(0.0, min(self.duration, current + seconds))
        if self.playback_channel == channel and target < self.duration:
            self.play(channel, (target, self.duration))
        else:
            if self.playback_channel == channel:
                self.stop_playback()
            view.position_var.set(f"At: {format_time(target)}")
            view.update_playhead(target)

    def update_playback(self) -> None:
        self.playback_timer = None
        if self.player is None or self.playback_channel is None:
            return
        position = min(self.playback_end, self.playback_start + time.monotonic() - self.playback_started_at)
        view = self.views[self.playback_channel]
        view.position_var.set(f"Now: {format_time(position)}")
        view.update_playhead(position)
        if self.player.poll() is not None:
            channel = self.playback_channel
            self.stop_playback()
            self.views[channel].position_var.set(f"Finished: {format_time(position)}")
            self.status.set("Playback finished")
            return
        self.playback_timer = self.after(100, self.update_playback)

    def play(self, channel: int, selected: tuple[float, float] | None = None) -> None:
        if not self.source:
            return
        if selected is None:
            start = self.views[channel].playhead_position or 0.0
            selected = (0.0 if start >= self.duration else start, self.duration)
        player = shutil.which("aplay")
        if not player:
            messagebox.showerror("Playback unavailable", "Install ALSA's aplay command to preview audio.")
            return
        self.stop_playback()
        start, end = selected
        decoder = ["ffmpeg", "-v", "error", "-i", str(self.source), "-ss", f"{start:.6f}",
                   "-t", f"{end - start:.6f}", "-af", f"pan=mono|c0=c{channel}",
                   "-f", "wav", "-acodec", "pcm_s16le", "pipe:1"]
        try:
            self.decoder = subprocess.Popen(decoder, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self.player = subprocess.Popen([player, "-q", "-"], stdin=self.decoder.stdout,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if self.decoder.stdout:
                self.decoder.stdout.close()
        except OSError as exc:
            self.stop_playback()
            messagebox.showerror("Playback error", str(exc))
            return
        self.playback_channel = channel
        self.playback_start, self.playback_end = start, end
        self.playback_started_at = time.monotonic()
        self.views[channel].position_var.set(f"Now: {format_time(start)}")
        self.views[channel].update_playhead(start)
        self.playback_timer = self.after(100, self.update_playback)
        self.status.set(f"Playing {self.channel_name(channel).lower()} {format_time(start)}–{format_time(end)}")

    def export_all(self) -> None:
        if self.busy or not self.source or not self.cuts:
            if not self.cuts:
                messagebox.showinfo("No cuts", "Add at least one cut first.")
            return
        folder = filedialog.askdirectory(title="Choose output folder")
        if not folder:
            return
        source = self.source
        cuts = self.cuts.copy()
        destination = Path(folder)
        outputs = [destination / f"{source.stem}_{self.channel_name(cut.channel).lower()}_{i:03d}.mp3"
                   for i, cut in enumerate(cuts, 1)]
        existing = [path.name for path in outputs if path.exists()]
        if existing and not messagebox.askyesno("Replace files?", f"{len(existing)} output file(s) already exist. Replace them?"):
            return
        self.status.set(f"Exporting {len(cuts)} cut(s)…")

        def job():
            for cut, path in zip(cuts, outputs):
                export_cut(source, path, cut)
            return outputs

        def done(paths):
            self.status.set(f"Exported {len(paths)} MP3 clip(s) to {destination}")
            messagebox.showinfo("Export complete", f"Saved {len(paths)} MP3 clip(s) in:\n{destination}")

        self.run_worker(job, done)

    def close(self) -> None:
        self.stop_playback()
        self.destroy()


if __name__ == "__main__":
    missing = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
    if missing:
        raise SystemExit("Install ffmpeg (including ffprobe) and put it on PATH.")
    StereoCutter().mainloop()
