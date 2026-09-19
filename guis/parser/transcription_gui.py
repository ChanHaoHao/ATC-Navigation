#!/usr/bin/env python3
"""GUI for channel-aware ATC transcription with recording-specific speaker ID."""

from __future__ import annotations

import csv
import json
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import uuid
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

from stereo_cutter import format_time, load_waveform, probe_audio


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "atc-map" / "backend"
TARGET_SR = 16000
FINGERPRINT_SIZE = 256


def decode_channel(path: Path, channel: int, channel_count: int) -> np.ndarray:
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn"]
    if channel_count == 2:
        command += ["-af", f"pan=mono|c0=c{channel}"]
    command += ["-ac", "1", "-ar", str(TARGET_SR), "-f", "f32le", "pipe:1"]
    result = subprocess.run(command, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def export_segment(
    source: Path,
    destination: Path,
    channel: int,
    channel_count: int,
    start: float,
    end: float,
) -> None:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
               "-map", "0:a:0", "-vn", "-ss", f"{start:.6f}", "-t", f"{end-start:.6f}"]
    if channel_count == 2:
        command += ["-af", f"pan=mono|c0=c{channel}"]
    command += ["-ac", "1", "-codec:a", "libmp3lame", "-q:a", "2", str(destination)]
    subprocess.run(command, capture_output=True, text=True, check=True)


def write_results_csv(path: Path, results: list[dict]) -> None:
    fields = ("segment", "start_s", "end_s", "duration_s", "speaker", "similarity",
              "transcript", "words_json", "audio_file")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "segment": result["segment"],
                "start_s": result["start_s"],
                "end_s": result["end_s"],
                "duration_s": result["duration_s"],
                "speaker": result["speaker"],
                "similarity": result["similarity"],
                "transcript": result["transcript"],
                "words_json": json.dumps(result["words"]),
                "audio_file": result["audio_file"],
            })


class TranscriptionGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.configure_fonts()
        self.title("ATC Audio Transcriber")
        self.geometry("1280x940")
        self.minsize(1000, 700)

        self.source: Path | None = None
        self.duration = 0.0
        self.channel_count = 0
        self.waveform: np.ndarray | None = None
        self.references: list[tuple[float, float]] = []
        self.loaded_fingerprint: np.ndarray | None = None
        self.selection: tuple[float, float] | None = None
        self.view_range: tuple[float, float] | None = None
        self.result_range: tuple[float, float] | None = None
        self.drag_start: float | None = None
        self.player: subprocess.Popen | None = None
        self.decoder: subprocess.Popen | None = None
        self.play_started_at = 0.0
        self.play_start = 0.0
        self.play_end = 0.0
        self.play_timer: str | None = None
        self.playhead: float | None = None
        self.busy = False

        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)
        top = ttk.Frame(root)
        top.pack(fill="x")
        ttk.Button(top, text="Open MP3", command=self.open_file).pack(side="left")
        self.file_label = ttk.Label(top, text="No file loaded")
        self.file_label.pack(side="left", padx=12)
        ttk.Label(top, text="Channel:").pack(side="left", padx=(20, 4))
        self.channel_var = tk.StringVar()
        self.channel_box = ttk.Combobox(top, textvariable=self.channel_var, state="readonly", width=12)
        self.channel_box.pack(side="left")
        self.channel_box.bind("<<ComboboxSelected>>", self.change_channel)

        self.waveform_view_var = tk.StringVar(value="Waveform: full recording")
        ttk.Label(root, textvariable=self.waveform_view_var).pack(anchor="w", pady=(10, 0))
        self.canvas = tk.Canvas(root, height=210, bg="#172033", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="x", pady=(4, 0))
        self.canvas.bind("<ButtonPress-1>", self.begin_selection)
        self.canvas.bind("<B1-Motion>", self.move_selection)
        self.canvas.bind("<ButtonRelease-1>", self.end_selection)
        self.canvas.bind("<Configure>", lambda _event: self.draw_waveform())

        select_bar = ttk.Frame(root)
        select_bar.pack(fill="x", pady=(7, 0))
        self.start_var = tk.StringVar(value="0.000")
        self.end_var = tk.StringVar(value="0.000")
        ttk.Label(select_bar, text="Start (s)").pack(side="left")
        ttk.Entry(select_bar, textvariable=self.start_var, width=10).pack(side="left", padx=(4, 12))
        ttk.Label(select_bar, text="End (s)").pack(side="left")
        ttk.Entry(select_bar, textvariable=self.end_var, width=10).pack(side="left", padx=(4, 12))
        ttk.Button(select_bar, text="Select range", command=self.select_from_fields).pack(side="left", padx=3)
        ttk.Button(select_bar, text="Add as ATC sample", command=self.add_reference).pack(side="left", padx=3)

        playback_bar = ttk.Frame(root)
        playback_bar.pack(fill="x", pady=(5, 0))
        ttk.Button(playback_bar, text="Play", command=self.play_channel).pack(side="left", padx=3)
        ttk.Button(playback_bar, text="Stop", command=self.stop_playback).pack(side="left", padx=3)
        ttk.Button(playback_bar, text="-5s", command=lambda: self.seek_channel(-5)).pack(side="left", padx=3)
        ttk.Button(playback_bar, text="+5s", command=lambda: self.seek_channel(5)).pack(side="left", padx=3)
        ttk.Button(playback_bar, text="Play selection", command=self.play_selection).pack(side="left", padx=3)
        ttk.Button(playback_bar, text="Show full audio", command=self.show_full_audio).pack(side="left", padx=3)
        self.position_var = tk.StringVar(value="Now: --:--.---")
        ttk.Label(playback_bar, textvariable=self.position_var).pack(side="left", padx=15)

        refs_frame = ttk.LabelFrame(root, text="Known ATC reference segments", padding=8)
        refs_frame.pack(fill="x", pady=(10, 0))
        self.refs_list = tk.Listbox(refs_frame, height=3, selectmode="extended")
        self.refs_list.pack(side="left", fill="x", expand=True)
        reference_buttons = ttk.Frame(refs_frame)
        reference_buttons.pack(side="left", padx=(8, 0))
        ttk.Button(reference_buttons, text="Remove selected", command=self.remove_references).pack(fill="x")
        ttk.Button(reference_buttons, text="Load ATC fingerprint...", command=self.load_fingerprint).pack(fill="x", pady=(4, 0))
        ttk.Button(reference_buttons, text="Clear fingerprint", command=self.clear_fingerprint).pack(fill="x", pady=(4, 0))
        self.fingerprint_var = tk.StringVar(value="Fingerprint: mark at least two ATC samples")
        ttk.Label(refs_frame, textvariable=self.fingerprint_var, wraplength=280).pack(side="left", padx=(10, 0))

        action_bar = ttk.Frame(root)
        action_bar.pack(fill="x", pady=(10, 5))
        self.transcribe_button = ttk.Button(action_bar, text="Split and transcribe selected channel", command=self.start_transcription)
        self.transcribe_button.pack(side="left")
        ttk.Button(action_bar, text="Export results", command=self.export_results).pack(side="right")
        self.status_var = tk.StringVar(value="Load an MP3, then mark ATC speech or load a saved fingerprint.")
        ttk.Label(action_bar, textvariable=self.status_var).pack(side="left", padx=12)

        columns = ("number", "time", "speaker", "similarity", "transcript")
        self.results = ttk.Treeview(root, columns=columns, show="headings", height=11)
        labels = ("#", "Start -> end", "Speaker", "Similarity", "Transcript")
        widths = (45, 145, 80, 85, 650)
        for column, label, width in zip(columns, labels, widths):
            self.results.heading(column, text=label)
            self.results.column(column, width=width, anchor="w" if column == "transcript" else "center")
        self.results.pack(fill="both", expand=True)
        self.results.bind("<Double-1>", self.play_result)
        self.results.bind("<<TreeviewSelect>>", self.load_result_editor)

        editor = ttk.LabelFrame(root, text="Edit selected transmission", padding=8)
        editor.pack(fill="x", pady=(8, 0))
        editor_top = ttk.Frame(editor)
        editor_top.pack(fill="x")
        ttk.Label(editor_top, text="Speaker").pack(side="left")
        self.speaker_var = tk.StringVar()
        self.speaker_box = ttk.Combobox(editor_top, textvariable=self.speaker_var,
                                        values=("ATC", "PILOT", "UNKNOWN"), width=12)
        self.speaker_box.pack(side="left", padx=(4, 12))
        ttk.Button(editor_top, text="Save speaker/transcript", command=self.save_result_edit).pack(side="left", padx=3)
        ttk.Button(editor_top, text="Merge selected rows", command=self.merge_results).pack(side="left", padx=3)
        ttk.Label(editor_top, text="Split at (s)").pack(side="left", padx=(15, 4))
        self.split_var = tk.StringVar()
        ttk.Entry(editor_top, textvariable=self.split_var, width=10).pack(side="left")
        ttk.Button(editor_top, text="Use playhead", command=self.use_playhead_for_split).pack(side="left", padx=3)
        ttk.Button(editor_top, text="Split selected row", command=self.split_result).pack(side="left", padx=3)
        editor_bottom = ttk.Frame(editor)
        editor_bottom.pack(fill="x", pady=(6, 0))
        ttk.Label(editor_bottom, text="Transcript").pack(side="left")
        self.transcript_var = tk.StringVar()
        ttk.Entry(editor_bottom, textvariable=self.transcript_var).pack(side="left", fill="x", expand=True, padx=(7, 0))

        self.result_data: list[dict] = []
        self.output_folder: Path | None = None
        self.working_directory: tempfile.TemporaryDirectory | None = None
        self.ui_events: queue.Queue[tuple[str, tuple]] = queue.Queue()
        self.after(50, self.process_ui_events)
        self.protocol("WM_DELETE_WINDOW", self.close)

    def configure_fonts(self) -> None:
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkIconFont"):
            tkfont.nametofont(name).configure(family="DejaVu Sans", size=12)
        tkfont.nametofont("TkHeadingFont").configure(family="DejaVu Sans", size=12, weight="bold")
        tkfont.nametofont("TkFixedFont").configure(family="DejaVu Sans Mono", size=12)
        style = ttk.Style(self)
        style.configure("Treeview", font=("DejaVu Sans", 12), rowheight=32)
        style.configure("Treeview.Heading", font=("DejaVu Sans", 12, "bold"))

    def post_ui_event(self, callback: str, *args) -> None:
        self.ui_events.put((callback, args))

    def process_ui_events(self) -> None:
        try:
            while True:
                callback, args = self.ui_events.get_nowait()
                getattr(self, callback)(*args)
        except queue.Empty:
            pass
        self.after(50, self.process_ui_events)

    def set_status(self, message: str) -> None:
        self.status_var.set(message)

    def current_channel(self) -> int:
        return 1 if self.channel_var.get() == "Right" else 0

    def open_file(self) -> None:
        if self.busy:
            return
        name = filedialog.askopenfilename(filetypes=[("MP3 audio", "*.mp3"), ("All files", "*")])
        if not name:
            return
        path = Path(name)
        self.status_var.set("Reading audio...")
        self.busy = True

        def worker() -> None:
            try:
                duration, channels = probe_audio(path)
                waveform = load_waveform(path, channels)
            except Exception as exc:
                self.post_ui_event("load_failed", exc)
                return
            self.post_ui_event("loaded", path, duration, channels, waveform)

        threading.Thread(target=worker, daemon=True).start()

    def load_failed(self, error: Exception) -> None:
        self.busy = False
        self.status_var.set("Could not load audio")
        messagebox.showerror("Audio error", str(error))

    def loaded(self, path: Path, duration: float, channels: int, waveform: np.ndarray) -> None:
        self.stop_playback()
        self.source, self.duration, self.channel_count, self.waveform = path, duration, channels, waveform
        self.channel_box["values"] = ("Mono",) if channels == 1 else ("Left", "Right")
        self.channel_var.set("Mono" if channels == 1 else "Left")
        self.file_label.config(text=f"{path.name}  |  {'mono' if channels == 1 else 'stereo'}  |  {format_time(duration)}")
        self.busy = False
        self.reset_channel_state()
        self.status_var.set("Mark two or more clear ATC segments, or load a saved ATC fingerprint.")

    def change_channel(self, _event=None) -> None:
        if self.source:
            self.stop_playback()
            self.reset_channel_state(keep_fingerprint=True)
            if self.loaded_fingerprint is None:
                self.status_var.set("Channel changed. Mark ATC segments or load a saved fingerprint.")
            else:
                self.status_var.set("Channel changed. The loaded fingerprint is still selected.")

    def reset_channel_state(self, keep_fingerprint: bool = False) -> None:
        self.cleanup_working_directory()
        self.references.clear()
        if not keep_fingerprint:
            self.loaded_fingerprint = None
            self.fingerprint_var.set("Fingerprint: mark at least two ATC samples")
        self.selection = None
        self.view_range = None
        self.result_range = None
        self.waveform_view_var.set("Waveform: full recording")
        self.playhead = None
        self.result_data.clear()
        self.output_folder = None
        self.results.delete(*self.results.get_children())
        self.refresh_references()
        self.start_var.set("0.000")
        self.end_var.set(f"{self.duration:.3f}")
        self.position_var.set("Now: --:--.---")
        self.draw_waveform()

    def cleanup_working_directory(self) -> None:
        if self.working_directory is not None:
            self.working_directory.cleanup()
            self.working_directory = None
        self.output_folder = None

    def seconds_at(self, x: float) -> float:
        view_start, view_end = self.waveform_bounds()
        position = view_start + x / max(1, self.canvas.winfo_width() - 1) * (view_end - view_start)
        return max(view_start, min(view_end, position))

    def waveform_bounds(self) -> tuple[float, float]:
        return self.view_range or (0.0, self.duration)

    def begin_selection(self, event: tk.Event) -> None:
        if not self.source:
            return
        selected_results = self.selected_result_indices()
        if len(selected_results) == 1 and self.result_range is not None:
            self.stop_playback()
            self.playhead = self.seconds_at(event.x)
            self.split_var.set(f"{self.playhead:.3f}")
            self.position_var.set(f"At: {format_time(self.playhead)}")
            self.draw_waveform()
            return
        self.drag_start = self.seconds_at(event.x)
        self.selection = (self.drag_start, self.drag_start)

    def move_selection(self, event: tk.Event) -> None:
        if self.drag_start is not None:
            self.selection = tuple(sorted((self.drag_start, self.seconds_at(event.x))))
            self.draw_waveform()

    def end_selection(self, event: tk.Event) -> None:
        self.move_selection(event)
        self.drag_start = None
        if self.selection:
            self.start_var.set(f"{self.selection[0]:.3f}")
            self.end_var.set(f"{self.selection[1]:.3f}")

    def show_full_audio(self) -> None:
        self.view_range = None
        self.result_range = None
        self.waveform_view_var.set("Waveform: full recording")
        self.draw_waveform()

    def get_range(self) -> tuple[float, float] | None:
        if not self.source:
            return None
        try:
            start, end = float(self.start_var.get()), float(self.end_var.get())
        except ValueError:
            messagebox.showerror("Invalid time", "Enter start and end times in seconds.")
            return None
        if not (0 <= start < end <= self.duration + .001):
            messagebox.showerror("Invalid range", f"Use 0 <= start < end <= {self.duration:.3f} seconds.")
            return None
        return start, min(end, self.duration)

    def select_from_fields(self) -> None:
        self.selection = self.get_range()
        self.draw_waveform()

    def add_reference(self) -> None:
        selected = self.get_range()
        if selected and selected not in self.references:
            self.loaded_fingerprint = None
            self.fingerprint_var.set("Fingerprint: using marked ATC samples")
            self.references.append(selected)
            self.references.sort()
            self.selection = None
            self.refresh_references()
            self.draw_waveform()

    def remove_references(self) -> None:
        for index in reversed(self.refs_list.curselection()):
            del self.references[index]
        self.refresh_references()
        self.draw_waveform()

    def refresh_references(self) -> None:
        self.refs_list.delete(0, "end")
        for i, (start, end) in enumerate(self.references, 1):
            self.refs_list.insert("end", f"ATC sample {i}: {format_time(start)} -> {format_time(end)}  ({end-start:.2f}s)")

    def load_fingerprint(self) -> None:
        name = filedialog.askopenfilename(
            title="Load ATC voice fingerprint",
            filetypes=[("NumPy fingerprint", "*.npy"), ("All files", "*")],
        )
        if not name:
            return
        try:
            fingerprint = np.asarray(np.load(name, allow_pickle=False), dtype=np.float32)
            if fingerprint.shape != (FINGERPRINT_SIZE,) or not np.all(np.isfinite(fingerprint)):
                raise ValueError(f"The fingerprint must be a numeric array with {FINGERPRINT_SIZE} values.")
            norm = float(np.linalg.norm(fingerprint))
            if norm == 0:
                raise ValueError("The fingerprint contains no usable voice data.")
            fingerprint = fingerprint / norm
        except Exception as exc:
            messagebox.showerror("Invalid fingerprint", str(exc))
            return
        self.loaded_fingerprint = fingerprint
        self.references.clear()
        self.selection = None
        self.refresh_references()
        self.draw_waveform()
        self.fingerprint_var.set(f"Fingerprint: {Path(name).name}")
        self.status_var.set("Saved ATC fingerprint loaded. You can start transcription.")

    def clear_fingerprint(self) -> None:
        self.loaded_fingerprint = None
        self.fingerprint_var.set("Fingerprint: mark at least two ATC samples")
        self.status_var.set("Loaded fingerprint cleared. Mark at least two ATC samples.")

    def draw_waveform(self) -> None:
        self.canvas.delete("all")
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width < 2 or height < 2:
            return
        mid = height / 2
        self.canvas.create_line(0, mid, width, mid, fill="#41516a")
        if self.waveform is None:
            self.canvas.create_text(width / 2, mid, text="Load an MP3", fill="#cbd5e1")
            return
        all_samples = self.waveform[:, self.current_channel()]
        view_start, view_end = self.waveform_bounds()
        sample_start = int(view_start / self.duration * len(all_samples))
        sample_end = max(sample_start + 1, int(view_end / self.duration * len(all_samples)))
        samples = all_samples[sample_start:sample_end]
        edges = np.linspace(0, len(samples), width + 1, dtype=int)
        for x in range(width):
            chunk = samples[edges[x]:edges[x + 1]]
            if len(chunk):
                self.canvas.create_line(x, mid - float(chunk.max()) * (mid - 12),
                                        x, mid - float(chunk.min()) * (mid - 12), fill="#5dd6ca")
        for start, end in self.references:
            if end <= view_start or start >= view_end:
                continue
            x1 = (max(start, view_start) - view_start) / (view_end - view_start) * width
            x2 = (min(end, view_end) - view_start) / (view_end - view_start) * width
            self.canvas.create_rectangle(x1, 0, x2, height, fill="#56d364", outline="#56d364", stipple="gray25")
        if self.selection:
            start, end = self.selection
            if end > view_start and start < view_end:
                x1 = (max(start, view_start) - view_start) / (view_end - view_start) * width
                x2 = (min(end, view_end) - view_start) / (view_end - view_start) * width
                self.canvas.create_rectangle(x1, 0, x2, height, fill="#ffd166", outline="#ffd166", stipple="gray25")
        if self.result_range:
            start, end = self.result_range
            x1 = (max(start, view_start) - view_start) / (view_end - view_start) * width
            x2 = (min(end, view_end) - view_start) / (view_end - view_start) * width
            self.canvas.create_rectangle(x1, 1, x2, height - 1, outline="#58a6ff", width=3)
        if self.playhead is not None and view_start <= self.playhead <= view_end:
            x = (self.playhead - view_start) / (view_end - view_start) * width
            self.canvas.create_line(x, 0, x, height, fill="#ff6b6b", width=2)
        self.canvas.create_text(5, height - 5, anchor="sw", text=format_time(view_start), fill="#cbd5e1")
        self.canvas.create_text(width - 5, height - 5, anchor="se", text=format_time(view_end), fill="#cbd5e1")

    def play_selection(self) -> None:
        selected = self.get_range()
        if selected:
            self.play(*selected)

    def play_channel(self) -> None:
        if not self.source:
            return
        start = self.playhead or 0.0
        if start >= self.duration:
            start = 0.0
        self.play(start, self.duration)

    def seek_channel(self, seconds: float) -> None:
        if not self.source:
            return
        playing = self.player is not None and self.player.poll() is None
        if playing:
            current = min(self.play_end, self.play_start + time.monotonic() - self.play_started_at)
        else:
            current = self.playhead or 0.0
        target = max(0.0, min(self.duration, current + seconds))
        if playing and target < self.duration:
            self.play(target, self.duration)
        else:
            if playing:
                self.stop_playback()
            self.playhead = target
            self.position_var.set(f"At: {format_time(target)}")
            self.draw_waveform()

    def play_result(self, _event=None) -> None:
        selected = self.results.selection()
        if selected:
            segment = self.result_data[int(selected[0])]
            self.play(segment["start_s"], segment["end_s"])

    def play(self, start: float, end: float) -> None:
        if not self.source:
            return
        player = shutil.which("aplay")
        if not player:
            messagebox.showerror("Playback unavailable", "Install ALSA's aplay command to preview audio.")
            return
        self.stop_playback()
        command = ["ffmpeg", "-v", "error", "-i", str(self.source), "-ss", f"{start:.6f}", "-t", f"{end-start:.6f}"]
        if self.channel_count == 2:
            command += ["-af", f"pan=mono|c0=c{self.current_channel()}"]
        command += ["-ac", "1", "-f", "wav", "-acodec", "pcm_s16le", "pipe:1"]
        try:
            self.decoder = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self.player = subprocess.Popen([player, "-q", "-"], stdin=self.decoder.stdout,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if self.decoder.stdout:
                self.decoder.stdout.close()
        except OSError as exc:
            self.stop_playback()
            messagebox.showerror("Playback error", str(exc))
            return
        self.play_start, self.play_end, self.play_started_at = start, end, time.monotonic()
        self.playhead = start
        self.position_var.set(f"Now: {format_time(start)}")
        self.update_playback()

    def update_playback(self) -> None:
        self.play_timer = None
        if self.player is None:
            return
        self.playhead = min(self.play_end, self.play_start + time.monotonic() - self.play_started_at)
        self.position_var.set(f"Now: {format_time(self.playhead)}")
        self.draw_waveform()
        if self.player.poll() is None:
            self.play_timer = self.after(100, self.update_playback)
        else:
            self.playhead = self.play_end
            self.position_var.set(f"Finished: {format_time(self.playhead)}")
            self.player = self.decoder = None

    def stop_playback(self) -> None:
        if self.play_timer:
            self.after_cancel(self.play_timer)
            self.play_timer = None
        was_playing = self.player is not None and self.player.poll() is None
        if was_playing:
            self.playhead = min(self.play_end, self.play_start + time.monotonic() - self.play_started_at)
        for process in (self.player, self.decoder):
            if process and process.poll() is None:
                process.terminate()
        self.player = self.decoder = None
        if was_playing:
            self.position_var.set(f"Stopped: {format_time(self.playhead or 0.0)}")
            self.draw_waveform()

    def start_transcription(self) -> None:
        if self.busy or not self.source:
            return
        if self.loaded_fingerprint is None and len(self.references) < 2:
            messagebox.showinfo("ATC reference needed", "Mark at least two clear ATC segments or load atc_fingerprint.npy.")
            return
        self.stop_playback()
        self.busy = True
        self.transcribe_button.state(["disabled"])
        self.channel_box.configure(state="disabled")
        self.result_data.clear()
        self.results.delete(*self.results.get_children())
        source, channel, count, references = self.source, self.current_channel(), self.channel_count, self.references.copy()
        loaded_fingerprint = None if self.loaded_fingerprint is None else self.loaded_fingerprint.copy()
        self.cleanup_working_directory()
        self.working_directory = tempfile.TemporaryDirectory(prefix="atc_transcription_")
        output_folder = Path(self.working_directory.name)
        segments_folder = output_folder / "segments"
        segments_folder.mkdir(parents=True)
        self.output_folder = output_folder
        self.status_var.set("Loading pipeline and building the ATC voice reference...")

        def worker() -> None:
            try:
                if str(BACKEND) not in sys.path:
                    sys.path.insert(0, str(BACKEND))
                import audio as pipeline_audio
                import asr
                import speaker

                samples = decode_channel(source, channel, count)
                if loaded_fingerprint is None:
                    chunks = [samples[int(start * TARGET_SR):int(end * TARGET_SR)] for start, end in references]
                    fingerprint = speaker.build_fingerprint(chunks)
                else:
                    fingerprint = loaded_fingerprint
                np.save(output_folder / "atc_fingerprint.npy", fingerprint)
                segments = pipeline_audio.vad_split(samples)
                self.post_ui_event("set_status", f"Found {len(segments)} transmissions. Transcribing...")
                completed = []
                for index, segment in enumerate(segments, 1):
                    chunk = samples[int(segment["start_s"] * TARGET_SR):int(segment["end_s"] * TARGET_SR)]
                    asr_result = asr.transcribe_segment(chunk, segment["start_s"])
                    identity = speaker.identify_speaker(chunk, asr_result["transcript"], fingerprint=fingerprint)
                    result = {**segment, **asr_result, **identity}
                    filename = f"segment_{segment['segment']:03d}.mp3"
                    relative_audio = Path("segments") / filename
                    export_segment(source, output_folder / relative_audio, channel, count,
                                   segment["start_s"], segment["end_s"])
                    result["audio_file"] = relative_audio.as_posix()
                    completed.append(result)
                    self.post_ui_event("add_result", result, index, len(segments))
                write_results_csv(output_folder / "transcription.csv", completed)
            except Exception as exc:
                self.post_ui_event("transcription_failed", exc)
                return
            self.post_ui_event("transcription_finished", completed)

        threading.Thread(target=worker, daemon=True).start()

    def add_result(self, result: dict, done: int, total: int) -> None:
        index = len(self.result_data)
        self.result_data.append(result)
        self.insert_result_row(index, result)
        self.status_var.set(f"Transcribing {done}/{total}...")

    def insert_result_row(self, index: int, result: dict) -> None:
        similarity = "" if result.get("similarity") is None else f"{result['similarity']:.3f}"
        self.results.insert("", "end", iid=str(index), values=(result["segment"],
            f"{format_time(result['start_s'])} -> {format_time(result['end_s'])}",
            result["speaker"], similarity, result["transcript"]))

    def refresh_result_rows(self, selected: int | None = None) -> None:
        self.results.delete(*self.results.get_children())
        for index, result in enumerate(self.result_data):
            result["segment"] = index + 1
            result["duration_s"] = round(result["end_s"] - result["start_s"], 2)
            self.insert_result_row(index, result)
        if selected is not None and 0 <= selected < len(self.result_data):
            self.results.selection_set(str(selected))
            self.results.focus(str(selected))
            self.results.see(str(selected))

    def selected_result_indices(self) -> list[int]:
        return sorted(int(item) for item in self.results.selection())

    def load_result_editor(self, _event=None) -> None:
        selected = self.selected_result_indices()
        if not selected:
            return
        first, last = self.result_data[selected[0]], self.result_data[selected[-1]]
        self.view_range = (first["start_s"], last["end_s"])
        self.result_range = self.view_range
        self.selection = None
        if len(selected) == 1:
            self.waveform_view_var.set(
                f"Waveform: transmission {selected[0] + 1} ({format_time(first['start_s'])} -> {format_time(first['end_s'])})"
            )
        else:
            self.waveform_view_var.set(
                f"Waveform: transmissions {selected[0] + 1}-{selected[-1] + 1}"
            )
        self.draw_waveform()
        if len(selected) != 1:
            return
        result = self.result_data[selected[0]]
        self.speaker_var.set(result["speaker"])
        self.transcript_var.set(result["transcript"])
        self.split_var.set(f"{(result['start_s'] + result['end_s']) / 2:.3f}")
        self.start_var.set(f"{result['start_s']:.3f}")
        self.end_var.set(f"{result['end_s']:.3f}")
        if self.playhead is None or not result["start_s"] <= self.playhead <= result["end_s"]:
            self.playhead = result["start_s"]
            self.position_var.set(f"At: {format_time(self.playhead)}")
            self.draw_waveform()

    def save_results_csv(self) -> None:
        if self.output_folder:
            write_results_csv(self.output_folder / "transcription.csv", self.result_data)

    def save_result_edit(self) -> None:
        selected = self.selected_result_indices()
        if self.busy or len(selected) != 1:
            messagebox.showinfo("Select one row", "Select exactly one transmission to edit.")
            return
        index = selected[0]
        result = self.result_data[index]
        transcript = self.transcript_var.get().strip()
        speaker = self.speaker_var.get().strip() or "UNKNOWN"
        if transcript != result["transcript"]:
            result["words"] = []
        result["transcript"] = transcript
        result["speaker"] = speaker
        self.refresh_result_rows(index)
        self.save_results_csv()
        self.status_var.set(f"Saved edits to transmission {index + 1} and transcription.csv")

    def new_edited_audio(self, start: float, end: float) -> str:
        if not self.source or not self.output_folder:
            raise ValueError("No transcription output folder is available.")
        relative = Path("segments") / f"edited_{uuid.uuid4().hex[:10]}.mp3"
        export_segment(self.source, self.output_folder / relative, self.current_channel(),
                       self.channel_count, start, end)
        return relative.as_posix()

    def remove_saved_audio(self, results: list[dict]) -> None:
        if not self.output_folder:
            return
        segments_folder = (self.output_folder / "segments").resolve()
        for result in results:
            path = (self.output_folder / result.get("audio_file", "")).resolve()
            if path.parent == segments_folder and path.exists():
                path.unlink()

    def merge_results(self) -> None:
        selected = self.selected_result_indices()
        if self.busy or len(selected) != 2 or selected[1] != selected[0] + 1:
            messagebox.showinfo("Select adjacent rows", "Select exactly two adjacent transmissions to merge.")
            return
        first_index, second_index = selected
        first, second = self.result_data[first_index], self.result_data[second_index]
        try:
            audio_file = self.new_edited_audio(first["start_s"], second["end_s"])
        except Exception as exc:
            messagebox.showerror("Merge failed", str(exc))
            return
        same_speaker = first["speaker"] == second["speaker"]
        if same_speaker and first.get("similarity") is not None and second.get("similarity") is not None:
            total = first["duration_s"] + second["duration_s"]
            similarity = round((first["similarity"] * first["duration_s"] +
                                second["similarity"] * second["duration_s"]) / total, 3)
        else:
            similarity = None
        merged = {
            "segment": first_index + 1,
            "start_s": first["start_s"],
            "end_s": second["end_s"],
            "duration_s": round(second["end_s"] - first["start_s"], 2),
            "speaker": first["speaker"] if same_speaker else "UNKNOWN",
            "similarity": similarity,
            "transcript": " ".join(text for text in (first["transcript"], second["transcript"]) if text),
            "words": sorted(first.get("words", []) + second.get("words", []), key=lambda word: word.get("t", 0)),
            "audio_file": audio_file,
        }
        self.result_data[first_index:second_index + 1] = [merged]
        self.remove_saved_audio([first, second])
        self.refresh_result_rows(first_index)
        self.save_results_csv()
        self.status_var.set(f"Merged transmissions {first_index + 1} and {second_index + 1}")

    def use_playhead_for_split(self) -> None:
        if self.playhead is not None:
            self.split_var.set(f"{self.playhead:.3f}")

    def split_result(self) -> None:
        selected = self.selected_result_indices()
        if self.busy or len(selected) != 1:
            messagebox.showinfo("Select one row", "Select exactly one transmission to split.")
            return
        index = selected[0]
        original = self.result_data[index]
        try:
            split_at = float(self.split_var.get())
        except ValueError:
            messagebox.showerror("Invalid split", "Enter an absolute timestamp in seconds.")
            return
        if not original["start_s"] < split_at < original["end_s"]:
            messagebox.showerror("Invalid split", "The split timestamp must be inside the selected transmission.")
            return
        created: list[str] = []
        try:
            created.append(self.new_edited_audio(original["start_s"], split_at))
            created.append(self.new_edited_audio(split_at, original["end_s"]))
        except Exception as exc:
            if self.output_folder:
                self.remove_saved_audio([{"audio_file": path} for path in created])
            messagebox.showerror("Split failed", str(exc))
            return
        words = original.get("words", [])
        left_words = [word for word in words if word.get("t", 0) < split_at]
        right_words = [word for word in words if word.get("t", 0) >= split_at]
        if words:
            left_text = " ".join(word.get("w", "") for word in left_words).strip()
            right_text = " ".join(word.get("w", "") for word in right_words).strip()
        else:
            left_text, right_text = original["transcript"], ""
        shared = {"speaker": original["speaker"], "similarity": original.get("similarity")}
        left = {"segment": index + 1, "start_s": original["start_s"], "end_s": split_at,
                "duration_s": round(split_at - original["start_s"], 2), "transcript": left_text,
                "words": left_words, "audio_file": created[0], **shared}
        right = {"segment": index + 2, "start_s": split_at, "end_s": original["end_s"],
                 "duration_s": round(original["end_s"] - split_at, 2), "transcript": right_text,
                 "words": right_words, "audio_file": created[1], **shared}
        self.result_data[index:index + 1] = [left, right]
        self.remove_saved_audio([original])
        self.refresh_result_rows(index)
        self.save_results_csv()
        self.status_var.set(f"Split transmission {index + 1} at {format_time(split_at)}")

    def transcription_failed(self, error: Exception) -> None:
        self.busy = False
        self.transcribe_button.state(["!disabled"])
        self.channel_box.configure(state="readonly")
        location = f" Partial output: {self.output_folder}" if self.output_folder else ""
        self.status_var.set(f"Transcription failed.{location}")
        messagebox.showerror("Pipeline error", str(error))

    def transcription_finished(self, _results: list[dict]) -> None:
        self.busy = False
        self.transcribe_button.state(["!disabled"])
        self.channel_box.configure(state="readonly")
        self.status_var.set(f"Done: {len(self.result_data)} transmissions. Review edits, then click Export results.")

    def export_results(self) -> None:
        if not self.result_data or not self.output_folder or not self.source:
            messagebox.showinfo("No results", "Transcribe the recording first.")
            return
        parent_name = filedialog.askdirectory(title="Choose where to export the results folder")
        if not parent_name:
            return
        channel_name = "mono" if self.channel_count == 1 else ("left" if self.current_channel() == 0 else "right")
        destination = Path(parent_name) / f"{self.source.stem}_{channel_name}_transcription"
        if destination.exists():
            destination = destination.with_name(f"{destination.name}_{datetime.now():%Y%m%d_%H%M%S}")
        try:
            self.save_results_csv()
            shutil.copytree(self.output_folder, destination)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self.status_var.set(f"Exported results to {destination}")
        messagebox.showinfo("Export complete", f"Saved the fingerprint, CSV, and split MP3 files in:\n{destination}")

    def close(self) -> None:
        self.stop_playback()
        self.cleanup_working_directory()
        self.destroy()


if __name__ == "__main__":
    missing = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
    if missing:
        raise SystemExit("Install ffmpeg (including ffprobe) and put it on PATH.")
    TranscriptionGUI().mainloop()
