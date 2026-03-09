#!/usr/bin/env python3
"""
Knob-controlled song crossfade mixer.

Reads a potentiometer value from an Arduino over serial and crossfades
between up to 6 local audio files.  Adjacent songs blend smoothly as
the knob turns using an equal-power crossfade curve.

Usage:
    python knob_mixer.py                        # Arduino connected
    python knob_mixer.py --mock                 # auto-sweep simulation
    python knob_mixer.py --mock --value 512     # fixed knob position

Audio files go in the songs/ directory (MP3, WAV, FLAC, OGG, AIFF).
They are loaded in alphabetical order.
"""

import argparse
import math
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf

from config import (
    BLOCK_SIZE,
    DEFAULT_BAUD_RATE,
    SAMPLE_RATE,
    SUPPORTED_AUDIO_EXTENSIONS,
    get_serial_port,
    get_songs_dir,
)


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def _resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample via linear interpolation (good enough for a crossfade mixer)."""
    if orig_sr == target_sr:
        return data
    ratio = target_sr / orig_sr
    n_out = int(len(data) * ratio)
    x_old = np.arange(len(data))
    x_new = np.linspace(0, len(data) - 1, n_out)
    if data.ndim == 1:
        return np.interp(x_new, x_old, data).astype(np.float32)
    out = np.empty((n_out, data.shape[1]), dtype=np.float32)
    for ch in range(data.shape[1]):
        out[:, ch] = np.interp(x_new, x_old, data[:, ch])
    return out


def _read_audio_file(path: Path) -> Tuple[np.ndarray, int]:
    """Read an audio file and return (samples_float32, sample_rate).

    Tries soundfile first (WAV/FLAC/OGG/AIFF).  Falls back to pydub for
    formats like MP3 that need ffmpeg decoding.
    """
    try:
        data, sr = sf.read(path, dtype="float32")
        return data, sr
    except Exception:
        pass

    from pydub import AudioSegment

    seg = AudioSegment.from_file(str(path))
    sr = seg.frame_rate
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    # pydub returns interleaved samples; reshape to (frames, channels)
    samples = samples.reshape((-1, seg.channels))
    # normalise int range to -1..1
    samples /= float(1 << (seg.sample_width * 8 - 1))
    return samples, sr


def load_songs(songs_dir: Path) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load audio files from *songs_dir* in alphabetical order.

    Each file is converted to stereo float32 at SAMPLE_RATE and peak-
    normalised so that all tracks have comparable loudness.

    Returns (list_of_arrays, list_of_display_names).
    """
    files = sorted(
        f for f in songs_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )
    if not files:
        return [], []

    songs: List[np.ndarray] = []
    names: List[str] = []

    for f in files[:6]:
        data, sr = _read_audio_file(f)

        if data.ndim == 1:
            data = np.column_stack([data, data])

        if sr != SAMPLE_RATE:
            data = _resample(data, sr, SAMPLE_RATE)

        peak = np.max(np.abs(data))
        if peak > 0:
            data *= 0.9 / peak

        songs.append(data)
        names.append(f.stem)
        print(f"  Loaded: {f.name}  ({len(data) / SAMPLE_RATE:.1f}s)")

    return songs, names


# ---------------------------------------------------------------------------
# Knob → gain mapping
# ---------------------------------------------------------------------------

def knob_to_gains(knob_value: int, num_songs: int) -> List[float]:
    """
    Map a raw 0-1023 knob reading to per-song gains.

    Only 2 adjacent songs are non-zero at any time.  The transition uses
    an equal-power (cos/sin) curve so perceived loudness stays constant
    through the crossfade zone.
    """
    position = (knob_value / 1023.0) * (num_songs - 1)
    idx = int(position)
    if idx >= num_songs - 1:
        idx = num_songs - 2
        blend = 1.0
    else:
        blend = position - idx

    gains = [0.0] * num_songs
    gains[idx] = math.cos(blend * math.pi / 2)
    gains[idx + 1] = math.sin(blend * math.pi / 2)
    return gains


# ---------------------------------------------------------------------------
# Knob readers (real serial + mock)
# ---------------------------------------------------------------------------

class KnobReader:
    """Background thread that reads ``K<value>\\n`` lines from Arduino."""

    def __init__(self, port: str, baudrate: int = DEFAULT_BAUD_RATE):
        self.value = 512
        self._port = port
        self._baudrate = baudrate
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        import serial as _serial

        with _serial.Serial(self._port, self._baudrate, timeout=0.1) as ser:
            while self._running:
                line = ser.readline().decode("ascii", errors="ignore").strip()
                if line.startswith("K"):
                    try:
                        self.value = max(0, min(1023, int(line[1:])))
                    except ValueError:
                        pass


class MockKnobReader:
    """Simulates the knob for testing without hardware.

    If *fixed_value* is given the knob stays there; otherwise it sweeps
    back and forth at a comfortable pace.
    """

    def __init__(self, fixed_value: Optional[int] = None):
        self.value = fixed_value if fixed_value is not None else 0
        self._fixed = fixed_value is not None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._fixed:
            return
        self._running = True
        self._thread = threading.Thread(target=self._sweep, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _sweep(self) -> None:
        direction = 1
        while self._running:
            self.value += direction * 3
            if self.value >= 1023:
                self.value = 1023
                direction = -1
            elif self.value <= 0:
                self.value = 0
                direction = 1
            time.sleep(0.03)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Knob-controlled song crossfade mixer",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Simulate the knob (auto-sweep or fixed --value)",
    )
    parser.add_argument(
        "--value", type=int, default=None,
        help="Fixed knob position 0-1023 (implies --mock)",
    )
    parser.add_argument(
        "--songs-dir", type=str, default=None,
        help="Directory containing audio files (default: ./songs)",
    )
    args = parser.parse_args()

    if args.value is not None:
        args.mock = True

    # ---- load songs -------------------------------------------------------
    songs_dir = Path(args.songs_dir) if args.songs_dir else get_songs_dir()

    if not songs_dir.exists():
        songs_dir.mkdir(parents=True)
        print(
            f"Created {songs_dir}/\n"
            "Drop 2-6 audio files (MP3 / WAV / FLAC / OGG / AIFF) in there,\n"
            "then run again."
        )
        return

    print(f"Loading songs from {songs_dir} ...")
    songs, names = load_songs(songs_dir)

    if len(songs) < 2:
        print(
            f"Need at least 2 audio files in {songs_dir}, found {len(songs)}.\n"
            "Supported formats: MP3, WAV, FLAC, OGG, AIFF."
        )
        return

    num_songs = len(songs)
    print(f"{num_songs} songs loaded.\n")

    # Per-song frame cursor — all advance continuously so turning the knob
    # back re-enters a song where it would have been, not from the start.
    frame_pos = [0] * num_songs

    # ---- knob reader ------------------------------------------------------
    if args.mock:
        knob: KnobReader | MockKnobReader = MockKnobReader(
            fixed_value=args.value,
        )
        print("Mock mode — ", end="")
        if args.value is not None:
            print(f"knob fixed at {args.value}")
        else:
            print("knob sweeping automatically")
    else:
        port = get_serial_port()
        knob = KnobReader(port, DEFAULT_BAUD_RATE)
        print(f"Reading knob from {port}")

    knob.start()

    # ---- audio callback ---------------------------------------------------
    def callback(
        outdata: np.ndarray,
        frames: int,
        _time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            print(f"  audio: {status}", flush=True)

        gains = knob_to_gains(knob.value, num_songs)
        mixed = np.zeros((frames, 2), dtype=np.float32)

        for i in range(num_songs):
            pos = frame_pos[i]
            song = songs[i]
            song_len = len(song)

            # Advance position regardless of gain (keeps songs time-aligned)
            end = pos + frames
            if end <= song_len:
                chunk = song[pos:end]
            else:
                chunk = np.concatenate([song[pos:], song[: end - song_len]])

            frame_pos[i] = end % song_len

            g = gains[i]
            if g > 0.001:
                mixed += chunk * g

        outdata[:] = mixed

    # ---- run --------------------------------------------------------------
    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=2,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=callback,
    ):
        print("Mixer running — Ctrl+C to stop.\n")
        try:
            while True:
                gains = knob_to_gains(knob.value, num_songs)
                parts = [
                    f"{names[i]}: {g:.0%}"
                    for i, g in enumerate(gains)
                    if g > 0.01
                ]
                label = " + ".join(parts) if parts else "silence"
                print(f"\r  Knob {knob.value:4d}  │  {label:<60s}", end="", flush=True)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")

    knob.stop()


if __name__ == "__main__":
    main()
