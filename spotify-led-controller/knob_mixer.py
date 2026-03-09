#!/usr/bin/env python3
"""
Knob-controlled song crossfade mixer with stem-based transitions.

Reads a potentiometer value from an Arduino over serial and transitions
between up to 6 local audio files.  Each song is separated into three
stems via demucs: vocals, beats (drums+bass), and tops (melodic
instruments).  Transitions swap one layer at a time — first the beats,
then the tops, then the vocals — creating a smooth mashup-style blend
between adjacent tracks.

Usage:
    python knob_mixer.py                        # Arduino connected
    python knob_mixer.py --mock                 # auto-sweep simulation
    python knob_mixer.py --mock --value 512     # fixed knob position
    python knob_mixer.py --no-stems             # simple volume crossfade

Audio files go in the songs/ directory (MP3, WAV, FLAC, OGG, AIFF).
They are loaded in alphabetical order.
"""

import argparse
import colorsys
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
    LED_BRIGHTNESS,
    NUM_NEOPIXELS,
    SAMPLE_RATE,
    SUPPORTED_AUDIO_EXTENSIONS,
    get_serial_port,
    get_songs_dir,
    get_stems_dir,
)
from stem_separator import ensure_stems, load_stems


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


def knob_to_stem_gains(
    knob_value: int, num_songs: int,
) -> List[Tuple[float, float, float]]:
    """Map a raw 0-1023 knob reading to per-song (vocal, beats, tops) gains.

    The transition between adjacent songs has three phases:
      1. First third  — beats crossfade A->B, vocals & tops stay on A.
      2. Middle third  — tops crossfade A->B, vocals stays A, beats stays B.
      3. Last third   — vocals crossfade A->B, beats & tops stay on B.

    Equal-power curves are used within each sub-transition.
    """
    position = (knob_value / 1023.0) * (num_songs - 1)
    idx = int(position)
    if idx >= num_songs - 1:
        idx = num_songs - 2
        blend = 1.0
    else:
        blend = position - idx

    third = 1.0 / 3.0
    gains: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * num_songs

    if blend <= third:
        # Phase 1: swap beats
        sub = blend / third
        c = math.cos(sub * math.pi / 2)
        s = math.sin(sub * math.pi / 2)
        gains[idx] = (1.0, c, 1.0)
        gains[idx + 1] = (0.0, s, 0.0)
    elif blend <= 2 * third:
        # Phase 2: swap tops
        sub = (blend - third) / third
        c = math.cos(sub * math.pi / 2)
        s = math.sin(sub * math.pi / 2)
        gains[idx] = (1.0, 0.0, c)
        gains[idx + 1] = (0.0, 1.0, s)
    else:
        # Phase 3: swap vocals
        sub = (blend - 2 * third) / third
        c = math.cos(sub * math.pi / 2)
        s = math.sin(sub * math.pi / 2)
        gains[idx] = (c, 0.0, 0.0)
        gains[idx + 1] = (s, 1.0, 1.0)

    return gains


class StemTransitionManager:
    """Direction-aware stem crossfade: always transitions beats→tops→vocals.

    Unlike the stateless ``knob_to_stem_gains``, this class tracks per-stem
    blend state so that reversing the knob still swaps beats before vocals.
    Each adjacent song-pair has three independent blend values (beats, tops,
    vocals) that are driven by knob *deltas*, using a cascading fill/drain
    model: forward fills beats first, then tops, then vocals; backward
    drains beats first, then tops, then vocals.
    """

    def __init__(self, num_songs: int):
        self.num_songs = num_songs
        # Per adjacent pair: [beats, tops, vocals] blend (0 = song A, 1 = song B)
        self._pairs: List[List[float]] = [
            [0.0, 0.0, 0.0] for _ in range(max(1, num_songs - 1))
        ]
        self._prev_position: Optional[float] = None
        self._gains: List[Tuple[float, float, float]] = [
            (0.0, 0.0, 0.0)
        ] * num_songs

    @property
    def gains(self) -> List[Tuple[float, float, float]]:
        """Last computed per-song (vocal, beats, tops) gains (read-only)."""
        return self._gains

    def update(self, knob_value: int) -> List[Tuple[float, float, float]]:
        """Advance state for *knob_value* and return per-song gains."""
        position = (knob_value / 1023.0) * (self.num_songs - 1)
        pair_idx = int(position)
        if pair_idx >= self.num_songs - 1:
            pair_idx = self.num_songs - 2
            blend = 1.0
        else:
            blend = position - pair_idx

        if self._prev_position is not None:
            prev_pair = int(self._prev_position)
            if prev_pair >= self.num_songs - 1:
                prev_pair = self.num_songs - 2
                prev_blend = 1.0
            else:
                prev_blend = self._prev_position - prev_pair

            if pair_idx == prev_pair:
                self._apply_delta(pair_idx, blend - prev_blend)
            elif pair_idx > prev_pair:
                for p in range(prev_pair, pair_idx):
                    self._pairs[p] = [1.0, 1.0, 1.0]
                self._fill_forward(pair_idx, blend)
            else:
                for p in range(pair_idx + 1, prev_pair + 1):
                    self._pairs[p] = [0.0, 0.0, 0.0]
                self._fill_backward(pair_idx, blend)
        else:
            self._fill_forward(pair_idx, blend)

        self._prev_position = position
        self._build_gains(pair_idx)
        return self._gains

    # -- internal helpers ---------------------------------------------------

    def _apply_delta(self, pair_idx: int, delta_blend: float) -> None:
        """Cascade a blend delta into the three stem slots (beats→tops→vocals)."""
        stems = self._pairs[pair_idx]
        budget = abs(delta_blend) * 3.0

        if delta_blend > 0:
            for i in range(3):
                space = 1.0 - stems[i]
                add = min(budget, space)
                stems[i] = min(1.0, stems[i] + add)
                budget -= add
                if budget <= 1e-9:
                    break
        elif delta_blend < 0:
            for i in range(3):
                drain = min(budget, stems[i])
                stems[i] = max(0.0, stems[i] - drain)
                budget -= drain
                if budget <= 1e-9:
                    break

    def _fill_forward(self, pair_idx: int, blend: float) -> None:
        """Set pair state as if arriving from blend=0 going forward."""
        budget = blend * 3.0
        stems = [0.0, 0.0, 0.0]
        for i in range(3):
            stems[i] = min(1.0, budget)
            budget = max(0.0, budget - 1.0)
        self._pairs[pair_idx] = stems

    def _fill_backward(self, pair_idx: int, blend: float) -> None:
        """Set pair state as if arriving from blend=1.0 going backward."""
        drain = (1.0 - blend) * 3.0
        stems = [1.0, 1.0, 1.0]
        for i in range(3):
            d = min(stems[i], drain)
            stems[i] -= d
            drain = max(0.0, drain - d)
        self._pairs[pair_idx] = stems

    def _build_gains(self, active_pair: int) -> None:
        """Convert raw stem blends into equal-power per-song gains."""
        gains: List[Tuple[float, float, float]] = [
            (0.0, 0.0, 0.0)
        ] * self.num_songs

        b_raw, t_raw, v_raw = self._pairs[active_pair]

        v_a = math.cos(v_raw * math.pi / 2)
        v_b = math.sin(v_raw * math.pi / 2)
        b_a = math.cos(b_raw * math.pi / 2)
        b_b = math.sin(b_raw * math.pi / 2)
        t_a = math.cos(t_raw * math.pi / 2)
        t_b = math.sin(t_raw * math.pi / 2)

        gains[active_pair] = (v_a, b_a, t_a)
        gains[active_pair + 1] = (v_b, b_b, t_b)
        self._gains = gains


# ---------------------------------------------------------------------------
# NeoPixel colors
# ---------------------------------------------------------------------------

def song_colors(num_songs: int, brightness: int = LED_BRIGHTNESS) -> List[Tuple[int, int, int]]:
    """Assign each song an evenly-spaced hue around the colour wheel."""
    colors = []
    for i in range(num_songs):
        h = i / num_songs
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        colors.append((int(r * brightness), int(g * brightness), int(b * brightness)))
    return colors


def led_command(gains: List[float], colors: List[Tuple[int, int, int]],
                num_pixels: int = NUM_NEOPIXELS,
                brightness: float = 1.0) -> str:
    """Build an ``L`` command string from per-song gains and colours.

    The 8 NeoPixels are split proportionally between the two active songs
    so you can *see* the crossfade position on the strip.  *brightness*
    (0.0-1.0) scales all RGB values -- used by BeatDetector to pulse on
    transients.
    """
    def _scale(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return (int(rgb[0] * brightness),
                int(rgb[1] * brightness),
                int(rgb[2] * brightness))

    active = [(i, g) for i, g in enumerate(gains) if g > 0.01]

    if not active:
        return f"L0,0,0,0,0,0,{num_pixels}"

    if len(active) == 1:
        r, g, b = _scale(colors[active[0][0]])
        return f"L{r},{g},{b},{r},{g},{b},{num_pixels}"

    idx_a, gain_a = active[0]
    idx_b, gain_b = active[1]
    r1, g1, b1 = _scale(colors[idx_a])
    r2, g2, b2 = _scale(colors[idx_b])
    split = round(gain_a / (gain_a + gain_b) * num_pixels)
    split = max(0, min(num_pixels, split))
    return f"L{r1},{g1},{b1},{r2},{g2},{b2},{split}"


# ---------------------------------------------------------------------------
# Beat detection
# ---------------------------------------------------------------------------

class BeatDetector:
    """Pulse a 0-1 brightness value in response to audio energy spikes.

    Keeps a slow-moving average of RMS energy.  When the instantaneous
    energy jumps above the average by *threshold*, a flash is triggered
    that decays exponentially.  A minimum floor keeps the LEDs always
    somewhat visible.
    """

    def __init__(self, threshold: float = 1.5, decay: float = 0.78,
                 floor: float = 0.20):
        self._avg = 0.0
        self._flash = 0.0
        self._threshold = threshold
        self._decay = decay
        self._floor = floor

    def update(self, rms: float) -> float:
        """Feed in current RMS energy, get back a brightness 0.0-1.0."""
        if self._avg < 1e-6:
            self._avg = rms
        elif rms > self._avg * self._threshold:
            self._flash = 1.0

        self._avg = self._avg * 0.93 + rms * 0.07
        self._flash *= self._decay

        return min(1.0, self._floor + self._flash * (1.0 - self._floor))


# ---------------------------------------------------------------------------
# Knob readers (real serial + mock)
# ---------------------------------------------------------------------------

class KnobReader:
    """Bidirectional serial link to the Arduino.

    A background thread reads ``K<value>\\n`` knob lines; the main thread
    can call :meth:`send` to push LED commands back.
    """

    def __init__(self, port: str, baudrate: int = DEFAULT_BAUD_RATE):
        self.value = 512
        self._port = port
        self._baudrate = baudrate
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._serial = None

    def start(self) -> None:
        import serial as _serial

        self._serial = _serial.Serial(self._port, self._baudrate, timeout=0.1)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._serial and self._serial.is_open:
            self._serial.close()

    def send(self, msg: str) -> None:
        """Write a command line to the Arduino (called from main thread)."""
        ser = self._serial
        if ser and ser.is_open:
            try:
                ser.write(f"{msg}\n".encode())
            except Exception:
                pass

    def _loop(self) -> None:
        while self._running:
            try:
                line = self._serial.readline().decode("ascii", errors="ignore").strip()
                if line.startswith("K"):
                    try:
                        self.value = max(0, min(1023, int(line[1:])))
                    except ValueError:
                        pass
            except Exception:
                if not self._running:
                    break


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

    def send(self, msg: str) -> None:  # noqa: D102
        pass

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

def _get_chunk(data: np.ndarray, pos: int, frames: int) -> np.ndarray:
    """Extract *frames* samples from *data* starting at *pos*, wrapping."""
    data_len = len(data)
    end = pos + frames
    if end <= data_len:
        return data[pos:end]
    return np.concatenate([data[pos:], data[: end - data_len]])


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
    parser.add_argument(
        "--no-stems", action="store_true",
        help="Disable stem separation; use simple volume crossfade instead",
    )
    args = parser.parse_args()

    if args.value is not None:
        args.mock = True

    use_stems = not args.no_stems

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

    if use_stems:
        stems_dir = get_stems_dir()
        print(f"Loading songs from {songs_dir} (stem mode) ...")
        print("Checking for cached stems ...")
        _files, names = ensure_stems(songs_dir, stems_dir)

        if len(names) < 2:
            print(
                f"Need at least 2 audio files in {songs_dir}, "
                f"found {len(names)}.\n"
                "Supported formats: MP3, WAV, FLAC, OGG, AIFF."
            )
            return

        vocals, beats, tops = load_stems(stems_dir, names)
        songs: Optional[List[np.ndarray]] = None
    else:
        print(f"Loading songs from {songs_dir} (simple crossfade) ...")
        songs, names = load_songs(songs_dir)
        vocals = None
        beats = None
        tops = None

        if len(names) < 2:
            print(
                f"Need at least 2 audio files in {songs_dir}, "
                f"found {len(names)}.\n"
                "Supported formats: MP3, WAV, FLAC, OGG, AIFF."
            )
            return

    num_songs = len(names)
    colors = song_colors(num_songs)
    for i, (name, c) in enumerate(zip(names, colors)):
        print(f"  #{i + 1} {name}  →  RGB({c[0]}, {c[1]}, {c[2]})")
    mode_label = "stem transitions" if use_stems else "simple crossfade"
    print(f"{num_songs} songs loaded ({mode_label}).\n")

    stem_mgr = StemTransitionManager(num_songs) if use_stems else None

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

    current_rms = [0.0]

    # ---- audio callbacks --------------------------------------------------

    def callback_stems(
        outdata: np.ndarray,
        frames: int,
        _time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            print(f"  audio: {status}", flush=True)

        stem_gains = stem_mgr.update(knob.value)
        mixed = np.zeros((frames, 2), dtype=np.float32)

        for i in range(num_songs):
            vg, bg, tg = stem_gains[i]
            pos = frame_pos[i]

            song_len = len(vocals[i])
            end = pos + frames
            frame_pos[i] = end % song_len

            if vg > 0.001:
                mixed += _get_chunk(vocals[i], pos, frames) * vg
            if bg > 0.001:
                mixed += _get_chunk(beats[i], pos, frames) * bg
            if tg > 0.001:
                mixed += _get_chunk(tops[i], pos, frames) * tg

        outdata[:] = mixed
        current_rms[0] = float(np.sqrt(np.mean(mixed * mixed)))

    def callback_simple(
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
            song_len = len(songs[i])
            end = pos + frames
            frame_pos[i] = end % song_len

            g = gains[i]
            if g > 0.001:
                mixed += _get_chunk(songs[i], pos, frames) * g

        outdata[:] = mixed
        current_rms[0] = float(np.sqrt(np.mean(mixed * mixed)))

    active_callback = callback_stems if use_stems else callback_simple

    # ---- run --------------------------------------------------------------
    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=2,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=active_callback,
    ):
        beat = BeatDetector()
        tick = 0
        print("Mixer running — Ctrl+C to stop.\n")
        try:
            while True:
                brightness = beat.update(current_rms[0])

                if use_stems:
                    stem_gains = stem_mgr.gains
                    led_gains = [max(vg, bg, tg) for vg, bg, tg in stem_gains]
                else:
                    led_gains = knob_to_gains(knob.value, num_songs)

                cmd = led_command(led_gains, colors, brightness=brightness)
                knob.send(cmd)

                tick += 1
                if tick % 4 == 0:
                    if use_stems:
                        stem_gains = stem_mgr.gains
                        parts = []
                        for i, (vg, bg, tg) in enumerate(stem_gains):
                            if vg > 0.01 or bg > 0.01 or tg > 0.01:
                                parts.append(
                                    f"{names[i]}[v:{vg:.0%} b:{bg:.0%} t:{tg:.0%}]"
                                )
                    else:
                        simple_gains = knob_to_gains(knob.value, num_songs)
                        parts = [
                            f"{names[i]}: {g:.0%}"
                            for i, g in enumerate(simple_gains)
                            if g > 0.01
                        ]
                    label = " + ".join(parts) if parts else "silence"
                    bar = "█" * int(brightness * 10)
                    print(
                        f"\r  Knob {knob.value:4d}  │  {label:<50s} │ ♪ {bar:<10s}",
                        end="", flush=True,
                    )
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nStopping...")

    knob.stop()


if __name__ == "__main__":
    main()
