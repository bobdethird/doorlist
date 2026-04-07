#!/usr/bin/env python3
"""Play the isolated drums stem for a single song.

This is a focused debugging tool for solenoid timing. It plays only the
``drums.wav`` stem for one song, while optionally sending solenoid pulses and
LED updates to the Arduino.

Usage:
    python drums_test.py --song 04_DieYoung
    python drums_test.py --song 04_DieYoung --no-solenoid
"""

import argparse
import colorsys
import random
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from config import (
    BLOCK_SIZE,
    DEFAULT_BAUD_RATE,
    SAMPLE_RATE,
    get_serial_port,
    get_stems_dir,
)
from knob_mixer import BeatDetector, KnobReader, SolenoidController, _resample, led_pairs_command


def load_drums(song_name: str, stems_dir: Path) -> np.ndarray:
    """Load the cached drums stem for *song_name* as stereo float32."""
    stem_path = stems_dir / "htdemucs" / song_name / "drums.wav"
    if not stem_path.is_file():
        raise FileNotFoundError(
            f"Missing drums stem: {stem_path}\n"
            "Run stem separation first, or choose a song with cached stems."
        )

    data, sr = sf.read(stem_path, dtype="float32")
    if data.ndim == 1:
        data = np.column_stack([data, data])
    if sr != SAMPLE_RATE:
        data = _resample(data, sr, SAMPLE_RATE)

    peak = np.max(np.abs(data))
    if peak > 0:
        data *= 0.9 / peak
    return data


def get_chunk(data: np.ndarray, pos: int, frames: int) -> np.ndarray:
    """Extract *frames* samples from *data* starting at *pos*, wrapping."""
    data_len = len(data)
    end = pos + frames
    if end <= data_len:
        return data[pos:end]
    return np.concatenate([data[pos:], data[: end - data_len]])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play a song's isolated drums stem for solenoid testing",
    )
    parser.add_argument(
        "--song", type=str, required=True,
        help="Song stem name under stems/htdemucs (for example: 04_DieYoung)",
    )
    parser.add_argument(
        "--no-solenoid", action="store_true",
        help="Disable solenoid pulses and Arduino LED output",
    )
    args = parser.parse_args()

    stems_dir = get_stems_dir()
    drums = load_drums(args.song, stems_dir)
    print(f"Loaded drums stem: {args.song}  ({len(drums) / SAMPLE_RATE:.1f}s)")

    use_solenoid = not args.no_solenoid
    knob: Optional[KnobReader] = None
    if use_solenoid:
        port = get_serial_port()
        knob = KnobReader(port, DEFAULT_BAUD_RATE)
        knob.start()
        print(f"Sending solenoid / LED commands to {port}")
    else:
        print("Solenoid disabled.")

    beat = BeatDetector()
    solenoid = SolenoidController() if use_solenoid else None
    current_rms = [0.0]
    pending_hits = [0]
    last_solenoid_fire = [0.0]
    flash_count = [0]
    prev_brightness = [0.0]
    lock = threading.Lock()
    frame_pos = [0]

    PARTY_HUES = [
        0.83,   # purple
        0.75,   # violet
        0.92,   # magenta / hot pink
        0.0,    # red
        0.97,   # rose
        0.66,   # blue
        0.58,   # electric blue
        0.05,   # red-orange
    ]

    def party_color() -> tuple[int, int, int]:
        hue = random.choice(PARTY_HUES) + random.uniform(-0.03, 0.03)
        sat = random.uniform(0.8, 1.0)
        val = random.uniform(0.4, 0.55)
        r, g, b = colorsys.hsv_to_rgb(hue % 1.0, sat, val)
        return (int(r * 255), int(g * 255), int(b * 255))

    def party_palette() -> list[tuple[int, int, int]]:
        return [party_color() for _ in range(4)]

    led_colors = party_palette()

    def callback(
        outdata: np.ndarray,
        frames: int,
        _time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            print(f"  audio: {status}", flush=True)

        pos = frame_pos[0]
        chunk = get_chunk(drums, pos, frames)
        frame_pos[0] = (pos + frames) % len(drums)
        outdata[:] = chunk

        rms = float(np.sqrt(np.mean(chunk * chunk)))
        current_rms[0] = rms

        brightness = beat.update(rms)
        if brightness > 0.55 and brightness > prev_brightness[0]:
            flash_count[0] += 1
        prev_brightness[0] = brightness

        if solenoid and solenoid.update(rms):
            with lock:
                pending_hits[0] += 1
                last_solenoid_fire[0] = time.time()

    print("Drums test running — Ctrl+C to stop.\n")
    try:
        with sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=2,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=callback,
        ):
            last_fc = 0
            while True:
                fc = flash_count[0]
                if fc != last_fc:
                    led_colors = party_palette()
                    last_fc = fc

                brightness = beat._floor + beat._flash * (1.0 - beat._floor)

                palette_text = " ".join(
                    f"{r:02X}{g:02X}{b:02X}" for r, g, b in led_colors
                )
                if knob:
                    knob.send(led_pairs_command(led_colors, brightness=brightness))
                    with lock:
                        queued_hits = pending_hits[0]
                        pending_hits[0] = 0
                        recent_solenoid_fire = last_solenoid_fire[0]
                    for _ in range(queued_hits):
                        knob.send("S")
                else:
                    recent_solenoid_fire = 0.0

                solenoid_text = (
                    "SOL" if time.time() - recent_solenoid_fire < 0.2 else "   "
                )
                bar = "█" * int(brightness * 10)
                print(
                    f"\r  Song {args.song:<16s} │ {solenoid_text} │ "
                    f"{palette_text:<27s} │ ♪ {bar:<10s}",
                    end="", flush=True,
                )
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if knob:
            knob.stop()


if __name__ == "__main__":
    main()
