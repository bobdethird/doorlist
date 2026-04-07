#!/usr/bin/env python3
"""Pre-compute solenoid hit times for a drums stem.

Runs the same BeatDetector logic used for the LEDs offline over the
entire drums stem, recording every flash trigger as a solenoid hit.
This means the solenoid fires at exactly the same moment the LEDs
flash — one algorithm, zero latency at runtime.

Usage:
    python solenoid_precompute.py --song 04_DieYoung
    python solenoid_precompute.py --song 04_DieYoung --offset -0.02
    python solenoid_precompute.py --all
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from config import BLOCK_SIZE, SAMPLE_RATE, get_stems_dir
from knob_mixer import _resample


def load_drums_stereo(stem_path: Path) -> np.ndarray:
    data, sr = sf.read(stem_path, dtype="float32")
    if data.ndim == 1:
        data = np.column_stack([data, data])
    if sr != SAMPLE_RATE:
        data = _resample(data, sr, SAMPLE_RATE)
    return data


def detect_hits(audio: np.ndarray, offset_s: float = 0.0,
                threshold: float = 1.5, decay: float = 0.78) -> list[float]:
    """Simulate BeatDetector block-by-block; record every flash trigger.

    This is the exact same algorithm the LEDs use at runtime — a hit is
    registered whenever ``rms > avg * threshold``, which sets flash to 1.0.
    The exponential decay of flash acts as a natural refractory period:
    the next trigger can only happen once the energy has settled and then
    spikes again above the (continuously-updated) moving average.
    """
    n_blocks = len(audio) // BLOCK_SIZE
    avg = 0.0
    flash = 0.0
    hits: list[float] = []

    for i in range(n_blocks):
        start = i * BLOCK_SIZE
        chunk = audio[start : start + BLOCK_SIZE]
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        t = start / SAMPLE_RATE

        fired = False
        if avg < 1e-6:
            avg = rms
        elif rms > avg * threshold:
            if flash < 0.3:
                fired = True
            flash = 1.0

        avg = avg * 0.93 + rms * 0.07
        flash *= decay

        if fired:
            t_adj = round(t + offset_s, 6)
            if t_adj >= 0:
                hits.append(t_adj)

    return hits


def process_song(song_dir: Path, offset_s: float = 0.0) -> Path:
    stem_path = song_dir / "drums.wav"
    if not stem_path.is_file():
        raise FileNotFoundError(f"No drums.wav in {song_dir}")

    song_name = song_dir.name
    print(f"  {song_name}: loading...", end="", flush=True)
    audio = load_drums_stereo(stem_path)
    duration = len(audio) / SAMPLE_RATE
    print(f" ({duration:.1f}s)  detecting hits...", end="", flush=True)

    hits = detect_hits(audio, offset_s=offset_s)

    out_path = song_dir / "solenoid_hits.json"
    payload = {
        "song": song_name,
        "sample_rate": SAMPLE_RATE,
        "duration_s": round(duration, 3),
        "offset_s": offset_s,
        "n_hits": len(hits),
        "hits": hits,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  {len(hits)} hits → {out_path.name}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute solenoid hit times from drums stems",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--song", type=str, help="Single song to process")
    group.add_argument("--all", action="store_true", help="Process every song in stems/htdemucs")
    parser.add_argument(
        "--offset", type=float, default=0.0,
        help="Time offset in seconds applied to every hit "
             "(negative = fire earlier to compensate for mechanical lag, default: 0)",
    )
    args = parser.parse_args()

    stems_dir = get_stems_dir()
    htdemucs = stems_dir / "htdemucs"
    if not htdemucs.is_dir():
        raise FileNotFoundError(f"No htdemucs directory at {htdemucs}")

    if args.all:
        songs = sorted(p for p in htdemucs.iterdir() if p.is_dir())
        print(f"Processing {len(songs)} songs...\n")
        for song_dir in songs:
            try:
                process_song(song_dir, offset_s=args.offset)
            except FileNotFoundError as e:
                print(f"  SKIP: {e}")
    else:
        song_dir = htdemucs / args.song
        if not song_dir.is_dir():
            raise FileNotFoundError(f"Song directory not found: {song_dir}")
        process_song(song_dir, offset_s=args.offset)

    print("\nDone.")


if __name__ == "__main__":
    main()
