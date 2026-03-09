"""Stem separation and loading using demucs.

Splits each song into four raw stems (vocals, drums, bass, other) via
demucs, then recombines them into three mix-ready stems:

  - **vocals** — singing / lyrics
  - **beats**  — drums + bass
  - **tops**   — other (synths, guitars, melodies, effects)

Results are cached so separation only runs once per song.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

from config import SAMPLE_RATE, SUPPORTED_AUDIO_EXTENSIONS

DEMUCS_MODEL = "htdemucs"
STEM_FILES = ("vocals.wav", "drums.wav", "bass.wav", "other.wav")


def _resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
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


def _stem_dir(stems_dir: Path, song_name: str) -> Path:
    """Return the directory where demucs writes stems for a given song."""
    return stems_dir / DEMUCS_MODEL / song_name


def _stems_exist(stems_dir: Path, song_name: str) -> bool:
    d = _stem_dir(stems_dir, song_name)
    return all((d / f).is_file() for f in STEM_FILES)


def separate_song(song_path: Path, stems_dir: Path) -> None:
    """Run demucs on a single song file, writing stems into *stems_dir*."""
    print(f"  Separating stems: {song_path.name} (this may take a minute) ...")
    cmd = [
        sys.executable, "-m", "demucs",
        "-n", DEMUCS_MODEL,
        "-o", str(stems_dir),
        str(song_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"demucs failed for {song_path.name}:\n{result.stderr}"
        )
    print(f"  Done: {song_path.name}")


def ensure_stems(songs_dir: Path, stems_dir: Path) -> Tuple[List[Path], List[str]]:
    """Make sure every song in *songs_dir* has cached stems.

    Returns the list of song files (sorted, max 6) and their display names,
    same ordering used by the mixer.
    """
    files = sorted(
        f for f in songs_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )[:6]

    if not files:
        return [], []

    stems_dir.mkdir(parents=True, exist_ok=True)
    names: List[str] = []

    for f in files:
        name = f.stem
        names.append(name)
        if not _stems_exist(stems_dir, name):
            separate_song(f, stems_dir)

    return files, names


def _load_stem(path: Path) -> np.ndarray:
    """Load a single stem WAV, convert to stereo float32 at SAMPLE_RATE."""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim == 1:
        data = np.column_stack([data, data])
    if sr != SAMPLE_RATE:
        data = _resample(data, sr, SAMPLE_RATE)
    return data


def load_stems(
    stems_dir: Path, names: List[str],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Load and recombine stems for each song.

    Returns (vocals_list, beats_list, tops_list) where:
      - vocals = vocals stem
      - beats  = drums + bass stems summed
      - tops   = other stem

    Each entry is a stereo float32 array at SAMPLE_RATE.  All three stems
    are trimmed to the same length and peak-normalised as a group so the
    recombined signal matches the original loudness.
    """
    all_vocals: List[np.ndarray] = []
    all_beats: List[np.ndarray] = []
    all_tops: List[np.ndarray] = []

    for name in names:
        d = _stem_dir(stems_dir, name)
        v = _load_stem(d / "vocals.wav")
        drums = _load_stem(d / "drums.wav")
        bass = _load_stem(d / "bass.wav")
        other = _load_stem(d / "other.wav")

        # Align lengths (demucs may pad slightly differently per stem)
        min_len = min(len(v), len(drums), len(bass), len(other))
        v = v[:min_len]
        beats = drums[:min_len] + bass[:min_len]
        tops = other[:min_len]

        # Peak-normalise the combined signal so it matches original loudness
        combined_peak = np.max(np.abs(v + beats + tops))
        if combined_peak > 0:
            scale = 0.9 / combined_peak
            v *= scale
            beats *= scale
            tops *= scale

        all_vocals.append(v)
        all_beats.append(beats)
        all_tops.append(tops)
        print(f"  Loaded stems: {name}  ({min_len / SAMPLE_RATE:.1f}s)")

    return all_vocals, all_beats, all_tops
