"""Configuration and port detection for the LED controller and knob mixer."""

import os
from pathlib import Path
from typing import Optional

from serial.tools import list_ports

# Arduino UNO vendor ID (official Arduino boards)
ARDUINO_VID = 0x2341

# Default baud rate for Arduino serial communication
DEFAULT_BAUD_RATE = 9600

# Default poll interval in seconds (2-5 seconds per plan)
DEFAULT_POLL_INTERVAL = 3

# Audio mixer defaults
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".mp3"}

# NeoPixel defaults (must match #defines in knob_sender.ino)
NUM_NEOPIXELS = 8
LED_BRIGHTNESS = 127  # 0-255 per channel (127 ≈ 50 % brightness / current)


def find_arduino_port() -> Optional[str]:
    """
    Auto-detect Arduino UNO serial port by vendor ID.
    Returns the port path (e.g., /dev/cu.usbmodem14201) or None if not found.
    """
    for port in list_ports.comports():
        if port.vid == ARDUINO_VID:
            return port.device
    return None


def get_serial_port() -> str:
    """
    Get the serial port for Arduino communication.
    Uses ARDUINO_PORT env var if set, otherwise auto-detects.
    Raises RuntimeError if no port is found.
    """
    port = os.environ.get("ARDUINO_PORT")
    if port:
        return port

    port = find_arduino_port()
    if port:
        return port

    raise RuntimeError(
        "No Arduino found. Connect your Arduino UNO via USB, or set ARDUINO_PORT "
        "environment variable (e.g., /dev/cu.usbmodem14201 on macOS)."
    )


def get_poll_interval() -> float:
    """Get poll interval in seconds from env or default."""
    try:
        return float(os.environ.get("POLL_INTERVAL", DEFAULT_POLL_INTERVAL))
    except ValueError:
        return DEFAULT_POLL_INTERVAL


def get_songs_dir() -> Path:
    """Get the directory containing audio files for the knob mixer."""
    env_dir = os.environ.get("SONGS_DIR")
    if env_dir:
        return Path(env_dir)
    return Path(__file__).parent / "songs"
