#!/usr/bin/env python3
"""
Spotify-to-Arduino LED Controller.

Polls the currently playing track on Spotify and sends its dominant album
art color (as HSV hue) and popularity to an Arduino UNO over serial for
LED control.

Serial protocol:  H<hue 0-360>,P<popularity 0.00-1.00>\n
  - H = dominant hue from album artwork (0-360 degrees)
  - P = track popularity normalised to 0.0-1.0
"""

import colorsys
import io
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import serial
import spotipy
from spotipy.oauth2 import SpotifyOAuth

from config import (
    DEFAULT_BAUD_RATE,
    get_poll_interval,
    get_serial_port,
)


def dominant_hue_from_url(url: str) -> Optional[int]:
    """
    Download an image from *url* and return the dominant HSV hue (0-360).
    Uses a simple average-color approach on a small thumbnail.
    Returns None if the image can't be fetched.
    """
    try:
        import requests
        from PIL import Image
    except ImportError:
        return None

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
    except Exception:
        return None

    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    img = img.resize((16, 16))  # tiny thumbnail is enough for average color

    pixels = list(img.getdata())
    r_avg = sum(p[0] for p in pixels) / len(pixels) / 255.0
    g_avg = sum(p[1] for p in pixels) / len(pixels) / 255.0
    b_avg = sum(p[2] for p in pixels) / len(pixels) / 255.0

    h, _s, _v = colorsys.rgb_to_hsv(r_avg, g_avg, b_avg)
    return int(h * 360)


def format_led_message(hue: int, popularity: float) -> bytes:
    """Format as serial protocol: H180,P0.72\\n"""
    return f"H{hue},P{popularity:.2f}\n".encode()


def get_current_track_data(
    sp: spotipy.Spotify,
) -> Optional[Tuple[str, int, float, Optional[str]]]:
    """
    Get data for the currently playing track.
    Returns (track_id, hue, popularity, track_name) or None.
    """
    try:
        playing = sp.current_user_playing_track()
    except Exception as e:
        print(f"Spotify API error: {e}")
        return None

    if not playing or not playing.get("item"):
        return None

    item = playing["item"]
    if item.get("type") != "track":
        return None

    track_id = item.get("id")
    if not track_id:
        return None

    popularity = item.get("popularity", 50) / 100.0

    # Extract smallest album art image URL
    images = item.get("album", {}).get("images", [])
    art_url = images[-1]["url"] if images else None

    hue = dominant_hue_from_url(art_url) if art_url else 0

    track_name = item.get("name", "Unknown")
    artists = ", ".join(a["name"] for a in item.get("artists", []))

    return (track_id, hue if hue is not None else 0, popularity, f"{artists} - {track_name}")


def main() -> None:
    # Load .env and .env.local if python-dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv(".env.local", override=True)
        load_dotenv(override=False)
    except ImportError:
        pass

    # Validate credentials before connecting
    client_id = os.environ.get("SPOTIPY_CLIENT_ID", "").strip()
    client_secret = os.environ.get("SPOTIPY_CLIENT_SECRET", "").strip()
    redirect_uri = os.environ.get("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:8080").strip()
    if not client_id or not client_secret:
        print(
            "Error: SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET must be set.\n"
            "Add them to .env or .env.local (create an app at "
            "https://developer.spotify.com/dashboard)"
        )
        return

    # Use project-specific cache (easier to clear if auth breaks)
    cache_path = Path(__file__).parent / ".spotify_cache"
    scope = "user-read-currently-playing"

    auth_manager = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=scope,
        cache_path=str(cache_path),
    )
    sp = spotipy.Spotify(auth_manager=auth_manager)

    # Trigger auth early to catch INVALID_CLIENT before main loop
    try:
        sp.current_user_playing_track()
    except Exception as e:
        err_msg = str(e).lower()
        if "invalid_client" in err_msg or "invalid client" in err_msg:
            print(
                "INVALID_CLIENT: Spotify rejected the credentials.\n\n"
                "Try these steps:\n"
                "  1. In Spotify Dashboard → your app → Settings → reset the Client Secret, "
                "     then copy the NEW secret into .env.local\n"
                "  2. Delete the cache: rm .spotify_cache (from this project folder)\n"
                "  3. Ensure redirect URI in .env.local exactly matches the one in Dashboard "
                f"     (yours: {redirect_uri})\n"
                "  4. If using Python 3.13, try Python 3.11 (known compatibility issue)"
            )
        raise

    poll_interval = get_poll_interval()

    try:
        port = get_serial_port()
        print(f"Using Arduino port: {port}")
    except RuntimeError as e:
        print(e)
        return

    last_hue, last_popularity = 0, 0.5
    last_track_id: Optional[str] = None

    # Establish serial connection with retries
    ser: Optional[serial.Serial] = None
    while ser is None:
        try:
            ser = serial.Serial(
                port=port,
                baudrate=DEFAULT_BAUD_RATE,
                timeout=1,
                write_timeout=2,
            )
            print("Serial connection established. Polling Spotify...")
        except serial.SerialException as e:
            print(f"Serial error (retrying in 5s): {e}")
            time.sleep(5)

    try:
        while True:
            try:
                data = get_current_track_data(sp)
            except Exception as e:
                print(f"Unexpected error: {e}")
                time.sleep(poll_interval)
                continue

            if data is not None:
                track_id, hue, popularity, display = data
                if track_id != last_track_id:
                    print(f"Now playing: {display}  (hue={hue}, pop={popularity:.2f})")
                    last_track_id = track_id
                last_hue, last_popularity = hue, popularity

            # Always send values (use last known when nothing playing)
            try:
                msg = format_led_message(last_hue, last_popularity)
                ser.write(msg)
                ser.flush()
            except serial.SerialException as e:
                print(f"Serial write error (reconnecting in 5s): {e}")
                ser.close()
                ser = None
                while ser is None:
                    try:
                        ser = serial.Serial(
                            port=port,
                            baudrate=DEFAULT_BAUD_RATE,
                            timeout=1,
                            write_timeout=2,
                        )
                        print("Serial connection re-established.")
                    except serial.SerialException:
                        time.sleep(5)

            time.sleep(poll_interval)
    finally:
        if ser is not None and ser.is_open:
            ser.close()


if __name__ == "__main__":
    main()
