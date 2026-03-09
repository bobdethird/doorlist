# Spotify LED Controller

Polls Spotify’s currently playing track and drives an Arduino + NeoPixel strip with the album art’s dominant color. Also includes a **knob mixer** that crossfades between local audio files using a potentiometer.

## Hardware

- **Arduino UNO** (or compatible with vendor ID `0x2341`)
- **NeoPixel strip** (8 LEDs, WS2812B or compatible)
- **Potentiometer** (for knob mixer only)

### Wiring

| Component        | Arduino Pin |
|------------------|-------------|
| Potentiometer outer pins | 5V, GND |
| Potentiometer wiper      | A0 |
| NeoPixel DIN             | 12 |
| NeoPixel VCC / GND       | 5V, GND |

## Setup

### 1. Arduino firmware

1. Install the [Adafruit NeoPixel](https://github.com/adafruit/Adafruit_NeoPixel) library in the Arduino IDE.
2. Open `knob_sender/knob_sender.ino` and upload to the Arduino.

### 2. Python environment

```bash
cd spotify-led-controller
pip install -r requirements.txt
```

### 3. Environment variables

Copy the example env file and add your Spotify credentials:

```bash
cp .env.example .env.local
```

Edit `.env.local` and set:

- `SPOTIPY_CLIENT_ID` — from your [Spotify Developer Dashboard](https://developer.spotify.com/dashboard) app
- `SPOTIPY_CLIENT_SECRET` — from the same app
- `SPOTIPY_REDIRECT_URI` — must match the redirect URI in the app settings (e.g. `http://127.0.0.1:8888/callback`)

Optional:

- `ARDUINO_PORT` — override auto-detection (e.g. `/dev/cu.usbmodem14201` on macOS)
- `POLL_INTERVAL` — seconds between Spotify polls (default: 3)
- `SONGS_DIR` — path to audio files for knob mixer (default: `./songs`)

## Usage

### Spotify LED mode

```bash
python spotify_led.py
```

Polls the currently playing track and sends the dominant album art hue and popularity to the Arduino. On first run you’ll be prompted to authorize in the browser.

### Knob mixer mode

1. Create a `songs/` directory and add 2–6 audio files (MP3, WAV, FLAC, OGG, AIFF).
2. Run:

```bash
python knob_mixer.py
```

Turn the potentiometer to crossfade between songs. Use `--mock` to simulate the knob without hardware:

```bash
python knob_mixer.py --mock              # auto-sweep
python knob_mixer.py --mock --value 512  # fixed position
```

## Troubleshooting

**No Arduino found** — Connect the Arduino via USB. If auto-detection fails, set `ARDUINO_PORT` in `.env.local` to your serial port (e.g. `/dev/cu.usbmodem14201` on macOS, `COM3` on Windows).

**INVALID_CLIENT** — Spotify rejected the credentials. Try:
1. In Spotify Dashboard → your app → Settings → reset the Client Secret, then update `.env.local`
2. Delete the cache: `rm .spotify_cache`
3. Ensure the redirect URI in `.env.local` exactly matches the one in the Dashboard

**Python 3.13** — If you see auth issues, try Python 3.11.
