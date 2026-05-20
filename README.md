# Spotify LED Controller

A small toolkit for turning Spotify tracks (and local audio files) into a live light-and-percussion show driven by an Arduino, a NeoPixel strip, and an optional solenoid kicker.

There are a few different modes:

- **Spotify LED** — polls the currently-playing track on Spotify and pushes the album art's dominant color to the LEDs.
- **Knob mixer** — crossfades between up to 6 local songs using a potentiometer, with optional [demucs](https://github.com/adefossez/demucs)-based stem separation so the knob can isolate vocals / beats / tops. Drives the LEDs and (optionally) a solenoid on every kick.
- **Drums test** — plays just the isolated drums stem of one song, useful for tuning the solenoid timing.
- **Drums visualizer** — live scrolling matplotlib plot of RMS, brightness, and pre-computed solenoid hits.
- **Solenoid pre-compute** — offline pass that records every solenoid trigger time to a JSON sidecar, so playback has zero-latency mechanical hits.

## Hardware

- **Arduino UNO** (or compatible with vendor ID `0x2341`)
- **NeoPixel strip** (8 LEDs, WS2812B or compatible)
- **Potentiometer** (for knob mixer only)
- **Solenoid + N-channel MOSFET** with flyback diode (optional, for the percussion modes)

### Wiring

| Component                  | Arduino Pin |
|----------------------------|-------------|
| Potentiometer outer pins   | 5V, GND     |
| Potentiometer wiper        | A0          |
| NeoPixel DIN               | 12          |
| NeoPixel VCC / GND         | 5V, GND     |
| MOSFET gate (solenoid)     | 9           |
| MOSFET source / solenoid – | GND         |
| Solenoid +                 | external supply (with flyback diode across the coil) |

The solenoid pulse width is 60 ms (`SOLENOID_PULSE_MS` in both `config.py` and `knob_sender.ino`).

## Setup

### 1. Arduino firmware

1. Install the [Adafruit NeoPixel](https://github.com/adafruit/Adafruit_NeoPixel) library in the Arduino IDE.
2. Open `knob_sender/knob_sender.ino` and upload to the Arduino.

The firmware speaks a tiny line-based protocol over serial at 9600 baud:

- `K<0-1023>\n` — knob reading sent up at ~50 Hz
- `L<r1>,<g1>,<b1>,<r2>,<g2>,<b2>,<split>\n` — two-color split fill
- `P<r1>,<g1>,<b1>,…,<r4>,<g4>,<b4>\n` — 4 adjacent color pairs across the strip
- `S\n` — fire the solenoid for one pulse

### 2. Python environment

```bash
cd spotify-led-controller
pip install -r requirements.txt
```

Note: `demucs` pulls in `torch`, so the first install is fairly large. If you only want the Spotify LED mode, you can skip the demucs / numpy / sounddevice extras.

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
- `SONGS_DIR` — path to audio files for the knob mixer (default: `./songs`)
- `STEMS_DIR` — where demucs caches separated stems (default: `./stems`)

## Usage

### Spotify LED mode

```bash
python spotify_led.py
```

Polls the currently playing track and sends the dominant album art hue and popularity to the Arduino. On first run you'll be prompted to authorize in the browser.

### Knob mixer mode

1. Create a `songs/` directory and add 2–6 audio files (MP3, WAV, FLAC, OGG, AIFF).
2. Run:

```bash
python knob_mixer.py
```

On first run this will use demucs to split each song into `vocals` / `drums` / `bass` / `other`, caching the results under `stems/htdemucs/<song_name>/`. Subsequent runs reuse the cache.

Useful flags:

```bash
python knob_mixer.py --mock                 # auto-sweep the knob
python knob_mixer.py --mock --value 512     # fixed knob position
python knob_mixer.py --no-stems             # plain volume crossfade, skip demucs
python knob_mixer.py --no-solenoid          # silence the solenoid
python knob_mixer.py --songs-dir /path/to/songs
```

Turn the potentiometer to crossfade between songs. With stems enabled, the mixer separates the vocals / beats / tops layers so transitions blend smoothly; a beat detector drives the LED brightness and fires the solenoid on each kick.

### Drums test (single song)

Plays only the `drums.wav` stem for one song. This is the simplest way to dial in solenoid timing:

```bash
python drums_test.py --song 04_DieYoung
python drums_test.py --song 04_DieYoung --no-solenoid
```

`--song` is the directory name under `stems/htdemucs/`. The song's stems must already be cached (run `knob_mixer.py` once to generate them).

### Drums visualizer (live plot)

Plays the full mix and shows a live scrolling matplotlib plot of RMS energy, LED brightness, and every solenoid fire, using the pre-computed hit times from `solenoid_hits.json`:

```bash
python drums_visualizer.py --song 04_DieYoung
python drums_visualizer.py --song 04_DieYoung --window 15
python drums_visualizer.py --song 04_DieYoung --offset -0.02   # fire 20 ms earlier
python drums_visualizer.py --song 04_DieYoung --no-solenoid
```

Requires `solenoid_hits.json` for the song (see the next section).

### Solenoid pre-compute

Runs the same beat-detection logic the LEDs use, but offline over the full drums stem, and writes every trigger time to `stems/htdemucs/<song>/solenoid_hits.json`. The visualizer then fires the solenoid from these timestamps for sample-accurate, latency-free hits:

```bash
python solenoid_precompute.py --song 04_DieYoung
python solenoid_precompute.py --song 04_DieYoung --offset -0.02
python solenoid_precompute.py --all
```

Use a small negative `--offset` to compensate for mechanical lag in the solenoid.

## Troubleshooting

**No Arduino found** — Connect the Arduino via USB. If auto-detection fails, set `ARDUINO_PORT` in `.env.local` to your serial port (e.g. `/dev/cu.usbmodem14201` on macOS, `COM3` on Windows).

**INVALID_CLIENT** — Spotify rejected the credentials. Try:
1. In Spotify Dashboard → your app → Settings → reset the Client Secret, then update `.env.local`
2. Delete the cache: `rm .spotify_cache`
3. Ensure the redirect URI in `.env.local` exactly matches the one in the Dashboard

**Missing drums stem** — The drums test and visualizer require cached stems. Run `python knob_mixer.py` once first, or run `python -m demucs -n htdemucs -o stems /path/to/song.mp3` manually.

**Missing solenoid_hits.json** — Run `python solenoid_precompute.py --song <name>` (or `--all`) before launching the visualizer.

**Solenoid fires late / early** — Use the `--offset` flag on `solenoid_precompute.py` (and/or `drums_visualizer.py`) to shift every hit by a constant number of seconds. Negative values fire earlier.

**Python 3.13** — If you see auth or torch issues, try Python 3.11.
