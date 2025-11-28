Mission Control Audio Engine (Python)

Audio I/O + DSP service for Mission Control.
Opens 4 duplex streams (44.1 kHz / 16-bit / stereo), processes each through a Compressor → Limiter chain, and exposes a control/monitoring API:

GET /status

GET /vu

POST /config

POST /ptt

GET /self-check

Designed for low-latency, fault-tolerant audio operations with external UI integration.

 Features

4 logical audio channels, each mapped to physical input/output devices

Real-time DSP chain per channel:

Compressor (2:1 ratio, −20 dB threshold, 10/100 ms attack/release)

Limiter (−3 dB ceiling)

VU meters @10 Hz (RMS → dBFS with −60 dBFS floor)

Engine status reporting: xruns, CPU usage, uptime

Hot-reloadable config.json with validation

Fully instrumented API with proper HTTP error handling

CI workflow (syntax checks) via GitHub Actions

 Quick start
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt

cp config.example.json config.json     # edit device IDs for your system
python main.py devices                 # list available input/output hardware
python main.py run                     # start the engine + API (http://localhost:8000)

# test: play a 1 kHz tone for 5 seconds on channel 1
python main.py test-tone --channel 1 --seconds 5

 HTTP API
GET /status

Returns:

sample rate, blocksize

logical→physical device map

xruns per channel

CPU usage

uptime

GET /vu

Returns current VU levels in dBFS with −60 dBFS floor.

POST /config

Hot-reloads configuration.
Rejects invalid configs with correct 400 / 500 responses.

POST /ptt

Controls per-channel mute/gate:

{ "channel": 1, "mute": false, "gate_open": true }

GET /self-check

Internal DSP test:

compressor RMS validation

limiter peak ceiling check

 Config schema (config.json)

 You must update real device IDs using:

python main.py devices


Example (valid for one local setup):

{
  "sample_rate": 44100,
  "sample_format": "int16",
  "channels": 2,
  "blocksize": 512,

  "logical_channels": {
    "1": {"input_device_id": 1, "output_device_id": 3},
    "2": {"input_device_id": 1, "output_device_id": 3},
    "3": {"input_device_id": 1, "output_device_id": 3},
    "4": {"input_device_id": 1, "output_device_id": 3}
  },

  "dsp": {
    "comp": {
      "ratio": 2.0,
      "threshold_db": -20.0,
      "attack_ms": 10.0,
      "release_ms": 100.0
    },
    "limiter": {
      "ceiling_db": -3.0,
      "release_ms": 8.0
    }
  }
}

Repository layout
mission_control_audio/
├─ main.py               # CLI runner + engine launcher
├─ api.py                # FastAPI HTTP interface
├─ requirements.txt
├─ README.md
├─ config.example.json
├─ .github/
│   └─ workflows/ci.yml  # Syntax CI
└─ audio_engine/
    ├─ __init__.py
    ├─ audio_device.py   # Device enumeration
    ├─ dsp.py            # Compressor + Limiter implementation
    ├─ engine.py         # Core audio engine + validation
    └─ vu.py             # RMS → dBFS computation

⭐ My Contributions (Jun / minoverse) — STAR Summary
Situation

When I joined, the audio engine lacked error handling, proper configuration validation, stable VU metering, DSP testing, and CI support.
Local audio was not functioning due to invalid device IDs and missing validation logic.

Task

I needed to:

Implement robust API error handling

Validate all configuration inputs before applying

Fix VU metering to 10 Hz with correct dBFS floor

Add DSP sanity checks for compressor & limiter

Make the engine run on local hardware

Add CI workflow

Document everything clearly

Action

I delivered the following improvements:

🔧 API Hardening

Replaced "ok": true with proper structured responses

Added HTTPException handling

Implemented correct 400 vs. 500 failures

Secured /config, /ptt, /status, and /self-check

🔍 Full Configuration Validation

Verified device IDs exist (sd.query_devices)

Checked sample rate, blocksize, sample format, channel count

Ensured devices support requested audio format

Prevented partial/invalid configs from being applied

🎚 Stable VU Meter

Implemented 100 ms RMS window → 10 Hz refresh

Applied −60 dBFS floor

Ensured stable, noise-free output

🎛 DSP Self-Check

Added offline test in /self-check

Compressor: expected −16 dBFS vs actual RMS

Limiter: peak ceiling ≤ −3 dBFS

Improved tolerance and reporting

🔌 Local Audio Debugging

Enumerated real devices on my machine

Fixed wrong config IDs

Mapped mic (1) → speakers (3)

Tested loopback, VU, status, test-tone generator

Solved initial crash from missing or invalid config.json

⚙️ CI Setup

Added .github/workflows/ci.yml

Performs syntax check (python -m compileall .) on every push

Result

Engine now runs stably on local hardware

API fully compliant with project requirements

DSP processing validated

VU metering correct and stable

Configuration safe and validated

CI prevents regressions

System ready for integration with UI/backend team

📄 License

MIT — see LICENSE.
