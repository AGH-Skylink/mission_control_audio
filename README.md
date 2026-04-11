Mission Control Audio Engine (Python)

Audio I/O + DSP service for Mission Control.
The engine opens 4 duplex audio streams (44.1 kHz / 16-bit / stereo), processes each channel through a Compressor → Limiter DSP chain, and exposes monitoring and control via a small HTTP API.

Features

4 logical audio channels mapped to physical input/output devices

Real-time DSP per channel:

Compressor (2:1 ratio, −20 dBFS threshold, 10 ms attack / 100 ms release)

Limiter (−3 dBFS ceiling)

Stable VU meters @ ~10 Hz (RMS → dBFS, −60 dBFS floor)

JSON status reporting (xruns, CPU usage, uptime)

Hot-reloadable configuration with validation

Proper API error handling (HTTP 400 / 500)

DSP sanity self-checks

CI workflow for syntax safety

Quick Start
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt

cp config.example.json config.json
python main.py devices
python main.py run    # API at http://localhost:8000


Test audio path (1 kHz tone):

python main.py test-tone --channel 1 --seconds 5

HTTP API
Method	Endpoint	Description
GET	/status	Engine state, device map, xruns, CPU, uptime
GET	/vu	Current VU levels (dBFS)
GET	/self-check	DSP compressor & limiter sanity check
POST	/config	Hot-reload validated configuration
POST	/ptt	{ channel, mute, gate_open }
🛠 Configuration (config.json)

Device IDs must match your local machine.

List devices using:

python main.py devices


Example (local Windows test):

{
  "sample_rate": 44100,
  "sample_format": "int16",
  "channels": 2,
  "blocksize": 512,

  "logical_channels": {
    "1": { "input_device_id": 1, "output_device_id": 3 },
    "2": { "input_device_id": 1, "output_device_id": 3 },
    "3": { "input_device_id": 1, "output_device_id": 3 },
    "4": { "input_device_id": 1, "output_device_id": 3 }
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

Repository Layout
mission_control_audio/
├─ main.py               # CLI + engine launcher
├─ api.py                # FastAPI HTTP interface
├─ requirements.txt
├─ README.md
├─ config.example.json
├─ .github/workflows/
│  └─ ci.yml             # CI: syntax checks
└─ audio_engine/
   ├─ audio_device.py    # Device enumeration
   ├─ dsp.py             # Compressor & limiter
   ├─ engine.py          # Core engine + validation
   └─ vu.py              # RMS → dBFS calculation

My Work — Problems Faced & Solved (STAR Style)

This section documents the concrete engineering problems I encountered and how I solved them.

1️⃣ API Error Handling Improvements (api.py)
Problem

The API always returned:

{"ok": true}


even when requests failed. No HTTP status codes were used, making failures hard to detect.

Solution

Replaced all {"ok": true} responses with:

{"status": "ok"}


Added proper HTTPException handling:

400 → invalid config, unknown channel

500 → internal engine errors

Added strict error handling for:

/config

/ptt

/self-check

Result

The API is now predictable, debuggable, and follows REST semantics.

2️⃣ Full Configuration Validation (audio_engine/engine.py)
Problem

Invalid configs were accepted.

Wrong device IDs caused runtime crashes.

Unsupported sample rates or channel counts broke the engine.

Solution

Implemented validate_config() which verifies:

sample_rate > 0

blocksize > 0

channels ∈ {1, 2}

sample_format == "int16"

Exactly 4 logical channels are defined

Each input_device_id / output_device_id:

Exists (sd.query_devices)

Supports requested sample rate & channels
(sd.check_input_settings, sd.check_output_settings)

Result

Invalid configs are rejected before touching the running engine.
No partial restarts, no silent failures.

3️⃣ VU Meter Stabilization (10 Hz, −60 dBFS Floor)
Problem

VU readings were unstable and sometimes dropped below the required −60 dBFS floor.

Solution

Used RMS over a 100 ms window → ~10 Hz update rate

Applied strict −60 dBFS floor

Normalized values correctly to dBFS

Result

VU output is stable, readable, and meets the specification.

Example:

{"1": -60.0, "2": -60.0, "3": -60.0, "4": -60.0}

4️⃣ DSP Sanity Checks (Compressor & Limiter)
Problem

There was no automated way to verify DSP correctness.

Solution

Implemented engine.self_check_dsp():

Compressor

Input: −12 dBFS sine

Expected output: ~−16 dBFS (2:1 above −20 dBFS)

±2–3 dB tolerance

Limiter

Input: 0 dBFS sine

Verified peak ≤ −3 dBFS

Result

DSP behavior can be verified programmatically without external hardware.

5️⃣ Local Audio Pipeline Testing
Problem

Local testing failed due to invalid configs and incorrect device IDs.

Solution

Enumerated devices with python main.py devices

Selected valid hardware:

Input: device 1 (Microphone Array)

Output: device 3 (Realtek Speakers)

Updated config.json

Verified:

/status

/vu

/self-check

Real-time loopback

Test-tone playback

Result

The engine runs correctly on local hardware.

6️⃣ GitHub CI (Syntax & Safety Checks)
Problem

No automated checks → easy to break the main branch.

Solution

Added .github/workflows/ci.yml:

Python 3.10

Installs safe dependencies

Runs:

python -m compileall .

Result

Syntax errors are caught automatically on every push and pull request.

📄License

MIT — see LICENSE.
