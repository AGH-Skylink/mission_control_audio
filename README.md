
# Mission Control Audio Engine (Python)

Audio I/O + DSP service for Mission Control: opens 4 duplex streams (44.1 kHz / 16‑bit / stereo), runs **Compressor → Limiter** per channel, exposes **/status**, **/vu**, **/config**, **/ptt** via HTTP.

## Features
- 4 logical channels mapped to physical devices
- Real‑time DSP: Compressor (2:1, −20 dB, 10/100 ms), Limiter (−3 dB ceiling)
- VU meters @10 Hz (RMS → dBFS, −60..0 dB)
- JSON status: xruns, CPU, uptime
- Hot‑reloadable config (devices & DSP)

## Quick start
```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp config.example.json config.json             # edit device IDs for your machine
python main.py devices                         # list audio devices
python main.py run                             # run engine + API at http://localhost:8000
# sanity test: 1 kHz tone on channel 1
python main.py test-tone --channel 1 --seconds 5
```

## HTTP API
- `GET /status` → engine state, device map, xruns, uptime, CPU
- `GET /vu` → current VU dBFS per channel
- `POST /config` → hot-reload config (body: same schema as `config.json`)
- `POST /ptt` → `{ "channel": int, "mute": bool, "gate_open": bool }`

## Config schema (config.json)
```jsonc
{
  "sample_rate": 44100,
  "sample_format": "int16",
  "channels": 2,
  "blocksize": 512,
  "logical_channels": {
    "1": {"input_device_id": 1, "output_device_id": 3},
    "2": {"input_device_id": 2, "output_device_id": 4},
    "3": {"input_device_id": 5, "output_device_id": 6},
    "4": {"input_device_id": 7, "output_device_id": 8}
  },
  "dsp": {
    "comp": {"ratio": 2.0, "threshold_db": -20.0, "attack_ms": 10.0, "release_ms": 100.0},
    "limiter": {"ceiling_db": -3.0, "release_ms": 8.0}
  }
}
```

## Repo layout
```
mission_control_audio/
├─ main.py
├─ api.py
├─ requirements.txt
├─ README.md
├─ config.example.json
└─ audio_engine/
   ├─ __init__.py
   ├─ audio_device.py
   ├─ dsp.py
   ├─ engine.py
   └─ vu.py
```

## License
MIT — see `LICENSE`.
