import sounddevice as sd
from core.logger import monitor

def list_devices():
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    device_list = []

    for idx, dev in enumerate(devices):
        name = dev.get("name")
        ins = dev.get("max_input_channels")
        outs = dev.get("max_output_channels")
        api = hostapis[dev.get("hostapi")]["name"] if dev.get("hostapi") is not None else "?"
        srate = dev.get("default_sample_rate")
        info = {
            "id": idx,
            "name": name,
            "in":ins,
            "out": outs,
            "API": api,
            "default_sr": srate
        }
        device_list.append(info)

    monitor.log_event("AUDIO_DEVICE_SCAN", {"devices": device_list},
                      message=f"Detected {len(devices)} audio devices on host system")
