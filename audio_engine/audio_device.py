
from loguru import logger
import sounddevice as sd


def list_devices():
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    logger.info("=== Audio Devices ===")
    for idx, dev in enumerate(devices):
        name = dev.get("name")
        ins = dev.get("max_input_channels")
        outs = dev.get("max_output_channels")
        api = hostapis[dev.get("hostapi")]["name"] if dev.get("hostapi") is not None else "?"
        print(f"[{idx:2d}] {name} | in:{ins} out:{outs} | API:{api}")
