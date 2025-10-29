
import argparse
import json
import threading
import time
from pathlib import Path

from loguru import logger

from audio_engine.engine import AudioEngine
from api import create_app, run_api


DEFAULT_CONFIG = Path("config.json")


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cmd_devices(args):
    from audio_engine.audio_device import list_devices
    list_devices()


def cmd_test_tone(args):
    cfg = load_config(Path(args.config))
    eng = AudioEngine(cfg)
    eng.start()
    logger.info(f"Playing 1 kHz test tone on logical channel {args.channel} for {args.seconds}s")
    eng.play_test_tone(args.channel, duration=args.seconds)
    eng.stop()


def cmd_run(args):
    cfg = load_config(Path(args.config))
    eng = AudioEngine(cfg)
    eng.start()
    logger.info("Audio engine started")

    # Run API in a thread, sharing engine instance via app state
    app = create_app(eng)
    api_thread = threading.Thread(target=run_api, args=(app, args.host, args.port), daemon=True)
    api_thread.start()
    logger.info(f"HTTP API running at http://{args.host}:{args.port}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        eng.stop()


def main():
    parser = argparse.ArgumentParser(description="Mission Control Audio Engine")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.json")

    sub = parser.add_subparsers(dest="cmd")

    p_dev = sub.add_parser("devices", help="List audio devices")
    p_dev.set_defaults(func=cmd_devices)

    p_run = sub.add_parser("run", help="Run audio engine + HTTP API")
    p_run.add_argument("--host", default="0.0.0.0")
    p_run.add_argument("--port", type=int, default=8000)
    p_run.set_defaults(func=cmd_run)

    p_tone = sub.add_parser("test-tone", help="Play 1 kHz test tone on a channel")
    p_tone.add_argument("--channel", type=int, required=True)
    p_tone.add_argument("--seconds", type=float, default=3.0)
    p_tone.set_defaults(func=cmd_test_tone)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
