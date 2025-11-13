# ──────────────────────────────────────────────────────────────────────────────
# File: audio_engine/engine.py
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import numpy as np
import psutil
import sounddevice as sd
from loguru import logger

from .dsp import DSPChain
from .vu import VUMeter


@dataclass
class ChannelConfig:
    input_device_id: int
    output_device_id: int


def _device_exists(dev_id: int) -> bool:
    """Check if a device ID exists in sounddevice."""
    try:
        sd.query_devices(dev_id)
        return True
    except Exception:
        return False


def _check_stream_settings(dev_id: int, samplerate: int, channels: int, is_input: bool):
    """Use sounddevice's own checks to validate device supports our settings."""
    try:
        if is_input:
            sd.check_input_settings(
                device=dev_id,
                samplerate=samplerate,
                channels=channels,
                dtype="int16",
            )
        else:
            sd.check_output_settings(
                device=dev_id,
                samplerate=samplerate,
                channels=channels,
                dtype="int16",
            )
    except Exception as e:
        role = "input" if is_input else "output"
        raise ValueError(f"Unsupported {role} settings for device {dev_id}: {e}")


class AudioChannel:
    def __init__(self, chan_id: int, cfg: ChannelConfig, fs: int, blocksize: int, dsp_cfg: dict):
        self.chan_id = chan_id
        self.fs = fs
        self.blocksize = blocksize
        self.cfg = cfg
        self.mute = False
        self.gate_open = True
        self.xruns = 0

        # ~10 Hz VU with floor -60 dBFS (handled inside VUMeter)
        self.vu = VUMeter(window_s=0.1, fs=fs)
        self.dsp = DSPChain(dsp_cfg, fs)

        self._stream = sd.Stream(
            samplerate=fs,
            blocksize=blocksize,
            dtype="int16",
            channels=2,
            device=(cfg.input_device_id, cfg.output_device_id),
            callback=self._callback,
        )

    def _callback(self, indata, outdata, frames, time_info, status):
        # xrun tracking
        if (
            status.input_underflow
            or status.input_overflow
            or status.output_underflow
            or status.output_overflow
        ):
            self.xruns += 1

        # int16 -> float32 [-1, 1]
        x = indata.astype(np.float32) / 32768.0

        # loopback through DSP
        y = x
        if self.gate_open:
            y = self.dsp.process(y)
        if self.mute:
            y = np.zeros_like(y)

        # update VU
        self.vu.update(y)

        # float32 -> int16
        outdata[:] = np.clip(y * 32767.0, -32768, 32767).astype(np.int16)

    def start(self):
        self._stream.start()

    def stop(self):
        self._stream.stop()
        self._stream.close()


class AudioEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.fs = int(cfg.get("sample_rate", 44100))
        self.blocksize = int(cfg.get("blocksize", 512))
        self.channels: dict[int, AudioChannel] = {}
        self.start_time = None
        self._lock = threading.RLock()

        # build channels from logical_channels
        for k, v in cfg.get("logical_channels", {}).items():
            ch_id = int(k)
            ch = AudioChannel(
                ch_id,
                ChannelConfig(v["input_device_id"], v["output_device_id"]),
                fs=self.fs,
                blocksize=self.blocksize,
                dsp_cfg=cfg.get("dsp", {}),
            )
            self.channels[ch_id] = ch

    # ── Validation helpers ────────────────────────────────────────────────────
    def validate_config(self, cfg: dict):
        """Validate configuration WITHOUT touching running engine."""
        sr = int(cfg.get("sample_rate", 0))
        chs = int(cfg.get("channels", 0))
        fmt = str(cfg.get("sample_format", ""))
        bs = int(cfg.get("blocksize", 0))

        if sr <= 0:
            raise ValueError("sample_rate must be > 0")
        if chs not in (1, 2):
            raise ValueError("channels must be 1 or 2 (spec uses stereo=2)")
        if fmt != "int16":
            raise ValueError("sample_format must be 'int16'")
        if bs <= 0:
            raise ValueError("blocksize must be > 0")

        lch = cfg.get("logical_channels", {})
        if not lch or len(lch) != 4:
            raise ValueError("logical_channels must define exactly 4 entries (1..4)")

        # Check each device exists and supports given settings
        for key, pair in lch.items():
            try:
                inp = int(pair["input_device_id"])
                outp = int(pair["output_device_id"])
            except Exception:
                raise ValueError(
                    f"logical_channels[{key}] must contain integer input/output_device_id"
                )

            if not _device_exists(inp):
                raise ValueError(f"input_device_id {inp} does not exist")
            if not _device_exists(outp):
                raise ValueError(f"output_device_id {outp} does not exist")

            _check_stream_settings(inp, sr, chs, is_input=True)
            _check_stream_settings(outp, sr, chs, is_input=False)

        return True

    def channel_keys(self):
        """Return set of channel IDs as strings, used by API for validation."""
        return {str(k) for k in self.channels.keys()}

    def start(self):
        for ch in self.channels.values():
            ch.start()
        self.start_time = time.time()

    def stop(self):
        for ch in self.channels.values():
            ch.stop()

    # ── Public control ────────────────────────────────────────────────────────
    def play_test_tone(self, channel: int, duration: float = 3.0, freq: float = 1000.0):
        """Simple 1 kHz test tone generator on a given channel output."""
        ch = self.channels[channel]
        frames = int(duration * self.fs)
        t = (np.arange(frames) / self.fs).astype(np.float32)
        tone = 0.2 * np.sin(2 * np.pi * freq * t)

        def cb(indata, outdata, frames, time_info, status):
            idx = cb.idx
            end = min(idx + frames, tone.size)
            block = np.zeros((frames, 2), dtype=np.float32)
            if idx < tone.size:
                sl = tone[idx:end]
                block[: end - idx, 0] = sl
                block[: end - idx, 1] = sl
                cb.idx = end
            outdata[:] = np.clip(block * 32767.0, -32768, 32767).astype(np.int16)

        cb.idx = 0
        with sd.OutputStream(
            samplerate=self.fs,
            blocksize=self.blocksize,
            dtype="int16",
            channels=2,
            device=ch.cfg.output_device_id,
            callback=cb,
        ):
            while cb.idx < tone.size:
                time.sleep(0.05)

    def set_ptt(self, channel: int, *, mute: bool, gate_open: bool):
        ch = self.channels.get(channel)
        if not ch:
            raise ValueError(f"unknown channel {channel}")
        ch.mute = mute
        ch.gate_open = gate_open

    def reload_config(self, new_cfg: dict):
        """
        Validate the new config and only if valid,
        stop + rebuild + restart engine.
        """
        self.validate_config(new_cfg)
        with self._lock:
            self.stop()
            self.__init__(new_cfg)
            self.start()

    # ── Monitoring ────────────────────────────────────────────────────────────
    def get_status(self) -> dict:
        uptime = 0.0 if not self.start_time else time.time() - self.start_time
        return {
            "sample_rate": self.fs,
            "blocksize": self.blocksize,
            "channels": {
                str(k): {
                    "input_device_id": v.cfg.input_device_id,
                    "output_device_id": v.cfg.output_device_id,
                    "xruns": v.xruns,
                }
                for k, v in self.channels.items()
            },
            "cpu_percent": psutil.cpu_percent(interval=None),
            "uptime_s": round(uptime, 2),
        }

    def get_vu_levels(self) -> dict:
        # VUMeter itself keeps floor at -60 dBFS and ~10 Hz refresh
        return {str(k): self.channels[k].vu.value() for k in sorted(self.channels)}

    # ── DSP self-check for spec sanity ────────────────────────────────────────
    def self_check_dsp(self) -> dict:
        """
        Offline checks that do not touch hardware.

        - Compressor sanity:
          input:  -12 dBFS 1 kHz sine
          expected output ~ -16 dBFS with 2:1 compression above -20 dBFS.
        - Limiter sanity:
          input:  0 dBFS 1 kHz sine
          expected peak <= -3 dBFS.
        """
        fs = self.fs
        dsp = DSPChain(self.cfg.get("dsp", {}), fs)

        def rms_dbfs(x: np.ndarray) -> float:
            rms = float(np.sqrt(np.mean(np.square(x))) + 1e-12)
            return 20.0 * math.log10(rms)

        # 1) Compressor sanity check
        dur = 1.0
        t = np.arange(int(fs * dur)) / fs
        amp_in = 10 ** (-12.0 / 20.0)  # -12 dBFS
        x = (amp_in * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        x_st = np.column_stack([x, x])  # stereo
        y = dsp.comp.process(x_st)
        measured = rms_dbfs(np.mean(y, axis=1))
        expected = -16.0  # threshold -20 + (8 dB over)/2 = -16 dBFS
        compressor_ok = abs(measured - expected) <= 2.0  # ±2 dB tolerance

        # 2) Limiter sanity check
        x2 = np.sin(2 * np.pi * 1000 * t).astype(np.float32)  # ~0 dBFS
        x2_st = np.column_stack([x2, x2])
        y2 = dsp.lim.process(x2_st)
        peak = float(np.max(np.abs(y2)))
        peak_db = 20.0 * math.log10(max(peak, 1e-12))
        limiter_ok = peak_db <= -3.0 + 0.2  # 0.2 dB slack

        return {
            "compressor_rms_dbfs": round(measured, 2),
            "compressor_expected_dbfs": expected,
            "compressor_ok": compressor_ok,
            "limiter_peak_dbfs": round(peak_db, 2),
            "limiter_ok": limiter_ok,
        }
