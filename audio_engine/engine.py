from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import numpy as np
import psutil
import sounddevice as sd


@dataclass
class ChannelConfig:
    input_device_id: int
    output_device_id: int


def _device_exists(dev_id: int) -> bool:
    try:
        sd.query_devices(dev_id)
        return True
    except Exception:
        return False


def _check_stream_settings(dev_id: int, samplerate: int, channels: int, is_input: bool) -> None:
    try:
        if is_input:
            sd.check_input_settings(device=dev_id, samplerate=samplerate, channels=channels, dtype="int16")
        else:
            sd.check_output_settings(device=dev_id, samplerate=samplerate, channels=channels, dtype="int16")
    except Exception as e:
        role = "input" if is_input else "output"
        raise ValueError(f"Unsupported {role} settings for device {dev_id}: {e}")


class VUMeter:
    """
    VU meter with ~10Hz refresh:
    - window_s = 0.1 => 100 ms RMS window
    - floor at -60 dBFS
    """
    def __init__(self, *, window_s: float, fs: int, floor_db: float = -60.0):
        self.fs = fs
        self.floor_db = float(floor_db)
        self.win = max(1, int(window_s * fs))
        self._buf = np.zeros((self.win, 2), dtype=np.float32)
        self._idx = 0
        self._filled = 0
        self._last_db = self.floor_db

    def update(self, stereo_f32: np.ndarray) -> None:
        # stereo_f32 shape: (frames, 2), float32 in [-1..1]
        n = stereo_f32.shape[0]
        for i in range(n):
            self._buf[self._idx] = stereo_f32[i]
            self._idx = (self._idx + 1) % self.win
            self._filled = min(self._filled + 1, self.win)

        if self._filled < self.win:
            self._last_db = self.floor_db
            return

        # RMS on summed stereo (or you can do per-channel; spec allows either)
        x = self._buf[:self.win]
        mono = 0.5 * (x[:, 0] + x[:, 1])
        rms = float(np.sqrt(np.mean(mono * mono)) + 1e-12)
        db = 20.0 * math.log10(rms)
        if db < self.floor_db:
            db = self.floor_db
        if db > 0.0:
            db = 0.0
        self._last_db = float(db)

    def value(self) -> float:
        return float(self._last_db)


class Compressor:
    """
    Simple envelope follower compressor:
    - threshold_db (e.g. -20 dBFS)
    - ratio (e.g. 2.0)
    - attack_ms (e.g. 10 ms)
    - release_ms (e.g. 100 ms)
    """
    def __init__(self, *, fs: int, ratio: float, threshold_db: float, attack_ms: float, release_ms: float):
        self.fs = fs
        self.ratio = float(ratio)
        self.threshold_db = float(threshold_db)
        self.attack_a = math.exp(-1.0 / (max(1e-6, attack_ms) * 0.001 * fs))
        self.release_a = math.exp(-1.0 / (max(1e-6, release_ms) * 0.001 * fs))
        self.env = 0.0  # linear envelope

    def process(self, x: np.ndarray) -> np.ndarray:
        # x: (frames, 2) float32
        y = np.empty_like(x)
        for i in range(x.shape[0]):
            s = 0.5 * (abs(float(x[i, 0])) + abs(float(x[i, 1])))
            if s > self.env:
                self.env = self.attack_a * self.env + (1.0 - self.attack_a) * s
            else:
                self.env = self.release_a * self.env + (1.0 - self.release_a) * s

            env_db = 20.0 * math.log10(max(self.env, 1e-12))
            if env_db <= self.threshold_db:
                gain_db = 0.0
            else:
                over_db = env_db - self.threshold_db
                compressed_over_db = over_db / self.ratio
                gain_db = compressed_over_db - over_db  # negative

            gain = 10.0 ** (gain_db / 20.0)
            y[i, 0] = x[i, 0] * gain
            y[i, 1] = x[i, 1] * gain
        return y


class Limiter:
    """
    Simple peak limiter:
    - ceiling_db (e.g. -3 dBFS)
    - release_ms (e.g. 8 ms)
    """
    def __init__(self, *, fs: int, ceiling_db: float, release_ms: float):
        self.fs = fs
        self.ceiling = 10.0 ** (float(ceiling_db) / 20.0)
        self.release_a = math.exp(-1.0 / (max(1e-6, release_ms) * 0.001 * fs))
        self.gain = 1.0

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        for i in range(x.shape[0]):
            peak = max(abs(float(x[i, 0])), abs(float(x[i, 1])), 1e-12)
            needed = self.ceiling / peak if peak > self.ceiling else 1.0
            # instant attack: drop gain immediately if needed
            if needed < self.gain:
                self.gain = needed
            else:
                # release back to 1.0
                self.gain = self.release_a * self.gain + (1.0 - self.release_a) * 1.0

            y[i, 0] = x[i, 0] * self.gain
            y[i, 1] = x[i, 1] * self.gain
        return y


class DSPChain:
    def __init__(self, dsp_cfg: dict, fs: int):
        comp_cfg = (dsp_cfg or {}).get("comp", {})
        lim_cfg = (dsp_cfg or {}).get("limiter", {})
        self.comp = Compressor(
            fs=fs,
            ratio=float(comp_cfg.get("ratio", 2.0)),
            threshold_db=float(comp_cfg.get("threshold_db", -20.0)),
            attack_ms=float(comp_cfg.get("attack_ms", 10.0)),
            release_ms=float(comp_cfg.get("release_ms", 100.0)),
        )
        self.lim = Limiter(
            fs=fs,
            ceiling_db=float(lim_cfg.get("ceiling_db", -3.0)),
            release_ms=float(lim_cfg.get("release_ms", 8.0)),
        )

    def process(self, x: np.ndarray) -> np.ndarray:
        y = self.comp.process(x)
        y = self.lim.process(y)
        return y


class AudioChannel:
    def __init__(self, chan_id: int, cfg: ChannelConfig, fs: int, blocksize: int, dsp_cfg: dict):
        self.chan_id = chan_id
        self.fs = fs
        self.blocksize = blocksize
        self.cfg = cfg
        self.mute = False
        self.gate_open = True
        self.xruns = 0

        self.vu = VUMeter(window_s=0.1, fs=fs, floor_db=-60.0)
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
        if status.input_underflow or status.input_overflow or status.output_underflow or status.output_overflow:
            self.xruns += 1

        x = indata.astype(np.float32) / 32768.0  # int16 -> float32
        y = x

        if self.gate_open:
            y = self.dsp.process(y)
        else:
            y = np.zeros_like(y)

        if self.mute:
            y = np.zeros_like(y)

        self.vu.update(y)

        outdata[:] = np.clip(y * 32767.0, -32768, 32767).astype(np.int16)

    def start(self):
        self._stream.start()

    def stop(self):
        try:
            self._stream.stop()
        finally:
            self._stream.close()


class AudioEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.fs = int(cfg.get("sample_rate", 44100))
        self.blocksize = int(cfg.get("blocksize", 512))
        self._lock = threading.RLock()
        self.start_time = None

        self.channels: dict[int, AudioChannel] = {}
        for k, v in cfg.get("logical_channels", {}).items():
            ch_id = int(k)
            ch = AudioChannel(
                ch_id,
                ChannelConfig(int(v["input_device_id"]), int(v["output_device_id"])),
                fs=self.fs,
                blocksize=self.blocksize,
                dsp_cfg=cfg.get("dsp", {}),
            )
            self.channels[ch_id] = ch

    # ---- validation (no side effects) ----
    def validate_config(self, cfg: dict) -> bool:
        sr = int(cfg.get("sample_rate", 0))
        chs = int(cfg.get("channels", 0))
        fmt = str(cfg.get("sample_format", ""))
        bs = int(cfg.get("blocksize", 0))

        if sr <= 0:
            raise ValueError("sample_rate must be > 0")
        if chs not in (1, 2):
            raise ValueError("channels must be 1 or 2")
        if fmt != "int16":
            raise ValueError("sample_format must be 'int16'")
        if bs <= 0:
            raise ValueError("blocksize must be > 0")

        lch = cfg.get("logical_channels", {})
        if not isinstance(lch, dict) or len(lch) != 4:
            raise ValueError("logical_channels must define exactly 4 entries (1..4)")

        for key, pair in lch.items():
            try:
                inp = int(pair["input_device_id"])
                outp = int(pair["output_device_id"])
            except Exception:
                raise ValueError(f"logical_channels[{key}] must contain integer input/output_device_id")

            if not _device_exists(inp):
                raise ValueError(f"input_device_id {inp} does not exist")
            if not _device_exists(outp):
                raise ValueError(f"output_device_id {outp} does not exist")

            _check_stream_settings(inp, sr, chs, is_input=True)
            _check_stream_settings(outp, sr, chs, is_input=False)

        return True

    def channel_keys(self):
        return {str(k) for k in self.channels.keys()}

    # ---- lifecycle ----
    def start(self):
        for ch in self.channels.values():
            ch.start()
        self.start_time = time.time()

    def stop(self):
        for ch in self.channels.values():
            ch.stop()

    def reload_config(self, new_cfg: dict):
        # validate before touching streams
        self.validate_config(new_cfg)
        with self._lock:
            self.stop()
            self.__init__(new_cfg)
            self.start()

    # ---- control ----
    def set_ptt(self, channel: int, *, mute: bool, gate_open: bool):
        ch = self.channels.get(int(channel))
        if not ch:
            raise ValueError(f"unknown channel {channel}")
        ch.mute = bool(mute)
        ch.gate_open = bool(gate_open)

    def play_test_tone(self, channel: int, duration: float = 3.0, freq: float = 1000.0):
        ch = self.channels[int(channel)]
        frames = int(duration * self.fs)
        t = (np.arange(frames) / self.fs).astype(np.float32)
        tone = 0.2 * np.sin(2 * np.pi * float(freq) * t)

        def cb(indata, outdata, f, time_info, status):
            idx = cb.idx
            end = min(idx + f, tone.size)
            block = np.zeros((f, 2), dtype=np.float32)
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

    # ---- monitoring ----
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
        return {str(k): self.channels[k].vu.value() for k in sorted(self.channels)}

    # ---- DSP sanity check ----
    def self_check_dsp(self) -> dict:
        fs = self.fs
        dsp = DSPChain(self.cfg.get("dsp", {}), fs)

        def rms_dbfs(x: np.ndarray) -> float:
            rms = float(np.sqrt(np.mean(x * x)) + 1e-12)
            return 20.0 * math.log10(rms)

        dur = 1.0
        t = np.arange(int(fs * dur)) / fs

        # Compressor check: -12 dBFS sine
        amp_in = 10 ** (-12.0 / 20.0)
        x = (amp_in * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        x_st = np.column_stack([x, x])
        y = dsp.comp.process(x_st)
        measured = rms_dbfs(np.mean(y, axis=1))
        expected = -16.0
        compressor_ok = abs(measured - expected) <= 3.0  # tolerate envelope differences

        # Limiter check: 0 dBFS sine
        x2 = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        x2_st = np.column_stack([x2, x2])
        y2 = dsp.lim.process(x2_st)
        peak = float(np.max(np.abs(y2)))
        peak_db = 20.0 * math.log10(max(peak, 1e-12))
        limiter_ok = peak_db <= (-3.0 + 0.2)

        return {
            "compressor_rms_dbfs": round(measured, 2),
            "compressor_expected_dbfs": expected,
            "compressor_ok": compressor_ok,
            "limiter_peak_dbfs": round(peak_db, 2),
            "limiter_ok": limiter_ok,
        }
