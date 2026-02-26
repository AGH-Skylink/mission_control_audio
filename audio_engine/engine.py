# audio_engine/engine.py
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import psutil
import sounddevice as sd
from loguru import logger


# -----------------------------
# DSP helpers (float32)
# -----------------------------
def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def lin_to_db(x: float, eps: float = 1e-12) -> float:
    return 20.0 * math.log10(max(x, eps))


@dataclass
class CompressorParams:
    ratio: float = 2.0
    threshold_db: float = -20.0
    attack_ms: float = 10.0
    release_ms: float = 100.0


@dataclass
class LimiterParams:
    ceiling_db: float = -3.0
    release_ms: float = 8.0


class Compressor:
    """
    Simple feed-forward compressor:
    - level detector: abs + envelope follower
    - gain computer in dB with ratio above threshold
    """
    def __init__(self, params: CompressorParams, sample_rate: int):
        self.p = params
        self.sr = sample_rate
        self.env = 0.0  # linear envelope
        self._attack_a = self._coef(self.p.attack_ms)
        self._release_a = self._coef(self.p.release_ms)

    def _coef(self, ms: float) -> float:
        tau = max(ms, 0.1) / 1000.0
        return math.exp(-1.0 / (self.sr * tau))

    def reset(self):
        self.env = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        # x: float32 [-1..1], shape (frames, channels)
        if x.size == 0:
            return x

        th_lin = db_to_lin(self.p.threshold_db)

        y = np.empty_like(x)
        for i in range(x.shape[0]):
            # peak detector across channels (mono detector)
            level = float(np.max(np.abs(x[i, :])))
            a = self._attack_a if level > self.env else self._release_a
            self.env = a * self.env + (1.0 - a) * level

            # gain computer
            if self.env <= th_lin or self.env <= 1e-12:
                gain = 1.0
            else:
                env_db = lin_to_db(self.env)
                # output_db = th + (in_db - th)/ratio
                out_db = self.p.threshold_db + (env_db - self.p.threshold_db) / max(self.p.ratio, 1e-6)
                gain_db = out_db - env_db
                gain = db_to_lin(gain_db)

            y[i, :] = x[i, :] * gain

        return y


class Limiter:
    """
    Simple peak limiter:
    - detects peak per sample across channels
    - applies instantaneous gain reduction if above ceiling
    - release via envelope follower
    """
    def __init__(self, params: LimiterParams, sample_rate: int):
        self.p = params
        self.sr = sample_rate
        self.g = 1.0
        self._release_a = self._coef(self.p.release_ms)
        self.ceiling = db_to_lin(self.p.ceiling_db)

    def _coef(self, ms: float) -> float:
        tau = max(ms, 0.1) / 1000.0
        return math.exp(-1.0 / (self.sr * tau))

    def reset(self):
        self.g = 1.0

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x

        y = np.empty_like(x)
        for i in range(x.shape[0]):
            peak = float(np.max(np.abs(x[i, :])))
            if peak > self.ceiling and peak > 1e-12:
                target = self.ceiling / peak
                self.g = min(self.g, target)  # instant attack
            else:
                # release back toward 1.0
                self.g = self._release_a * self.g + (1.0 - self._release_a) * 1.0

            y[i, :] = x[i, :] * self.g

        return y


# -----------------------------
# VU meter helper
# -----------------------------
class VUAccumulator:
    """Accumulate RMS for ~100ms window, then convert to dBFS."""
    def __init__(self):
        self.sumsq = 0.0
        self.count = 0

    def add_block(self, x_float: np.ndarray):
        # x_float: float32 [-1..1]
        if x_float.size == 0:
            return
        # mono RMS across both channels
        mono = np.mean(x_float, axis=1)
        self.sumsq += float(np.sum(mono * mono))
        self.count += int(mono.shape[0])

    def consume_dbfs(self) -> float:
        if self.count <= 0:
            return -60.0
        rms = math.sqrt(self.sumsq / max(self.count, 1))
        # Convert to dBFS relative to full-scale=1.0
        dbfs = 20.0 * math.log10(max(rms, 1e-12))
        # Floor at -60 dBFS, cap at 0
        dbfs = max(-60.0, min(0.0, dbfs))
        # reset
        self.sumsq = 0.0
        self.count = 0
        return float(dbfs)


# -----------------------------
# Audio channel + engine
# -----------------------------
class AudioChannel:
    def __init__(
        self,
        engine: "AudioEngine",
        channel_id: str,
        input_device_id: int,
        output_device_id: int,
        sample_rate: int,
        blocksize: int,
        channels: int,
        dsp_comp: CompressorParams,
        dsp_lim: LimiterParams,
    ):
        self.engine = engine
        self.channel_id = str(channel_id)
        self.in_dev = int(input_device_id)
        self.out_dev = int(output_device_id)
        self.sr = int(sample_rate)
        self.blocksize = int(blocksize)
        self.nch = int(channels)

        self.xruns = 0
        self.mute = False
        self.gate_open = True

        self._lock = threading.Lock()

        self._comp = Compressor(dsp_comp, self.sr)
        self._lim = Limiter(dsp_lim, self.sr)
        self._vu = VUAccumulator()

        # test tone state
        self._tone_enabled = False
        self._tone_freq = 1000.0
        self._tone_phase = 0.0

        self._stream: Optional[sd.Stream] = None

    def set_ptt(self, mute: bool, gate_open: bool):
        with self._lock:
            self.mute = bool(mute)
            self.gate_open = bool(gate_open)

    def enable_tone(self, enable: bool):
        with self._lock:
            self._tone_enabled = bool(enable)

    def _callback(self, indata, outdata, frames, time_info, status: sd.CallbackFlags):
        # Real-time callback: keep it lightweight
        if status:
            # any under/over flow counts as xrun
            self.xruns += 1

        # timing hook for jitter measurement (Phase4)
        self.engine._rt_hook_on_callback_start(self.channel_id, time.perf_counter())

        with self._lock:
            mute = self.mute
            gate_open = self.gate_open
            tone = self._tone_enabled

        if tone:
            # generate tone, ignore input
            t = (np.arange(frames, dtype=np.float32) + 0.0) / self.sr
            phase = self._tone_phase
            y = np.sin(2.0 * math.pi * self._tone_freq * t + phase).astype(np.float32)
            self._tone_phase = float((phase + 2.0 * math.pi * self._tone_freq * (frames / self.sr)) % (2.0 * math.pi))
            y = (0.2 * y).reshape(-1, 1)  # -14 dBFS-ish
            if self.nch == 2:
                y = np.repeat(y, 2, axis=1)
            out_float = y
        else:
            if indata is None:
                outdata.fill(0)
                return

            # convert int16 -> float32
            x = indata.astype(np.float32) / 32768.0

            if mute or (not gate_open):
                out_float = np.zeros_like(x)
            else:
                # DSP chain
                y = self._comp.process(x)
                y = self._lim.process(y)
                out_float = y

        # VU accumulation @ callback rate; published at 10Hz by engine thread
        self._vu.add_block(out_float)

        # float32 -> int16
        out = np.clip(out_float, -1.0, 0.9999695)
        out_i16 = (out * 32768.0).astype(np.int16)
        outdata[:] = out_i16

    def open(self):
        logger.info(f"[ch{self.channel_id}] opening stream in={self.in_dev} out={self.out_dev}")
        self._stream = sd.Stream(
            device=(self.in_dev, self.out_dev),
            samplerate=self.sr,
            blocksize=self.blocksize,
            dtype="int16",
            channels=self.nch,
            callback=self._callback,
        )
        self._stream.start()
        logger.info(f"[ch{self.channel_id}] stream started")

    def close(self):
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def consume_vu_dbfs(self) -> float:
        return self._vu.consume_dbfs()


class AudioEngine:
    def __init__(self, cfg: Dict[str, Any]):
        self._cfg = cfg
        self._proc = psutil.Process()

        self.sample_rate = int(cfg.get("sample_rate", 44100))
        self.sample_format = str(cfg.get("sample_format", "int16"))
        self.channels = int(cfg.get("channels", 2))
        self.blocksize = int(cfg.get("blocksize", 512))

        dsp_cfg = cfg.get("dsp", {}) or {}
        comp_cfg = (dsp_cfg.get("comp", {}) or {})
        lim_cfg = (dsp_cfg.get("limiter", {}) or {})

        self.comp_params = CompressorParams(
            ratio=float(comp_cfg.get("ratio", 2.0)),
            threshold_db=float(comp_cfg.get("threshold_db", -20.0)),
            attack_ms=float(comp_cfg.get("attack_ms", 10.0)),
            release_ms=float(comp_cfg.get("release_ms", 100.0)),
        )
        self.lim_params = LimiterParams(
            ceiling_db=float(lim_cfg.get("ceiling_db", -3.0)),
            release_ms=float(lim_cfg.get("release_ms", 8.0)),
        )

        self._channels: Dict[str, AudioChannel] = {}
        self._vu_levels: Dict[str, float] = {str(i): -60.0 for i in range(1, 5)}

        self._start_time = None
        self._running = False

        # callback jitter stats (Phase4)
        self._cb_lock = threading.Lock()
        self._cb_times: Dict[str, list[float]] = {str(i): [] for i in range(1, 5)}

        # publisher thread for VU @ 10Hz
        self._vu_thread: Optional[threading.Thread] = None
        self._vu_stop = threading.Event()

        # build channels from cfg
        logical = cfg.get("logical_channels", {}) or {}
        for k in ["1", "2", "3", "4"]:
            ch_cfg = logical.get(k)
            if not ch_cfg:
                continue
            self._channels[k] = AudioChannel(
                engine=self,
                channel_id=k,
                input_device_id=int(ch_cfg["input_device_id"]),
                output_device_id=int(ch_cfg["output_device_id"]),
                sample_rate=self.sample_rate,
                blocksize=self.blocksize,
                channels=self.channels,
                dsp_comp=self.comp_params,
                dsp_lim=self.lim_params,
            )

        # prime cpu_percent (first call returns 0)
        try:
            self._proc.cpu_percent(interval=None)
        except Exception:
            pass

    # -------------------------
    # Config validation
    # -------------------------
    def validate_config(self, cfg: Dict[str, Any]) -> None:
        # basic checks
        sr = int(cfg.get("sample_rate", 0))
        bs = int(cfg.get("blocksize", 0))
        chn = int(cfg.get("channels", 0))
        fmt = str(cfg.get("sample_format", ""))

        if sr <= 0:
            raise ValueError("sample_rate must be > 0")
        if bs <= 0:
            raise ValueError("blocksize must be > 0")
        if chn not in (1, 2):
            raise ValueError("channels must be 1 or 2")
        if fmt != "int16":
            raise ValueError("sample_format must be 'int16'")

        logical = cfg.get("logical_channels", {})
        if not isinstance(logical, dict):
            raise ValueError("logical_channels must be a dict")
        if set(logical.keys()) != {"1", "2", "3", "4"}:
            raise ValueError("logical_channels must define exactly keys: '1','2','3','4'")

        # device existence + capability checks
        for k, v in logical.items():
            in_id = int(v.get("input_device_id", -1))
            out_id = int(v.get("output_device_id", -1))

            try:
                sd.query_devices(in_id)
            except Exception as e:
                raise ValueError(f"channel {k}: input_device_id {in_id} not found ({e})")
            try:
                sd.query_devices(out_id)
            except Exception as e:
                raise ValueError(f"channel {k}: output_device_id {out_id} not found ({e})")

            # supported settings
            try:
                sd.check_input_settings(device=in_id, samplerate=sr, channels=chn, dtype="int16")
            except Exception as e:
                raise ValueError(f"channel {k}: input device {in_id} does not support sr={sr}, ch={chn}, int16 ({e})")

            try:
                sd.check_output_settings(device=out_id, samplerate=sr, channels=chn, dtype="int16")
            except Exception as e:
                raise ValueError(f"channel {k}: output device {out_id} does not support sr={sr}, ch={chn}, int16 ({e})")

        # DSP params sanity (optional but helpful)
        dsp = cfg.get("dsp", {}) or {}
        comp = (dsp.get("comp", {}) or {})
        lim = (dsp.get("limiter", {}) or {})

        ratio = float(comp.get("ratio", 2.0))
        if ratio <= 1.0:
            raise ValueError("dsp.comp.ratio must be > 1.0")
        ceiling = float(lim.get("ceiling_db", -3.0))
        if ceiling > 0.0:
            raise ValueError("dsp.limiter.ceiling_db must be <= 0.0")

    def reload_config(self, cfg: Dict[str, Any]) -> None:
        # assume validate_config already called by API; still safe to call again
        self.validate_config(cfg)

        was_running = self._running
        if was_running:
            self.stop()

        # re-init state
        self.__init__(cfg)

        if was_running:
            self.start()

    # -------------------------
    # Engine control
    # -------------------------
    def start(self):
        if self._running:
            return
        if len(self._channels) != 4:
            raise RuntimeError("Engine requires 4 logical channels configured")

        self._start_time = time.time()
        self._running = True
        self._vu_stop.clear()

        # open streams
        for ch in self._channels.values():
            ch.open()

        # start VU thread @ 10Hz
        self._vu_thread = threading.Thread(target=self._vu_publisher_loop, daemon=True)
        self._vu_thread.start()

        logger.info("AudioEngine started (4 streams)")

    def stop(self):
        if not self._running:
            return
        self._running = False
        self._vu_stop.set()

        # close streams
        for ch in self._channels.values():
            ch.close()

        logger.info("AudioEngine stopped")

    def channel_keys(self):
        return list(self._channels.keys())

    def set_ptt(self, channel: int, mute: bool, gate_open: bool):
        ch_id = str(channel)
        ch = self._channels.get(ch_id)
        if not ch:
            raise ValueError(f"unknown channel {channel}")
        ch.set_ptt(mute=mute, gate_open=gate_open)

    def play_test_tone(self, channel: int, duration: float = 3.0):
        ch_id = str(channel)
        ch = self._channels.get(ch_id)
        if not ch:
            raise ValueError(f"unknown channel {channel}")

        def _run():
            ch.enable_tone(True)
            time.sleep(max(0.0, float(duration)))
            ch.enable_tone(False)

        threading.Thread(target=_run, daemon=True).start()

    # -------------------------
    # VU + status
    # -------------------------
    def _vu_publisher_loop(self):
        # publish every 100ms (~10Hz)
        while not self._vu_stop.is_set():
            for k, ch in self._channels.items():
                try:
                    self._vu_levels[k] = ch.consume_vu_dbfs()
                except Exception:
                    self._vu_levels[k] = -60.0
            time.sleep(0.1)

    def get_vu_levels(self) -> Dict[str, float]:
        # stable 10Hz updated values
        return {k: float(v) for k, v in self._vu_levels.items()}

    def get_status(self) -> Dict[str, Any]:
        uptime = 0.0
        if self._start_time is not None:
            uptime = time.time() - self._start_time

        # CPU % (non-blocking)
        try:
            cpu = float(self._proc.cpu_percent(interval=None))
        except Exception:
            cpu = 0.0

        ch_status: Dict[str, Any] = {}
        for k, ch in self._channels.items():
            ch_status[k] = {
                "input_device_id": ch.in_dev,
                "output_device_id": ch.out_dev,
                "xruns": int(ch.xruns),
            }

        return {
            "sample_rate": self.sample_rate,
            "blocksize": self.blocksize,
            "channels": ch_status,
            "cpu_percent": cpu,
            "uptime_s": round(uptime, 2),
        }

    # -------------------------
    # DSP self-check
    # -------------------------
    def self_check_dsp(self) -> Dict[str, Any]:
        sr = self.sample_rate
        n = sr  # 1 sec
        t = np.arange(n, dtype=np.float32) / float(sr)

        # --- compressor check: -12 dBFS sine, expect ~ -16 dBFS RMS after 2:1 above -20 dB
        x_amp = db_to_lin(-12.0)
        x = (x_amp * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
        x = x.reshape(-1, 1)
        if self.channels == 2:
            x = np.repeat(x, 2, axis=1)

        comp = Compressor(self.comp_params, sr)
        y = comp.process(x)

        # RMS in dBFS (FS=1.0)
        rms = float(np.sqrt(np.mean((np.mean(y, axis=1)) ** 2) + 1e-12))
        comp_rms_db = 20.0 * math.log10(max(rms, 1e-12))
        expected_db = -20.0 + ((-12.0 - (-20.0)) / max(self.comp_params.ratio, 1e-6))  # -16 for 2:1
        comp_ok = abs(comp_rms_db - expected_db) <= 2.5  # tolerance

        # --- limiter check: 0 dBFS sine -> peak <= ceiling_db
        x2_amp = db_to_lin(0.0)
        x2 = (x2_amp * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
        x2 = x2.reshape(-1, 1)
        if self.channels == 2:
            x2 = np.repeat(x2, 2, axis=1)

        lim = Limiter(self.lim_params, sr)
        y2 = lim.process(x2)
        peak = float(np.max(np.abs(y2)) + 1e-12)
        peak_db = 20.0 * math.log10(peak)
        limiter_ok = peak_db <= (self.lim_params.ceiling_db + 0.2)  # small tolerance

        return {
            "compressor_rms_dbfs": round(comp_rms_db, 2),
            "compressor_expected_dbfs": round(expected_db, 2),
            "compressor_ok": bool(comp_ok),
            "limiter_peak_dbfs": round(peak_db, 2),
            "limiter_ok": bool(limiter_ok),
        }

    # -------------------------
    # Phase4: callback jitter hooks
    # -------------------------
    def _rt_hook_on_callback_start(self, channel_id: str, t: float):
        with self._cb_lock:
            buf = self._cb_times.get(str(channel_id))
            if buf is None:
                return
            buf.append(float(t))
            # keep recent
            if len(buf) > 10000:
                del buf[:2000]

    def get_callback_jitter(self) -> Dict[str, Any]:
        import statistics as stats

        out: Dict[str, Any] = {}
        with self._cb_lock:
            for ch, ts in self._cb_times.items():
                if len(ts) < 5:
                    out[ch] = {"n": len(ts)}
                    continue
                dt = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
                dt_sorted = sorted(dt)
                p95 = dt_sorted[int(0.95 * len(dt_sorted)) - 1]
                out[ch] = {
                    "n": len(dt),
                    "mean_ms": round(stats.mean(dt) * 1000.0, 4),
                    "p95_ms": round(p95 * 1000.0, 4),
                    "max_ms": round(max(dt) * 1000.0, 4),
                    "jitter_std_ms": round(stats.pstdev(dt) * 1000.0, 4),
                }
        return out
