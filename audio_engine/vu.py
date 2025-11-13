
import numpy as np
import time


class VUMeter:
    """Keeps windowed RMS and emits dBFS at ~10 Hz."""

    def __init__(self, window_s=0.1, fs=44100):
        self.fs = fs
        self.window = max(1, int(window_s * fs))
        self.buf = np.zeros((self.window, 2), dtype=np.float32)
        self.pos = 0
        self.last_emit = 0.0
        self.dbfs = -60.0

    def update(self, block: np.ndarray):
        # block shape: (frames, channels)
        n = block.shape[0]
        if n >= self.window:
            slice_ = block[-self.window:]
            self.buf[:] = slice_
            self.pos = 0
        else:
            end = self.pos + n
            if end <= self.window:
                self.buf[self.pos:end] = block
            else:
                first = self.window - self.pos
                self.buf[self.pos:] = block[:first]
                self.buf[: n - first] = block[first:]
            self.pos = (self.pos + n) % self.window

        now = time.time()
        if now - self.last_emit >= 0.1:
            mono = np.mean(self.buf, axis=1)
            rms = float(np.sqrt(np.mean(mono * mono)) + 1e-12)
            db = 20.0 * np.log10(rms)
            self.dbfs = max(-60.0, min(0.0, db))
            self.last_emit = now

    def value(self) -> float:
        return self.dbfs
