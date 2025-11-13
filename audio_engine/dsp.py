
import numpy as np


class Compressor:
    def __init__(self, ratio=2.0, threshold_db=-20.0, attack_ms=10.0, release_ms=100.0, fs=44100):
        self.ratio = ratio
        self.threshold = 10 ** (threshold_db / 20.0)
        self.alpha_a = np.exp(-1.0 / (attack_ms * 0.001 * fs))
        self.alpha_r = np.exp(-1.0 / (release_ms * 0.001 * fs))
        self.env = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        # x: float32 [-1,1], shape (frames, channels)
        x_mono = np.mean(np.abs(x), axis=1)
        y = np.empty_like(x)
        for i, s in enumerate(x_mono):
            # envelope follower
            if s > self.env:
                self.env = self.alpha_a * self.env + (1 - self.alpha_a) * s
            else:
                self.env = self.alpha_r * self.env + (1 - self.alpha_r) * s
            gain = 1.0
            if self.env > self.threshold:
                over = self.env / self.threshold
                target = over ** (1.0 - 1.0 / self.ratio)
                gain = 1.0 / max(target, 1e-6)
            y[i] = x[i] * gain
        return y


class Limiter:
    def __init__(self, ceiling_db=-3.0):
        self.ceiling = 10 ** (ceiling_db / 20.0)

    def process(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, -self.ceiling, self.ceiling)


class DSPChain:
    def __init__(self, cfg: dict, fs: int):
        c = cfg.get("comp", {})
        l = cfg.get("limiter", {})
        self.comp = Compressor(
            ratio=c.get("ratio", 2.0),
            threshold_db=c.get("threshold_db", -20.0),
            attack_ms=c.get("attack_ms", 10.0),
            release_ms=c.get("release_ms", 100.0),
            fs=fs,
        )
        self.lim = Limiter(ceiling_db=l.get("ceiling_db", -3.0))

    def process(self, block: np.ndarray) -> np.ndarray:
        y = self.comp.process(block)
        y = self.lim.process(y)
        return y
