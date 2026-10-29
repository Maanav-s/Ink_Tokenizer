"""Digitizer artefacts: what the capture device does to the pen trajectory.

Pens and tablets don't record the ideal trajectory. A `Device` draws one
capture profile per page seed and applies it to each timed stroke:

- sample rate: 60 to 200 Hz;
- pen-down and pen-up hooks: the first or last one or two samples jump off
  the stroke by about a millimetre, as the tip skids on contact and lift-off;
- dropped samples: isolated ones, and occasional bursts (lost packets);
- spatial quantization: coordinates snapped to the device's resolution;
- timestamp jitter, keeping timestamps strictly increasing.

Level 0 is an ideal device: 100 Hz and no artefacts, so a clean render stays
clean. The profile is seeded separately from the handwriting noise, so it
doesn't change the strokes themselves.
"""
import math
import random
import zlib


class Device:
    def __init__(self, seed, level):
        self.level = level
        r = random.Random(zlib.crc32(f"{seed}:device".encode()))
        self.rng = random.Random(zlib.crc32(f"{seed}:device:samples".encode()))
        if level <= 0:
            self.rate, self.hook_p, self.hook_mm, self.drop_p, self.burst_p, self.quant, self.jitter = \
                100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            return
        self.rate = r.choice([60.0, 90.0, 100.0, 120.0, 133.0, 200.0])
        self.hook_p = min(1.0, level * r.uniform(0.1, 0.6))   # per stroke end
        self.hook_mm = level * r.uniform(0.4, 1.4)
        self.drop_p = level * r.uniform(0.0, 0.03)            # per interior sample
        self.burst_p = level * r.uniform(0.0, 0.01)           # per interior sample
        self.quant = r.choice([0.0, 0.05, 0.1, 0.2])          # mm
        self.jitter = level * r.choice([0.0, 0.001, 0.002])   # s

    def profile(self):
        return {"sample_rate_hz": self.rate, "hook_prob": round(self.hook_p, 3),
                "hook_mm": round(self.hook_mm, 2), "drop_prob": round(self.drop_p, 4),
                "burst_prob": round(self.burst_p, 4), "quantization_mm": self.quant,
                "time_jitter_s": self.jitter}

    def process(self, pts):
        """pts: [(x, y, t)] for one stroke, sampled at self.rate."""
        if self.level <= 0 or len(pts) < 2:
            return pts
        r = self.rng
        pts = list(pts)
        # Hooks: displace the first or last samples, the second one less.
        for end in (0, -1):
            if r.random() < self.hook_p and len(pts) >= 4:
                a = r.uniform(0, 2 * math.pi)
                d = self.hook_mm * r.uniform(0.5, 1.0)
                hx, hy = d * math.cos(a), d * math.sin(a)
                idx = (0, 1) if end == 0 else (-1, -2)
                for k, f in zip(idx, (1.0, 0.4)):
                    x, y, t = pts[k]
                    pts[k] = (x + f * hx, y + f * hy, t)
        # Dropped samples, never the first or last.
        if len(pts) > 4:
            keep = [True] * len(pts)
            i = 1
            while i < len(pts) - 1:
                if r.random() < self.burst_p:
                    for k in range(i, min(len(pts) - 1, i + r.randint(2, 6))):
                        keep[k] = False
                    i += 6
                elif r.random() < self.drop_p:
                    keep[i] = False
                i += 1
            pts = [p for p, k in zip(pts, keep) if k]
        # Quantization and timestamp jitter.
        out = []
        for i, (x, y, t) in enumerate(pts):
            if self.quant:
                x, y = round(x / self.quant) * self.quant, round(y / self.quant) * self.quant
            if self.jitter and 0 < i < len(pts) - 1:
                t += r.uniform(-self.jitter, self.jitter)
            if out and t < out[-1][2] + 0.002:
                t = out[-1][2] + 0.002
            out.append((x, y, t))
        return out
