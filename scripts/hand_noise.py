"""Handwriting-like variation for the naive ink renderer.

Real handwriting is not clean geometry plus independent per-point noise. Its
randomness is structured and correlated at several scales, so this module
models it in layers, each smooth and each scaled by one `level` knob
(0 = clean geometry and constant pen speed, 1 = default amount):

- Writer: a persistent style shared by everything that writer writes: slant,
  letter size, pen speed.
- Line or element: the baseline tilts and curves slightly, and the whole
  element sits a little off its layout position.
- Character: each glyph varies a little in size, rotation, baseline offset
  and spacing.
- Stroke:
  - a smooth random displacement field warps shapes ("elastic distortion",
    Simard et al. 2003), applied to point coordinates rather than pixels;
  - a slow, low-amplitude wobble runs along the stroke;
  - endpoints land off target, and pen-down and pen-up overshoot or stop
    short along the tangent;
  - straight diagram lines bow slightly;
  - circles come out elliptical and don't close exactly.
- Kinematics: the pen speeds up at the start of a stroke and slows at its end
  and in tight curves (roughly the two-thirds power law), so the T channel
  resembles real pen dynamics. Pauses between strokes vary randomly.

Every random draw comes from one seeded random.Random, so a (seed, level)
pair always produces the same page.
"""
import math
import random
import zlib


class Smooth1D:
    """Smooth random function of arc length: random knots, cosine-interpolated."""

    def __init__(self, rng, wavelength, amp):
        self.rng, self.lam, self.amp, self.knots = rng, wavelength, amp, []

    def __call__(self, s):
        i = int(s / self.lam)
        while len(self.knots) < i + 2:
            self.knots.append(self.rng.gauss(0, self.amp))
        f = (1 - math.cos(math.pi * (s / self.lam - i))) / 2
        return self.knots[i] * (1 - f) + self.knots[i + 1] * f


class Field2D:
    """Smooth random displacement field: a sum of random plane waves."""

    def __init__(self, rng, wavelength, amp, modes=6):
        self.waves = []
        for _ in range(modes):
            ang = rng.uniform(0, 2 * math.pi)
            k = 2 * math.pi / (wavelength * rng.uniform(0.7, 1.4))
            # Random direction for the displacement each wave produces.
            d = rng.uniform(0, 2 * math.pi)
            a = rng.gauss(0, amp / math.sqrt(modes / 2))
            self.waves.append((k * math.cos(ang), k * math.sin(ang), rng.uniform(0, 2 * math.pi),
                               a * math.cos(d), a * math.sin(d)))

    def __call__(self, x, y):
        dx = dy = 0.0
        for kx, ky, ph, ax, ay in self.waves:
            v = math.sin(kx * x + ky * y + ph)
            dx += ax * v
            dy += ay * v
        return dx, dy


def densify(points, step):
    out = [points[0]]
    for a, b in zip(points, points[1:]):
        n = max(1, math.ceil(math.dist(a, b) / step))
        out += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(1, n + 1)]
    return out


def arc_lengths(points):
    s = [0.0]
    for a, b in zip(points, points[1:]):
        s.append(s[-1] + math.dist(a, b))
    return s


def unit(a, b):
    d = math.dist(a, b)
    return ((b[0] - a[0]) / d, (b[1] - a[1]) / d) if d > 1e-9 else (1.0, 0.0)


class Hand:
    """Noise model for one page. Writer styles are drawn lazily per writer id."""

    def __init__(self, seed, level):
        self.seed, self.level = seed, level
        self.rng = random.Random(seed)
        self.writers = {}

    def g(self, mu, sd):
        """Gaussian scaled toward its mean by the noise level."""
        return mu + self.level * self.rng.gauss(0, sd)

    # -------------------------------------------------------------- writer

    def writer(self, wid):
        if wid not in self.writers:
            r = random.Random(zlib.crc32(f"{self.seed}:{wid}".encode()))
            lv = self.level
            self.writers[wid] = {
                "slant": lv * r.gauss(0.12, 0.08),         # shear, + leans right
                "size": 1 + lv * r.gauss(0, 0.04),
                "speed": 150.0 * math.exp(lv * r.gauss(0, 0.2)),  # mm/s
                "tremor": lv * r.uniform(0.04, 0.12),      # mm
            }
        return self.writers[wid]

    # ---------------------------------------------------------------- text

    def text_line(self, wid, h, length):
        """Per-line placement: offset, baseline tilt and curvature."""
        return {
            "dx": self.g(0, 0.04 * h), "dy": self.g(0, 0.04 * h),
            "tilt": self.g(0, math.radians(1.2)),
            "curve": self.g(0, 0.03 * h) / max(length, h) ** 2,
            "w": self.writer(wid),
        }

    def glyph(self, h):
        """Per-character variation: scale, rotation, baseline jitter, spacing."""
        return {"scale": self.g(1, 0.06), "rot": self.g(0, math.radians(3)),
                "dy": self.g(0, 0.03 * h), "space": self.g(0, 0.04 * h)}

    # -------------------------------------------------------------- strokes

    def deform(self, stroke, scale, wid, kind):
        """Stroke-level variation. scale is the element's size (text height in mm,
        or a fixed reference for diagram primitives)."""
        if self.level == 0:
            return stroke
        lv, w = self.level, self.writer(wid)
        pts = densify(stroke, max(0.3, scale / 40))
        s = arc_lengths(pts)
        L = s[-1]
        if L < 1e-6:
            return pts
        text = kind == "text"

        # Overshoot or stop short along the tangent at both ends.
        if text:
            e0, e1 = self.g(0, 0.025 * scale), self.g(0, 0.025 * scale)
        else:
            e0, e1 = self.g(0.4, 0.8), self.g(0.6, 0.9)  # diagram lines tend to overshoot
        t0, t1 = unit(pts[1], pts[0]), unit(pts[-2], pts[-1])
        e0, e1 = max(e0, -0.3 * L), max(e1, -0.3 * L)
        if e0 > 0:
            pts.insert(0, (pts[0][0] + t0[0] * e0, pts[0][1] + t0[1] * e0))
        elif e0 < 0:
            pts = [p for p, si in zip(pts, s) if si >= -e0] or pts
        if e1 > 0:
            pts.append((pts[-1][0] + t1[0] * e1, pts[-1][1] + t1[1] * e1))
        elif e1 < 0:
            s1 = arc_lengths(pts)
            keep = [p for p, si in zip(pts, s1) if si <= s1[-1] + e1]
            pts = keep if len(keep) >= 2 else pts
        s = arc_lengths(pts)
        L = s[-1]

        # Endpoint error: each end lands off target, blended linearly along the
        # stroke, so long lines also tilt a little.
        err = 0.015 * scale if text else 0.7
        a = (self.g(0, err), self.g(0, err))
        b = (self.g(0, err), self.g(0, err))

        field = Field2D(self.rng, 0.9 * scale if text else 60.0,
                        lv * (0.03 * scale if text else 0.8))
        wobble = Smooth1D(self.rng, 0.6 * scale if text else 25.0,
                          lv * (0.012 * scale if text else 0.5))
        tremor = Smooth1D(self.rng, 2.0, w["tremor"])
        out = []
        for i, ((x, y), si) in enumerate(zip(pts, s)):
            u = si / L
            nx, ny = self._normal(pts, i)
            fx, fy = field(x, y)
            n = wobble(si) + tremor(si)
            out.append((x + fx + n * nx + (1 - u) * a[0] + u * b[0],
                        y + fy + n * ny + (1 - u) * a[1] + u * b[1]))
        return out

    @staticmethod
    def _normal(pts, i):
        a, b = pts[max(0, i - 1)], pts[min(len(pts) - 1, i + 1)]
        tx, ty = unit(a, b)
        return -ty, tx

    def line(self, polyline):
        """A hand-drawn polyline: each straight segment bows slightly."""
        if self.level == 0:
            return [tuple(p) for p in polyline]
        out = [tuple(polyline[0])]
        for a, b in zip(polyline, polyline[1:]):
            L = math.dist(a, b)
            if L < 1e-9:
                continue
            bow = max(-3.0, min(3.0, self.g(0, 0.008 * L)))
            nx, ny = -(b[1] - a[1]) / L, (b[0] - a[0]) / L
            n = max(1, math.ceil(L / 2))
            for k in range(1, n + 1):
                u = k / n
                off = bow * math.sin(math.pi * u)
                out.append((a[0] + (b[0] - a[0]) * u + off * nx, a[1] + (b[1] - a[1]) * u + off * ny))
        return out

    def circle_params(self):
        """Ellipse distortion and closure error for hand-drawn circles."""
        if self.level == 0:
            return 1.0, 0.0, 0.0, 0.0
        return (self.g(1, 0.08), self.rng.uniform(0, math.pi),   # aspect, its axis
                self.g(math.radians(15), math.radians(15)),        # closure overrun (+) / gap (-)
                self.g(0, 0.4))                                    # start-angle shift (rad)

    # ----------------------------------------------------------- kinematics

    def time_stroke(self, pts, t0, wid, rate):
        """Sample a stroke at `rate` Hz. Returns [(x, y, t)] and the end time."""
        w = self.writer(wid)
        dense = densify(pts, 0.25)
        s = arc_lengths(dense)
        L = s[-1]
        v_peak = w["speed"] * (math.exp(self.level * self.rng.gauss(0, 0.15)) if self.level else 1)
        if self.level == 0 or L < 1e-6:
            speeds = [v_peak] * len(dense)
        else:
            # Turning angle per mm, smoothed over ~1.5 mm, as a curvature estimate.
            turn = [0.0] * len(dense)
            for i in range(1, len(dense) - 1):
                a, b = unit(dense[i - 1], dense[i]), unit(dense[i], dense[i + 1])
                turn[i] = abs(math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])) / 0.25
            k = 3
            curv = [sum(turn[max(0, i - k):i + k + 1]) / (2 * k + 1) for i in range(len(dense))]
            ramp = 4.0  # mm to accelerate from pen-down / decelerate to pen-up
            speeds = []
            for si, c in zip(s, curv):
                v = v_peak * (1 + 4.0 * c) ** (-1 / 3)  # two-thirds power law, softened
                ease = min(1.0, math.sqrt(max(si, 0.05) / ramp), math.sqrt(max(L - si, 0.05) / ramp))
                speeds.append(v * max(0.15, ease))
            speeds = [v_peak + self.level * (v - v_peak) for v in speeds]
        tcum = [0.0]
        for i in range(1, len(dense)):
            tcum.append(tcum[-1] + (s[i] - s[i - 1]) / ((speeds[i] + speeds[i - 1]) / 2))
        T = tcum[-1]
        # Fixed-rate samples, then the stroke's end. Drop the last regular sample if
        # it would sit within 2 ms of the end, so timestamps stay strictly
        # increasing after rounding to 1 ms in the file.
        times = [kk / rate for kk in range(math.ceil(T * rate))]
        if len(times) > 1 and T - times[-1] < 0.002:
            times.pop()
        times.append(T)
        out, j = [], 0
        for t in times:
            while j < len(tcum) - 2 and tcum[j + 1] < t:
                j += 1
            span = tcum[j + 1] - tcum[j] if len(tcum) > 1 else 0
            f = (t - tcum[j]) / span if span > 0 else 0.0
            a, b = dense[j], dense[min(j + 1, len(dense) - 1)]
            out.append((a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), t0 + t))
        return out, t0 + T

    def pause(self, base):
        return base * (math.exp(self.level * self.rng.gauss(0, 0.5)) if self.level else 1)
