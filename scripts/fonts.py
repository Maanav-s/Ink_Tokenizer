"""Single-stroke fonts and text metrics for the ink renderer and the layout engine.

Every font is normalized to the same vertical frame, in units of the text
height h (the distance from the descender line to the cap line):

- the baseline is at BASE * h above the descender line;
- capitals (H, E) reach exactly 1.0 * h.

So text of a given text_height has the same capital size in every font, and
descenders keep each font's own proportions (some reach below the descender
line). Greek letters come from the Hershey `greeks` font, and the characters
in SYMBOL_ADVANCE are drawn as primitives (see symbol_strokes()).

`TextRenderer.glyphs()` returns each character's clean strokes in line
coordinates (x right from the start of the text, y up from the baseline);
render_inkml.py adds the handwriting variation and maps them to the page.
`PoolMetrics` gives widths that are safe for every font in a pool, for the
layout engine and the validator.
"""
import math
import os
import re
import xml.etree.ElementTree as ET

from HersheyFonts import HersheyFonts

BASE = 0.25  # baseline height above the descender line, fraction of text height

# The batch font pool: fonts that looked believably handwritten (fonts/README.md).
FONT_POOL = ["futural", "cursive", "EMSReadability", "EMSReadabilityItalic", "EMSTech", "EMSAllure"]

GREEK = dict(zip("αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ",
                 "abcdefghijklmnopqrstuvwxABCDEFGHIJKLMNOPQRSTUVWX"))  # Hershey 'greeks' keys

# Symbols no font has, drawn as primitives: char -> advance in units of text
# height (None: the advance of "O"). Their strokes come from symbol_strokes().
SYMBOL_ADVANCE = {"⊕": None, "→": 0.75, "≤": 0.6, "≥": 0.6, "≠": 0.6, "±": 0.6, "·": 0.3,
                  "×": 0.55, "∞": 0.75, "∫": 0.45, "√": 0.55, "≈": 0.6, "∑": 0.7, "∂": 0.55,
                  "∆": 0.7, "°": 0.35, "←": 0.75, "↔": 0.85, "⇒": 0.75, "∈": 0.5}

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "svg")


class Glyph:
    """A glyph normalized to the common frame: strokes in units of h, x from
    the glyph's left edge, y up from the baseline."""

    def __init__(self, advance, strokes):
        self.advance, self.strokes = advance, strokes


def _normalize(raw, adv_of, strokes_of, cap):
    """raw: dict char -> font glyph. cap: the font's cap height above its baseline."""
    k = (1.0 - BASE) / cap
    return {ch: Glyph(adv_of(g) * k, [[(x * k, y * k) for x, y in st] for st in strokes_of(g)])
            for ch, g in raw.items()}


def hershey_glyphs(name, chars):
    font = HersheyFonts()
    font.load_default_font(name)
    font.normalize_rendering(1.0)
    o = font.render_options
    raw = {}
    for ch in chars:
        gl = list(font.glyphs_for_text(ch))
        if gl:
            g = gl[0]
            raw[ch] = (g.char_width * o.scalex,
                       [[((px - g.left_offset) * o.scalex, o.yofs + py * o.scaley) for px, py in st]
                        for st in g.strokes])
    ref = raw["A" if name == "greeks" else "H"][1]
    base = min(y for st in ref for _, y in st)
    cap = max(y for st in ref for _, y in st) - base
    raw = {ch: (adv, [[(x, y - base) for x, y in st] for st in sts]) for ch, (adv, sts) in raw.items()}
    return _normalize(raw, lambda g: g[0], lambda g: g[1], cap)


def svg_glyphs(path):
    """A single-line SVG font (e.g. the EMS fonts from Inkscape's Hershey Text
    extension). Paths use absolute M, L and (in a few accented glyphs) C, and
    commands may be glued to their first number ("M1022 40")."""
    svg = "{http://www.w3.org/2000/svg}"
    font = ET.parse(path).getroot().find(f".//{svg}font")
    default_adv = float(font.get("horiz-adv-x", 500))
    raw = {}
    for g in font.findall(f"{svg}glyph"):
        ch = g.get("unicode")
        if ch is None or len(ch) != 1:
            continue
        strokes, cur, cmd = [], None, None
        tok = re.findall(r"[A-Za-z]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", g.get("d") or "")
        i = 0
        while i < len(tok):
            if tok[i].isalpha():
                cmd = tok[i]
                if cmd not in "MLC":
                    raise ValueError(f"{path}: unsupported path command {cmd!r} in glyph {ch!r}")
                i += 1
                continue
            if cmd == "C":  # cubic Bezier, flattened to 8 segments
                p0 = cur[-1]
                p1, p2, p3 = [(float(tok[i + k]), float(tok[i + k + 1])) for k in (0, 2, 4)]
                i += 6
                for j in range(1, 9):
                    t = j / 8
                    cur.append(tuple((1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t * t * c + t ** 3 * d
                                     for a, b, c, d in zip(p0, p1, p2, p3)))
                continue
            pt = (float(tok[i]), float(tok[i + 1]))
            i += 2
            if cmd == "M":
                cur = [pt]
                strokes.append(cur)
                cmd = "L"  # further pairs after a moveto are linetos
            else:
                cur.append(pt)
        raw[ch] = (float(g.get("horiz-adv-x", default_adv)), [st for st in strokes if len(st) > 1])
    # SVG font units already have the baseline at y = 0, y up.
    cap = max(y for c in "HE" for st in raw[c][1] for _, y in st)
    return _normalize(raw, lambda g: g[0], lambda g: g[1], cap)


_CACHE = {}
HERSHEY_CHARS = "".join(chr(c) for c in range(32, 127))


def load_glyphs(name):
    if name not in _CACHE:
        path = os.path.join(FONT_DIR, f"{name}.svg")
        if os.path.exists(path):
            _CACHE[name] = svg_glyphs(path)
        else:
            _CACHE[name] = hershey_glyphs(name, HERSHEY_CHARS if name != "greeks" else "".join(GREEK.values()))
    return _CACHE[name]


class TextRenderer:
    """Clean glyph strokes and metrics for one font.

    `font` is a Hershey font name (e.g. futural, cursive) or the name of an SVG
    font in fonts/svg (e.g. EMSReadability). Any character that is neither in
    the font, Greek, nor in SYMBOL_ADVANCE is an error.
    """

    base = BASE

    def __init__(self, font):
        self.name = font
        self.main = load_glyphs(font)
        self.greek = load_glyphs("greeks")
        ys = [y for st in self.main["+"].strokes for _, y in st]
        self.axis = (min(ys) + max(ys)) / 2  # operator centre above the baseline, units of h
        xs = [y for st in self.main["x"].strokes for _, y in st]
        self.x_height = max(xs)

    def _glyph(self, ch):
        if ch in GREEK:
            return self.greek.get(GREEK[ch])
        return self.main.get(ch)

    def has(self, ch):
        return ch in SYMBOL_ADVANCE or self._glyph(ch) is not None

    def advance(self, ch, h):
        if ch in SYMBOL_ADVANCE:
            adv = SYMBOL_ADVANCE[ch]
            if adv is not None:
                return adv * h
            ch = "O"
        g = self._glyph(ch)
        if g is None:
            raise ValueError(f"font {self.name!r} has no glyph for {ch!r}")
        return g.advance * h

    def width(self, text, h):
        return sum(self.advance(ch, h) for ch in text)

    def char_strokes(self, ch, h):
        """Clean strokes of one character at text height h, y up from the baseline."""
        if ch in SYMBOL_ADVANCE:
            return symbol_strokes(ch, self.advance(ch, h), h, self.axis * h, (1.0 - BASE) / 2 * h)
        g = self._glyph(ch)
        if g is None:
            raise ValueError(f"font {self.name!r} has no glyph for {ch!r}")
        return [[(x * h, y * h) for x, y in st] for st in g.strokes]


class PoolMetrics:
    """Widths that are safe for every font in a pool: each character's advance
    is the maximum over the pool. Same vertical frame as TextRenderer."""

    base = BASE

    def __init__(self, fonts=FONT_POOL):
        self.fonts = [TextRenderer(f) for f in fonts]
        self.axis = sum(f.axis for f in self.fonts) / len(self.fonts)
        self.x_height = max(f.x_height for f in self.fonts)
        self._adv = {}

    def has(self, ch):
        return all(f.has(ch) for f in self.fonts)

    def advance(self, ch, h):
        if ch not in self._adv:
            self._adv[ch] = max(f.advance(ch, 1.0) for f in self.fonts)
        return self._adv[ch] * h

    def width(self, text, h):
        return sum(self.advance(ch, h) for ch in text)


# -------------------------------------------------------------- symbols

def bezier(p0, p1, p2, n=16):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1])
            for t in (i / n for i in range(n + 1))]


def ring(cx, cy, r, n=24):
    """A circle starting at the top and running counter-clockwise (y up)."""
    return [(cx - r * math.sin(2 * math.pi * i / n), cy + r * math.cos(2 * math.pi * i / n)) for i in range(n + 1)]


def symbol_strokes(ch, w, h, c, cap_mid):
    """Strokes for a primitive-drawn symbol, in y-up glyph coordinates.

    w is the advance and h the text height (mm). c is the height of an
    operator's centre above the baseline, and cap_mid the middle of a capital.
    """
    m, d = w / 2, 0.11 * h
    if ch == "⊕":
        r = 0.27 * h
        return [ring(m, cap_mid, r), [(m - r, cap_mid), (m + r, cap_mid)], [(m, cap_mid + r), (m, cap_mid - r)]]
    if ch in "→←":
        a, b = (0.08 * w, 0.92 * w) if ch == "→" else (0.92 * w, 0.08 * w)
        s = 0.16 * h if ch == "→" else -0.16 * h
        return [[(a, c), (b, c)], [(b - s, c + 0.1 * h), (b, c), (b - s, c - 0.1 * h)]]
    if ch == "↔":
        a, b = 0.08 * w, 0.92 * w
        return [[(a, c), (b, c)], [(b - 0.16 * h, c + 0.1 * h), (b, c), (b - 0.16 * h, c - 0.1 * h)],
                [(a + 0.16 * h, c + 0.1 * h), (a, c), (a + 0.16 * h, c - 0.1 * h)]]
    if ch == "⇒":
        b = 0.92 * w
        return [[(0.08 * w, c + 0.07 * h), (b - 0.1 * h, c + 0.07 * h)],
                [(0.08 * w, c - 0.07 * h), (b - 0.1 * h, c - 0.07 * h)],
                [(b - 0.2 * h, c + 0.17 * h), (b, c), (b - 0.2 * h, c - 0.17 * h)]]
    if ch in "≤≥":
        a, b = (0.85 * w, 0.15 * w) if ch == "≤" else (0.15 * w, 0.85 * w)
        return [[(a, c + 0.24 * h), (b, c + 0.06 * h), (a, c - 0.12 * h)], [(b, c - 0.24 * h), (a, c - 0.24 * h)]]
    if ch == "≠":
        return [[(0.12 * w, c + 0.08 * h), (0.88 * w, c + 0.08 * h)],
                [(0.12 * w, c - 0.08 * h), (0.88 * w, c - 0.08 * h)],
                [(0.66 * w, c + 0.24 * h), (0.34 * w, c - 0.24 * h)]]
    if ch == "≈":
        def wave(y):
            return [(0.12 * w + 0.76 * w * i / 8, y + 0.05 * h * math.sin(2 * math.pi * i / 8)) for i in range(9)]
        return [wave(c + 0.08 * h), wave(c - 0.08 * h)]
    if ch == "±":
        cy = c + 0.08 * h
        return [[(m - 0.17 * h, cy), (m + 0.17 * h, cy)], [(m, cy + 0.17 * h), (m, cy - 0.17 * h)],
                [(m - 0.17 * h, c - 0.2 * h), (m + 0.17 * h, c - 0.2 * h)]]
    if ch == "·":
        r = 0.03 * h
        return [[(m + r * math.cos(a * math.pi / 4), c + r * math.sin(a * math.pi / 4)) for a in range(9)]]
    if ch == "°":
        return [ring(m, 0.68 * h, 0.08 * h, n=12)]
    if ch == "×":
        return [[(m - d, c + d), (m + d, c - d)], [(m + d, c + d), (m - d, c - d)]]
    if ch == "∞":
        a = 0.42 * w
        pts = []
        for i in range(33):
            t = 2 * math.pi * i / 32
            k = 1 + math.sin(t) ** 2
            pts.append((m + a * math.cos(t) / k, c + 0.9 * a * math.sin(t) * math.cos(t) / k))
        return [pts]
    if ch == "∫":
        top, bot = 0.75 * h, -0.32 * h
        return [bezier((0.92 * w, top - 0.06 * h), (0.75 * w, top + 0.05 * h), (0.6 * w, top - 0.12 * h))
                + [(0.42 * w, bot + 0.12 * h)]
                + bezier((0.42 * w, bot + 0.12 * h), (0.32 * w, bot - 0.05 * h), (0.1 * w, bot + 0.06 * h))[1:]]
    if ch == "√":
        return [[(0.0, 0.25 * h), (0.15 * w, 0.32 * h), (0.42 * w, -0.05 * h), (0.75 * w, 0.72 * h), (w, 0.72 * h)]]
    if ch == "∑":
        x0, x1, top = 0.1 * w, 0.9 * w, 0.75 * h
        return [[(x1, top), (x0, top), (m, top / 2), (x0, 0.0), (x1, 0.0)]]
    if ch in "∆":
        return [[(m, 0.75 * h), (0.08 * w, 0.0), (0.92 * w, 0.0), (m, 0.75 * h)]]
    if ch == "∂":
        r = 0.2 * h
        loop = [(m + r * math.cos(a), r + r * math.sin(a)) for a in (2 * math.pi * i / 16 for i in range(17))]
        return [bezier((m - 0.15 * h, 0.68 * h), (m + 0.3 * h, 0.85 * h), (m + r, r)) + loop[1:]]
    if ch == "∈":
        r = 0.22 * h
        arc = [(m + 0.1 * h - r * math.sin(a), c + r * math.cos(a)) for a in (math.pi * i / 12 for i in range(13))]
        return [arc, [(m + 0.1 * h - r, c), (m + 0.1 * h, c)]]
    raise ValueError(ch)
