"""Single-stroke fonts and text metrics for the ink renderer and the layout engine.

Every font is normalized to the same vertical frame, in units of the text
height h (the distance from the descender line to the cap line):

- the baseline is at BASE * h above the descender line;
- capitals (H, E) reach exactly 1.0 * h.

So text of a given text_height has the same capital size in every font, and
descenders keep each font's own proportions (some reach below the descender
line). Greek letters come from the Hershey `greeks` font, the characters
in SYMBOL_ADVANCE are drawn as primitives (see symbol_strokes()), and
double-struck capitals, ℏ, ϖ and ς are a font letter plus extra strokes (see
built_strokes()).

`TextRenderer.glyphs()` returns each character's clean strokes in line
coordinates (x right from the start of the text, y up from the baseline);
render_inkml.py adds the handwriting variation and maps them to the page.
`PoolMetrics` gives widths that are safe for every font in a pool, for the
layout engine and the validator.
"""
import math
import os
import re
import unicodedata
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
                  "∆": 0.7, "°": 0.35, "←": 0.75, "↔": 0.85, "⇒": 0.75, "∈": 0.5,
                  "≡": 0.6, "∉": 0.5, "∋": 0.5, "⊂": 0.55, "⊃": 0.55, "⊆": 0.55, "⊇": 0.55, "⊊": 0.55,
                  "⊑": 0.55, "↦": 0.75, "⟶": 1.0, "⇔": 0.85, "⟺": 1.1, "↪": 0.8, "⇌": 0.8, "≅": 0.6,
                  "≃": 0.6, "≜": 0.6, "∝": 0.6, "⊥": 0.6, "⊤": 0.6, "⊢": 0.5, "⊩": 0.6, "⊨": 0.55,
                  "≪": 0.75, "≫": 0.75, "⊗": None, "⊙": None, "⊖": None, "∧": 0.55, "∨": 0.55,
                  "∩": 0.55, "∪": 0.55, "∓": 0.6, "÷": 0.6, "∙": 0.3, "†": 0.4, "◁": 0.6, "△": 0.7,
                  "◯": 0.9, "∠": 0.6, "∇": 0.7, "∀": 0.6, "∃": 0.5, "∅": 0.6, "ℵ": 0.6, "⋮": 0.3,
                  "‖": 0.3, "⟨": 0.35, "⟩": 0.35, "⌊": 0.35, "⌋": 0.35, "⌈": 0.35, "⌉": 0.35, "¬": 0.6}

# Characters drawn as a font letter plus a few extra strokes (see built_strokes()).
# Double-struck capitals (ℝ, 𝔸, ...) are found by their Unicode name instead.
BUILT_ON = {"ℏ": "h", "ϖ": "ω", "ς": "c"}

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
        ch = built_on(ch) or ch
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
        if built_on(ch):
            return built_strokes(ch, self.char_strokes(built_on(ch), h), h)
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


def wave(x0, x1, y, amp):
    return [(x0 + (x1 - x0) * i / 8, y + amp * math.sin(2 * math.pi * i / 8)) for i in range(9)]


def mirrored(strokes, w):
    return [[(w - x, y) for x, y in st] for st in strokes]


def built_on(ch):
    """The font letter that `ch` is drawn on, or None."""
    name = unicodedata.name(ch, "")
    if "DOUBLE-STRUCK CAPITAL " in name:
        return name[-1]
    return BUILT_ON.get(ch)


def built_strokes(ch, strokes, h):
    """A BUILT_ON or double-struck character: its base letter's strokes (y up,
    mm) plus the strokes that make it this character."""
    pts = [p for st in strokes for p in st]
    x0, x1 = min(x for x, _ in pts), max(x for x, _ in pts)
    y0, y1 = min(y for _, y in pts), max(y for _, y in pts)
    if ch == "ℏ":
        return strokes + [[(x0 - 0.06 * h, 0.5 * h), (x0 + 0.3 * h, 0.64 * h)]]
    if ch == "ϖ":
        return strokes + [[(x0 - 0.04 * h, y1 + 0.08 * h), (x1 + 0.04 * h, y1 + 0.08 * h)]]
    if ch == "ς":
        bx, by = min(pts, key=lambda p: p[1])
        return strokes + [bezier((bx, by), (bx + 0.16 * h, by - 0.1 * h), (bx, by - 0.28 * h), 8)]
    # Blackboard bold, as written by hand: the letter's leftmost long stem or
    # diagonal is doubled, or curved letters (C, Q, S, ...) get an inner stem.
    segments = [(p, q) for st in strokes for p, q in zip(st, st[1:])
                if math.dist(p, q) >= 0.4 * h and abs(q[1] - p[1]) >= 0.5 * math.dist(p, q)]
    shift = 0.09 * h
    if segments:
        p, q = min(segments, key=lambda s: s[0][0] + s[1][0])
        return strokes + [[(p[0] + shift, p[1]), (q[0] + shift, q[1])]]
    left = [y for x, y in pts if x <= x0 + 0.12 * h] or [y0, y1]
    return strokes + [[(x0 + shift, max(left)), (x0 + shift, min(left))]]


def symbol_strokes(ch, w, h, c, cap_mid):
    """Strokes for a primitive-drawn symbol, in y-up glyph coordinates.

    w is the advance and h the text height (mm). c is the height of an
    operator's centre above the baseline, and cap_mid the middle of a capital.
    """
    m, d = w / 2, 0.11 * h
    if ch == "⊕":
        r = 0.27 * h
        return [ring(m, cap_mid, r), [(m - r, cap_mid), (m + r, cap_mid)], [(m, cap_mid + r), (m, cap_mid - r)]]
    if ch in "⊗⊙⊖":
        r, k = 0.27 * h, 0.27 * h / math.sqrt(2)
        inner = {"⊗": [[(m - k, cap_mid + k), (m + k, cap_mid - k)], [(m + k, cap_mid + k), (m - k, cap_mid - k)]],
                 "⊙": [ring(m, cap_mid, 0.04 * h, n=8)],
                 "⊖": [[(m - r, cap_mid), (m + r, cap_mid)]]}[ch]
        return [ring(m, cap_mid, r)] + inner
    if ch in "→←⟶":
        a, b = (0.92 * w, 0.08 * w) if ch == "←" else (0.08 * w, 0.92 * w)
        s = -0.16 * h if ch == "←" else 0.16 * h
        return [[(a, c), (b, c)], [(b - s, c + 0.1 * h), (b, c), (b - s, c - 0.1 * h)]]
    if ch == "↦":
        a, b = 0.08 * w, 0.92 * w
        return [[(a, c + 0.1 * h), (a, c - 0.1 * h)], [(a, c), (b, c)],
                [(b - 0.16 * h, c + 0.1 * h), (b, c), (b - 0.16 * h, c - 0.1 * h)]]
    if ch == "↪":
        a, b, r = 0.08 * w, 0.92 * w, 0.08 * h
        hook = [(a + r - r * math.sin(t), c + r + r * math.cos(t)) for t in (math.pi * i / 8 for i in range(9))]
        return [hook + [(b, c)], [(b - 0.16 * h, c + 0.1 * h), (b, c), (b - 0.16 * h, c - 0.1 * h)]]
    if ch == "⇌":
        a, b, y = 0.08 * w, 0.92 * w, 0.07 * h
        return [[(a, c + y), (b, c + y), (b - 0.16 * h, c + y + 0.1 * h)],
                [(b, c - y), (a, c - y), (a + 0.16 * h, c - y - 0.1 * h)]]
    if ch == "↔":
        a, b = 0.08 * w, 0.92 * w
        return [[(a, c), (b, c)], [(b - 0.16 * h, c + 0.1 * h), (b, c), (b - 0.16 * h, c - 0.1 * h)],
                [(a + 0.16 * h, c + 0.1 * h), (a, c), (a + 0.16 * h, c - 0.1 * h)]]
    if ch == "⇒":
        b = 0.92 * w
        return [[(0.08 * w, c + 0.07 * h), (b - 0.1 * h, c + 0.07 * h)],
                [(0.08 * w, c - 0.07 * h), (b - 0.1 * h, c - 0.07 * h)],
                [(b - 0.2 * h, c + 0.17 * h), (b, c), (b - 0.2 * h, c - 0.17 * h)]]
    if ch in "⇔⟺":
        a, b = 0.08 * w, 0.92 * w
        return [[(a + 0.1 * h, c + 0.07 * h), (b - 0.1 * h, c + 0.07 * h)],
                [(a + 0.1 * h, c - 0.07 * h), (b - 0.1 * h, c - 0.07 * h)],
                [(a + 0.2 * h, c + 0.17 * h), (a, c), (a + 0.2 * h, c - 0.17 * h)],
                [(b - 0.2 * h, c + 0.17 * h), (b, c), (b - 0.2 * h, c - 0.17 * h)]]
    if ch == "¬":
        return [[(0.12 * w, c + 0.1 * h), (0.88 * w, c + 0.1 * h), (0.88 * w, c - 0.1 * h)]]
    if ch == "≡":
        return [[(0.12 * w, y), (0.88 * w, y)] for y in (c + 0.13 * h, c, c - 0.13 * h)]
    if ch in "≅≃":
        lines = (c + 0.02 * h, c - 0.14 * h) if ch == "≅" else (c - 0.08 * h,)
        top = c + 0.18 * h if ch == "≅" else c + 0.08 * h
        return [wave(0.12 * w, 0.88 * w, top, 0.05 * h)] + [[(0.12 * w, y), (0.88 * w, y)] for y in lines]
    if ch == "≜":
        return [[(m, c + 0.38 * h), (m - 0.1 * h, c + 0.2 * h), (m + 0.1 * h, c + 0.2 * h), (m, c + 0.38 * h)],
                [(0.12 * w, c + 0.06 * h), (0.88 * w, c + 0.06 * h)],
                [(0.12 * w, c - 0.1 * h), (0.88 * w, c - 0.1 * h)]]
    if ch == "∝":
        a = 0.42 * w
        lobe = []
        for i in range(17):
            t = math.pi / 2 + math.pi * i / 16
            k = 1 + math.sin(t) ** 2
            lobe.append((m + a * math.cos(t) / k, c + 0.9 * a * math.sin(t) * math.cos(t) / k))
        return [[(0.92 * w, c + 0.16 * h)] + lobe + [(0.92 * w, c - 0.16 * h)]]
    if ch in "⊂⊃⊆⊇⊊⊑":
        low = ch in "⊆⊇⊊⊑"
        cy, r = (c + 0.08 * h, 0.17 * h) if low else (c, 0.2 * h)
        xo, xr = 0.88 * w, 0.12 * w
        if ch == "⊑":
            strokes = [[(xo, cy + r), (xr, cy + r), (xr, cy - r), (xo, cy - r)]]
        else:
            xc = xr + r
            arc = [(xc - r * math.sin(t), cy + r * math.cos(t)) for t in (math.pi * i / 12 for i in range(13))]
            strokes = [[(xo, cy + r)] + arc + [(xo, cy - r)]]
        if low:
            y = c - 0.2 * h
            strokes.append([(xr, y), (xo, y)])
            if ch == "⊊":
                strokes.append([(m + 0.06 * h, y + 0.07 * h), (m - 0.06 * h, y - 0.07 * h)])
        return mirrored(strokes, w) if ch in "⊃⊇" else strokes
    if ch in "≪≫":
        strokes = [[(x + 0.26 * h, c + 0.2 * h), (x, c), (x + 0.26 * h, c - 0.2 * h)] for x in (0.08 * w, 0.08 * w + 0.22 * h)]
        return mirrored(strokes, w) if ch == "≫" else strokes
    if ch in "∧∨":
        strokes = [[(0.1 * w, c - 0.22 * h), (m, c + 0.26 * h), (0.9 * w, c - 0.22 * h)]]
        return [[(x, 2 * c - y) for x, y in st] for st in strokes] if ch == "∨" else strokes
    if ch in "∩∪":
        r, yc = 0.38 * w, c + 0.24 * h - 0.38 * w
        arch = [(m - r * math.cos(t), yc + r * math.sin(t)) for t in (math.pi * i / 12 for i in range(13))]
        strokes = [[(m - r, c - 0.22 * h)] + arch + [(m + r, c - 0.22 * h)]]
        return [[(x, 2 * c - y) for x, y in st] for st in strokes] if ch == "∪" else strokes
    if ch == "∓":
        cy = c - 0.08 * h
        return [[(m - 0.17 * h, c + 0.2 * h), (m + 0.17 * h, c + 0.2 * h)],
                [(m - 0.17 * h, cy), (m + 0.17 * h, cy)], [(m, cy + 0.17 * h), (m, cy - 0.17 * h)]]
    if ch == "÷":
        return [[(0.12 * w, c), (0.88 * w, c)], ring(m, c + 0.15 * h, 0.03 * h, n=8), ring(m, c - 0.15 * h, 0.03 * h, n=8)]
    if ch == "∙":
        return [ring(m, c, 0.06 * h, n=12), ring(m, c, 0.03 * h, n=8)]
    if ch == "◁":
        return [[(0.88 * w, c + 0.25 * h), (0.12 * w, c), (0.88 * w, c - 0.25 * h), (0.88 * w, c + 0.25 * h)]]
    if ch == "◯":
        return [ring(m, c, 0.4 * w, n=32)]
    if ch == "⊥":
        return [[(m, 0.72 * h), (m, 0.0)], [(0.1 * w, 0.0), (0.9 * w, 0.0)]]
    if ch == "⊤":
        return [[(0.1 * w, 0.72 * h), (0.9 * w, 0.72 * h)], [(m, 0.72 * h), (m, 0.0)]]
    if ch in "⊢⊩⊨":
        stems = [0.15 * w] if ch != "⊩" else [0.12 * w, 0.3 * w]
        bars = [0.45 * h, 0.25 * h] if ch == "⊨" else [0.35 * h]
        return [[(x, 0.7 * h), (x, 0.0)] for x in stems] + [[(stems[-1], y), (0.9 * w, y)] for y in bars]
    if ch == "†":
        return [[(m, 0.78 * h), (m, -0.15 * h)], [(m - 0.13 * h, 0.5 * h), (m + 0.13 * h, 0.5 * h)]]
    if ch == "∠":
        return [[(0.85 * w, 0.6 * h), (0.1 * w, 0.0), (0.9 * w, 0.0)]]
    if ch == "∇":
        return [[(0.08 * w, 0.75 * h), (0.92 * w, 0.75 * h), (m, 0.0), (0.08 * w, 0.75 * h)]]
    if ch == "∀":
        return [[(0.1 * w, 0.75 * h), (m, 0.0), (0.9 * w, 0.75 * h)], [(0.29 * w, 0.4 * h), (0.71 * w, 0.4 * h)]]
    if ch == "∃":
        return [[(0.15 * w, 0.75 * h), (0.85 * w, 0.75 * h), (0.85 * w, 0.0), (0.15 * w, 0.0)],
                [(0.2 * w, 0.375 * h), (0.85 * w, 0.375 * h)]]
    if ch == "∅":
        return [ring(m, cap_mid, 0.28 * h), [(m + 0.24 * h, cap_mid + 0.4 * h), (m - 0.24 * h, cap_mid - 0.4 * h)]]
    if ch == "ℵ":
        return [[(0.15 * w, 0.72 * h), (0.85 * w, 0.0)], [(0.78 * w, 0.72 * h), (0.75 * w, 0.45 * h), (0.62 * w, 0.24 * h)],
                [(0.4 * w, 0.46 * h), (0.25 * w, 0.28 * h), (0.25 * w, 0.0)]]
    if ch == "⋮":
        return [ring(m, y * h, 0.03 * h, n=8) for y in (0.62, 0.36, 0.1)]
    if ch == "‖":
        return [[(x, 0.88 * h), (x, -0.2 * h)] for x in (0.35 * w, 0.65 * w)]
    if ch in "⟨⟩":
        strokes = [[(0.85 * w, 0.88 * h), (0.15 * w, 0.34 * h), (0.85 * w, -0.2 * h)]]
        return mirrored(strokes, w) if ch == "⟩" else strokes
    if ch in "⌊⌋⌈⌉":
        floor = [[(0.3 * w, 0.88 * h), (0.3 * w, -0.2 * h), (0.85 * w, -0.2 * h)]]
        strokes = floor if ch in "⌊⌋" else [[(x, 0.68 * h - y) for x, y in st] for st in floor]
        return mirrored(strokes, w) if ch in "⌋⌉" else strokes
    if ch in "≤≥":
        a, b = (0.85 * w, 0.15 * w) if ch == "≤" else (0.15 * w, 0.85 * w)
        return [[(a, c + 0.24 * h), (b, c + 0.06 * h), (a, c - 0.12 * h)], [(b, c - 0.24 * h), (a, c - 0.24 * h)]]
    if ch == "≠":
        return [[(0.12 * w, c + 0.08 * h), (0.88 * w, c + 0.08 * h)],
                [(0.12 * w, c - 0.08 * h), (0.88 * w, c - 0.08 * h)],
                [(0.66 * w, c + 0.24 * h), (0.34 * w, c - 0.24 * h)]]
    if ch == "≈":
        return [wave(0.12 * w, 0.88 * w, c + 0.08 * h, 0.05 * h), wave(0.12 * w, 0.88 * w, c - 0.08 * h, 0.05 * h)]
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
    if ch in "∆△":
        return [[(m, 0.75 * h), (0.08 * w, 0.0), (0.92 * w, 0.0), (m, 0.75 * h)]]
    if ch == "∂":
        r = 0.2 * h
        loop = [(m + r * math.cos(a), r + r * math.sin(a)) for a in (2 * math.pi * i / 16 for i in range(17))]
        return [bezier((m - 0.15 * h, 0.68 * h), (m + 0.3 * h, 0.85 * h), (m + r, r)) + loop[1:]]
    if ch in "∈∉∋":
        r = 0.22 * h
        arc = [(m + 0.1 * h - r * math.sin(a), c + r * math.cos(a)) for a in (math.pi * i / 12 for i in range(13))]
        strokes = [arc, [(m + 0.1 * h - r, c), (m + 0.1 * h, c)]]
        if ch == "∉":
            strokes.append([(m + 0.12 * h, c + 0.32 * h), (m - 0.12 * h, c - 0.32 * h)])
        return mirrored(strokes, w) if ch == "∋" else strokes
    raise ValueError(ch)
