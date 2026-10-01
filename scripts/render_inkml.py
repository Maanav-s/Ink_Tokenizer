"""Naive ink renderer: content.json + layout.json -> page.inkml.

Proof of concept for the "text to ink" and "diagrams to ink" stages in
docs/synthetic_data.md, without any handwriting model:

- Text is drawn with a single-stroke Hershey font, scaled to the layout's
  text_height. Characters the font lacks (only "⊕" on page_0001) are drawn
  as primitives.
- Diagrams are drawn from geometric primitives (gate outlines, pin circles,
  wire polylines, junction dots).
- Elements are written in layout.json's writing_order. Each stroke is
  sampled at a fixed rate while the pen moves at constant speed, and pen-up
  travel adds time between strokes, so the T channel is a plausible timeline.
- Stroke width is carried by InkML brushes: elements reference "#pen", and
  title text references the wider "#bold" brush.
- With --noise > 0, scripts/hand_noise.py adds handwriting-like variation
  (writer slant, baseline drift, per-glyph jitter, elastic warping, bowed
  lines, overshoot, variable pen speed). --noise 0 gives clean geometry.

Usage: uv run --extra render python scripts/render_inkml.py [page_dir]
           [--font NAME] [--noise LEVEL] [--seed N] [--out FILE]
"""
import argparse
import json
import math
import os
import re
import xml.etree.ElementTree as ET

from HersheyFonts import HersheyFonts

from hand_noise import Hand

INKML_NS = "http://www.w3.org/2003/InkML"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"

SAMPLE_RATE = 100.0    # Hz, typical digitizer rate
TRAVEL_SPEED = 400.0   # mm/s while the pen is lifted
PEN_UP_PAUSE = 0.08    # s, minimum gap between strokes
ELEMENT_PAUSE = 0.4    # s, extra gap between writing_order entries
JUNCTION_RADIUS = 1.5  # mm
PEN_WIDTH = 2.0        # mm, normal stroke width (a whiteboard marker)
BOLD_WIDTH = 3.5       # mm, stroke width for bold elements (the title)
DIAGRAM_SCALE = 50.0   # mm, reference size passed to the noise model for diagram strokes
CIRCLE_SEGMENTS = 24


# ------------------------------------------------------------------ text

GREEK = dict(zip("αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ",
                 "abcdefghijklmnopqrstuvwxABCDEFGHIJKLMNOPQRSTUVWX"))  # Hershey 'greeks' keys

# Symbols no Hershey text font has, drawn as primitives: char -> advance in units of
# text height. Their strokes come from symbol_strokes().
SYMBOL_ADVANCE = {"⊕": None, "→": 0.75, "≤": 0.6, "≥": 0.6, "≠": 0.6, "±": 0.6, "·": 0.3,
                  "×": 0.55, "∞": 0.75, "∫": 0.45, "√": 0.55}


FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "svg")


class HersheySource:
    """A Hershey font from the HersheyFonts package, normalized so glyph y runs
    from 0 at the descender line to 1 at the cap line."""

    def __init__(self, name):
        self.font = HersheyFonts()
        self.font.load_default_font(name)
        self.font.normalize_rendering(1.0)
        o = self.font.render_options
        self.sx, self.sy, self.y0 = o.scalex, o.scaley, o.yofs
        # Baseline measured from the glyph data (the bottom of H, or A for fonts
        # where H is another letter). render_options' base_line doesn't give it.
        g = self.glyph("H" if name != "greeks" else "A")
        self.base = min(self.y0 + py * self.sy for st in g.strokes for _, py in st)

    def glyph(self, ch):
        glyphs = list(self.font.glyphs_for_text(ch))
        return glyphs[0] if glyphs else None


class SvgGlyph:
    def __init__(self, adv, strokes):
        self.char_width, self.left_offset, self.strokes = adv, 0.0, strokes


class SvgSource:
    """A single-line SVG font (fonts/svg/<name>.svg, e.g. the EMS fonts from
    Inkscape's Hershey Text extension).

    It is scaled so its measured cap height (baseline to the top of H and E)
    and baseline match the reference Hershey font's. Text in any font then
    has the same capital size for a given text_height, while descenders keep
    the font's own proportions. The font-face metrics vary too much between
    these fonts to use instead."""

    def __init__(self, path, ref_base):
        svg = "{http://www.w3.org/2000/svg}"
        font = ET.parse(path).getroot().find(f".//{svg}font")
        default_adv = float(font.get("horiz-adv-x", 500))
        self.glyphs = {}
        for g in font.findall(f"{svg}glyph"):
            ch = g.get("unicode")
            if ch is None or len(ch) != 1:
                continue
            # These fonts use absolute M, L and (in a few accented glyphs) C.
            # Commands may be glued to their first number ("M1022 40"), so
            # tokenize with a regex.
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
            self.glyphs[ch] = SvgGlyph(float(g.get("horiz-adv-x", default_adv)),
                                       [st for st in strokes if len(st) > 1])
        cap = max(y for c in "HE" for st in self.glyphs[c].strokes for _, y in st)
        self.sx = self.sy = (1.0 - ref_base) / cap
        self.y0 = self.base = ref_base

    def glyph(self, ch):
        return self.glyphs.get(ch)


def font_source(name):
    path = os.path.join(FONT_DIR, f"{name}.svg")
    if os.path.exists(path):
        return SvgSource(path, ref_base=HersheySource("futural").base)
    return HersheySource(name)


class TextRenderer:
    """Lays out strings with a single-stroke font; text_height spans descender to cap.

    `font` is a Hershey font name (e.g. futural, cursive) or the name of an SVG
    font in fonts/svg (e.g. EMSReadability). Greek letters fall back to the
    Hershey 'greeks' font, and the characters in SYMBOL_ADVANCE are drawn as
    primitives. Any other character missing from the font is an error.
    """

    def __init__(self, font):
        self.main = font
        self.sources = {font: font_source(font), "greeks": HersheySource("greeks")}
        self.base = self.sources[font].base  # baseline height, fraction of text_height
        g, sx, sy, y0, base = self._lookup("+")
        ys = [y0 + py * sy for st in g.strokes for _, py in st]
        self.op_y = (min(ys) + max(ys)) / 2 - base  # operator centre above baseline

    def _lookup(self, ch):
        """(glyph, sx, sy, y0, base) for a character, or None."""
        name, key = (self.main, ch) if ch not in GREEK else ("greeks", GREEK[ch])
        src = self.sources[name]
        g = src.glyph(key)
        return (g, src.sx, src.sy, src.y0, src.base) if g else None

    def width(self, text, h):
        return sum(self._advance(ch, h) for ch in text)

    def _advance(self, ch, h):
        if ch in SYMBOL_ADVANCE:
            adv = SYMBOL_ADVANCE[ch]
            if adv is not None:
                return adv * h
            ch = "O"
        g = self._lookup(ch)
        return g[0].char_width * g[1] * h if g else 0.55 * h

    def strokes(self, text, x0, y_bottom, h, hand, wid, max_width=None):
        """Strokes (lists of (x, y) in page mm) for text whose box bottom is y_bottom.

        Glyphs are built in line coordinates (X right from x0, Y up from the
        baseline), varied per glyph and per line by `hand`, then mapped to the page.
        If the text would be wider than max_width, it is squeezed horizontally
        (to no less than 75%), as a writer running out of room would.
        """
        natural = self.width(text, h)
        squeeze = 1.0 if not max_width or natural <= max_width else max(0.75, max_width / natural)
        line = hand.text_line(wid, h, natural * squeeze)
        w = line["w"]
        hs = h * w["size"]
        glyphs, x = [], 0.0
        for ch in text:
            adv = self._advance(ch, hs)
            if ch in SYMBOL_ADVANCE:
                local = symbol_strokes(ch, adv, hs, self.op_y * hs, (0.68 - self.base) * hs, hand)
            else:
                found = self._lookup(ch)
                if found is None:
                    raise ValueError(f"font {self.main!r} has no glyph for {ch!r}")
                g, sx, sy, y0, base = found
                local = [[((px - g.left_offset) * sx * hs, (y0 + py * sy - base) * hs)
                          for px, py in st] for st in g.strokes]
            # The writer's persistent shape for this character, then this
            # occurrence's own variation, both about the glyph's middle.
            inst = hand.glyph(hs)
            local = hand.shape_glyph(local, [hand.letter(wid, ch), inst], adv / 2, 0.3 * hs, hs)
            glyphs += [[(x + px, py + inst["dy"]) for px, py in st] for st in local]
            x += adv + inst["space"]
        y_base = y_bottom - self.base * h
        ct, st_ = math.cos(line["tilt"]), math.sin(line["tilt"])
        out = []
        for st in glyphs:
            pts = []
            for X, Y in st:
                X = X * squeeze + w["slant"] * Y
                Y += line["curve"] * X * X
                X, Y = ct * X - st_ * Y, st_ * X + ct * Y
                pts.append((x0 + line["dx"] + X, y_base + line["dy"] - Y))
            out.append(hand.deform(pts, h, wid, "text"))
        return out


def oplus(cx, cy, r, hand):
    """⊕ in y-up glyph coordinates: a circle, then a horizontal and a vertical bar."""
    # circle() works in y-down page coordinates; mirror it (and reverse it, to
    # keep the drawing direction) so it still starts at the top here.
    ring = [(x, 2 * cy - y) for x, y in reversed(circle(cx, cy, r, hand))]
    return [ring, [(cx - r, cy), (cx + r, cy)], [(cx, cy + r), (cx, cy - r)]]


def symbol_strokes(ch, w, h, c, cap_mid, hand):
    """Strokes for a primitive-drawn symbol, in y-up glyph coordinates.

    w is the advance and h the text height (mm). c is the height of an
    operator's centre above the baseline, and cap_mid the middle of a capital.
    """
    if ch == "⊕":
        return oplus(w / 2, cap_mid, 0.27 * h, hand)
    m, d = w / 2, 0.11 * h
    if ch == "→":
        tip = 0.92 * w
        return [[(0.08 * w, c), (tip, c)], [(tip - 0.16 * h, c + 0.1 * h), (tip, c), (tip - 0.16 * h, c - 0.1 * h)]]
    if ch in "≤≥":
        a, b = (0.85 * w, 0.15 * w) if ch == "≤" else (0.15 * w, 0.85 * w)
        return [[(a, c + 0.24 * h), (b, c + 0.06 * h), (a, c - 0.12 * h)], [(b, c - 0.24 * h), (a, c - 0.24 * h)]]
    if ch == "≠":
        return [[(0.12 * w, c + 0.08 * h), (0.88 * w, c + 0.08 * h)],
                [(0.12 * w, c - 0.08 * h), (0.88 * w, c - 0.08 * h)],
                [(0.66 * w, c + 0.24 * h), (0.34 * w, c - 0.24 * h)]]
    if ch == "±":
        cy = c + 0.08 * h
        return [[(m - 0.17 * h, cy), (m + 0.17 * h, cy)], [(m, cy + 0.17 * h), (m, cy - 0.17 * h)],
                [(m - 0.17 * h, c - 0.2 * h), (m + 0.17 * h, c - 0.2 * h)]]
    if ch == "·":
        r = 0.03 * h
        return [[(m + r * math.cos(a * math.pi / 4), c + r * math.sin(a * math.pi / 4)) for a in range(9)]]
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
    raise ValueError(ch)


# ------------------------------------------------------------ primitives

def circle(cx, cy, r, hand, n=CIRCLE_SEGMENTS):
    """Starts at the top and runs counter-clockwise on the page, like most writers.
    `hand` makes it slightly elliptical and over- or under-closed."""
    aspect, axis, overrun, start = hand.circle_params()
    sweep = 2 * math.pi + overrun
    m = max(3, round(n * sweep / (2 * math.pi)))
    ca, sa = math.cos(axis), math.sin(axis)
    pts = []
    for i in range(m + 1):
        a = start + sweep * i / m
        ex, ey = -r * math.sin(a) * aspect, -r * math.cos(a)
        pts.append((cx + ca * ex - sa * ey, cy + sa * ex + ca * ey))
    return pts


def bezier(p0, p1, p2, n=16):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1])
            for t in (i / n for i in range(n + 1))]


def back_curve_x(x, y0, y1, bulge, y):
    """x of the concave back curve drawn by bezier((x, y1), (x + 2 * bulge, ym), (x, y0))."""
    u = (y - y0) / (y1 - y0)  # the Bezier parameter is linear in y here
    return x + 4 * u * (1 - u) * bulge


def gate_strokes(kind, bbox, ports):
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    ym = (y0 + y1) / 2
    if kind == "and2":
        r = h / 2
        xa = x1 - r
        arc = [(xa + r * math.sin(math.pi * i / 16), ym - r * math.cos(math.pi * i / 16))
               for i in range(17)]
        body = [(x0, y0), (xa, y0)] + arc + [(x0, y1), (x0, y0)]
        return [body]
    if kind in ("or2", "xor2"):
        gap = 0.12 * w if kind == "xor2" else 0.0
        bx = x0 + gap  # back of the OR body
        bulge = 0.12 * w
        back = bezier((bx, y1), (bx + 2 * bulge, ym), (bx, y0))
        body = (bezier((bx, y0), (bx + 0.6 * (x1 - bx), y0), (x1, ym))
                + bezier((x1, ym), (bx + 0.6 * (x1 - bx), y1), (bx, y1))[1:]
                + back[1:])
        strokes = [body]
        if kind == "xor2":
            strokes.append(bezier((x0, y0), (x0 + 2 * bulge, ym), (x0, y1)))
        # Input stubs from the port to the body's back curve, so wires touch the gate.
        for p in ("a", "b"):
            px, py = ports[p]
            strokes.append([(px, py), (back_curve_x(bx, y0, y1, bulge, py), py)])
        return strokes
    raise ValueError(f"unknown gate type {kind}")


# --------------------------------------------------------------- page

def on_polyline(p, polyline, tol=1e-6):
    for a, b in zip(polyline, polyline[1:]):
        L = math.dist(a, b)
        if L and abs(math.dist(a, p) + math.dist(p, b) - L) < tol:
            return True
    return False


# --------------------------------------------------------------- sketch

def rounded_rect(bbox, r):
    """Closed stroke starting at the top-left, clockwise on the page."""
    x0, y0, x1, y1 = bbox
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    if r <= 0:
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    pts = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        pts += [(cx + r * math.cos(math.radians(a0 + 90 * i / 6)), cy + r * math.sin(math.radians(a0 + 90 * i / 6)))
                for i in range(7)]
    return pts + [pts[0]]


def sketch_item_strokes(item, text, hand, wid):
    """Strokes for one generic sketch item (see artifacts/README.md, "sketch")."""
    kind = item["kind"]
    lines = lambda pls: [hand.deform(hand.line(pl), DIAGRAM_SCALE, wid, "line") for pl in pls]
    if kind == "text":
        x0, _, x1, y1 = item["bbox"]
        h = item["text_height"]
        width = min(text.width(item["text"], h), x1 - x0)
        if item.get("align", "left") == "center":
            x0 = (x0 + x1) / 2 - width / 2
        return text.strokes(item["text"], x0, y1, h, hand, wid, max_width=x1 - x0)
    if kind == "polyline":
        pts = [tuple(p) for p in item["points"]]
        return lines([pts + [pts[0]] if item.get("closed") else pts])
    if kind == "rect":
        return [hand.deform(rounded_rect(item["bbox"], item.get("corner_radius", 0)), DIAGRAM_SCALE, wid, "line")]
    if kind == "diamond":
        x0, y0, x1, y1 = item["bbox"]
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        return lines([[(xm, y0), (x1, ym), (xm, y1), (x0, ym), (xm, y0)]])
    if kind == "ellipse":
        x0, y0, x1, y1 = item["bbox"]
        rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
        pts = circle((x0 + x1) / 2, (y0 + y1) / 2, 1.0, hand, n=40)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        pts = [(cx + (x - cx) * rx, cy + (y - cy) * ry) for x, y in pts]
        return [hand.deform(pts, DIAGRAM_SCALE, wid, "line")]
    if kind == "arrow":
        pts = [tuple(p) for p in item["points"]]
        shaft = lines([pts])[0]
        # The head follows the drawn shaft's end, so it stays attached under noise.
        tip, back = shaft[-1], shaft[max(0, len(shaft) - 4)]
        ux, uy = tip[0] - back[0], tip[1] - back[1]
        n = math.hypot(ux, uy) or 1.0
        ux, uy = ux / n, uy / n
        hs = item.get("head", 9.0)
        a = math.radians(hand.g(25, 4))
        left = (tip[0] - hs * (ux * math.cos(a) - uy * math.sin(a)), tip[1] - hs * (uy * math.cos(a) + ux * math.sin(a)))
        right = (tip[0] - hs * (ux * math.cos(a) + uy * math.sin(a)), tip[1] - hs * (uy * math.cos(a) - ux * math.sin(a)))
        return [shaft, hand.deform([left, tip, right], DIAGRAM_SCALE, wid, "line")]
    raise ValueError(f"unknown sketch item kind {kind!r}")


def sketch_item(eid, layout):
    for e in layout["elements"].values():
        if eid in e.get("items", {}):
            return e["items"][eid]
    return None


def element_strokes(eid, content, layout, text, hand):
    """Strokes for one writing_order entry, in the order they are drawn."""
    els = layout["elements"]
    wid = writer_of(eid, layout)
    lines = lambda pls: [hand.deform(hand.line(pl), DIAGRAM_SCALE, wid, "line") for pl in pls]
    if eid in els:  # text or math line
        cel = content[eid]
        s = cel.get("written", cel.get("text"))
        x0, _, x1, y1 = els[eid]["bbox"]
        return text.strokes(s, x0, y1, els[eid]["text_height"], hand, wid, max_width=x1 - x0)

    item = sketch_item(eid, layout)
    if item is not None:
        return sketch_item_strokes(item, text, hand, wid)

    container, _, _ = eid.partition(".")
    if container in els and eid in els[container].get("cells", {}):
        tt = els[container]
        cell = tt["cells"][eid]
        s = content[eid]["text"]
        h = tt["text_height"]
        tx0, _, tx1, ty1 = cell["text_bbox"]
        x = (tx0 + tx1) / 2 - text.width(s, h) / 2
        return text.strokes(s, x, ty1, h, hand, wid)
    if container in els and eid in els[container].get("rules", {}):
        return lines([els[container]["rules"][eid]["polyline"]])

    for dia in (e for e in els.values() if "components" in e):
        if eid in dia["components"]:
            comp, ccomp = dia["components"][eid], content[eid]
            if ccomp["type"] in ("input_pin", "output_pin"):
                lx0, _, _, ly1 = comp["label_bbox"]
                (px, py), = comp["ports"].values()
                r = (comp["bbox"][2] - comp["bbox"][0]) / 2
                return (text.strokes(ccomp["label"], lx0, ly1, comp["label_text_height"], hand, wid)
                        + [hand.deform(circle(px, py, r, hand), r * 4, wid, "text")])
            return lines(gate_strokes(ccomp["type"], comp["bbox"], comp["ports"]))
        if eid in dia["wires"]:
            pl = [tuple(p) for p in dia["wires"][eid]["polyline"]]
            # Wires are stored point to point, so a fan-out branch repeats its
            # net's trunk. A writer draws the trunk once: if an earlier wire of the
            # same net already passes through one of this wire's junctions, the
            # stroke starts there, and the junction dot goes in with it.
            net = content[eid]["net"]
            earlier = layout["writing_order"][:layout["writing_order"].index(eid)]
            drawn = [dia["wires"][w]["polyline"] for w in earlier
                     if w in dia["wires"] and content[w]["net"] == net]
            dots = []
            for j in dia["junctions"]:
                jp = tuple(j["point"])
                if j["net"] == net and jp in pl[1:-1] and any(on_polyline(jp, d) for d in drawn):
                    pl = pl[pl.index(jp):]
                    dots = [jp]
            strokes = lines([pl])
            for jp in dots:
                strokes.append(circle(*strokes[0][0], JUNCTION_RADIUS, hand, n=8))
            return strokes
        if eid in dia["labels"]:
            lab = dia["labels"][eid]
            x0, _, _, y1 = lab["bbox"]
            return text.strokes(content[eid]["text"], x0, y1, lab["text_height"], hand, wid)
    raise KeyError(f"no layout for {eid}")


def index_content(content):
    """Flat id -> content dict for every element and sub-element."""
    idx = {}
    for e in content["elements"]:
        idx[e["id"]] = e
        for h in e.get("header", []):
            idx[h["id"]] = h
        for r in e.get("rows", []):
            for c in r["cells"]:
                idx[c["id"]] = c
        for key in ("components", "wires", "labels", "rules"):
            for sub in e.get(key, []):
                idx[sub["id"]] = sub
    return idx


def writer_of(eid, layout):
    els = layout["elements"]
    if eid in els:
        return els[eid]["writer"]
    for e in els.values():
        for key in ("cells", "rules", "components", "wires", "labels", "items"):
            if eid in e.get(key, {}):
                return e[key][eid]["writer"]
    raise KeyError(eid)


def render(page_dir, font, noise, seed, out):
    content = json.load(open(os.path.join(page_dir, "content.json")))
    layout = json.load(open(os.path.join(page_dir, "layout.json")))
    cidx = index_content(content)
    text = TextRenderer(font)
    hand = Hand(seed, noise)

    ET.register_namespace("", INKML_NS)
    q = lambda tag: f"{{{INKML_NS}}}{tag}"
    ink = ET.Element(q("ink"))
    defs = ET.SubElement(ink, q("definitions"))
    ctx = ET.SubElement(defs, q("context"), {XML_ID: "ctx0"})
    fmt = ET.SubElement(ctx, q("traceFormat"))
    for name, units in (("X", "mm"), ("Y", "mm"), ("T", "s")):
        ET.SubElement(fmt, q("channel"), {"name": name, "type": "decimal", "units": units})
    for bid, width in (("pen", PEN_WIDTH), ("bold", BOLD_WIDTH)):
        brush = ET.SubElement(defs, q("brush"), {XML_ID: bid})
        for prop in ("width", "height"):
            ET.SubElement(brush, q("brushProperty"), {"name": prop, "value": f"{width:g}", "units": "mm"})
    page = layout["page"]
    for k, v in (("page_id", layout["page_id"]), ("page_width_mm", page["width"]),
                 ("page_height_mm", page["height"]), ("y_axis", page["y_axis"]),
                 ("generator", f"scripts/render_inkml.py hershey:{font}"),
                 ("noise_level", noise), ("noise_seed", seed)):
        a = ET.SubElement(ink, q("annotation"), {"type": k})
        a.text = str(v)

    traces, groups, t, pen = [], [], 0.0, None
    for eid in layout["writing_order"]:
        refs, wid = [], writer_of(eid, layout)
        for stroke in element_strokes(eid, cidx, layout, text, hand):
            stroke = [tuple(p) for p in stroke]
            if pen is not None:
                t += hand.pause(PEN_UP_PAUSE) + math.dist(pen, stroke[0]) / TRAVEL_SPEED
            pts, t = hand.time_stroke(stroke, t, wid, SAMPLE_RATE)
            pen = stroke[-1]
            tid = f"t{len(traces)}"
            bold = cidx.get(eid, {}).get("role") == "title" or (sketch_item(eid, layout) or {}).get("bold")
            traces.append((tid, pts, "bold" if bold else "pen"))
            refs.append(tid)
        groups.append((eid, wid, refs))
        t += hand.pause(ELEMENT_PAUSE)

    for tid, pts, brush in traces:
        el = ET.SubElement(ink, q("trace"), {XML_ID: tid, "contextRef": "#ctx0", "brushRef": f"#{brush}"})
        el.text = ", ".join(f"{x:.2f} {y:.2f} {tt:.3f}" for x, y, tt in pts)
    root = ET.SubElement(ink, q("traceGroup"), {XML_ID: "elements"})
    for eid, writer, refs in groups:
        g = ET.SubElement(root, q("traceGroup"))
        ET.SubElement(g, q("annotation"), {"type": "element_id"}).text = eid
        ET.SubElement(g, q("annotation"), {"type": "writer"}).text = writer
        for tid in refs:
            ET.SubElement(g, q("traceView"), {"traceDataRef": f"#{tid}"})

    ET.indent(ink)
    if not out:
        if noise == 0 and font == "futural":
            out = os.path.join(page_dir, "page.inkml")
        else:
            os.makedirs(os.path.join(page_dir, "renders"), exist_ok=True)
            out = os.path.join(page_dir, "renders", f"{font}_noise{noise:g}_seed{seed}.inkml")
    ET.ElementTree(ink).write(out, encoding="utf-8", xml_declaration=True)
    n_pts = sum(len(p) for _, p, _ in traces)
    print(f"wrote {out}: {len(groups)} elements, {len(traces)} traces, {n_pts} points, {t:.1f} s of writing")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("page_dir", nargs="?", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "artifacts", "page_0001"))
    ap.add_argument("--font", default="futural",
                    help="Hershey font name, or an SVG font in fonts/svg (default: futural)")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="handwriting variation level: 0 = clean geometry, 1 = default amount")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="output path (default: page.inkml for clean futural, "
                         "else renders/<font>_noise<L>_seed<N>.inkml)")
    args = ap.parse_args()
    render(args.page_dir, args.font, args.noise, args.seed, args.out)
