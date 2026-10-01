"""Naive ink renderer: content.json + layout.json -> page.inkml.

Proof of concept for the "text to ink" and "diagrams to ink" stages in
docs/synthetic_data.md, without any handwriting model or jitter:

- Text is drawn with a single-stroke Hershey font, scaled to the layout's
  text_height. Characters the font lacks (only "⊕" on page_0001) are drawn
  as primitives.
- Diagrams are drawn from geometric primitives (gate outlines, pin circles,
  wire polylines, junction dots).
- Elements are written in layout.json's writing_order. Each stroke is
  sampled at a fixed rate while the pen moves at constant speed, and pen-up
  travel adds time between strokes, so the T channel is a plausible timeline.

Usage: uv run --extra render python scripts/render_inkml.py [page_dir] [--font NAME]
"""
import argparse
import json
import math
import os
import xml.etree.ElementTree as ET

from HersheyFonts import HersheyFonts

INKML_NS = "http://www.w3.org/2003/InkML"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"

PEN_SPEED = 150.0      # mm/s while the pen is down
SAMPLE_RATE = 100.0    # Hz, typical digitizer rate
TRAVEL_SPEED = 400.0   # mm/s while the pen is lifted
PEN_UP_PAUSE = 0.08    # s, minimum gap between strokes
ELEMENT_PAUSE = 0.4    # s, extra gap between writing_order entries
JUNCTION_RADIUS = 1.5  # mm
CIRCLE_SEGMENTS = 24


# ------------------------------------------------------------------ text

class TextRenderer:
    """Lays out strings with a Hershey font; text_height spans descender to cap."""

    def __init__(self, font):
        self.font = HersheyFonts()
        self.font.load_default_font(font)
        self.font.normalize_rendering(1.0)  # glyph y in [0, 1]: descender bottom to cap
        opts = self.font.render_options
        self.sx, self.sy, self.y0 = opts.scalex, opts.scaley, opts.yofs

    def _glyph(self, ch):
        glyphs = list(self.font.glyphs_for_text(ch))
        return glyphs[0] if glyphs else None

    def width(self, text, h):
        return sum(self._advance(ch, h) for ch in text)

    def _advance(self, ch, h):
        g = self._glyph("O" if ch == "⊕" else ch)
        return g.char_width * self.sx * h if g else 0.55 * h

    def strokes(self, text, x0, y_bottom, h):
        """Strokes (lists of (x, y) in page mm) for text whose box bottom is y_bottom."""
        out, x = [], x0
        for ch in text:
            adv = self._advance(ch, h)
            if ch == "⊕":
                out += oplus(x + adv / 2, y_bottom - 0.68 * h, 0.27 * h)
            else:
                g = self._glyph(ch)
                if g is None:
                    raise ValueError(f"font has no glyph for {ch!r}")
                for s in g.strokes:
                    out.append([(x + (px - g.left_offset) * self.sx * h,
                                 y_bottom - (self.y0 + py * self.sy) * h) for px, py in s])
            x += adv
        return out


def oplus(cx, cy, r):
    return [circle(cx, cy, r), [(cx - r, cy), (cx + r, cy)], [(cx, cy - r), (cx, cy + r)]]


# ------------------------------------------------------------ primitives

def circle(cx, cy, r, n=CIRCLE_SEGMENTS):
    # Starts at the top and runs counter-clockwise on the page, like most writers.
    return [(cx - r * math.sin(2 * math.pi * i / n), cy - r * math.cos(2 * math.pi * i / n))
            for i in range(n + 1)]


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

def element_strokes(eid, content, layout, text):
    """Strokes for one writing_order entry, in the order they are drawn."""
    els = layout["elements"]
    if eid in els:  # text or math line
        cel = content[eid]
        s = cel.get("written", cel.get("text"))
        x0, _, _, y1 = els[eid]["bbox"]
        return text.strokes(s, x0, y1, els[eid]["text_height"])

    container, _, _ = eid.partition(".")
    if container in els and eid in els[container].get("cells", {}):
        tt = els[container]
        cell = tt["cells"][eid]
        s = content[eid]["text"]
        h = tt["text_height"]
        tx0, _, tx1, ty1 = cell["text_bbox"]
        x = (tx0 + tx1) / 2 - text.width(s, h) / 2
        return text.strokes(s, x, ty1, h)
    if container in els and eid in els[container].get("rules", {}):
        return [els[container]["rules"][eid]["polyline"]]

    for dia in (e for e in els.values() if "components" in e):
        if eid in dia["components"]:
            comp, ccomp = dia["components"][eid], content[eid]
            if ccomp["type"] in ("input_pin", "output_pin"):
                lx0, _, _, ly1 = comp["label_bbox"]
                (px, py), = comp["ports"].values()
                r = (comp["bbox"][2] - comp["bbox"][0]) / 2
                return text.strokes(ccomp["label"], lx0, ly1, comp["label_text_height"]) + [circle(px, py, r)]
            return gate_strokes(ccomp["type"], comp["bbox"], comp["ports"])
        if eid in dia["wires"]:
            strokes = [dia["wires"][eid]["polyline"]]
            # A junction dot goes in with the wire that branches off it.
            for j in dia["junctions"]:
                pl = dia["wires"][eid]["polyline"]
                if content[eid]["net"] == j["net"] and list(j["point"]) in [list(p) for p in pl[1:-1]]:
                    strokes.append(circle(*j["point"], JUNCTION_RADIUS, n=8))
            return strokes
        if eid in dia["labels"]:
            lab = dia["labels"][eid]
            x0, _, _, y1 = lab["bbox"]
            return text.strokes(content[eid]["text"], x0, y1, lab["text_height"])
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
        for key in ("cells", "rules", "components", "wires", "labels"):
            if eid in e.get(key, {}):
                return e[key][eid]["writer"]
    raise KeyError(eid)


def sample_stroke(points, t0):
    """Constant-speed resampling at SAMPLE_RATE. Returns [(x, y, t)] and end time."""
    seg = [math.dist(a, b) for a, b in zip(points, points[1:])]
    total = sum(seg)
    step = PEN_SPEED / SAMPLE_RATE
    n = max(1, math.ceil(total / step))
    out, i, acc = [], 0, 0.0
    for k in range(n + 1):
        d = min(total, k * total / n)
        while i < len(seg) - 1 and acc + seg[i] < d:
            acc += seg[i]
            i += 1
        f = 0.0 if not seg or seg[i] == 0 else (d - acc) / seg[i]
        a, b = points[i], points[min(i + 1, len(points) - 1)]
        out.append((a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), t0 + k * (total / n) / PEN_SPEED))
    return out, out[-1][2]


def render(page_dir, font):
    content = json.load(open(os.path.join(page_dir, "content.json")))
    layout = json.load(open(os.path.join(page_dir, "layout.json")))
    cidx = index_content(content)
    text = TextRenderer(font)

    ET.register_namespace("", INKML_NS)
    q = lambda tag: f"{{{INKML_NS}}}{tag}"
    ink = ET.Element(q("ink"))
    defs = ET.SubElement(ink, q("definitions"))
    ctx = ET.SubElement(defs, q("context"), {XML_ID: "ctx0"})
    fmt = ET.SubElement(ctx, q("traceFormat"))
    for name, units in (("X", "mm"), ("Y", "mm"), ("T", "s")):
        ET.SubElement(fmt, q("channel"), {"name": name, "type": "decimal", "units": units})
    page = layout["page"]
    for k, v in (("page_id", layout["page_id"]), ("page_width_mm", page["width"]),
                 ("page_height_mm", page["height"]), ("y_axis", page["y_axis"]),
                 ("generator", f"scripts/render_inkml.py hershey:{font}")):
        a = ET.SubElement(ink, q("annotation"), {"type": k})
        a.text = str(v)

    traces, groups, t, pen = [], [], 0.0, None
    for eid in layout["writing_order"]:
        refs = []
        for stroke in element_strokes(eid, cidx, layout, text):
            stroke = [tuple(p) for p in stroke]
            if pen is not None:
                t += PEN_UP_PAUSE + math.dist(pen, stroke[0]) / TRAVEL_SPEED
            pts, t = sample_stroke(stroke, t)
            pen = stroke[-1]
            tid = f"t{len(traces)}"
            traces.append((tid, pts))
            refs.append(tid)
        groups.append((eid, writer_of(eid, layout), refs))
        t += ELEMENT_PAUSE

    for tid, pts in traces:
        el = ET.SubElement(ink, q("trace"), {XML_ID: tid, "contextRef": "#ctx0"})
        el.text = ", ".join(f"{x:.2f} {y:.2f} {tt:.3f}" for x, y, tt in pts)
    root = ET.SubElement(ink, q("traceGroup"), {XML_ID: "elements"})
    for eid, writer, refs in groups:
        g = ET.SubElement(root, q("traceGroup"))
        ET.SubElement(g, q("annotation"), {"type": "element_id"}).text = eid
        ET.SubElement(g, q("annotation"), {"type": "writer"}).text = writer
        for tid in refs:
            ET.SubElement(g, q("traceView"), {"traceDataRef": f"#{tid}"})

    ET.indent(ink)
    out = os.path.join(page_dir, "page.inkml")
    ET.ElementTree(ink).write(out, encoding="utf-8", xml_declaration=True)
    n_pts = sum(len(p) for _, p in traces)
    print(f"wrote {out}: {len(groups)} elements, {len(traces)} traces, {n_pts} points, {t:.1f} s of writing")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("page_dir", nargs="?", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "artifacts", "page_0001"))
    ap.add_argument("--font", default="futural", help="Hershey font name (default: futural)")
    args = ap.parse_args()
    render(args.page_dir, args.font)
