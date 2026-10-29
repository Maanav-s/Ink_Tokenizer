"""Ink renderer: layout.json -> InkML.

A naive stand-in for the "text to ink" and "diagrams to ink" stages in
docs/synthetic_data.md, without any handwriting model:

- Text and math are drawn with single-stroke fonts (scripts/fonts.py); math
  is typeset by scripts/math_typeset.py. Diagrams are drawn from geometric
  shapes (scripts/shapes.py).
- Items are written in layout.json's writing_order. Each writer has their own
  clock, so on multi-writer pages their strokes interleave in time. Each
  stroke is sampled by the device (scripts/digitizer.py) while the pen moves
  at a varying speed, and pen-up travel adds time between strokes.
- With --noise > 0, scripts/hand_noise.py adds handwriting-like variation and
  the digitizer adds capture artefacts. --noise 0 gives clean geometry.
- Every stroke is labelled: one <traceGroup> per writing_order item, holding
  one <traceGroup> per character (text, math) or part (shapes). Strokes the
  writer crossed out are labelled with a status.
- If layout.json has completion targets, a sidecar <out>.targets.json
  resolves them to times and trace ids for this render.

Usage: uv run --extra render python scripts/render_inkml.py [page_dir]
           [--layout FILE] [--font NAME[,NAME...]] [--noise LEVEL] [--seed N]
           [--out FILE] [--no-device]
--font takes one font per writer, in the layout's writer order (cycled).
"""
import argparse
import json
import math
import os
import xml.etree.ElementTree as ET

import shapes
from digitizer import Device
from fonts import TextRenderer
from hand_noise import Hand

INKML_NS = "http://www.w3.org/2003/InkML"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"

TRAVEL_SPEED = 400.0   # mm/s while the pen is lifted
PEN_UP_PAUSE = 0.08    # s, minimum gap between strokes
ELEMENT_PAUSE = 0.4    # s, extra gap between writing_order entries
JUNCTION_RADIUS = 1.5  # mm
PEN_WIDTH = 2.0        # mm, normal stroke width (a whiteboard marker)
BOLD_WIDTH = 3.5       # mm, stroke width for bold items (titles)
DIAGRAM_SCALE = 50.0   # mm, reference size passed to the noise model for diagram strokes
MIN_SQUEEZE = 0.75     # text never gets narrower than this to fit its box


class Unit:
    """Strokes for one labelled piece of an item: a character or a shape part."""

    def __init__(self, label, strokes):
        self.label, self.strokes = label, strokes


# ------------------------------------------------------------------ text

def hand_circle(cx, cy, r, hand, n=24):
    """A hand-drawn circle: starts near the top, runs counter-clockwise on the
    page, comes out slightly elliptical and over- or under-closed."""
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


def glyph_unit(font, ch, x, y, size, hand, wid, label):
    """One character in line coordinates (y up), at pen position x and baseline
    y, with the writer's allograph and this occurrence's variation. Returns
    the unit and the occurrence's extra spacing (mm)."""
    adv = font.advance(ch, size)
    local = font.char_strokes(ch, size)
    inst = hand.glyph(size)
    local = hand.shape_glyph(local, [hand.letter(wid, ch), inst], adv / 2, 0.3 * size, size)
    return Unit(label, [[(x + px, y + py + inst["dy"]) for px, py in st] for st in local]), inst["space"]


def text_plan(text, corrections):
    """The order the writer produces a line in, including crossed-out attempts.

    Returns a list of steps: ("char", i, ch, status, anchor) and
    ("mark", style, first_step, last_step). i is the index in `text` (None for
    a wrong character), status is None or "deleted", and anchor is the step
    whose position an overwriting character reuses (else None).

    A correction {index, wrong, style} means the writer first wrote `wrong`
    where text[index:] starts. "strike" and "scribble" cross it out and carry
    on to its right; "overwrite" writes the right characters (`replace` of
    them, default len(wrong)) on top of the wrong ones.
    """
    steps, by_index = [], {c["index"]: c for c in corrections or []}
    i = 0
    while i < len(text):
        c = by_index.get(i)
        if c:
            first = len(steps)
            for ch in c["wrong"]:
                steps.append(("char", None, ch, "deleted", None))
            if c["style"] == "overwrite":
                n = c.get("replace", len(c["wrong"]))
                for k in range(n):
                    steps.append(("char", i + k, text[i + k], None, first + k if k < len(c["wrong"]) else None))
                i += n
                continue
            steps.append(("mark", c["style"], first, len(steps) - 1))
        steps.append(("char", i, text[i], None, None))
        i += 1
    return steps


def mark_strokes(style, x0, x1, h, hand):
    """A strike-through line or a scribble over [x0, x1], line coordinates."""
    if style == "strike":
        y = 0.32 * h
        return [[(x0 - 0.05 * h, y + hand.g(0, 0.03 * h)), (x1 + 0.05 * h, y + hand.g(0, 0.03 * h))]]
    if style == "scribble":
        n = max(3, round((x1 - x0) / (0.12 * h)))
        return [[(x0 + (x1 - x0) * k / n + hand.g(0, 0.02 * h), (0.75 * h if k % 2 else -0.02 * h) + hand.g(0, 0.04 * h))
                 for k in range(n + 1)]]
    raise ValueError(f"unknown correction style {style!r}")


def text_line_units(font, text, h, hand, wid, corrections=None):
    """Units for one line of text in line coordinates, and its natural width."""
    hs = h * hand.writer(wid)["size"]
    units, xs, x = [], [], 0.0  # one units/xs entry per plan step
    for step in text_plan(text, corrections):
        if step[0] == "mark":
            _, style, a, b = step
            x_b = xs[b] + font.advance(units[b].label["truth"], hs)
            units.append(Unit({"truth": style, "status": "mark"}, mark_strokes(style, xs[a], x_b, hs, hand)))
            xs.append(xs[a])
            x += 0.25 * hs
            continue
        _, i, ch, status, anchor = step
        xp = xs[anchor] if anchor is not None else x
        label = {"truth": ch}
        if i is not None:
            label["char_index"] = i
        if status:
            label["status"] = status
        xs.append(xp)
        if ch == " ":
            units.append(Unit(label, []))
            space = 0.0
        else:
            u, space = glyph_unit(font, ch, xp, 0.0, hs, hand, wid, label)
            units.append(u)
        if anchor is None:
            x += font.advance(ch, hs) + space
    return [u for u in units if u.strokes], x


def place_line(units, x0, y_base, h, width, hand, wid, squeeze=1.0, max_drift=0.3):
    """Map units from line coordinates to the page, with the writer's slant and
    this line's offset, tilt and curvature; then stroke-level variation."""
    line = hand.text_line(wid, h, width * squeeze, max_drift)
    w = line["w"]
    ct, st_ = math.cos(line["tilt"]), math.sin(line["tilt"])
    out = []
    for u in units:
        strokes = []
        for st in u.strokes:
            pts = []
            for X, Y in st:
                X = X * squeeze + w["slant"] * Y
                Y += line["curve"] * X * X
                X, Y = ct * X - st_ * Y, st_ * X + ct * Y
                pts.append((x0 + line["dx"] + X, y_base + line["dy"] - Y))
            strokes.append(hand.deform(pts, h, wid, "text"))
        out.append(Unit(u.label, strokes))
    return out


def text_units(item, font, hand, wid):
    h = item["text_height"]
    x0, _, x1, y1 = item["bbox"]
    units, natural = text_line_units(font, item["text"], h, hand, wid, item.get("corrections"))
    box = x1 - x0
    squeeze = 1.0 if natural <= box else max(MIN_SQUEEZE, box / natural)
    width = natural * squeeze
    align = item.get("align", "left")
    if align == "center":
        x0 = (x0 + x1) / 2 - width / 2
    elif align == "right":
        x0 = x1 - width
    drift = 0.12 if item.get("role") in ("cell", "header_cell") else 0.3
    return place_line(units, x0, y1 - font.base * h, h, natural, hand, wid, squeeze, drift)


def math_units(item, font, hand, wid):
    """An equation typeset in line coordinates, then placed like a text line, so
    the whole equation shares one slant, tilt and offset."""
    import math_typeset
    h = item["text_height"]
    eq = math_typeset.typeset(item["latex"], h * hand.writer(wid)["size"], font, item.get("options"))
    units = []
    for atom in eq.atoms:
        if atom["type"] == "char":
            label = {"truth": atom["ch"]}
            if "src" in atom:
                label["src"] = atom["src"]
            u, _ = glyph_unit(font, atom["ch"], atom["x"], atom["y"], atom["size"], hand, wid, label)
        else:
            u = Unit({"truth": atom["part"]}, [hand.line(st) for st in atom["strokes"]])
        units.append(u)
    x0, _, x1, _ = item["bbox"]
    box = x1 - x0
    squeeze = 1.0 if eq.width <= box else max(MIN_SQUEEZE, box / eq.width)
    return [u for u in place_line(units, x0, item["baseline"], h, eq.width, hand, wid, squeeze) if u.strokes]


# --------------------------------------------------------------- shapes

def item_units(item, fonts, hand):
    """Units for one layout item, in drawing order. fonts: writer id -> TextRenderer."""
    kind, wid = item["kind"], item["writer"]
    font = fonts[wid]

    def lines(part, pls):
        return Unit({"truth": part}, [hand.deform(hand.line(pl), DIAGRAM_SCALE, wid, "line") for pl in pls])

    def outline(part, pts):
        return Unit({"truth": part}, [hand.deform(pts, DIAGRAM_SCALE, wid, "line")])

    if kind == "text":
        return text_units(item, font, hand, wid)
    if kind == "math":
        return math_units(item, font, hand, wid)
    if kind == "polyline":
        pts = [tuple(p) for p in item["points"]]
        return [lines(item.get("part", "line"), [pts + [pts[0]] if item.get("closed") else pts])]
    if kind == "rect":
        return [outline("rect", shapes.rounded_rect(item["bbox"], item.get("corner_radius", 0)))]
    if kind == "diamond":
        return [lines("diamond", [shapes.diamond(item["bbox"])])]
    if kind == "parallelogram":
        return [lines("parallelogram", [shapes.parallelogram(item["bbox"], item.get("skew", 8))])]
    if kind == "ellipse":
        x0, y0, x1, y1 = item["bbox"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        pts = [(cx + (x - cx) * (x1 - x0) / 2, cy + (y - cy) * (y1 - y0) / 2)
               for x, y in hand_circle(cx, cy, 1.0, hand, n=40)]
        return [outline("ellipse", pts)]
    if kind == "arrow":
        pts = [tuple(p) for p in item["points"]]
        shaft = hand.deform(hand.line(pts), DIAGRAM_SCALE, wid, "line")
        # The head follows the drawn shaft's end, so it stays attached under noise.
        tip, back = shaft[-1], shaft[max(0, len(shaft) - 4)]
        ux, uy = tip[0] - back[0], tip[1] - back[1]
        n = math.hypot(ux, uy) or 1.0
        ux, uy = ux / n, uy / n
        hs = item.get("head", 9.0)
        a = math.radians(hand.g(25, 4))
        left = (tip[0] - hs * (ux * math.cos(a) - uy * math.sin(a)), tip[1] - hs * (uy * math.cos(a) + ux * math.sin(a)))
        right = (tip[0] - hs * (ux * math.cos(a) + uy * math.sin(a)), tip[1] - hs * (uy * math.cos(a) - ux * math.sin(a)))
        return [Unit({"truth": "shaft"}, [shaft]),
                Unit({"truth": "head"}, [hand.deform([left, tip, right], DIAGRAM_SCALE, wid, "line")])]
    if kind == "wire":
        units = [lines("wire", [[tuple(p) for p in item["points"]]])]
        for p in item.get("dots", []):
            units.append(Unit({"truth": "junction"}, [hand_circle(p[0], p[1], JUNCTION_RADIUS, hand, n=8)]))
        return units
    if kind == "dot":
        p = item["point"]
        return [Unit({"truth": "junction"}, [hand_circle(p[0], p[1], JUNCTION_RADIUS, hand, n=8)])]
    if kind == "pin":
        (px, py), r = item["point"], item.get("radius", 4.0)
        return [Unit({"truth": "pin"}, [hand.deform(hand_circle(px, py, r, hand), r * 4, wid, "text")])]
    if kind == "gate":
        return [lines(part, strokes) for part, strokes in shapes.gate_shape(item["gate"], item["bbox"])]
    if kind == "component":
        return [lines(part, strokes) for part, strokes in
                shapes.component_shape(item["component"], item["a"], item.get("b"), item.get("size", 1.0),
                                       item.get("style"))]
    raise ValueError(f"unknown item kind {kind!r}")


# --------------------------------------------------------------- page

def default_out(page_dir, layout, fonts, noise, seed):
    if noise == 0 and list(fonts) == ["futural"] and layout.get("layout_seed", 0) == 0:
        return os.path.join(page_dir, "page.inkml")
    os.makedirs(os.path.join(page_dir, "renders"), exist_ok=True)
    return os.path.join(page_dir, "renders",
                        f"L{layout.get('layout_seed', 0)}_{'+'.join(fonts)}_noise{noise:g}_seed{seed}.inkml")


def render(page_dir, fonts, noise, seed, out=None, layout_path=None, device=True, quiet=False, layout=None,
           write=True):
    """Render one page. fonts: one font name per writer (cycled). `layout` may
    be given as a dict instead of a file. With write=False nothing is written.

    Returns {"out": path, "traces": {trace id: [(x, y, t)]}, "groups": [(item
    id, item, [(label, [trace ids])])]} with groups in time order."""
    if isinstance(fonts, str):
        fonts = fonts.split(",")
    if layout is None:
        layout = json.load(open(layout_path or os.path.join(page_dir, "layout.json")))
    writers = layout["writers"]
    font_of = {w["id"]: fonts[i % len(fonts)] for i, w in enumerate(writers)}
    renderers = {f: TextRenderer(f) for f in set(font_of.values())}
    wfont = {wid: renderers[f] for wid, f in font_of.items()}
    hand = Hand(seed, noise)
    dev = Device(seed, noise if device else 0)
    items = layout["items"]

    clock, pen, end_time, unit_end = {}, {}, {}, {}
    traces, groups = [], []  # traces: (start, pts, brush); groups: (eid, item, [(label, [trace idx])])
    for eid in layout["writing_order"]:
        item = items[eid]
        wid = item["writer"]
        if wid not in clock:
            after = next(w for w in writers if w["id"] == wid).get("start_after")
            clock[wid] = end_time[after] + hand.pause(ELEMENT_PAUSE) if after else 0.0
        t = clock[wid]
        brush = "bold" if item.get("bold") else "pen"
        group = []
        for unit in item_units(item, wfont, hand):
            refs = []
            for stroke in unit.strokes:
                stroke = [tuple(p) for p in stroke]
                if wid in pen:
                    t += hand.pause(PEN_UP_PAUSE) + math.dist(pen[wid], stroke[0]) / TRAVEL_SPEED
                pts, t = hand.time_stroke(stroke, t, wid, dev.rate)
                pts = dev.process(pts)
                t = max(t, pts[-1][2])
                pen[wid] = stroke[-1]
                refs.append(len(traces))
                traces.append((pts[0][2], pts, brush))
            if "char_index" in unit.label:
                unit_end[(eid, unit.label["char_index"])] = t
            group.append((unit.label, refs))
        end_time[eid] = t
        clock[wid] = t + hand.pause(ELEMENT_PAUSE)
        groups.append((eid, item, group))

    # Trace ids follow start time, so interleaved writers stay in time order.
    order = sorted(range(len(traces)), key=lambda i: traces[i][0])
    tid = {i: f"t{n}" for n, i in enumerate(order)}

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
                 ("generator", "scripts/render_inkml.py"), ("layout_seed", layout.get("layout_seed", 0)),
                 ("writer_fonts", ",".join(f"{w}:{f}" for w, f in font_of.items())),
                 ("noise_level", noise), ("noise_seed", seed), ("device", json.dumps(dev.profile()))):
        ET.SubElement(ink, q("annotation"), {"type": k}).text = str(v)

    for i in order:
        _, pts, brush = traces[i]
        el = ET.SubElement(ink, q("trace"), {XML_ID: tid[i], "contextRef": "#ctx0", "brushRef": f"#{brush}"})
        el.text = ", ".join(f"{x:.2f} {y:.2f} {tt:.3f}" for x, y, tt in pts)
    root = ET.SubElement(ink, q("traceGroup"), {XML_ID: "elements"})
    first = lambda g: min((traces[r][0] for _, refs in g[2] for r in refs), default=0.0)
    for eid, item, group in sorted(groups, key=first):
        g = ET.SubElement(root, q("traceGroup"))
        notes = [("element_id", eid), ("element", item.get("element", eid)), ("writer", item["writer"]),
                 ("kind", item["kind"])]
        if item.get("role"):
            notes.append(("role", item["role"]))
        if item["kind"] == "text":
            notes.append(("truth", item["text"]))
        elif item["kind"] == "math":
            notes.append(("truth", item["latex"]))
        for k, v in notes:
            ET.SubElement(g, q("annotation"), {"type": k}).text = str(v)
        for label, refs in group:
            u = ET.SubElement(g, q("traceGroup"))
            for k, v in label.items():
                ET.SubElement(u, q("annotation"), {"type": k}).text = str(v)
            for r in refs:
                ET.SubElement(u, q("traceView"), {"traceDataRef": f"#{tid[r]}"})

    result = {"out": None, "traces": {tid[i]: traces[i][1] for i in order},
              "groups": [(eid, item, [(label, [tid[r] for r in refs]) for label, refs in group])
                         for eid, item, group in sorted(groups, key=first)]}
    if not write:
        return result
    ET.indent(ink)
    out = out or default_out(page_dir, layout, fonts, noise, seed)
    result["out"] = out
    ET.ElementTree(ink).write(out, encoding="utf-8", xml_declaration=True)

    if layout.get("targets"):
        resolved = resolve_targets(layout["targets"], groups, traces, tid, end_time, unit_end)
        with open(out[:-len(".inkml")] + ".targets.json", "w") as f:
            json.dump({"inkml": os.path.basename(out), "targets": resolved}, f, indent=1)
    if not quiet:
        n_pts = sum(len(p) for _, p, _ in traces)
        span = max((p[-1][2] for _, p, _ in traces), default=0)
        print(f"wrote {out}: {len(groups)} items, {len(traces)} traces, {n_pts} points, {span:.1f} s of writing")
    return result


def resolve_targets(targets, groups, traces, tid, end_time, unit_end):
    """Each target's context ends when its `after` item (or character) is
    finished. A target is kept only if every target stroke starts later, which
    can fail when another writer's strokes interleave."""
    refs_of = {}
    for eid, _, group in groups:
        for label, refs in group:
            refs_of.setdefault(eid, []).extend(refs)
            if "char_index" in label:
                refs_of.setdefault((eid, label["char_index"]), []).extend(refs)
    out = []
    for tg in targets:
        after = tg["after"]
        if "char" in after:  # the last drawn character at or before it (spaces have no ink)
            c = max(k for e, k in unit_end if e == after["id"] and k <= after["char"])
            t_ctx = unit_end[(after["id"], c)]
        else:
            t_ctx = end_time[after["id"]]
        refs = []
        for part in tg["target"]:
            if "chars" in part:
                for c in range(*part["chars"]):
                    refs += refs_of.get((part["id"], c), [])
            else:
                refs += refs_of.get(part["id"], [])
        if not refs or min(traces[r][0] for r in refs) <= t_ctx:
            continue
        out.append({**tg, "t_context_end": round(t_ctx, 3),
                    "target_traces": [tid[r] for r in sorted(refs, key=lambda r: traces[r][0])]})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("page_dir", nargs="?", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "artifacts", "page_0001"))
    ap.add_argument("--layout", help="layout file (default: <page_dir>/layout.json)")
    ap.add_argument("--font", default="futural",
                    help="font per writer, comma separated: Hershey names or SVG fonts in fonts/svg")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="handwriting variation level: 0 = clean geometry, 1 = deliberately messy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-device", action="store_true", help="no digitizer artefacts")
    ap.add_argument("--out", help="output path (default: page.inkml for a clean futural render of "
                         "layout seed 0, else renders/L<layout>_<fonts>_noise<L>_seed<N>.inkml)")
    args = ap.parse_args()
    render(args.page_dir, args.font.split(","), args.noise, args.seed, args.out, args.layout,
           not args.no_device)
