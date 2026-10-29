"""Helpers shared by the layout engine, the diagram layouts and the validator.

Layout items (layout.json v1, see artifacts/README.md) are plain dicts with a
`kind` and geometry in page mm, y down. Geometry lives only in these keys, so
translate() can move any item: bbox, points, point, a, b, baseline, dots.

Text sizes come from fonts.PoolMetrics, so a box sized here fits the ink in
every font of the pool.
"""
import math

import shapes
from fonts import BASE, PoolMetrics

_METRICS = None


def pool_metrics():
    global _METRICS
    if _METRICS is None:
        _METRICS = PoolMetrics()
    return _METRICS


def is_math(s):
    """Labels written as $...$ are LaTeX, typeset as math items."""
    return len(s) > 2 and s[0] == "$" and s[-1] == "$"


def text_item(text, x, y_bottom, h, metrics=None, align="left", width=None, **extra):
    """A text item whose box starts at x (or is centred on x / ends at x for
    align center / right) with its bottom (the descender line) at y_bottom.
    width overrides the measured width (e.g. a table cell's full width)."""
    m = metrics or pool_metrics()
    w = m.width(text, h) if width is None else width
    x0 = {"left": x, "center": x - w / 2, "right": x - w}[align]
    item = {"kind": "text", "text": text, "bbox": [x0, y_bottom - h, x0 + w, y_bottom], "text_height": h}
    if align != "left":
        item["align"] = align
    item.update(extra)
    return item


def math_box(latex, h, metrics=None, options=None):
    """(width, ascent, descent) of typeset LaTeX, in mm."""
    import math_typeset
    eq = math_typeset.typeset(latex, h, metrics or pool_metrics(), options)
    return eq.width, eq.ascent, eq.descent


def math_item(latex, x, baseline, h, metrics=None, align="left", options=None, **extra):
    """A math item with its baseline at `baseline`. Its bbox spans the typeset
    ascent and descent, at least the extent of a text line of height h."""
    w, asc, desc = math_box(latex, h, metrics, options)
    asc, desc = max(asc, (1 - BASE) * h), max(desc, BASE * h)
    x0 = {"left": x, "center": x - w / 2, "right": x - w}[align]
    item = {"kind": "math", "latex": latex, "bbox": [x0, baseline - asc, x0 + w, baseline + desc],
            "baseline": baseline, "text_height": h}
    if options:
        item["options"] = options
    item.update(extra)
    return item


def label_item(s, x, y_bottom, h, metrics=None, align="left", **extra):
    """A text item, or a math item when s is $...$ (then y_bottom is still the
    bottom of a text line of height h, i.e. the baseline is BASE * h above it)."""
    if is_math(s):
        return math_item(s[1:-1], x, y_bottom - BASE * h, h, metrics, align, **extra)
    return text_item(s, x, y_bottom, h, metrics, align, **extra)


def label_width(s, h, metrics=None):
    if is_math(s):
        return math_box(s[1:-1], h, metrics)[0]
    return (metrics or pool_metrics()).width(s, h)


def translate(item, dx, dy):
    """Move an item in place by (dx, dy) and return it."""
    if "bbox" in item:
        x0, y0, x1, y1 = item["bbox"]
        item["bbox"] = [x0 + dx, y0 + dy, x1 + dx, y1 + dy]
    for k in ("points", "dots"):
        if k in item:
            item[k] = [[p[0] + dx, p[1] + dy] for p in item[k]]
    for k in ("point", "a", "b"):
        if k in item:
            item[k] = [item[k][0] + dx, item[k][1] + dy]
    if "baseline" in item:
        item["baseline"] += dy
    return item


def round_item(item, nd=2):
    """Round geometry for a compact layout.json."""
    def r(v):
        if isinstance(v, float):
            return round(v, nd)
        if isinstance(v, list):
            return [r(x) for x in v]
        return v
    for k in ("bbox", "points", "dots", "point", "a", "b", "baseline", "text_height"):
        if k in item:
            item[k] = r(item[k])
    return item


def item_bbox(item):
    """[x0, y0, x1, y1] covering an item's clean geometry."""
    kind = item["kind"]
    if "bbox" in item:
        return list(item["bbox"])
    pts = []
    if kind in ("polyline", "arrow", "wire"):
        pts = [tuple(p) for p in item["points"]]
        if kind == "arrow":
            head = item.get("head", 9.0)
            x, y = pts[-1]
            pts += [(x - head, y - head), (x + head, y + head)]
    elif kind in ("dot", "pin"):
        (x, y), r = item["point"], item.get("radius", 2.0)
        pts = [(x - r, y - r), (x + r, y + r)]
    elif kind == "component":
        for _, strokes in shapes.component_shape(item["component"], item["a"], item.get("b"),
                                                 item.get("size", 1.0), item.get("style")):
            for st in strokes:
                pts += st
    else:
        raise ValueError(f"item_bbox: unknown kind {kind!r}")
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def union_bbox(boxes):
    boxes = list(boxes)
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def overlap(a, b, margin=0.0):
    """True if boxes a and b, each grown by margin, intersect."""
    return a[0] - margin < b[2] and b[0] - margin < a[2] and a[1] - margin < b[3] and b[1] - margin < a[3]


def dist(a, b):
    return math.dist(a, b)
