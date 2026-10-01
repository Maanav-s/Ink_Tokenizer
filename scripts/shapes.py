"""Clean geometry for drawn shapes: gates, schematic components and outlines.

All coordinates are page mm with y down. Each function returns a list of
(part, strokes) pairs in drawing order, where `part` names the piece for
the ink's per-part labels (e.g. "body", "bubble", "lead_a") and strokes is a
list of polylines. render_inkml.py adds the handwriting variation.

gate_ports() and component_ports() are the single definition of where the
pins are, shared by the layout engine (for routing) and the renderer.
"""
import math

from fonts import bezier

GATES = ("and2", "or2", "xor2", "nand2", "nor2", "xnor2", "not", "buf")
COMPONENTS = ("resistor", "capacitor", "inductor", "diode", "voltage_source", "current_source",
              "battery", "ground", "terminal")
BUBBLE_R = 0.07  # output bubble radius, fraction of the gate's height


def circle_pts(cx, cy, r, n=24, start=0.0):
    """Starts at angle `start` from the top and runs counter-clockwise on the page."""
    return [(cx - r * math.sin(start + 2 * math.pi * i / n), cy - r * math.cos(start + 2 * math.pi * i / n))
            for i in range(n + 1)]


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


def diamond(bbox):
    x0, y0, x1, y1 = bbox
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    return [(xm, y0), (x1, ym), (xm, y1), (x0, ym), (xm, y0)]


def parallelogram(bbox, skew):
    """skew: horizontal offset of the top edge to the right, mm."""
    x0, y0, x1, y1 = bbox
    return [(x0 + skew, y0), (x1, y0), (x1 - skew, y1), (x0, y1), (x0 + skew, y0)]


# ------------------------------------------------------------------ gates

def _bubble_r(bbox):
    return BUBBLE_R * (bbox[3] - bbox[1])


def gate_ports(kind, bbox):
    """{port: (x, y)}: inputs a and b (or just a) on the left edge, output y on the right."""
    x0, y0, x1, y1 = bbox
    h, ym = y1 - y0, (y0 + y1) / 2
    if kind in ("not", "buf"):
        return {"a": (x0, ym), "y": (x1, ym)}
    if kind not in GATES:
        raise ValueError(f"unknown gate type {kind!r}")
    return {"a": (x0, y0 + 0.25 * h), "b": (x0, y0 + 0.75 * h), "y": (x1, ym)}


def _back_curve_x(x, y0, y1, bulge, y):
    """x of the concave back curve drawn by bezier((x, y1), (x + 2 * bulge, ym), (x, y0))."""
    u = (y - y0) / (y1 - y0)  # the Bezier parameter is linear in y here
    return x + 4 * u * (1 - u) * bulge


def gate_shape(kind, bbox):
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    ym = (y0 + y1) / 2
    ports = gate_ports(kind, bbox)
    inverted = kind in ("nand2", "nor2", "xnor2", "not")
    br = _bubble_r(bbox)
    xe = x1 - 2 * br if inverted else x1  # where the body ends
    parts = []
    if kind in ("and2", "nand2"):
        r = h / 2
        xa = xe - r
        arc = [(xa + r * math.sin(math.pi * i / 16), ym - r * math.cos(math.pi * i / 16)) for i in range(17)]
        parts.append(("body", [[(x0, y0), (xa, y0)] + arc + [(x0, y1), (x0, y0)]]))
    elif kind in ("or2", "nor2", "xor2", "xnor2"):
        gap = 0.12 * w if kind in ("xor2", "xnor2") else 0.0
        bx = x0 + gap  # back of the OR body
        bulge = 0.12 * w
        back = bezier((bx, y1), (bx + 2 * bulge, ym), (bx, y0))
        body = (bezier((bx, y0), (bx + 0.6 * (xe - bx), y0), (xe, ym))
                + bezier((xe, ym), (bx + 0.6 * (xe - bx), y1), (bx, y1))[1:]
                + back[1:])
        parts.append(("body", [body]))
        if gap:
            parts.append(("xor_curve", [bezier((x0, y0), (x0 + 2 * bulge, ym), (x0, y1))]))
        # Input stubs from the port to the body's back curve, so wires touch the gate.
        stubs = [[(px, py), (_back_curve_x(bx, y0, y1, bulge, py), py)] for p, (px, py) in ports.items() if p != "y"]
        parts.append(("stubs", stubs))
    elif kind in ("not", "buf"):
        parts.append(("body", [[(x0, y0), (xe, ym), (x0, y1), (x0, y0)]]))
    else:
        raise ValueError(f"unknown gate type {kind!r}")
    if inverted:
        parts.append(("bubble", [circle_pts(xe + br, ym, br, n=12)]))
    return parts


# ------------------------------------------------------------- components

def component_ports(kind, a, b=None):
    """Two-terminal components run from port a to port b; ground and terminal
    have the single port a."""
    if kind in ("ground", "terminal"):
        return {"a": tuple(a)}
    return {"a": tuple(a), "b": tuple(b)}


def component_shape(kind, a, b=None, size=1.0, style=None):
    """Strokes for a schematic component.

    Two-terminal kinds are drawn along the segment a -> b: a lead, the body
    (about 30 mm long times `size`, centred), and a lead. For a voltage
    source, a is the + terminal. ground hangs below a, and terminal is an
    open circle at a. style picks a variant: resistor "zigzag" (default) or
    "box".
    """
    if kind == "ground":
        x, y = a
        s = 8 * size
        return [("lead", [[(x, y), (x, y + s)]]),
                ("body", [[(x - s, y + s), (x + s, y + s)], [(x - 0.6 * s, y + 1.4 * s), (x + 0.6 * s, y + 1.4 * s)],
                          [(x - 0.25 * s, y + 1.8 * s), (x + 0.25 * s, y + 1.8 * s)]])]
    if kind == "terminal":
        return [("body", [circle_pts(a[0], a[1], 2.5 * size, n=12)])]
    L = math.dist(a, b)
    if L < 1e-6:
        raise ValueError(f"{kind}: zero-length component")
    ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
    nx, ny = -uy, ux
    body = min(30 * size, 0.7 * L)
    s0, s1 = (L - body) / 2, (L + body) / 2

    def P(s, n=0.0):  # point at distance s along the axis, n across it
        return (a[0] + ux * s + nx * n, a[1] + uy * s + ny * n)

    lead_a, lead_b = [P(0), P(s0)], [P(s1), P(L)]
    if kind == "resistor":
        if style == "box":
            w = 0.2 * body
            return [("lead_a", [lead_a]), ("body", [[P(s0, -w), P(s1, -w), P(s1, w), P(s0, w), P(s0, -w)]]),
                    ("lead_b", [lead_b])]
        n, amp = 6, 0.17 * body
        zig = [P(s0)] + [P(s0 + body * (i + 0.5) / n, amp * (1 if i % 2 == 0 else -1)) for i in range(n)] + [P(s1)]
        return [("body", [lead_a + zig[1:] + lead_b[1:]])]
    if kind in ("capacitor", "battery"):
        gap, plate = 0.12 * body, 0.45 * body
        m = L / 2
        la, lb = [P(0), P(m - gap / 2)], [P(m + gap / 2), P(L)]
        if kind == "capacitor":
            plates = [[P(m - gap / 2, -plate / 2), P(m - gap / 2, plate / 2)],
                      [P(m + gap / 2, -plate / 2), P(m + gap / 2, plate / 2)]]
        else:  # long plate is +, at a
            plates = [[P(m - gap / 2, -plate / 2), P(m - gap / 2, plate / 2)],
                      [P(m + gap / 2, -plate / 4), P(m + gap / 2, plate / 4)]]
        return [("lead_a", [la]), ("body", plates), ("lead_b", [lb])]
    if kind == "inductor":
        n = 4
        r = body / (2 * n)
        coil = [P(s0)]
        for i in range(n):
            c = s0 + r * (2 * i + 1)
            coil += [P(c - r * math.cos(math.pi * k / 8), -r * 1.6 * math.sin(math.pi * k / 8)) for k in range(1, 9)]
        return [("body", [lead_a + coil[1:] + lead_b[1:]])]
    if kind == "diode":
        w = 0.3 * body
        tri = [P(s0 + 0.15 * body, -w), P(s0 + 0.15 * body, w), P(s1 - 0.15 * body, 0), P(s0 + 0.15 * body, -w)]
        return [("lead_a", [[P(0), P(s0 + 0.15 * body)]]), ("body", [tri, [P(s1 - 0.15 * body, -w), P(s1 - 0.15 * body, w)]]),
                ("lead_b", [[P(s1 - 0.15 * body), P(L)]])]
    if kind in ("voltage_source", "current_source"):
        r = body / 2
        m = L / 2
        ring = circle_pts(*P(m), r, n=28)
        parts = [("lead_a", [[P(0), P(m - r)]]), ("body", [ring])]
        if kind == "voltage_source":
            k = 0.25 * r
            parts.append(("sign", [[P(m - 0.5 * r, -k), P(m - 0.5 * r, k)], [P(m - 0.5 * r - k), P(m - 0.5 * r + k)],
                                   [P(m + 0.5 * r, -k), P(m + 0.5 * r, k)]]))
        else:  # arrow from b towards a, i.e. current leaves at a
            parts.append(("sign", [[P(m + 0.55 * r), P(m - 0.55 * r)],
                                   [P(m - 0.2 * r, -0.3 * r), P(m - 0.55 * r), P(m - 0.2 * r, 0.3 * r)]]))
        parts.append(("lead_b", [[P(m + r), P(L)]]))
        return parts
    raise ValueError(f"unknown component type {kind!r}")
