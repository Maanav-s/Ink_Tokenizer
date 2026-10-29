"""Automatic layout for diagram content blocks: flowcharts, gate-level circuits
and analog schematics.

The content (content.json v1) only says what is connected to what; this module
places the shapes and routes the connections, randomized by a layout rng, so
one block yields many distinct pages. Each layout_* function returns local
geometry (mm, y down, everything inside [0, w] x [0, h]):

    {"width": w, "height": h, "items": {leaf_id: item}, "order": [leaf ids]}

Connections are routed orthogonally on a grid by A* (Router) with bend,
crossing and crowding penalties. Different edges or nets may cross at right
angles but never share a segment or touch; a net with fan-out is routed as a
tree whose later branches start at a junction dot on the part already drawn.

Items carry a few non-geometry annotations used by check_diagram: arrows
"from"/"to" (node ids), circuit wires "net"/"from"/"to", schematic wires
"net"; a schematic result also has "nets" {net: [ports]}.
"""
import heapq
import math
import random

import shapes
from layout_common import item_bbox, label_item, label_width, overlap, translate

R, D, L, U = 0, 1, 2, 3  # move directions on the grid (y down)
STEP = ((1, 0), (0, 1), (-1, 0), (0, -1))
INF = float("inf")


def opp(d):
    return (d + 2) % 4


def perp(d):
    return {(d + 1) % 4, (d + 3) % 4}


class RouteError(Exception):
    pass


def simplify(pts):
    """Drop repeated and collinear interior points of an orthogonal polyline."""
    out = [tuple(pts[0])]
    for p in pts[1:]:
        p = tuple(p)
        if abs(p[0] - out[-1][0]) < 1e-9 and abs(p[1] - out[-1][1]) < 1e-9:
            continue
        if len(out) >= 2:
            a, b = out[-2], out[-1]
            if (abs(a[0] - b[0]) < 1e-9 and abs(b[0] - p[0]) < 1e-9) or \
               (abs(a[1] - b[1]) < 1e-9 and abs(b[1] - p[1]) < 1e-9):
                out[-1] = p
                continue
        out.append(p)
    return [list(p) for p in out]


def grow(box, m):
    return [box[0] - m, box[1] - m, box[2] + m, box[3] + m]


# ------------------------------------------------------------------ router

class Router:
    """Orthogonal maze router on a square grid of pitch s.

    Cells can be blocked (obstacles), owned by one net (ports and the lanes
    in front of them), restricted to some move directions (lanes) and carry
    a soft extra cost. occ records, per cell and net, which directions the
    routed wires leave the cell in: a cell used by another net may only be
    crossed straight through at a right angle.
    """

    def __init__(self, box, s):
        self.s = s
        self.gx = math.floor(box[0] / s) * s
        self.gy = math.floor(box[1] / s) * s
        self.nx = int(math.ceil((box[2] - self.gx) / s)) + 1
        self.ny = int(math.ceil((box[3] - self.gy) / s)) + 1
        self.blocked = bytearray(self.nx * self.ny)
        self.soft, self.owner, self.lane, self.occ = {}, {}, {}, {}
        self.ports = set()

    def cell(self, x, y):
        i, j = round((x - self.gx) / self.s), round((y - self.gy) / self.s)
        if not (0 <= i < self.nx and 0 <= j < self.ny):
            raise RouteError("point outside the routing grid")
        return j * self.nx + i

    def xy(self, c):
        return (self.gx + (c % self.nx) * self.s, self.gy + (c // self.nx) * self.s)

    def cells_in(self, box):
        x0, y0, x1, y1 = box
        i0 = max(0, math.ceil((x0 - self.gx) / self.s - 1e-9))
        i1 = min(self.nx - 1, math.floor((x1 - self.gx) / self.s + 1e-9))
        j0 = max(0, math.ceil((y0 - self.gy) / self.s - 1e-9))
        j1 = min(self.ny - 1, math.floor((y1 - self.gy) / self.s + 1e-9))
        return [j * self.nx + i for j in range(j0, j1 + 1) for i in range(i0, i1 + 1)]

    def block(self, box):
        for c in self.cells_in(box):
            self.blocked[c] = 1

    def add_soft(self, box, cost):
        for c in self.cells_in(box):
            self.soft[c] = max(self.soft.get(c, 0.0), cost)

    def reserve(self, c, net, dirs, port=False):
        self.blocked[c] = 0
        self.owner[c] = net
        self.lane[c] = set(dirs)
        if port:
            self.ports.add(c)

    def foreign(self, c, net):
        own = self.owner.get(c)
        return (own is not None and own != net) or any(n != net for n in self.occ.get(c, {}))

    def dirs(self, c, net):
        return self.occ.get(c, {}).get(net, set())

    def branch_sources(self, net, cells, cost=0.0):
        """Tree cells a branch may leave from (a T: one free side, at most 3 wires)."""
        out = []
        for c in cells:
            if c in self.owner or self.foreign(c, net) or c in self.lane:
                continue
            used = self.dirs(c, net)
            if len(used) <= 2:
                free = {0, 1, 2, 3} - used
                if free:
                    out.append((c, free, cost))
        return out

    def route(self, net, sources, targets, bend=3.0, cross=6.0, near=1.5):
        """Cheapest path from one of sources [(cell, out dirs, cost)] to one of
        targets {cell: (in dirs, cost)}; returns its cells, or None."""
        nx, ny, occ, blocked = self.nx, self.ny, self.occ, self.blocked
        owner, lane, soft, ports = self.owner, self.lane, self.soft, self.ports
        targets = {c: v for c, v in targets.items() if not self.foreign(c, net)}
        if not targets:
            return None
        tl = [(c % nx, c // nx) for c in targets]

        def h(c):
            i, j = c % nx, c // nx
            return min(abs(i - a) + abs(j - b) for a, b in tl)

        src = {}
        for c, dirs, cost in sources:
            if self.foreign(c, net):
                continue
            if c in src:
                src[c] = (src[c][0] | set(dirs), min(src[c][1], cost))
            else:
                src[c] = (set(dirs), cost)
        heap, best, parent = [], {}, {}
        for c, (dirs, cost) in src.items():
            best[c * 5 + 4] = cost
            heapq.heappush(heap, (cost + h(c), cost, c, 4))
        while heap:
            f, g, c, d = heapq.heappop(heap)
            k = c * 5 + d
            if g > best.get(k, INF):
                continue
            if d == 4:
                if c in targets and src[c][0] & targets[c][0]:
                    return [c]
                outs = src[c][0]
            else:
                if c in targets and d in targets[c][0]:
                    path = [c]
                    while k in parent:
                        k = parent[k]
                        path.append(k // 5)
                    return path[::-1]
                o = occ.get(c)
                if o and any(n != net for n in o):
                    outs = (d,)  # crossing another net: straight on
                else:
                    outs = (d, (d + 1) % 4, (d + 3) % 4)
                ln = lane.get(c)
                if ln is not None:
                    outs = [x for x in outs if x in ln]
            i, j = c % nx, c // nx
            for o in outs:
                ni, nj = i + STEP[o][0], j + STEP[o][1]
                if not (0 <= ni < nx and 0 <= nj < ny):
                    continue
                n = nj * nx + ni
                cost = g + 1.0 + soft.get(n, 0.0)
                if d != 4 and o != d:
                    cost += bend
                tg = targets.get(n)
                if tg is not None and o in tg[0]:
                    cost += tg[1]
                else:
                    if blocked[n] or n in ports:
                        continue
                    own = owner.get(n)
                    if own is not None and own != net:
                        continue
                    ln = lane.get(n)
                    if ln is not None and o not in ln:
                        continue
                    oc = occ.get(n)
                    if oc:
                        if net in oc:
                            continue
                        if set().union(*oc.values()) != perp(o):
                            continue
                        cost += cross
                    for p in perp(o):
                        pi, pj = ni + STEP[p][0], nj + STEP[p][1]
                        if 0 <= pi < nx and 0 <= pj < ny:
                            oc2 = occ.get(pj * nx + pi)
                            if oc2 and any(o in v or opp(o) in v for v in oc2.values()):
                                cost += near
                k2 = n * 5 + o
                if cost < best.get(k2, INF):
                    best[k2] = cost
                    parent[k2] = k
                    heapq.heappush(heap, (cost + h(n), cost, n, o))
        return None

    def commit(self, net, path):
        for c in path:
            self.occ.setdefault(c, {}).setdefault(net, set())
        for a, b in zip(path, path[1:]):
            d = {1: R, self.nx: D, -1: L, -self.nx: U}[b - a]
            self.occ[a][net].add(d)
            self.occ[b][net].add(opp(d))

    def points(self, path):
        return simplify([self.xy(c) for c in path])


def finalize(items, order, pad=1.0, **extra):
    """Translate items so their union box starts at (pad, pad)."""
    boxes = [item_bbox(it) for it in items.values()]
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    for it in items.values():
        translate(it, pad - x0, pad - y0)
    assert sorted(order) == sorted(items), "order must list every item once"
    return {"width": x1 - x0 + 2 * pad, "height": y1 - y0 + 2 * pad, "items": items, "order": order, **extra}


def _overflow(res, style):
    over = 0.0
    if style.get("max_width"):
        over += max(0.0, res["width"] / style["max_width"] - 1)
    if style.get("max_height"):
        over += max(0.0, res["height"] / style["max_height"] - 1)
    return over


def _fit(build, rng, style, attempts):
    """Run build(sub_rng, attempt) until a result fits style's max size; else
    return the one that overflows least."""
    best, best_over, err = None, INF, None
    for a in range(attempts):
        sub = random.Random(rng.getrandbits(64))
        try:
            res = build(sub, a)
        except RouteError as e:
            err = e
            continue
        over = _overflow(res, style)
        if over == 0:
            return res
        if over < best_over:
            best, best_over = res, over
    if best is None:
        raise RouteError(f"no layout found: {err}")
    return best


def _text_lines(lines, cx, cy, th, pitch, align="center", x=None):
    """Label items for lines stacked and centred vertically on cy."""
    n = len(lines)
    total = n * th + (n - 1) * (pitch - th)
    top = cy - total / 2 + 0.06 * th
    out = []
    for k, s in enumerate(lines):
        yb = top + k * pitch + th
        out.append(label_item(s, cx if x is None else x, yb, th, align=align))
    return out


# --------------------------------------------------------------- flowchart

def _wrap(text, max_chars):
    lines = []
    for para in text.split("\n"):
        if len(para) <= 16:
            lines.append(para)
            continue
        words, cur = para.split(" "), ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > max_chars:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}" if cur else w
        lines.append(cur)
    return lines


def _fc_node_geometry(node, th, P, max_chars):
    """Text lines and box size for a flowchart node."""
    text = node.get("text", "")
    lines = _wrap(text, max_chars)
    while len(lines) > 3 and max_chars < 60:
        max_chars += 4
        lines = _wrap(text, max_chars)
    pitch = th * P["line_pitch"]
    tw = max(label_width(s, th) for s in lines) if lines else th
    tb = len(lines) * th + (len(lines) - 1) * (pitch - th)
    shape = node.get("shape", "process")
    g = {"shape": shape, "lines": lines, "pitch": pitch, "skew": 0.0}
    px, py = P["pad_x"] * th, P["pad_y"] * th
    if shape == "decision":
        a, b = tw + 0.7 * th, tb + 0.45 * th
        k = P["diamond_k"]
        g["w"], g["h"] = a * k, b / (1 - 1 / k)
    elif shape == "terminator" and P["terminator"] == "ellipse":
        g["w"], g["h"] = (tw + 0.7 * th) * 1.45, (tb + 0.6 * th) * 1.5
    elif shape == "terminator":
        g["h"] = tb + 2 * py
        g["w"] = tw + g["h"] + 0.3 * th
    elif shape == "io":
        h = tb + 2 * py
        skew = h * P["skew"]
        g["h"], g["skew"] = h, skew
        g["w"] = tw + 2 * (skew * (h + tb) / (2 * h) + 0.45 * th)
    else:
        g["w"], g["h"] = tw + 2 * px, tb + 2 * py
    return g


def _fc_order(ids, edges):
    """DFS preorder from the sources, and the back edges (to an ancestor)."""
    out = {i: [] for i in ids}
    inc = {i: [] for i in ids}
    for e in edges:
        out[e["from"]].append(e["to"])
        inc[e["to"]].append(e["from"])
    order, state, back = [], {}, set()
    for r in [i for i in ids if not inc[i]] + ids:
        if r in state:
            continue
        state[r] = 1
        order.append(r)
        stack = [(r, iter(out[r]))]
        while stack:
            n, it = stack[-1]
            m = next(it, None)
            if m is None:
                state[n] = 2
                stack.pop()
            elif state.get(m) == 1:
                back.add((n, m))
            elif m not in state:
                state[m] = 1
                order.append(m)
                stack.append((m, iter(out[m])))
    return order, back, out, inc


def _fc_grid(ids, shape, edges, rng, P):
    """Greedy placement on an abstract grid: u along the flow, v across it."""
    order, back, out, inc = _fc_order(ids, edges)
    pos, at = {}, {}

    def fwd(du, dv, decision):
        if du >= 1:
            return 2.0 * (du - 1) + 2.0 * abs(dv) + (1.5 if dv else 0.0)
        if du == 0:
            return 1.0 + 2.5 * (abs(dv) - 1) + (0.0 if decision else 4.0)
        return 6.0 + 2.0 * abs(du) + 2.0 * abs(dv)

    def blocked_line(a, b):
        (au, av), (bu, bv) = a, b
        if au == bu:
            return sum(4.0 for v in range(min(av, bv) + 1, max(av, bv)) if (au, v) in at)
        if av == bv:
            return sum(4.0 for u in range(min(au, bu) + 1, max(au, bu)) if (u, av) in at)
        return 2.0 if (au, bv) in at and (bu, av) in at else 0.0

    for n in order:
        nb = [(m, "in") for m in inc[n] if m in pos] + [(m, "out") for m in out[n] if m in pos]
        if not pos:
            best = (0, 0)
        elif not nb:
            best = (0, max(v for _, v in pos.values()) + 2)
        else:
            us = [pos[m][0] for m, _ in nb]
            vs = [pos[m][1] for m, _ in nb]
            cu = [p[0] for p in pos.values()]
            cv = [p[1] for p in pos.values()]
            span_u, span_v = max(cu) - min(cu), max(cv) - min(cv)
            best, best_c = None, INF
            for u in range(min(us) - 1, max(us) + 3):
                for v in range(min(vs) - 3, max(vs) + 4):
                    if (u, v) in at:
                        continue
                    c = 0.0
                    for m, kind in nb:
                        mu, mv = pos[m]
                        if kind == "in":
                            c += fwd(u - mu, v - mv, shape[m] == "decision") + blocked_line((mu, mv), (u, v))
                        elif (n, m) in back:
                            c += 0.7 * (abs(u - mu) + abs(v - mv))
                        else:
                            c += fwd(mu - u, mv - v, shape[n] == "decision") + blocked_line((u, v), (mu, mv))
                    c += P["grow_v"] * max(0, max(max(cv), v) - min(min(cv), v) - span_v)
                    c += P["grow_u"] * max(0, max(max(cu), u) - min(min(cu), u) - span_u)
                    c += rng.uniform(0, P["noise"])
                    c += 0.8 * max(0, min(cv) - v)  # grow across the flow down / to the right
                    if "u_cap" in P:
                        c += 6.0 * max(0, max(max(cu), u) - min(min(cu), u) - P["u_cap"])
                    if c < best_c:
                        best, best_c = (u, v), c
        pos[n] = best
        at[best] = n
    # Compress to consecutive indices.
    us = sorted({p[0] for p in pos.values()})
    vs = sorted({p[1] for p in pos.values()})
    return {n: (us.index(u), vs.index(v)) for n, (u, v) in pos.items()}, order, back


def _fc_ports(g, s):
    """Attachment points of a node: (name, outward dir, outline point)."""
    cx, cy, w, h = g["cx"], g["cy"], g["w"], g["h"]
    ex = w / 2 - g["skew"] / 2
    ports = [("top", U, (cx, cy - h / 2)), ("bottom", D, (cx, cy + h / 2)),
             ("left", L, (cx - ex, cy)), ("right", R, (cx + ex, cy))]
    if g["shape"] in ("process", "io") or (g["shape"] == "terminator" and g.get("stadium")):
        lim = w / 2 - s - (g["skew"] if g["shape"] == "io" else 0) - (h / 2 if g["shape"] == "terminator" else 0)
        k = math.floor(lim / s)
        if k >= 2:
            off = max(2, round(0.6 * k)) * s
            for name, d, y in (("top", U, cy - h / 2), ("bottom", D, cy + h / 2)):
                ports += [(f"{name}-", d, (cx - off, y)), (f"{name}+", d, (cx + off, y))]
    return ports


def _lane_point(pt, d, s, router):
    """The first grid point at least s/2 outside the outline, along d."""
    x, y = pt
    if d in (U, D):
        t = (y + (0.5 * s if d == D else -0.5 * s) - router.gy) / s
        j = math.ceil(t - 1e-9) if d == D else math.floor(t + 1e-9)
        return (x, router.gy + j * s)
    t = (x + (0.5 * s if d == R else -0.5 * s) - router.gx) / s
    i = math.ceil(t - 1e-9) if d == R else math.floor(t + 1e-9)
    return (router.gx + i * s, y)


def _fc_box_item(g, eid):
    x0, y0 = g["cx"] - g["w"] / 2, g["cy"] - g["h"] / 2
    box = [x0, y0, x0 + g["w"], y0 + g["h"]]
    shape = g["shape"]
    if shape == "decision":
        it = {"kind": "diamond", "bbox": box}
    elif shape == "io":
        it = {"kind": "parallelogram", "bbox": box, "skew": g["skew"]}
    elif shape == "terminator" and not g.get("stadium"):
        it = {"kind": "ellipse", "bbox": box}
    elif shape == "terminator":
        it = {"kind": "rect", "bbox": box, "corner_radius": g["h"] / 2}
    else:
        it = {"kind": "rect", "bbox": box}
        if g.get("radius"):
            it["corner_radius"] = g["radius"]
    it.update(element=eid, role="node_box")
    return it


def _seg_hits_box(a, b, box, m=0.0):
    x0, x1 = min(a[0], b[0]), max(a[0], b[0])
    y0, y1 = min(a[1], b[1]), max(a[1], b[1])
    return x0 < box[2] + m and box[0] - m < x1 and y0 < box[3] + m and box[1] - m < y1


def _label_spot(text, pts, th, gap, sides, ok):
    """A box for an edge label beside the first (else second) segment of pts, or None."""
    for k in range(min(2, len(pts) - 1)):
        box = _label_spot_seg(text, pts[k], pts[k + 1], th, gap, sides, ok)
        if box:
            return box
    return None


def _label_spot_seg(text, p0, p1, th, gap, sides, ok):
    lw = label_width(text, th)
    (x0, y0), (x1, y1) = p0, p1
    seg = math.hypot(x1 - x0, y1 - y0)
    steps = [gap + k * 0.35 * th for k in range(8)]
    for side in sides:
        for t in steps:
            if abs(x0 - x1) < 1e-9:  # vertical
                sy = 1 if y1 > y0 else -1
                if t + th > seg - 0.3 * th:
                    break
                yt = y0 + t if sy > 0 else y0 - t - th
                box = [x0 + gap, yt, x0 + gap + lw, yt + th] if side > 0 else [x0 - gap - lw, yt, x0 - gap, yt + th]
            else:
                sx = 1 if x1 > x0 else -1
                if t + lw > seg - 0.4 * th:
                    break
                xs = x0 + t if sx > 0 else x0 - t - lw
                # Above the line, leave room for descenders (deep in script fonts).
                box = [xs, y0 - 0.6 * th - th, xs + lw, y0 - 0.6 * th] if side > 0 else \
                      [xs, y0 + 0.4 * th, xs + lw, y0 + 0.4 * th + th]
            if ok(box):
                return box
    return None


def layout_flowchart(block, rng, style):
    """Layered-ish flowchart layout: greedy grid placement in flow order, then
    A* routing of every edge between node outlines."""
    th = float(style["text_height"])
    want = block.get("direction")
    first = want if want in ("TB", "LR") else ("TB" if rng.random() < 0.65 else "LR")
    other = {"TB": "LR", "LR": "TB"}[first]

    def build(sub, attempt):
        direction = first if (want in ("TB", "LR") or attempt % 2 == 0) else other
        level = attempt // (1 if want in ("TB", "LR") else 2)
        limit = style.get("max_height" if direction == "TB" else "max_width")
        return _flowchart_once(block, sub, th, direction, compact=min(level, 3), spread=1.0 + 0.3 * (attempt // 8),
                               limit=limit)

    return _fit(build, rng, style, 10)


def _flowchart_once(block, rng, th, direction, compact, spread, limit=None):
    eid = block["id"]
    nodes = block["nodes"]
    ids = [n["id"] for n in nodes]
    edges = [dict(e, id=e.get("id") or f"e{i + 1}") for i, e in enumerate(block.get("edges", []))
             if e["from"] in ids and e["to"] in ids and e["from"] != e["to"]]
    s = round(th * rng.uniform(0.3, 0.38))
    P = {"line_pitch": rng.uniform(1.4, 1.6), "pad_x": rng.uniform(0.7, 1.1), "pad_y": rng.uniform(0.62, 0.85),
         "diamond_k": rng.uniform(1.6, 1.95), "skew": rng.uniform(0.3, 0.5),
         "terminator": "ellipse" if rng.random() < 0.25 else "stadium",
         "noise": rng.uniform(0.3, 1.5), "grow_u": 0.4, "grow_v": 0.9}
    if direction == "LR":
        P["grow_u"], P["grow_v"] = 0.9, 0.5
    P["grow_u"] += 0.8 * compact  # compact layouts fold the flow sideways
    max_chars = rng.randint(10, 13) if (compact or direction == "LR") else rng.randint(13, 18)
    if compact and direction == "TB":
        max_chars = rng.randint(16, 22)  # wider, flatter boxes save rows
    rect_radius = rng.choice([0.0, 0.0, 0.0, 0.25 * th])
    geo = {n["id"]: _fc_node_geometry(n, th, P, max_chars) for n in nodes}
    for g in geo.values():
        g["stadium"] = g["shape"] == "terminator" and P["terminator"] == "stadium"
        if g["shape"] == "process":
            g["radius"] = rect_radius
    shape = {i: geo[i]["shape"] for i in ids}
    if limit and compact:
        # Soft cap on the number of ranks along the flow, so long flows fold back.
        along = [g["h"] if direction == "TB" else g["w"] for g in geo.values()]
        pitch = sum(along) / len(along) + (1.7 if direction == "TB" else 2.2) * th
        P["u_cap"] = max(2, int(limit / pitch) - (1 if compact < 3 else 2))
    grid, order, back = _fc_grid(ids, shape, edges, rng, P)

    # Real coordinates: columns and rows sized by their largest node.
    cr = {n: ((v, u) if direction == "TB" else (u, v)) for n, (u, v) in grid.items()}
    ncol = max(c for c, _ in cr.values()) + 1
    nrow = max(r for _, r in cr.values()) + 1
    cw, rh = [0.0] * ncol, [0.0] * nrow
    for n, (c, r) in cr.items():
        cw[c] = max(cw[c], geo[n]["w"])
        rh[r] = max(rh[r], geo[n]["h"])
    lbl_w = max([label_width(e["label"], th) for e in edges if e.get("label")] or [0.0])
    k = (1.0 - 0.1 * compact) * spread
    head = min(rng.uniform(0.32, 0.42) * th, 1.25 * s)
    gx = max(rng.uniform(1.6, 2.8) * th * k, 6 * s, (lbl_w + 2.5 * s + head) if lbl_w else 0)
    gy = max(rng.uniform(1.3, 2.1) * th * k, 5 * s, th + 2.5 * s)
    xs, ys = [], []
    for c in range(ncol):
        x = 0.0 if c == 0 else xs[-1] + cw[c - 1] / 2 + gx + cw[c] / 2
        xs.append(math.ceil(x / s) * s)
    for r in range(nrow):
        y = 0.0 if r == 0 else ys[-1] + rh[r - 1] / 2 + gy + rh[r] / 2
        ys.append(math.ceil(y / s) * s)
    for n, (c, r) in cr.items():
        geo[n]["cx"], geo[n]["cy"] = xs[c], ys[r]

    boxes = {n: [g["cx"] - g["w"] / 2, g["cy"] - g["h"] / 2, g["cx"] + g["w"] / 2, g["cy"] + g["h"] / 2]
             for n, g in geo.items()}
    allbox = [min(b[0] for b in boxes.values()), min(b[1] for b in boxes.values()),
              max(b[2] for b in boxes.values()), max(b[3] for b in boxes.values())]
    router = Router(grow(allbox, 5 * s + lbl_w), s)
    for n, b in boxes.items():
        router.block(grow(b, 0.45 * s))
        router.add_soft(grow(b, 1.4 * s), 0.6)
    ports = {}
    for n, g in geo.items():
        ports[n] = []
        for name, d, pt in _fc_ports(g, s):
            lp = _lane_point(pt, d, s, router)
            c = router.cell(*lp)
            c2 = router.cell(lp[0] + STEP[d][0] * s, lp[1] + STEP[d][1] * s)
            router.blocked[c] = 1
            ports[n].append({"name": name, "dir": d, "pt": pt, "lane": c, "lane2": c2, "free": True})

    flow = D if direction == "TB" else R

    def src_cost(d, is_back, offset):
        if is_back:
            c = 0.0 if d in perp(flow) else (2.0 if d == flow else 5.0)
        else:
            c = 0.0 if d == flow else (2.5 if d in perp(flow) else 8.0)
        return c + (1.5 if offset else 0.0)

    def tgt_cost(d, is_back, offset):  # d: the move direction into the port
        if is_back:
            c = 0.0 if d in perp(flow) else (3.0 if d == opp(flow) else 5.0)
        else:
            c = 0.0 if d == flow else (2.5 if d in perp(flow) else 8.0)
        return c + (1.5 if offset else 0.0)

    rank = {n: i for i, n in enumerate(order)}
    route_order = sorted(edges, key=lambda e: ((e["from"], e["to"]) in back, rank[e["from"]], rank[e["to"]]))
    arrows, label_boxes = {}, {}
    side_pref = rng.choice([1, -1])

    def label_ok(box):
        for n, b in boxes.items():
            if overlap(box, b, 0.4 * th):
                return False
        for b in label_boxes.values():
            if overlap(box, b, 0.4 * th):
                return False
        for pts in arrows.values():
            for a, b in zip(pts, pts[1:]):
                if _seg_hits_box(a, b, box, 0.35 * th):
                    return False
        for pl in ports.values():
            for p in pl:
                if p["free"]:
                    x, y = router.xy(p["lane"])
                    if overlap(box, [x, y, x, y], 0.6 * s):
                        return False
        return True

    for e in route_order:
        a, b = e["from"], e["to"]
        is_back = (a, b) in back
        srcs, tgts, by_cell_s, by_cell_t = [], {}, {}, {}
        for p in ports[a]:
            if p["free"]:
                srcs.append((p["lane"], {p["dir"]}, src_cost(p["dir"], is_back, p["name"][-1] in "+-")))
                by_cell_s[p["lane"]] = p
        for p in ports[b]:
            if p["free"]:
                d = opp(p["dir"])
                tgts[p["lane2"]] = ({d}, tgt_cost(d, is_back, p["name"][-1] in "+-"))
                by_cell_t[p["lane2"]] = p
        path = router.route(e["id"], srcs, tgts, bend=3.0, cross=8.0, near=2.0)
        if path is None:
            raise RouteError(f"flowchart edge {e['id']} could not be routed")
        pt = by_cell_t[path[-1]]
        path = path + [pt["lane"]]  # straight on into the port, so the head has room
        router.commit(e["id"], path)
        ps = by_cell_s[path[0]]
        ps["free"] = pt["free"] = False
        arrows[e["id"]] = simplify([ps["pt"]] + [router.xy(c) for c in path] + [pt["pt"]])
        if e.get("label"):
            pts = arrows[e["id"]]
            box = _label_spot(e["label"], pts, th, 0.55 * th, (side_pref, -side_pref), label_ok)
            if box is None:
                raise RouteError(f"no room for the label of {e['id']}")
            label_boxes[e["id"]] = box
            # Block the label's ink extent for later routes: descenders reach
            # below the box, and script letters overhang on both sides.
            router.block(grow([box[0] - 0.3 * th, box[1] - 0.1 * th, box[2] + 0.3 * th, box[3] + 0.3 * th], 0.3 * s))

    # Items and writing order.
    items = {}
    node_leaves = {}
    for n in ids:
        g = geo[n]
        items[f"{eid}.{n}.box"] = _fc_box_item(g, eid)
        lines = _text_lines(g["lines"], g["cx"], g["cy"], th, g["pitch"])
        tids = [f"{eid}.{n}.text"] if len(lines) == 1 else [f"{eid}.{n}.text.{k}" for k in range(len(lines))]
        for t, it in zip(tids, lines):
            it.update(element=eid, role="node_text")
            items[t] = it
        node_leaves[n] = ([f"{eid}.{n}.box"] + tids) if rng.random() < 0.8 else (tids + [f"{eid}.{n}.box"])
    edge_leaves = {}
    for e in edges:
        items[f"{eid}.{e['id']}.arrow"] = {"kind": "arrow", "points": arrows[e["id"]], "head": head,
                                          "element": eid, "role": "edge", "from": e["from"], "to": e["to"]}
        edge_leaves[e["id"]] = [f"{eid}.{e['id']}.arrow"]
        if e.get("label"):
            x0, y0, x1, y1 = label_boxes[e["id"]]
            it = label_item(e["label"], x0, y1, th)
            it.update(element=eid, role="edge_label")
            items[f"{eid}.{e['id']}.label"] = it
            edge_leaves[e["id"]].append(f"{eid}.{e['id']}.label")
    p_early = rng.choice([0.0, 0.3, 0.7])
    early = {e["id"]: rng.random() < p_early for e in edges}
    drawn, done, seq = set(), set(), []
    for n in order:
        seq += node_leaves[n]
        drawn.add(n)
        for e in edges:
            if e["id"] in done:
                continue
            if (e["from"] in drawn and e["to"] in drawn) or (e["from"] == n and early[e["id"]]):
                seq += edge_leaves[e["id"]]
                done.add(e["id"])
    return finalize(items, seq)


# ----------------------------------------------------------------- circuit

PIN_TYPES = ("input_pin", "output_pin")


def _gate_dims(kind, s, P):
    if kind in ("not", "buf"):
        return P["w1"] * s, 6 * s, {"a": 3 * s, "y": 3 * s}
    return P["w2"] * s, 8 * s, {"a": 2 * s, "b": 6 * s, "y": 4 * s}


def _pav(desired):
    """Least-squares non-decreasing fit (pool adjacent violators)."""
    blocks = []  # [sum, count]
    for d in desired:
        blocks.append([d, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s_, c_ = blocks.pop()
            blocks[-1][0] += s_
            blocks[-1][1] += c_
    out = []
    for s_, c_ in blocks:
        out += [s_ / c_] * c_
    return out


def layout_circuit(block, rng, style):
    """Gate-level circuit: columns by logic depth, barycenter ordering,
    aligned stacking, then tree routing of every net."""
    th = float(style["text_height"])

    def build(sub, attempt):
        return _circuit_once(block, sub, th, shrink=attempt >= 2, extra=attempt // 2)

    return _fit(build, rng, style, 6)


def _circuit_once(block, rng, th, shrink, extra):
    eid = block["id"]
    comps = {c["id"]: c for c in block["components"]}
    wires = [dict(w, id=w.get("id") or f"w_{i + 1}") for i, w in enumerate(block.get("wires", []))]
    split = lambda p: tuple(p.rsplit(".", 1))
    s = max(5.0, round(th * 0.27 * 2 + rng.choice([-1, 0, 0, 1])) / 2 - (0.5 if shrink else 0.0))
    P = {"w2": rng.choice([9, 10, 10, 11]), "w1": rng.choice([6, 7]), "gap": rng.choice([3, 4, 4, 5]),
         "pin_r": rng.uniform(2.5, 3.5), "chan": rng.choice([2, 3, 3, 4]) + extra}
    pin_gap = max(4, math.ceil(th * 1.35 / s))
    kind = {c: comps[c]["type"] for c in comps}
    is_gate = lambda c: kind[c] in shapes.GATES

    def dims(c):
        if is_gate(c):
            return _gate_dims(kind[c], s, P)
        return 0.0, 0.0, {"a": 0.0, "y": 0.0}

    drivers = {c: [] for c in comps}  # c -> [(port, driver comp, driver port)]
    sinks = {c: [] for c in comps}    # c -> [(sink comp, sink port)]
    for w in wires:
        (fc, fp), (tc, tp) = split(w["from"]), split(w["to"])
        drivers[tc].append((tp, fc, fp))
        sinks[fc].append((tc, tp))

    # Logic depth; cycles (latches) are cut where the DFS meets them.
    level, onstack = {}, set()

    def depth(c):
        if c in level:
            return level[c]
        if kind[c] == "input_pin":
            level[c] = 0
            return 0
        onstack.add(c)
        lv = 1
        for _, dc, _ in drivers[c]:
            if dc not in onstack:
                lv = max(lv, depth(dc) + (1 if is_gate(c) else 0))
        onstack.discard(c)
        level[c] = lv
        return lv

    for c in comps:
        depth(c)
    last = max([level[c] for c in comps if is_gate(c)] or [0]) + 1
    for c in comps:
        if kind[c] == "output_pin":
            level[c] = last
    cols = [[] for _ in range(last + 1)]
    for c in comps:
        cols[level[c]].append(c)
    cols = [col for col in cols if col]
    colof = {c: i for i, col in enumerate(cols) for c in col}
    real_col = dict(colof)

    # Wires spanning several columns pass through dummy nodes (one per net
    # and column), which reserve a track between the gates.
    drivers = {c: [] for c in comps}
    sinks = {c: [] for c in comps}

    def link(u, up, v, vp):
        drivers[v].append((vp, u, up))
        sinks[u].append((v, vp))

    for w in wires:
        (fc, fp), (tc, tp) = split(w["from"]), split(w["to"])
        prev = (fc, fp)
        for ci in range(colof[fc] + 1, colof[tc]):
            dmy = f"~{w['from']}~{ci}"
            if dmy not in kind:
                kind[dmy] = "dummy"
                drivers[dmy], sinks[dmy] = [], []
                cols[ci].append(dmy)
                colof[dmy] = ci
                link(*prev, dmy, "a")
            prev = (dmy, "y")
        link(*prev, tc, tp)

    def gap_between(a, b):
        ga, gb = is_gate(a), is_gate(b)
        if kind[a] == "dummy" or kind[b] == "dummy":
            return (3 if ga or gb else 2) * s
        return (pin_gap if not ga and not gb else max(P["gap"], 3)) * s

    top = {}

    def stack(col, desired=None):
        offs, o = [], 0.0
        for i, c in enumerate(col):
            if i:
                o += dims(col[i - 1])[1] + gap_between(col[i - 1], c)
            offs.append(o)
        if desired is None:
            base = -(offs[-1] + dims(col[-1])[1]) / 2
            z = [base] * len(col)
        else:
            z = _pav([d - o for d, o in zip(desired, offs)])
        z = [round(v / s) * s for v in z]
        for c, v, o in zip(col, z, offs):
            top[c] = v + o

    def port_y(c, p):
        return top[c] + dims(c)[2].get(p, 0.0)

    for col in cols:
        stack(col)

    def key_from(c, use_drivers):
        ys = [port_y(dc, dp) for _, dc, dp in drivers[c]] if use_drivers else [port_y(tc, tp) for tc, tp in sinks[c]]
        return sum(ys) / len(ys) if ys else None

    def crossings():
        segs = [(colof[u], port_y(u, up), colof[v], port_y(v, vp)) for u in sinks for v, vp in sinks[u]
                for up in [next(p for p, uu, _ in drivers[v] if uu == u)]]
        n = 0
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                a, b = segs[i], segs[j]
                lo, hi = max(a[0], b[0]), min(a[2], b[2])
                if hi <= lo:
                    continue
                f = lambda sg, x: sg[1] + (sg[3] - sg[1]) * (x - sg[0]) / ((sg[2] - sg[0]) or 1)
                d0, d1 = f(a, lo) - f(b, lo), f(a, hi) - f(b, hi)
                if d0 * d1 < 0:
                    n += 1
        return n

    best = (crossings(), [list(c) for c in cols])
    for it in range(6):
        rng_jit = 0.01 * s
        for rng_dir in ((range(1, len(cols)), True), (range(len(cols) - 2, -1, -1), False)):
            for ci in rng_dir[0]:
                col = cols[ci]
                keys = {}
                for i, c in enumerate(col):
                    k = key_from(c, rng_dir[1])
                    keys[c] = (k if k is not None else top[c]) + rng.uniform(-rng_jit, rng_jit)
                col.sort(key=lambda c: keys[c])
                stack(col)
            n = crossings()
            if n < best[0]:
                best = (n, [list(c) for c in cols])
    cols = best[1]
    for col in cols:
        stack(col)

    # Align: pull each column towards its drivers / sinks, keeping order.
    def align(ci, use_drivers):
        col = cols[ci]
        desired = []
        for c in col:
            if use_drivers:
                cand = [port_y(dc, dp) - dims(c)[2].get(p, 0.0) for p, dc, dp in drivers[c]]
            else:
                cand = [port_y(tc, tp) - dims(c)[2].get("y", 0.0) for tc, tp in sinks[c]]
            desired.append(sum(cand) / len(cand) if cand else top[c])
        stack(col, desired)

    for _ in range(3):
        for ci in range(1, len(cols)):
            align(ci, True)
        for ci in range(len(cols) - 2, -1, -1):
            align(ci, False)
    for ci in range(1, len(cols)):
        align(ci, True)

    # Straighten: small shifts that make one input wire straight.
    def try_straighten(ci, use_drivers):
        col = cols[ci]
        for i, c in enumerate(col):
            if use_drivers:
                cand = [port_y(dc, dp) - port_y(c, p) for p, dc, dp in drivers[c]]
            else:
                cand = [port_y(tc, tp) - port_y(c, "y") for tc, tp in sinks[c]]
            for dlt in sorted(cand, key=abs):
                if dlt == 0:
                    break
                if abs(dlt) > 4 * s:
                    continue
                nt = top[c] + dlt
                if i > 0 and nt < top[col[i - 1]] + dims(col[i - 1])[1] + gap_between(col[i - 1], c) - 1e-6:
                    continue
                if i + 1 < len(col) and nt + dims(c)[1] + gap_between(c, col[i + 1]) > top[col[i + 1]] + 1e-6:
                    continue
                top[c] = nt
                break

    for ci in range(1, len(cols)):
        try_straighten(ci, True)
    for ci in range(len(cols) - 2, -1, -1):
        try_straighten(ci, False)
    for ci in range(1, len(cols)):
        try_straighten(ci, True)

    # Columns in x: channels sized by the nets passing through them.
    chan_nets = [set() for _ in cols]
    for w in wires:
        (fc, fp), (tc, _) = split(w["from"]), split(w["to"])
        a, b = sorted((colof[fc], colof[tc]))
        for ci in range(a, max(b, a + 1)):
            chan_nets[ci].add(w["from"])
    xcol, x = [], 0.0
    for ci, col in enumerate(cols):
        xcol.append(x)
        width = max(dims(c)[0] for c in col)
        x += width + (len(chan_nets[ci]) + 3 + P["chan"]) * s
    box, ports = {}, {}
    for ci, col in enumerate(cols):
        for c in col:
            w_, h_, offs = dims(c)
            x0 = xcol[ci]
            if is_gate(c):
                box[c] = [x0, top[c], x0 + w_, top[c] + h_]
                ports[c] = {p: tuple(v) for p, v in shapes.gate_ports(kind[c], box[c]).items()}
            else:
                box[c] = [x0, top[c], x0, top[c]]
                ports[c] = {"y" if kind[c] == "input_pin" else "a": (x0, top[c])}

    # Pin labels.
    items = {}
    r_pin = P["pin_r"]
    for c in comps:
        if is_gate(c):
            items[f"{eid}.{c}"] = {"kind": "gate", "gate": kind[c], "bbox": box[c], "element": eid, "role": "gate"}
            continue
        (px, py), = ports[c].values()
        items[f"{eid}.{c}"] = {"kind": "pin", "point": [px, py], "radius": r_pin, "element": eid, "role": "pin"}
        lbl = comps[c].get("label")
        if lbl:
            gap = r_pin + 0.25 * th
            if kind[c] == "input_pin":
                it = label_item(lbl, px - gap, py + 0.62 * th, th, align="right")
            else:
                it = label_item(lbl, px + gap, py + 0.62 * th, th)
            it.update(element=eid, role="pin_label")
            items[f"{eid}.{c}.label"] = it

    # Routing.
    nets = {}
    for w in wires:
        nets.setdefault(w["from"], []).append(w)
    allb = [item_bbox(it) for it in items.values()]
    area = [min(b[0] for b in allb), min(b[1] for b in allb), max(b[2] for b in allb), max(b[3] for b in allb)]

    # Net labels get a spot reserved beside their driver's output before routing.
    labels = block.get("labels", [])
    reserved = {}
    solid = [grow(box[c], s) for c in comps if is_gate(c)] + \
            [items[f"{eid}.{c}.label"]["bbox"] for c in comps if f"{eid}.{c}.label" in items]
    for lb in labels:
        att = lb.get("attached_to") or ""
        drv = next((w["from"] for w in wires if att == f"net:{w.get('net') or w['from']}"), None)
        if drv is None:
            continue
        px, py = ports[split(drv)[0]][split(drv)[1]]
        lw = label_width(lb["text"], th)
        x0 = px + (1.5 * s if is_gate(split(drv)[0]) else 2.5 * s)
        sides = [[x0, py - 0.4 * th - th, x0 + lw, py - 0.4 * th], [x0, py + 0.35 * th, x0 + lw, py + 1.35 * th]]
        if rng.random() < 0.25:
            sides.reverse()
        for bx in sides:
            if not any(overlap(bx, b, 0.3 * s) for b in solid + list(reserved.values())):
                reserved[lb["id"]] = bx
                break

    def make_router():
        rt = Router(grow(area, 8 * s), s)
        for bx in reserved.values():
            rt.block(grow(bx, 0.3 * s))
        for c in comps:
            if is_gate(c):
                rt.add_soft(grow(box[c], 3 * s), 0.5)  # jogs go mid-channel, not at the gate
                rt.block(grow(box[c], s))
            else:
                (px, py), = ports[c].values()
                rt.block([px - s, py - s, px + s, py + s])
            lb = items.get(f"{eid}.{c}.label")
            if lb:
                rt.block(grow(lb["bbox"], 0.5 * s))
        for w in wires:
            (fc, fp), (tc, tp) = split(w["from"]), split(w["to"])
            net = w["from"]
            for comp, p, out in ((fc, fp, True), (tc, tp, False)):
                x, y = ports[comp][p]
                rt.reserve(rt.cell(x, y), net, {R}, port=True)
                rt.reserve(rt.cell(x + s if out else x - s, y), net, {R})
        return rt

    net_order = sorted(nets, key=lambda n: (max(colof[split(w["to"])[0]] for w in nets[n]) - colof[split(n)[0]],
                                            rng.random()))
    routed = None
    for attempt in range(8):
        rt = make_router()
        result, failed = {}, None
        for net in net_order:
            fc, fp = split(net)
            src_cell = rt.cell(*ports[fc][fp])
            tree = []
            todo = sorted(nets[net], key=lambda w: abs(ports[split(w["to"])[0]][split(w["to"])[1]][0] - ports[fc][fp][0])
                          + abs(ports[split(w["to"])[0]][split(w["to"])[1]][1] - ports[fc][fp][1]))
            for w in todo:
                tc, tp = split(w["to"])
                tcell = rt.cell(*ports[tc][tp])
                if not tree:
                    srcs = [(src_cell, {R}, 0.0)]
                else:
                    srcs = rt.branch_sources(net, tree, 1.0)
                path = rt.route(net, srcs, {tcell: ({R}, 0.0)}, bend=3.0, cross=5.0, near=3.5)
                if path is None:
                    failed = net
                    break
                rt.commit(net, path)
                pts = rt.points(path)
                pts[-1] = list(ports[tc][tp])
                junction = None
                if tree:
                    junction = pts[0]
                else:
                    pts[0] = list(ports[fc][fp])
                tree += path
                result[w["id"]] = (pts, junction, len(result))
            if failed:
                break
        if not failed:
            routed = (rt, result)
            break
        net_order.remove(failed)
        net_order.insert(0, failed)
    if routed is None:
        raise RouteError("circuit could not be routed")
    rt, result = routed

    wire_net = {}
    for w in wires:
        pts, junction, _ = result[w["id"]]
        it = {"kind": "wire", "points": pts, "element": eid, "role": "wire",
              "net": w.get("net") or w["from"], "from": w["from"], "to": w["to"]}
        if junction:
            it["dots"] = [list(junction)]
        items[f"{eid}.{w['id']}"] = it
        wire_net[w["id"]] = it["net"]

    # Net labels and caption.
    taken = [it["bbox"] for it in items.values() if "bbox" in it]
    segs = [(a, b) for it in items.values() if it["kind"] == "wire" for a, b in zip(it["points"], it["points"][1:])]
    dots = [d for it in items.values() for d in it.get("dots", [])]
    pins = [it["point"] for it in items.values() if it["kind"] == "pin"]

    def free(bx):
        if any(overlap(bx, t, 0.3 * th) for t in taken):
            return False
        if any(_seg_hits_box(a, b, bx, 0.3 * th) for a, b in segs):
            return False
        return not any(overlap(bx, [p[0], p[1], p[0], p[1]], 0.6 * s) for p in dots + pins)

    captions = []
    for lb in labels:
        att = lb.get("attached_to") or ""
        lid = f"{eid}.{lb['id']}"
        if att.startswith("net:"):
            net = att[4:]
            nws = [w for w in wires if wire_net[w["id"]] == net]
            lw = label_width(lb["text"], th)
            cands = [reserved[lb["id"]]] if lb["id"] in reserved else []
            for w in nws:
                pts = items[f"{eid}.{w['id']}"]["points"]
                for k, (a, b) in enumerate(zip(pts, pts[1:])):
                    if abs(a[1] - b[1]) < 1e-9:
                        x0, x1 = sorted((a[0], b[0]))
                        for t in (0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0):
                            xs = x0 + 0.5 * s + t * max(0.0, x1 - x0 - lw - s)
                            cands.append([xs, a[1] - 0.4 * th - th, xs + lw, a[1] - 0.4 * th])
                            cands.append([xs, a[1] + 0.35 * th, xs + lw, a[1] + 1.35 * th])
                    else:
                        y0, y1 = sorted((a[1], b[1]))
                        for t in (0.5, 0.25, 0.75):
                            ym = y0 + t * (y1 - y0)
                            cands.append([a[0] + 0.4 * th, ym - th / 2, a[0] + 0.4 * th + lw, ym + th / 2])
                            cands.append([a[0] - 0.4 * th - lw, ym - th / 2, a[0] - 0.4 * th, ym + th / 2])
            spot = next((c for c in cands if free(c)), None)
            if spot is None and cands:
                spot = cands[0]
            if spot is None:
                captions.append(lb)
                continue
            it = label_item(lb["text"], spot[0], spot[3], th)
            it.update(element=eid, role="net_label")
            items[lid] = it
            taken.append(it["bbox"])
        elif att in comps:
            b = item_bbox(items[f"{eid}.{att}"])
            lw = label_width(lb["text"], th)
            cx = (b[0] + b[2]) / 2
            cands = [[cx - lw / 2, b[1] - 0.25 * th - th, cx + lw / 2, b[1] - 0.25 * th],
                     [cx - lw / 2, b[3] + 0.25 * th, cx + lw / 2, b[3] + 1.25 * th]]
            spot = next((c for c in cands if free(c)), cands[0])
            it = label_item(lb["text"], spot[0], spot[3], th)
            it.update(element=eid, role="net_label")
            items[lid] = it
            taken.append(it["bbox"])
        else:
            captions.append(lb)
    if captions:
        allb = [item_bbox(it) for it in items.values()]
        x0, y0 = min(b[0] for b in allb), min(b[1] for b in allb)
        x1, y1 = max(b[2] for b in allb), max(b[3] for b in allb)
        below = rng.random() < 0.6
        align = rng.choice(["center", "left"])
        y = y1 + 0.6 * th + th if below else y0 - 0.6 * th
        for lb in captions:
            x = (x0 + x1) / 2 if align == "center" else x0 + 0.5 * th
            it = label_item(lb["text"], x, y, th, align=align)
            it.update(element=eid, role="caption")
            items[f"{eid}.{lb['id']}"] = it
            y += 1.3 * th if below else -1.3 * th

    # Writing order.
    seq = []
    seen = set()
    pin_leaves = lambda c: [x for x in ([f"{eid}.{c}.label", f"{eid}.{c}"] if rng.random() < 0.35
                                        else [f"{eid}.{c}", f"{eid}.{c}.label"]) if x in items]
    gate_seq = [c for col in cols for c in col if is_gate(c)]
    parent = {}  # branch wire -> the earlier wire of its net it starts on
    for w in wires:
        pts, junction, idx = result[w["id"]]
        if junction is None:
            continue
        for o in sorted(nets[w["from"]], key=lambda o: result[o["id"]][2]):
            op = result[o["id"]][0]
            if result[o["id"]][2] < idx and any(_on_segment(junction, a, b) for a, b in zip(op, op[1:])):
                parent[w["id"]] = o["id"]
                break
    net_labels = {}
    for lb in labels:
        att = lb.get("attached_to") or ""
        if att.startswith("net:") and f"{eid}.{lb['id']}" in items and items[f"{eid}.{lb['id']}"]["role"] == "net_label":
            net_labels.setdefault(att[4:], []).append(f"{eid}.{lb['id']}")
    labels_inline = rng.random() < 0.5

    def emit(leaves):
        for x in leaves:
            if x not in seen:
                seen.add(x)
                seq.append(x)

    def emit_wire(w):
        if f"{eid}.{w['id']}" in seen:
            return
        if w["id"] in parent:
            emit_wire(next(o for o in wires if o["id"] == parent[w["id"]]))
        emit([f"{eid}.{w['id']}"])
        net = wire_net[w["id"]]
        if labels_inline and net in net_labels:
            emit(net_labels[net])

    if rng.random() < 0.5:  # pins, gates, then wires net by net
        ins = [c for c in cols[0] if kind[c] == "input_pin"]
        outs = [c for c in comps if kind[c] == "output_pin"]
        for c in ins + (outs if rng.random() < 0.5 else []):
            emit(pin_leaves(c))
        for c in gate_seq:
            emit([f"{eid}.{c}"])
        for c in outs:
            emit(pin_leaves(c))
        for net in sorted(nets, key=lambda n: (colof[split(n)[0]], top[split(n)[0]])):
            for w in nets[net]:
                emit_wire(w)
    else:  # gate by gate with their input wires
        for c in gate_seq + [c for c in comps if kind[c] == "output_pin"]:
            for w in wires:
                dc = split(w["from"])[0]
                if split(w["to"])[0] == c and kind[dc] == "input_pin":
                    emit(pin_leaves(dc))
            emit(pin_leaves(c) if not is_gate(c) else [f"{eid}.{c}"])
            for w in wires:
                if split(w["to"])[0] == c:
                    emit_wire(w)
        for c in comps:
            emit(pin_leaves(c) if not is_gate(c) else [f"{eid}.{c}"])
        for w in wires:
            emit_wire(w)
    emit([k for k in items if k not in seen])
    return finalize(items, seq)


def _on_segment(p, a, b, tol=0.01):
    return (min(a[0], b[0]) - tol <= p[0] <= max(a[0], b[0]) + tol and
            min(a[1], b[1]) - tol <= p[1] <= max(a[1], b[1]) + tol and
            (abs(a[0] - b[0]) < tol or abs(a[1] - b[1]) < tol))


# --------------------------------------------------------------- schematic

TWO_TERMINAL = ("resistor", "capacitor", "inductor", "diode", "voltage_source", "current_source", "battery")


def layout_schematic(block, rng, style):
    """Analog schematic: components on the LLM's coarse grid with a random
    pitch, labels beside their bodies, nets routed as trees."""
    th = float(style["text_height"])

    def build(sub, attempt):
        return _schematic_once(block, sub, th, grow_pitch=1.0 + 0.2 * (attempt // 2), shrink=attempt % 2 == 1)

    return _fit(build, rng, style, 8)


def _schematic_once(block, rng, th, grow_pitch, shrink):
    eid = block["id"]
    s = 5.0
    comps = [dict(c) for c in block["components"]]
    # Components without a placement hint go in a new row below.
    rows = [c["at"][1] for c in comps if c.get("at")]
    nxt = [0, (max(rows) + 1) if rows else 0]
    for c in comps:
        if not c.get("at"):
            c["at"] = list(nxt)
            nxt[0] += 1
        if c["type"] in TWO_TERMINAL and c.get("orient") not in ("h", "v"):
            c["orient"] = "v" if c["type"] in ("voltage_source", "current_source", "battery") else "h"
    ncol = max(c["at"][0] for c in comps) + 1
    nrow = max(c["at"][1] for c in comps) + 1
    lo, hi = (13, 17) if shrink else (14, 21)
    lo, hi = round(lo * grow_pitch), round(hi * grow_pitch)
    px = [rng.randint(lo, hi) * s for _ in range(ncol)]
    py = [rng.randint(lo, hi) * s for _ in range(nrow)]
    xs = [sum(px[:i]) for i in range(ncol)]
    ys = [sum(py[:i]) for i in range(nrow)]
    min_pitch = min(px[1:] + py[1:] or [hi * s])
    Lc = min(2 * rng.choice([6, 7, 7, 8]), int((min_pitch - 4 * s) / (2 * s)) * 2)
    Lc = max(Lc, 8)
    Llen = Lc * s
    size = min(rng.uniform(1.1, 1.45), (Llen - 20) / 30)
    res_style = rng.choice(["zigzag", "zigzag", "box"])
    fmt = rng.choice(["two_lines", "two_lines", "equals", "equals", "label_only"])
    term_r = 2.5 * size

    items, ports, cbox, orient = {}, {}, {}, {}
    for c in comps:
        cid, t = c["id"], c["type"]
        x, y = xs[c["at"][0]], ys[c["at"][1]]
        it = {"kind": "component", "component": t, "element": eid, "role": "component"}
        if t in TWO_TERMINAL:
            if c["orient"] == "v":
                a, b = [x, y - Llen / 2], [x, y + Llen / 2]
            else:
                a, b = [x - Llen / 2, y], [x + Llen / 2, y]
            it.update(a=a, b=b)
            ports[cid] = {"a": tuple(a), "b": tuple(b)}
            orient[cid] = c["orient"]
        else:
            it["a"] = [x, y]
            ports[cid] = {"a": (x, y)}
            orient[cid] = t
        if size != 1.0:
            it["size"] = size
        if t == "resistor" and res_style != "zigzag":
            it["style"] = res_style
        items[f"{eid}.{cid}"] = it
        cbox[cid] = item_bbox(it)
    for i, a in enumerate(comps):
        for b in comps[i + 1:]:
            if overlap(cbox[a["id"]], cbox[b["id"]], 2 * s):
                raise RouteError(f"components {a['id']} and {b['id']} overlap")

    nets = [dict(n) for n in block.get("nets", [])]
    port_net = {}
    for n in nets:
        for p in n["connects"]:
            port_net[p] = n["id"]

    def port_dirs(cid, p):
        """(move dirs into the port, out of it) for a wire."""
        o = orient[cid]
        if o == "v":
            return ({D}, {U}) if p == "a" else ({U}, {D})
        if o == "h":
            return ({R}, {L}) if p == "a" else ({L}, {R})
        if o == "ground":
            return {D}, {U}
        return None  # terminal: decided by its label side

    # Label text per component.
    def label_lines(c, fmt):
        lbl, val = c.get("label"), c.get("value")
        if c["type"] == "ground" and not lbl:
            return []
        if c["type"] == "terminal":
            return [lbl] if lbl else []
        if not lbl:
            return [val] if val else []
        if not val or fmt == "label_only":
            return [lbl]
        if fmt == "equals" and not lbl.startswith("$"):
            return [f"{lbl} = {val}"]
        return [lbl, val]

    pitch = 1.4 * th
    gap = 0.4 * th

    def label_box(c, side, lines):
        cid = c["id"]
        b = cbox[cid]
        w = max(label_width(t, th) for t in lines)
        hgt = len(lines) * th + (len(lines) - 1) * (pitch - th)
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if c["type"] == "terminal" or c["type"] == "ground":
            cy = ports[cid]["a"][1] if c["type"] == "terminal" else cy
        if side == "right":
            return [b[2] + gap, cy - hgt / 2, b[2] + gap + w, cy + hgt / 2]
        if side == "left":
            return [b[0] - gap - w, cy - hgt / 2, b[0] - gap, cy + hgt / 2]
        if side == "above":
            return [cx - w / 2, b[1] - gap - hgt, cx + w / 2, b[1] - gap]
        return [cx - w / 2, b[3] + gap, cx + w / 2, b[3] + gap + hgt]

    # Rough wire corridors (L routes between the ports of each net), to keep labels off them.
    corridors = []
    for n in nets:
        pts = [ports[p.rsplit(".", 1)[0]][p.rsplit(".", 1)[1]] for p in n["connects"]
               if p.rsplit(".", 1)[0] in ports]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                (ax, ay), (bx, by) = pts[i], pts[j]
                corridors += [((ax, ay), (bx, ay)), ((bx, ay), (bx, by)), ((ax, ay), (ax, by)), ((ax, by), (bx, by))]
    centre = ((min(b[0] for b in cbox.values()) + max(b[2] for b in cbox.values())) / 2,
              (min(b[1] for b in cbox.values()) + max(b[3] for b in cbox.values())) / 2)

    lbl_side, lbl_lines, lbl_boxes = {}, {}, {}
    for c in comps:
        cid = c["id"]
        o = orient[cid]
        sides = {"v": ["right", "left"], "h": ["above", "below"], "terminal": ["right", "left", "above", "below"],
                 "ground": ["right", "left"]}[o]
        best, best_c = None, INF
        for f in (fmt, "two_lines", "label_only"):
            lines = label_lines(c, f)
            if not lines:
                break
            for side in sides:
                bx = label_box(c, side, lines)
                cost = rng.uniform(0, 1.5)
                cost += sum(10 for o2 in comps if o2["id"] != cid and overlap(bx, cbox[o2["id"]], 2 * s))
                cost += sum(10 for b2 in lbl_boxes.values() if overlap(bx, b2, s))
                cost += sum(3 for a, b in corridors if _seg_hits_box(a, b, bx, s))
                if o == "v":
                    cost += 1.0 if (side == "right") == (cbox[cid][0] < centre[0]) and len(comps) > 3 else 0.0
                if o == "terminal":
                    mine = ports[cid]["a"]
                    for p, n in port_net.items():
                        pc, pp = p.rsplit(".", 1)
                        if pc == cid or pc not in ports or n != port_net.get(f"{cid}.a"):
                            continue
                        ox, oy = ports[pc][pp]
                        toward = {"right": ox > mine[0] + 1, "left": ox < mine[0] - 1,
                                  "above": oy < mine[1] - 1, "below": oy > mine[1] + 1}[side]
                        cost += 6 if toward else 0
                if cost < best_c:
                    best, best_c = (side, lines), cost
            if best_c < 10:
                break
        if best is None:
            continue
        if best_c >= 10:
            raise RouteError(f"no room for the label of {cid}")
        lbl_side[cid], lbl_lines[cid] = best
        lbl_boxes[cid] = label_box(c, *best)

    allb = list(cbox.values()) + list(lbl_boxes.values())
    area = [min(b[0] for b in allb), min(b[1] for b in allb), max(b[2] for b in allb), max(b[3] for b in allb)]

    def make_router():
        rt = Router(grow(area, 8 * s), s)
        for c in comps:
            cid = c["id"]
            if c["type"] == "terminal":
                x, y = ports[cid]["a"]
                rt.block([x - 0.5 * s, y - 0.5 * s, x + 0.5 * s, y + 0.5 * s])
            else:
                rt.block(grow(cbox[cid], 0.8 * s))
        for b in lbl_boxes.values():
            rt.block(grow(b, 0.5 * s))
        term_dirs = {}
        for c in comps:
            cid = c["id"]
            for p, (x, y) in ports[cid].items():
                net = port_net.get(f"{cid}.{p}")
                cell = rt.cell(x, y)
                if net is None:
                    continue
                pd = port_dirs(cid, p)
                if pd is None:  # terminal: any side but the label's
                    away = {"right": L, "left": R, "above": D, "below": U}.get(lbl_side.get(cid), None)
                    ins = {R, D, L, U} - ({away} if away is not None else set())
                    pd = (ins, {opp(d) for d in ins})
                    rt.reserve(cell, net, ins | pd[1], port=True)
                else:
                    rt.reserve(cell, net, pd[0] | pd[1], port=True)
                    (od,) = pd[1]
                    rt.reserve(rt.cell(x + STEP[od][0] * s, y + STEP[od][1] * s), net, pd[0] | pd[1])
                term_dirs[(cid, p)] = pd
        return rt, term_dirs

    order_nets = sorted(nets, key=lambda n: (len(n["connects"]), rng.random()))
    routed = None
    for attempt in range(6):
        rt, pdirs = make_router()
        result, failed = {}, None
        for n in order_nets:
            net = n["id"]
            pl = [tuple(p.rsplit(".", 1)) for p in n["connects"] if p.rsplit(".", 1)[0] in ports]
            if len(pl) < 2:
                continue
            first = pl[0]
            rest = pl[1:]
            tree, branches = [], []
            connected = [ports[first[0]][first[1]]]
            while rest:
                nearest = min(rest, key=lambda q: min(abs(ports[q[0]][q[1]][0] - x) + abs(ports[q[0]][q[1]][1] - y)
                                                      for x, y in connected + [rt.xy(c) for c in tree[::4]]))
                rest.remove(nearest)
                tc = rt.cell(*ports[nearest[0]][nearest[1]])
                if not tree:
                    srcs = [(rt.cell(*ports[first[0]][first[1]]), pdirs[first][1], 0.0)]
                else:
                    srcs = rt.branch_sources(net, tree, 1.0)
                path = rt.route(net, srcs, {tc: (pdirs[nearest][0], 0.0)}, bend=3.0, cross=6.0, near=2.0)
                if path is None:
                    failed = n
                    break
                rt.commit(net, path)
                pts = rt.points(path)
                pts[-1] = list(ports[nearest[0]][nearest[1]])
                if tree:
                    branches.append((pts, pts[0]))
                else:
                    pts[0] = list(ports[first[0]][first[1]])
                    branches.append((pts, None))
                tree += path
                connected.append(ports[nearest[0]][nearest[1]])
            if failed:
                break
            result[net] = branches
        if not failed:
            routed = result
            break
        order_nets.remove(failed)
        order_nets.insert(0, failed)
    if routed is None:
        raise RouteError("schematic could not be routed")

    # Label items.
    lbl_leaves = {}
    for c in comps:
        cid = c["id"]
        if cid not in lbl_boxes:
            continue
        x0, y0, x1, y1 = lbl_boxes[cid]
        side = lbl_side[cid]
        align = {"right": "left", "left": "right"}.get(side, "center")
        xa = {"left": x0, "right": x1, "center": (x0 + x1) / 2}[align]
        its = _text_lines(lbl_lines[cid], 0, (y0 + y1) / 2, th, pitch, align=align, x=xa)
        names = [f"{eid}.{cid}.label", f"{eid}.{cid}.value"][:len(its)]
        for nm, it in zip(names, its):
            it.update(element=eid, role="component_label")
            items[nm] = it
        lbl_leaves[cid] = names

    wire_leaves = []  # (leaf, ports it ends on, parent leaf)
    for n in nets:
        net = n["id"]
        br = routed.get(net, [])
        for k, (pts, junction) in enumerate(br):
            leaf = f"{eid}.{net}" if len(br) == 1 else f"{eid}.{net}.{k + 1}"
            it = {"kind": "wire", "points": pts, "element": eid, "role": "wire", "net": net}
            if junction:
                it["dots"] = [list(junction)]
            items[leaf] = it
            ends = [c for c in ports for p, xy in ports[c].items()
                    if any(abs(xy[0] - q[0]) < 1e-6 and abs(xy[1] - q[1]) < 1e-6 for q in (pts[0], pts[-1]))]
            parent = None
            if junction:
                for j in range(k):
                    pl = br[j][0]
                    if any(_on_segment(junction, a, b) for a, b in zip(pl, pl[1:])):
                        parent = f"{eid}.{net}" if len(br) == 1 else f"{eid}.{net}.{j + 1}"
                        break
            wire_leaves.append((leaf, ends, parent))

    # Writing order: components roughly left to right, wires once their ends exist.
    jit = rng.uniform(0, 0.4) * min_pitch
    comp_seq = sorted(comps, key=lambda c: (cbox[c["id"]][0] + cbox[c["id"]][2]) / 2 + rng.uniform(0, jit)
                      + 0.1 * (cbox[c["id"]][1]))
    wires_last = rng.random() < 0.25
    seq, drawn, done = [], set(), set()

    def flush():
        changed = True
        while changed:
            changed = False
            for leaf, ends, parent in wire_leaves:
                if leaf in done or not all(e in drawn for e in ends) or (parent and parent not in done):
                    continue
                seq.append(leaf)
                done.add(leaf)
                changed = True

    for c in comp_seq:
        cid = c["id"]
        lab = lbl_leaves.get(cid, [])
        if rng.random() < 0.15:
            seq += lab + [f"{eid}.{cid}"]
        else:
            seq += [f"{eid}.{cid}"] + lab
        drawn.add(cid)
        if not wires_last:
            flush()
    flush()
    seq += [leaf for leaf, _, _ in wire_leaves if leaf not in done]
    netmap = {n["id"]: list(n["connects"]) for n in nets}
    return finalize(items, seq, nets=netmap)


# ------------------------------------------------------------------- check

BOX_KINDS = ("rect", "diamond", "parallelogram", "ellipse", "gate", "component", "pin")


def _segments(it):
    pts = it.get("points") or []
    return list(zip(pts, pts[1:]))


def _seg_conflict(s1, s2, tol=0.01):
    """'overlap' / 'touch' for two orthogonal segments, None if apart or a
    proper right-angle crossing."""
    (a, b), (c, d) = s1, s2
    h1, h2 = abs(a[1] - b[1]) < tol, abs(c[1] - d[1]) < tol
    if h1 == h2:
        k = 1 if h1 else 0  # the shared coordinate
        if abs(a[k] - c[k]) > tol:
            return None
        j = 1 - k
        lo = max(min(a[j], b[j]), min(c[j], d[j]))
        hi = min(max(a[j], b[j]), max(c[j], d[j]))
        if hi - lo > tol:
            return "overlap"
        return "touch" if hi - lo > -tol else None
    hs, vs = (s1, s2) if h1 else (s2, s1)
    x, y = vs[0][0], hs[0][1]
    hx0, hx1 = sorted((hs[0][0], hs[1][0]))
    vy0, vy1 = sorted((vs[0][1], vs[1][1]))
    if not (hx0 - tol <= x <= hx1 + tol and vy0 - tol <= y <= vy1 + tol):
        return None
    if hx0 + tol < x < hx1 - tol and vy0 + tol < y < vy1 - tol:
        return None
    return "touch"


def _inside(it, p, m=0.5):
    """Whether point p is inside a shape item by more than m mm."""
    x, y = p
    kind = it["kind"]
    b = item_bbox(it)
    if not (b[0] + m < x < b[2] - m and b[1] + m < y < b[3] - m):
        return False
    if kind == "diamond":
        cx, cy, hw, hh = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, (b[2] - b[0]) / 2, (b[3] - b[1]) / 2
        return abs(x - cx) / (hw - m) + abs(y - cy) / (hh - m) < 1
    if kind == "ellipse":
        cx, cy, hw, hh = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, (b[2] - b[0]) / 2, (b[3] - b[1]) / 2
        return ((x - cx) / (hw - m)) ** 2 + ((y - cy) / (hh - m)) ** 2 < 1
    if kind == "parallelogram":
        sk = it.get("skew", 8)
        t = (b[3] - y) / (b[3] - b[1])  # 0 at the bottom, 1 at the top
        return b[0] + sk * t + m < x < b[2] - sk * (1 - t) - m
    return True


def _outline_dist(it, p):
    b = item_bbox(it)
    if it["kind"] == "diamond":
        poly = shapes.diamond(b)
    elif it["kind"] == "parallelogram":
        poly = shapes.parallelogram(b, it.get("skew", 8))
    else:
        poly = [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3]), (b[0], b[1])]
    best = INF
    for a, c in zip(poly, poly[1:]):
        dx, dy = c[0] - a[0], c[1] - a[1]
        t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / ((dx * dx + dy * dy) or 1)))
        best = min(best, math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy))
    return best


def _near(p, q, tol=0.01):
    return abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol


def check_diagram(result):
    """Problems with a layout_* result (empty list if none): order, bounds,
    connectivity, text overlaps, lines through shapes, nets/edges touching."""
    probs = []
    items, order = result["items"], result["order"]
    W, H = result["width"], result["height"]
    if sorted(order) != sorted(items) or len(set(order)) != len(order):
        probs.append("order does not list every item exactly once")
    for k, it in items.items():
        b = item_bbox(it)
        if b[0] < -0.01 or b[1] < -0.01 or b[2] > W + 0.01 or b[3] > H + 0.01:
            probs.append(f"{k} outside [0, w] x [0, h]")
        if "element" not in it or "role" not in it:
            probs.append(f"{k} lacks element/role")
    lines = {k: it for k, it in items.items() if it["kind"] in ("arrow", "wire")}
    texts = {k: it for k, it in items.items() if it["kind"] in ("text", "math")}
    shapes_ = {k: it for k, it in items.items() if it["kind"] in BOX_KINDS}

    def group(k):  # leaf ids of one node/component share a prefix
        return k.split(".")[1] if "." in k else k

    # Text clear of everything.
    tk = list(texts)
    for i, a in enumerate(tk):
        ba = texts[a]["bbox"]
        for b in tk[i + 1:]:
            if overlap(ba, texts[b]["bbox"]):
                probs.append(f"text {a} overlaps {b}")
        for k, it in shapes_.items():
            if group(k) == group(a) and texts[a].get("role") == "node_text":
                continue
            if overlap(ba, item_bbox(it)):
                probs.append(f"text {a} overlaps {k}")
        for k, it in lines.items():
            if any(_seg_hits_box(p, q, ba) for p, q in _segments(it)):
                probs.append(f"text {a} overlaps line {k}")

    # Lines through shapes.
    for k, it in lines.items():
        for p, q in _segments(it):
            n = max(1, int(math.hypot(q[0] - p[0], q[1] - p[1])))
            for sk, sh in shapes_.items():
                if sh["kind"] == "pin" or not overlap(item_bbox(sh), [min(p[0], q[0]), min(p[1], q[1]),
                                                                       max(p[0], q[0]), max(p[1], q[1])]):
                    continue
                if sh["kind"] == "component" and sh["component"] == "terminal":
                    continue
                if any(_inside(sh, (p[0] + (q[0] - p[0]) * t / n, p[1] + (q[1] - p[1]) * t / n)) for t in range(n + 1)):
                    probs.append(f"{k} passes through {sk}")
    # Different nets / edges never overlap or touch.
    net_of = {k: it.get("net", k) for k, it in lines.items()}
    lk = list(lines)
    for i, a in enumerate(lk):
        for b in lk[i + 1:]:
            if net_of[a] == net_of[b]:
                continue
            for s1 in _segments(lines[a]):
                for s2 in _segments(lines[b]):
                    c = _seg_conflict(s1, s2)
                    if c:
                        probs.append(f"{a} and {b} {c} at {s1} / {s2}")
    for k, it in lines.items():
        for pk, pin in items.items():
            if pin["kind"] == "pin" and any(_on_segment(pin["point"], p, q) for p, q in _segments(it)):
                same = any(_near(pin["point"], x) for x in (it["points"][0], it["points"][-1]))
                if not same:
                    probs.append(f"{k} runs through pin {pk}")
        for p, q in _segments(it):
            if abs(p[0] - q[0]) > 1e-6 and abs(p[1] - q[1]) > 1e-6:
                probs.append(f"{k} has a diagonal segment")

    roles = {it.get("role") for it in items.values()}
    eid = next(iter(items.values()))["element"] if items else ""
    if "edge" in roles:
        for k, it in lines.items():
            src, dst = items.get(f"{eid}.{it['from']}.box"), items.get(f"{eid}.{it['to']}.box")
            if src is None or dst is None:
                probs.append(f"{k}: unknown endpoints")
                continue
            if _outline_dist(src, it["points"][0]) > 0.05:
                probs.append(f"{k} does not start on {it['from']}")
            if _outline_dist(dst, it["points"][-1]) > 0.05:
                probs.append(f"{k} tip not on {it['to']}")
            if src["kind"] == "diamond" and min(math.dist(it["points"][0], v) for v in shapes.diamond(src["bbox"])) > 0.05:
                probs.append(f"{k} does not leave from a vertex")
    if "gate" in roles or "pin" in roles:
        def port(ref):
            c, p = ref.rsplit(".", 1)
            comp = items.get(f"{eid}.{c}")
            if comp is None:
                return None
            if comp["kind"] == "gate":
                return shapes.gate_ports(comp["gate"], comp["bbox"]).get(p)
            return tuple(comp["point"])
        bynet = {}
        for k, it in lines.items():
            bynet.setdefault(it["from"], []).append(k)
            a, b = port(it["from"]), port(it["to"])
            if b is None or not _near(it["points"][-1], b):
                probs.append(f"{k} does not end on {it['to']}")
            if a is None:
                probs.append(f"{k}: unknown driver {it['from']}")
        for net, ks in bynet.items():
            src = port(net)
            linked = {k for k in ks if src and _near(lines[k]["points"][0], src)}
            changed = True
            while changed:
                changed = False
                for k in ks:
                    if k in linked:
                        continue
                    st = lines[k]["points"][0]
                    if any(_on_segment(st, p, q) for o in linked for p, q in _segments(lines[o])):
                        if not any(_near(st, d) for d in lines[k].get("dots", [])):
                            probs.append(f"{k} branches without a junction dot")
                        linked.add(k)
                        changed = True
            for k in set(ks) - linked:
                probs.append(f"{k} is not connected to its driver {net}")
    if "component" in roles:
        comp_ports = {}
        for k, it in items.items():
            if it["kind"] == "component":
                for p, xy in shapes.component_ports(it["component"], it["a"], it.get("b")).items():
                    comp_ports[f"{k.split('.', 1)[1]}.{p}"] = xy
        for net, conn in result.get("nets", {}).items():
            ws = [k for k, it in lines.items() if it.get("net") == net]
            pts = [comp_ports[p] for p in conn if p in comp_ports]
            if len(pts) < 2:
                continue
            segs = [(k, s_) for k in ws for s_ in _segments(lines[k])]
            # Union-find over this net's segments and ports.
            parent = list(range(len(segs) + len(pts)))

            def find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            for i, (_, (a, b)) in enumerate(segs):
                for j in range(i + 1, len(segs)):
                    c, d = segs[j][1]
                    if any(_on_segment(p, c, d) for p in (a, b)) or any(_on_segment(p, a, b) for p in (c, d)):
                        parent[find(i)] = find(j)
                for j, p in enumerate(pts):
                    if _near(p, a) or _near(p, b):
                        parent[find(i)] = find(len(segs) + j)
            roots = {find(len(segs) + j) for j in range(len(pts))}
            if len(roots) > 1:
                probs.append(f"net {net} is not fully connected")
            for k in ws:
                for end in (lines[k]["points"][0], lines[k]["points"][-1]):
                    if any(_near(end, p) for p in pts):
                        continue
                    if not any(_on_segment(end, a, b) for o in ws if o != k for a, b in _segments(lines[o])):
                        probs.append(f"{k} has a dangling end")
                    elif not any(_near(end, d) for o in ws for d in lines[o].get("dots", [])):
                        probs.append(f"{k} joins its net without a junction dot")
        for k, it in lines.items():
            for p, xy in comp_ports.items():
                if result.get("nets") and p not in result["nets"].get(it.get("net"), []) and \
                        any(_on_segment(xy, a, b) for a, b in _segments(it)):
                    probs.append(f"{k} touches port {p} of another net")
    return probs
