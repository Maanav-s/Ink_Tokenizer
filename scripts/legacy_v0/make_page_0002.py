"""Hand-authored synthetic page 0002 (flowchart: CI deploy with canary + rollback).

Emits artifacts/page_0002/content.json and layout.json. The flowchart is held
semantically in content.json (nodes and edges) and drawn in layout.json as a
generic `sketch` element. Every sketch item id is `<flowchart id>.<node or edge
id>.<part>` (part: box | text for nodes, arrow | label for edges), so the
item -> content mapping is derivable from the id alone.

The script asserts its structural and geometric checks before writing.
Stdlib only.
"""
import json, math, os, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "artifacts", "page_0002")
PAGE_W, PAGE_H = 1200, 800
W = "writer_0"
# Glyph advance as a fraction of text height used to size text boxes. The
# default Hershey font (futural) measures 0.54-0.67 on these strings; cursive
# is narrower (~0.45). 0.68 keeps the boxes conservative.
CHAR_W = 0.68
NODE_H = 20    # node text height (mm)
LABEL_H = 20   # edge label text height
NOTE_H = 22
TITLE_H = 30
FC = "fc_1"

# ---------------------------------------------------------------- content
title = {"id": "title", "type": "text", "role": "title",
         "text": "Deploy pipeline - canary + rollback"}
notes = [
    {"id": "note_1", "type": "text", "role": "bullet", "text": "- canary: 10% traffic first"},
    {"id": "note_2", "type": "text", "role": "bullet", "text": "- gate: 5xx err rate, 5 min"},
    {"id": "note_3", "type": "text", "role": "bullet", "text": "- rollback is automatic"},
]
nodes = [  # (id, shape, text, terminal role)
    ("n1", "terminator", "push to main", "start"),
    ("n2", "io", "read deploy.yaml", None),
    ("n3", "process", "build + unit tests", None),
    ("n4", "decision", "tests pass?", None),
    ("n5", "process", "deploy canary 10%", None),
    ("n6", "process", "wait 5 min", None),
    ("n7", "decision", "err < 1%?", None),
    ("n8", "decision", "at 100%?", None),
    ("n9", "process", "traffic += 30%", None),
    ("n10", "process", "rollback", None),
    ("n11", "terminator", "stop + notify", "end"),
    ("n12", "terminator", "done", "end"),
]
edges = [  # (id, from, to, label)
    ("e1", "n1", "n2", None),
    ("e2", "n2", "n3", None),
    ("e3", "n3", "n4", None),
    ("e4", "n4", "n5", "yes"),
    ("e5", "n4", "n11", "no"),
    ("e6", "n5", "n6", None),
    ("e7", "n6", "n7", None),
    ("e8", "n7", "n8", "yes"),
    ("e9", "n7", "n10", "no"),
    ("e10", "n8", "n12", "yes"),
    ("e11", "n8", "n9", "no"),
    ("e12", "n9", "n6", None),      # loop back: wait again at the new traffic level
    ("e13", "n10", "n11", None),
]
flowchart = {
    "id": FC, "type": "flowchart",
    "nodes": [dict({"id": i, "shape": s, "text": t}, **({"terminal": r} if r else {}))
              for i, s, t, r in nodes],
    "edges": [dict({"id": i, "from": a, "to": b}, **({"label": l} if l else {}))
              for i, a, b, l in edges],
    "layout_item_ids": {
        "pattern": FC + ".<node_or_edge_id>.<part>",
        "node_parts": ["box", "text"], "edge_parts": ["arrow", "label"],
    },
}
content = {
    "schema": "ink_tokenizer.synthetic_page.content/v0",
    "page_id": "page_0002",
    "topic": "software process: CI deploy pipeline with canary rollout and automatic rollback",
    "provenance": {"generator": "hand-produced by LLM agent (Claude)", "automated": False},
    "elements": [title, *notes, flowchart],
}

# ---------------------------------------------------------------- layout
def r1(v):
    return round(v, 1)

def tw(text, h):
    return len(text) * h * CHAR_W

# Node geometry: column centre x, row centre y, width, height.
CX = {1: 185, 2: 500, 3: 790, 4: 1040}
CY = {1: 110, 2: 205, 3: 305, 4: 430, 5: 570, 6: 700}
GEOM = {  # id: (col, row, w, h)
    "n1": (1, 1, 190, 46), "n2": (1, 2, 270, 50), "n3": (1, 3, 270, 50),
    "n4": (1, 4, 220, 90), "n5": (2, 4, 260, 50), "n6": (2, 5, 180, 50),
    "n7": (2, 6, 200, 90), "n8": (3, 6, 200, 90), "n9": (3, 5, 220, 50),
    "n10": (1, 6, 160, 50), "n11": (1, 5, 200, 46), "n12": (4, 5, 130, 46),
}
IO_SKEW = 22  # horizontal offset of the parallelogram's top edge
NODE = {i: {"shape": s, "text": t} for i, s, t, _ in nodes}

def node_box(nid):
    c, r, w, h = GEOM[nid]
    return [CX[c] - w / 2, CY[r] - h / 2, CX[c] + w / 2, CY[r] + h / 2]

def anchor(nid, side):
    """Point on the node outline where an edge attaches (edge midpoints / vertices)."""
    x0, y0, x1, y1 = node_box(nid)
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    return {"top": [xm, y0], "bottom": [xm, y1], "left": [x0, ym], "right": [x1, ym]}[side]

def outline(nid):
    """Closed outline polygon of a node shape, used by the checks."""
    x0, y0, x1, y1 = node_box(nid)
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    shape = NODE[nid]["shape"]
    if shape == "decision":
        return [(xm, y0), (x1, ym), (xm, y1), (x0, ym)]
    if shape == "io":
        return [tuple(p) for p in io_points(nid)]
    r = min(corner_radius(nid), (x1 - x0) / 2, (y1 - y0) / 2)
    if r <= 0:
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    pts = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        pts += [(cx + r * math.cos(math.radians(a0 + 90 * k / 24)),
                 cy + r * math.sin(math.radians(a0 + 90 * k / 24))) for k in range(25)]
    return pts

def corner_radius(nid):
    return {"terminator": 23, "process": 0}.get(NODE[nid]["shape"], 0)

def io_points(nid):
    x0, y0, x1, y1 = node_box(nid)
    return [[x0 + IO_SKEW, y0], [x1, y0], [x1 - IO_SKEW, y1], [x0, y1]]

items = {}
def item(iid, **kw):
    items[iid] = dict(kw, writer=W)

for nid, shape, text, _ in nodes:
    box = node_box(nid)
    if shape == "decision":
        item(f"{FC}.{nid}.box", kind="diamond", bbox=box)
    elif shape == "io":
        item(f"{FC}.{nid}.box", kind="polyline", points=io_points(nid), closed=True)
    elif shape == "terminator":
        item(f"{FC}.{nid}.box", kind="rect", bbox=box, corner_radius=corner_radius(nid))
    else:
        item(f"{FC}.{nid}.box", kind="rect", bbox=box)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w = tw(text, NODE_H)
    # The box bottom is the descender line; shift it down so the
    # cap-to-baseline band is roughly centred on the node.
    yb = cy + 0.45 * NODE_H
    item(f"{FC}.{nid}.text", kind="text", text=text, align="center", text_height=NODE_H,
         bbox=[r1(cx - w / 2), r1(yb - NODE_H), r1(cx + w / 2), r1(yb)])

ROUTE = {  # edge: (from side, via points, to side)
    "e1": ("bottom", [], "top"), "e2": ("bottom", [], "top"), "e3": ("bottom", [], "top"),
    "e4": ("right", [], "left"), "e5": ("bottom", [], "top"), "e6": ("bottom", [], "top"),
    "e7": ("bottom", [], "top"), "e8": ("right", [], "left"), "e9": ("left", [], "right"),
    "e10": ("right", [[CX[4], CY[6]]], "bottom"), "e11": ("top", [], "bottom"),
    "e12": ("left", [], "right"), "e13": ("top", [], "bottom"),
}
for eid, a, b, label in edges:
    s_side, via, t_side = ROUTE[eid]
    pts = [anchor(a, s_side), *via, anchor(b, t_side)]
    item(f"{FC}.{eid}.arrow", kind="arrow", points=[[r1(x), r1(y)] for x, y in pts], head=8)
    if label:
        (xa, ya), (xb, yb) = pts[0], pts[1]
        w = tw(label, LABEL_H)
        if ya == yb:   # horizontal first segment: label above it, near the start
            xs = xa + (12 if xb > xa else -12 - w)
            bb = [xs, ya - 5 - LABEL_H, xs + w, ya - 5]
        else:          # vertical first segment: label to its right, near the start
            ys = ya + 8 if yb > ya else ya - 8 - LABEL_H
            bb = [xa + 7, ys, xa + 7 + w, ys + LABEL_H]
        item(f"{FC}.{eid}.label", kind="text", text=label, text_height=LABEL_H,
             bbox=[r1(v) for v in bb])

el = {}
el["title"] = {"bbox": [40, 22, r1(40 + tw(title["text"], TITLE_H)), 22 + TITLE_H],
               "text_height": TITLE_H, "writer": W}
for k, n in enumerate(notes):
    y = 90 + 40 * k
    el[n["id"]] = {"bbox": [650, y, r1(650 + tw(n["text"], NOTE_H)), y + NOTE_H],
                   "text_height": NOTE_H, "writer": W}

def item_bounds(it):
    if "bbox" in it:
        return it["bbox"]
    xs = [p[0] for p in it["points"]]; ys = [p[1] for p in it["points"]]
    return [min(xs), min(ys), max(xs), max(ys)]
bs = [item_bounds(it) for it in items.values()]
el[FC] = {"bbox": [min(b[0] for b in bs) - 5, min(b[1] for b in bs) - 5,
                   max(b[2] for b in bs) + 5, max(b[3] for b in bs) + 5],
          "items": items, "writer": W}

# Writing order: the main path top to bottom, node by node (shape then text,
# except terminators, whose text is written first and then circled), each
# arrow after its source node; side branches are filled in afterwards.
def node_ids(nid):
    if NODE[nid]["shape"] == "terminator":
        return [f"{FC}.{nid}.text", f"{FC}.{nid}.box"]
    return [f"{FC}.{nid}.box", f"{FC}.{nid}.text"]
def edge_ids(eid):
    return [f"{FC}.{eid}.arrow"] + ([f"{FC}.{eid}.label"] if f"{FC}.{eid}.label" in items else [])

order = ["title", "note_1"]
for step in ["n1", "e1", "n2", "e2", "n3", "e3", "n4", "e4", "n5", "e6", "n6", "e7", "n7",
             "e8", "n8", "e10", "n12", "e11", "n9", "e12", "note_2",
             "e5", "n11", "e9", "n10", "e13", "note_3"]:
    if step.startswith("note"):
        order.append(step)
    elif step.startswith("n"):
        order += node_ids(step)
    else:
        order += edge_ids(step)

layout = {
    "schema": "ink_tokenizer.synthetic_page.layout/v0",
    "page_id": "page_0002",
    "unit": "mm",
    "page": {"width": PAGE_W, "height": PAGE_H, "origin": "top-left", "y_axis": "down",
             "surface": "whiteboard"},
    "writers": [{"id": W, "style": None}],
    "text_metrics": {"char_advance_per_height": CHAR_W,
                     "note": "bboxes are layout estimates sized from the futural font's measured "
                             "advance; renderer may refit"},
    "elements": el,
    "writing_order": order,
}

# ---------------------------------------------------------------- checks
TOL = 1.0
node_map = {n["id"]: n for n in flowchart["nodes"]}
edge_map = {e["id"]: e for e in flowchart["edges"]}

# 1. Item <-> content mapping.
expected = set()
for n in flowchart["nodes"]:
    expected |= {f"{FC}.{n['id']}.box", f"{FC}.{n['id']}.text"}
for e in flowchart["edges"]:
    expected.add(f"{FC}.{e['id']}.arrow")
    if "label" in e:
        expected.add(f"{FC}.{e['id']}.label")
assert set(items) == expected, set(items) ^ expected
for iid, it in items.items():
    fc, ref, part = iid.split(".")
    assert fc == FC and (ref in node_map or ref in edge_map), iid
    if part == "text":
        assert it["text"] == node_map[ref]["text"], iid
    if part == "label":
        assert it["text"] == edge_map[ref]["label"], iid

# 2. Graph structure.
for e in flowchart["edges"]:
    assert e["from"] in node_map and e["to"] in node_map, e
starts = [n for n in flowchart["nodes"] if n.get("terminal") == "start"]
assert len(starts) == 1 and starts[0]["shape"] == "terminator"
assert all(n["shape"] == "terminator" for n in flowchart["nodes"] if "terminal" in n)
assert all("terminal" in n for n in flowchart["nodes"] if n["shape"] == "terminator")
out = {n: [e for e in flowchart["edges"] if e["from"] == n] for n in node_map}
for n in flowchart["nodes"]:
    if n["shape"] == "decision":
        assert len(out[n["id"]]) >= 2 and all(e.get("label") for e in out[n["id"]]), n
        assert len({e["label"] for e in out[n["id"]]}) == len(out[n["id"]]), n
    elif n.get("terminal") == "end":
        assert not out[n["id"]], n
    else:
        assert len(out[n["id"]]) == 1, n
seen, stack = set(), [starts[0]["id"]]
while stack:
    n = stack.pop()
    if n not in seen:
        seen.add(n); stack += [e["to"] for e in out[n]]
assert seen == set(node_map), set(node_map) - seen
assert sum(n["shape"] == "decision" for n in flowchart["nodes"]) >= 2
# at least one loop: some edge goes back to a node that reaches its source
def reaches(a, b):
    s, st = set(), [a]
    while st:
        x = st.pop()
        if x == b:
            return True
        if x not in s:
            s.add(x); st += [e["to"] for e in out[x]]
    return False
assert any(reaches(e["to"], e["from"]) for e in flowchart["edges"])

# 3. Arrow geometry: orthogonal segments, endpoints on the outlines.
def seg_dist(p, a, b):
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / ((dx * dx + dy * dy) or 1)))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)
def outline_dist(nid, p):
    poly = outline(nid)
    return min(seg_dist(p, poly[k], poly[(k + 1) % len(poly)]) for k in range(len(poly)))
def inside(nid, p):
    poly, (x, y), c = outline(nid), p, False
    for k in range(len(poly)):
        (xa, ya), (xb, yb) = poly[k], poly[(k + 1) % len(poly)]
        if (ya > y) != (yb > y) and x < xa + (y - ya) * (xb - xa) / (yb - ya):
            c = not c
    return c
for e in flowchart["edges"]:
    pts = items[f"{FC}.{e['id']}.arrow"]["points"]
    assert outline_dist(e["from"], pts[0]) <= TOL, e
    assert outline_dist(e["to"], pts[-1]) <= TOL, e
    for p, q in zip(pts, pts[1:]):
        assert (p[0] == q[0]) != (p[1] == q[1]), (e, p, q)
        assert math.dist(p, q) >= 20, (e, p, q)   # room for the 8 mm head
    # the shaft must not cut through any node it does not connect
    for nid in node_map:
        for p, q in zip(pts, pts[1:]):
            for k in range(1, 20):
                m = (p[0] + (q[0] - p[0]) * k / 20, p[1] + (q[1] - p[1]) * k / 20)
                assert not inside(nid, m) or nid in (e["from"], e["to"]) and \
                    outline_dist(nid, m) <= TOL, (e, nid)

# 4. Text: no overlaps, node text inside its shape, labels clear of shapes and shafts.
texts = {k: v["bbox"] for k, v in el.items() if k != FC}
texts.update({k: v["bbox"] for k, v in items.items() if v["kind"] == "text"})
def overlap(a, b, gap=0):
    return a[0] < b[2] + gap and b[0] < a[2] + gap and a[1] < b[3] + gap and b[1] < a[3] + gap
ks = sorted(texts)
for i, a in enumerate(ks):
    for b in ks[i + 1:]:
        assert not overlap(texts[a], texts[b], 2), (a, b)
for n in flowchart["nodes"]:
    x0, y0, x1, y1 = items[f"{FC}.{n['id']}.text"]["bbox"]
    for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        assert inside(n["id"], p) and outline_dist(n["id"], p) >= 3, (n["id"], p)
def rect_hits_seg(bb, p, q, pad=2):
    x0, y0, x1, y1 = bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad
    return min(p[0], q[0]) <= x1 and max(p[0], q[0]) >= x0 and min(p[1], q[1]) <= y1 and max(p[1], q[1]) >= y0
for iid, it in items.items():
    if iid.endswith(".label"):
        bb = it["bbox"]
        for nid in node_map:
            nb = node_box(nid)
            assert not any(inside(nid, c) for c in ((bb[0], bb[1]), (bb[2], bb[1]), (bb[2], bb[3]), (bb[0], bb[3]))), (iid, nid)
            assert not overlap(bb, nb) or NODE[nid]["shape"] == "decision", (iid, nid)
        for aid, a in items.items():
            if a["kind"] == "arrow":
                for p, q in zip(a["points"], a["points"][1:]):
                    assert not rect_hits_seg(bb, p, q), (iid, aid)
for k, bb in texts.items():   # page-level text must stay clear of the flowchart shapes
    if not k.startswith(FC):
        for nid in node_map:
            assert not overlap(bb, node_box(nid), 5), (k, nid)

# 5. Everything on the page (with a small margin for noise).
M = 10
for k, v in el.items():
    b = v["bbox"]
    assert M <= b[0] < b[2] <= PAGE_W - M and M <= b[1] < b[3] <= PAGE_H - M, (k, b)
for iid, it in items.items():
    b = item_bounds(it)
    assert M <= b[0] <= b[2] <= PAGE_W - M and M <= b[1] <= b[3] <= PAGE_H - M, (iid, b)

# 6. Writing order lists every leaf exactly once.
leaves = [e["id"] for e in content["elements"] if e["type"] == "text"] + list(items)
assert sorted(order) == sorted(leaves) and len(order) == len(set(order))
assert FC not in order  # containers never appear

os.makedirs(OUT, exist_ok=True)
for name, obj in (("content.json", content), ("layout.json", layout)):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
print(f"checks OK; wrote {OUT} ({len(flowchart['nodes'])} nodes, {len(flowchart['edges'])} edges, "
      f"{len(order)} writing_order entries)")
