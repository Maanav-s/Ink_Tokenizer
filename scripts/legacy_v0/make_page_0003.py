"""Hand-authored synthetic page 0003: design-review meeting notes (writing-heavy).

Emits artifacts/page_0003/content.json and layout.json. The content is a
sensor-board power-subsystem design review: title, date/attendees line, four
bullet sections with nested sub-bullets, an action-item table (owner / task /
due), a week timeline table, and a few sketch annotations (underlines, a box
around a key decision, an arrow from a late note to a table row).

The script asserts its own layout and content checks before writing anything.
Standard library only.
"""
import json, os, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "artifacts", "page_0003")
PAGE_ID = "page_0003"
PAGE_W, PAGE_H = 1200, 800
MARGIN = 15  # mm, minimum distance from every box to the page edge
W = "writer_0"
CHAR_W = 0.55  # nominal layout estimate from the schema (recorded, not used for sizing)

# Glyph advance per character in units of text height: the maximum over the
# Hershey fonts futural, cursive and scripts, measured with
# scripts/render_inkml.py's TextRenderer. Symbols drawn as primitives use the
# renderer's own advances. Boxes are sized from these, plus noise margin.
ADVANCE = {
    ' ': 0.58, '!': 0.36, '"': 0.58, '#': 0.76, '$': 0.72, '%': 0.86, '&': 0.93, "'": 0.36,
    '(': 0.51, ')': 0.51, '*': 0.58, '+': 0.93, ',': 0.34, '-': 0.93, '.': 0.31, '/': 0.79,
    '0': 0.72, '1': 0.72, '2': 0.72, '3': 0.72, '4': 0.72, '5': 0.72, '6': 0.72, '7': 0.72,
    '8': 0.72, '9': 0.72, ':': 0.34, ';': 0.34, '<': 0.86, '=': 0.93, '>': 0.86, '?': 0.65,
    '@': 0.97, 'A': 0.65, 'B': 0.76, 'C': 0.76, 'D': 0.76, 'E': 0.68, 'F': 0.65, 'G': 0.76,
    'H': 0.79, 'I': 0.52, 'J': 0.58, 'K': 0.76, 'L': 0.61, 'M': 1.0, 'N': 0.79, 'O': 0.79,
    'P': 0.76, 'Q': 0.79, 'R': 0.76, 'S': 0.72, 'T': 0.58, 'U': 0.79, 'V': 0.7, 'W': 0.86,
    'X': 0.73, 'Y': 0.7, 'Z': 0.72, '[': 0.51, '\\': 0.51, ']': 0.51, '^': 0.58, '_': 0.65,
    '`': 0.34, 'a': 0.68, 'b': 0.68, 'c': 0.65, 'd': 0.68, 'e': 0.65, 'f': 0.43, 'g': 0.68,
    'h': 0.68, 'i': 0.29, 'j': 0.36, 'k': 0.61, 'l': 0.29, 'm': 1.08, 'n': 0.68, 'o': 0.68,
    'p': 0.68, 'q': 0.68, 'r': 0.47, 's': 0.61, 't': 0.43, 'u': 0.68, 'v': 0.58, 'w': 0.79,
    'x': 0.61, 'y': 0.58, 'z': 0.61, '{': 0.51, '|': 0.29, '}': 0.51, '~': 0.86,
    'μ': 0.76, 'Δ': 0.65, 'η': 0.72,
    '→': 0.75, '≤': 0.6, '≥': 0.6, '≠': 0.6, '±': 0.6, '·': 0.3, '×': 0.55, '∞': 0.75,
    '∫': 0.45, '√': 0.55,
}
SUPPORTED_SYMBOLS = set("⊕→≤≥≠±·×∞∫√")
GREEK = set("αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ")


def text_width(s, h):
    """Conservative ink width: font advance, +5% writer size, +0.4 h for slant/spacing noise."""
    return sum(ADVANCE.get(ch, 1.0) for ch in s) * h * 1.05 + 0.4 * h


def tbox(x, y, s, h):
    return [x, y, round(x + text_width(s, h), 1), y + h]


# ================================================================ content
# Text heights by role (mm).
H = {"title": 28, "heading": 22, "bullet": 18, "subbullet": 16, "note": 17}
IND = 30  # mm, indent per sub-bullet level

elements = []      # content elements, in reading order
el = {}            # layout elements
text_ids = []      # every text element id (for checks)


def text(tid, role, s, x, y, **extra):
    e = {"id": tid, "type": "text", "role": role, "text": s, **extra}
    elements.append(e)
    h = H[role]
    el[tid] = {"bbox": tbox(x, y, s, h), "text_height": h, "writer": W}
    text_ids.append(tid)
    return e


def section(sid, heading, items, x, y, step, list_style):
    """A heading plus list items. items: (id, indent, text, marker[, gap_before])."""
    sec = {"id": sid, "type": "section", "heading": heading[0], "list_style": list_style,
           "items": []}
    text(heading[0], "heading", heading[1], x, y, section=sid)
    y += H["heading"] + 14
    parents = {0: heading[0]}
    for it in items:
        iid, indent, s, marker = it[:4]
        y += it[4] if len(it) > 4 else 0
        role = "bullet" if indent == 0 else "subbullet"
        parent = parents[indent]
        text(iid, role, s, x + IND * indent, y, section=sid, indent=indent,
             parent=parent, marker=marker)
        parents[indent + 1] = iid
        sec["items"].append({"id": iid, "indent": indent, "parent": parent})
        y += H[role] + step
    ids = [heading[0]] + [i["id"] for i in sec["items"]]
    boxes = [el[i]["bbox"] for i in ids]
    el[sid] = {"bbox": [min(b[0] for b in boxes) - 5, min(b[1] for b in boxes) - 5,
                        max(b[2] for b in boxes) + 5, max(b[3] for b in boxes) + 5],
               "writer": W}
    elements.append(sec)
    return y


LX, RX = 40, 640   # left and right column x

text("title", "title", "Sensor brd rev B - power design review", LX, 25)
text("meta", "note", "Thu 1 Oct   att: MK, JR, AS, DP, LT", LX, 69,
     meta={"date": "Thu 1 Oct", "attendees": ["MK", "JR", "AS", "DP", "LT"]})

# ---- left column: agenda (numbered) and decisions
section("sec_agenda", ("h_agenda", "Agenda"), [
    ("ag_1", 0, "1. 3V3 rail: LDO vs buck", "1."),
    ("ag_2", 0, "2. Vin range + rev-pol prot.", "2."),
    ("ag_3", 0, "3. sleep current budget", "3."),
    ("ag_4", 0, "4. rev B schedule / owners", "4."),
], LX, 100, 15, "numbered")

section("sec_decisions", ("h_dec", "Decisions"), [
    ("dec_1", 0, "- 3V3 → sync buck, 2 MHz", "-"),
    ("dec_1a", 1, "- LDO only as post-reg for AFE", "-", 4),
    ("dec_1b", 1, "- target η ≥ 85% @ 50 mA", "-"),
    ("dec_2", 0, "- Vin 2.7-5.5 V, P-FET rev-pol", "-"),
    ("dec_3", 0, "- keep 2V5 ref ±0.1%, split AGND", "-"),
], LX, 275, 15, "dash")

# ---- right column: open questions and risks
section("sec_questions", ("h_oq", "Open questions"), [
    ("oq_1", 0, "- buck Iq in sleep ≤ 1 μA?", "-"),
    ("oq_1a", 1, "- DS: 0.6 μA typ, no max given", "-"),
    ("oq_2", 0, "- inrush on USB hot-plug?", "-"),
    ("oq_2a", 1, "- soft-start ≥ 1 ms enough?", "-"),
    ("oq_3", 0, "- EMI vs RF front end ??", "-"),
], RX, 100, 15, "dash")

section("sec_risks", ("h_risk", "Risks"), [
    ("rk_1", 0, "- buck ripple → ADC noise", "-"),
    ("rk_1a", 1, "- spec ripple ≤ 10 mVpp", "-"),
    ("rk_1b", 1, "- LC filter + keep-out", "-"),
    ("rk_2", 0, "- inductor lead time 12 wk", "-"),
    ("rk_3", 0, "- ΔT @ 300 mA burst ~8 C ?", "-"),
], RX, 305, 15, "dash")


# ---- tables (generic "table" element, same structure as truth_table)
def table(tid, header, rows, rules, kind, columns):
    t = {"id": tid, "type": "table", "kind": kind, "columns": columns,
         "header": [{"id": f"{tid}.h{j}", "text": s} for j, s in enumerate(header)],
         "rows": [{"id": f"{tid}.r{i}",
                   "cells": [{"id": f"{tid}.r{i}c{j}", "text": s} for j, s in enumerate(r)]}
                  for i, r in enumerate(rows)],
         "row_order": "as_written",
         "rules": rules}
    elements.append(t)
    return t


PAD = 6  # mm, minimum horizontal padding between cell text and cell edge


def table_layout(t, x, y, col_w, row_h, th, align):
    col_x = [x]
    for w in col_w:
        col_x.append(col_x[-1] + w)
    n_rows = len(t["rows"]) + 1
    row_y = [y + row_h * i for i in range(n_rows + 1)]
    cells, row_boxes = {}, {}

    def cell(cid, j, i, s):
        x0, x1, y0, y1 = col_x[j], col_x[j + 1], row_y[i], row_y[i + 1]
        tw = text_width(s, th)
        cx = x0 + PAD if align[j] == "left" else x0 + (x1 - x0 - tw) / 2
        cy = y0 + (row_h - th) / 2
        cells[cid] = {"cell_box": [x0, y0, x1, y1],
                      "text_bbox": [round(cx, 1), cy, round(cx + tw, 1), cy + th],
                      "row": i - 1 if i else "header", "col": j, "writer": W}

    for j, h in enumerate(t["header"]):
        cell(h["id"], j, 0, h["text"])
    for i, r in enumerate(t["rows"]):
        row_boxes[r["id"]] = [col_x[0], row_y[i + 1], col_x[-1], row_y[i + 2]]
        for j, c in enumerate(r["cells"]):
            cell(c["id"], j, i + 1, c["text"])
    rules = {}
    for r in t["rules"]:
        if r["kind"] == "horizontal":
            yy = row_y[1] if r["after_row"] == "header" else row_y[r["after_row"] + 2]
            rules[r["id"]] = {"polyline": [[col_x[0] - 4, yy], [col_x[-1] + 4, yy]], "writer": W}
        else:
            xx = col_x[r["after_column"] + 1]
            rules[r["id"]] = {"polyline": [[xx, row_y[0] - 3], [xx, row_y[-1] + 3]], "writer": W}
    el[t["id"]] = {"bbox": [col_x[0] - 5, row_y[0] - 5, col_x[-1] + 5, row_y[-1] + 5],
                   "column_x": col_x, "row_y": row_y, "text_height": th,
                   "cells": cells, "rules": rules, "row_boxes": row_boxes, "writer": W}


text("h_act", "heading", "Action items", LX, 482, section=None)
act = table("act", ["who", "task", "due"], [
    ["MK", "buck schem + BOM", "Wk41"],
    ["JR", "inrush / soft-start sim", "Wk41"],
    ["AS", "Iq meas. on eval brd", "Wk42"],
    ["DP", "layout keep-out vs ADC", "Wk42"],
    ["LT", "order L samples", "Wk41"],
], [
    {"id": "act.rule_header", "kind": "horizontal", "after_row": "header"},
    {"id": "act.rule_c0", "kind": "vertical", "after_column": 0},
    {"id": "act.rule_c1", "kind": "vertical", "after_column": 1},
], "action_items", ["owner", "task", "due"])
table_layout(act, LX, 517, [70, 310, 95], 35, 18, ["center", "left", "center"])

text("h_sched", "heading", "Schedule", RX, 510, section=None)
sched = table("sched", ["Wk41", "Wk42", "Wk43", "Wk44"], [
    ["schem", "layout", "fab", "bring-up"],
], [
    {"id": "sched.rule_header", "kind": "horizontal", "after_row": "header"},
    {"id": "sched.rule_c0", "kind": "vertical", "after_column": 0},
    {"id": "sched.rule_c1", "kind": "vertical", "after_column": 1},
    {"id": "sched.rule_c2", "kind": "vertical", "after_column": 2},
], "timeline", ["Wk41", "Wk42", "Wk43", "Wk44"])
table_layout(sched, RX, 545, [125, 125, 125, 125], 35, 18, ["center"] * 4)

# Late additions (written after the tables): a note that links the lead-time
# risk to the "order L samples" action row, and the next-review line.
text("note_lead", "note", "! L lead time gates bring-up", RX, 652,
     refers_to=["rk_2", "act.r4", "sched.r0c3"])
text("note_next", "note", "next review: Wk42 Thu", RX, 735)

# ---- sketch annotations
def pad_box(b, p):
    return [b[0] - p, b[1] - p, b[2] + p, b[3] + p]


hb = el["h_agenda"]["bbox"]
ha = el["h_act"]["bbox"]
db = pad_box(el["dec_1"]["bbox"], 5)
nb = el["note_lead"]["bbox"]
r4 = el["act"]["row_boxes"]["act.r4"]
arrow_pts = [[nb[0] - 6, (nb[1] + nb[3]) / 2], [r4[2] + 40, (r4[1] + r4[3]) / 2 - 8],
             [r4[2] + 8, (r4[1] + r4[3]) / 2]]
sketch_items = {
    "sk.ul_agenda": {"kind": "polyline", "points": [[hb[0], hb[3] + 4], [hb[2], hb[3] + 4]]},
    "sk.ul_act": {"kind": "polyline", "points": [[ha[0], ha[3] + 4], [ha[2], ha[3] + 4]]},
    "sk.box_dec_1": {"kind": "rect", "bbox": db, "corner_radius": 4},
    "sk.arrow_lead": {"kind": "arrow", "points": arrow_pts, "head": 10},
}
for v in sketch_items.values():
    v["writer"] = W
pts = [p for v in sketch_items.values() for p in v.get("points", [])] + \
      [p for v in sketch_items.values() if "bbox" in v for p in (v["bbox"][:2], v["bbox"][2:])]
el["sk"] = {"bbox": [min(p[0] for p in pts) - 3, min(p[1] for p in pts) - 3,
                     max(p[0] for p in pts) + 3, max(p[1] for p in pts) + 3],
            "items": sketch_items, "writer": W}
elements.append({"id": "sk", "type": "sketch", "items": [
    {"id": "sk.ul_agenda", "kind": "underline", "target": "h_agenda"},
    {"id": "sk.ul_act", "kind": "underline", "target": "h_act"},
    {"id": "sk.box_dec_1", "kind": "box", "target": "dec_1", "meaning": "key decision"},
    {"id": "sk.arrow_lead", "kind": "arrow", "from": "note_lead", "to": "act.r4"},
]})

content = {
    "schema": "ink_tokenizer.synthetic_page.content/v0",
    "page_id": PAGE_ID,
    "topic": "meeting notes: sensor board rev B power subsystem design review",
    "provenance": {"generator": "hand-produced by LLM agent (Claude)", "automated": False},
    "elements": elements,
}

# ================================================================ writing order
# Top to bottom, left column then right column, tables header first then row
# by row; the decision box, the lead-time note with its arrow, and the
# next-review line are later additions.
order = ["title", "meta", "h_agenda", "sk.ul_agenda", "ag_1", "ag_2", "ag_3", "ag_4",
         "h_dec", "dec_1", "dec_1a", "dec_1b", "dec_2", "dec_3",
         "h_oq", "oq_1", "oq_1a", "oq_2", "oq_2a", "oq_3",
         "h_risk", "rk_1", "rk_1a", "rk_1b", "rk_2", "rk_3",
         "h_act", "sk.ul_act"]
order += [h["id"] for h in act["header"]] + ["act.rule_header", "act.rule_c0", "act.rule_c1"]
for r in act["rows"]:
    order += [c["id"] for c in r["cells"]]
order += ["h_sched"] + [h["id"] for h in sched["header"]] + [r["id"] for r in sched["rules"]]
for r in sched["rows"]:
    order += [c["id"] for c in r["cells"]]
order += ["sk.box_dec_1", "note_lead", "sk.arrow_lead", "note_next"]

layout = {
    "schema": "ink_tokenizer.synthetic_page.layout/v0",
    "page_id": PAGE_ID,
    "unit": "mm",
    "page": {"width": PAGE_W, "height": PAGE_H, "origin": "top-left", "y_axis": "down",
             "surface": "whiteboard"},
    "writers": [{"id": W, "style": None}],
    "text_metrics": {"char_advance_per_height": CHAR_W,
                     "note": "bboxes sized from measured Hershey advances (max of futural, "
                             "cursive, scripts) +5% and +0.4 h noise margin; renderer may refit"},
    "elements": el,
    "writing_order": order,
}

# ================================================================ checks
def overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def inside(inner, outer):
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def seg_hits_box(p, q, b, n=200):
    return any(b[0] <= p[0] + (q[0] - p[0]) * k / n <= b[2] and
               b[1] <= p[1] + (q[1] - p[1]) * k / n <= b[3] for k in range(n + 1))


# content ids, leaf ids and the layout ids that place them
cidx, leaves = {}, set()
for e in elements:
    cidx[e["id"]] = e
    if e["type"] == "text":
        leaves.add(e["id"])
    for h in e.get("header", []):
        cidx[h["id"]] = h; leaves.add(h["id"])
    for r in e.get("rows", []):
        cidx[r["id"]] = r
        for c in r["cells"]:
            cidx[c["id"]] = c; leaves.add(c["id"])
    for key in ("rules", "items") if e["type"] != "section" else ():
        for s in e.get(key, []):
            cidx[s["id"]] = s; leaves.add(s["id"])
placed = set(el)
for e in el.values():
    for key in ("cells", "rules", "items", "row_boxes"):
        placed |= set(e.get(key, {}))
missing = set(cidx) - placed
assert not missing, f"content ids without layout: {missing}"
assert not placed - set(cidx), f"layout ids without content: {placed - set(cidx)}"
assert len(order) == len(set(order)), "writing_order has duplicates"
assert set(order) == leaves, f"writing_order mismatch: {set(order) ^ leaves}"

# text boxes: text elements, table cell text, sketch text items
tboxes = {i: el[i]["bbox"] for i in text_ids}
for t in ("act", "sched"):
    for cid, c in el[t]["cells"].items():
        tboxes[cid] = c["text_bbox"]
page = [MARGIN, MARGIN, PAGE_W - MARGIN, PAGE_H - MARGIN]
for i, b in tboxes.items():
    assert inside(b, page), f"{i} outside page: {b}"
ids = sorted(tboxes)
for a in range(len(ids)):
    for b in range(a + 1, len(ids)):
        assert not overlap(tboxes[ids[a]], tboxes[ids[b]]), f"overlap {ids[a]} / {ids[b]}"
for e in el.values():
    assert inside(e["bbox"], [0, 0, PAGE_W, PAGE_H]), f"element bbox outside page: {e['bbox']}"
# vertical gap between consecutive lines of a column (noise moves baselines)
for i in text_ids:
    for j in text_ids:
        bi, bj = tboxes[i], tboxes[j]
        if i != j and bi[0] < bj[2] and bj[0] < bi[2] and bi[3] <= bj[1]:
            assert bj[1] - bi[3] >= 12, f"lines {i}, {j} closer than 6 mm"

# indentation: a sub-bullet starts right of its parent
for i in text_ids:
    p = cidx[i].get("parent")
    if p and cidx[i]["indent"] > 0:
        assert tboxes[i][0] > tboxes[p][0], f"{i} not indented past {p}"
    if cidx[i]["role"] == "subbullet":
        assert cidx[i]["indent"] >= 1

# tables: row width matches header, cell text fits its cell box with padding
for t in (act, sched):
    n = len(t["header"])
    for r in t["rows"]:
        assert len(r["cells"]) == n, f"{r['id']} has {len(r['cells'])} cells, header {n}"
    for cid, c in el[t["id"]]["cells"].items():
        tb, cb = c["text_bbox"], c["cell_box"]
        s = cidx[cid]["text"]
        assert tb[2] - tb[0] >= text_width(s, el[t["id"]]["text_height"]) - 0.2
        assert inside(pad_box(tb, 0), [cb[0] + PAD - 0.1, cb[1] + 4, cb[2] - PAD + 0.1, cb[3] - 4]), \
            f"cell text {cid} {s!r} does not fit {cb}"

# sketch items: underlines and arrow cross no text except their own targets;
# the decision box encloses dec_1 and clears every other text box
skc = {s["id"]: s for s in cidx["sk"]["items"]}
for sid, it in sketch_items.items():
    own = {skc[sid].get("target"), skc[sid].get("from")}
    if it["kind"] in ("polyline", "arrow"):
        for p, q in zip(it["points"], it["points"][1:]):
            for i, b in tboxes.items():
                if i not in own:
                    assert not seg_hits_box(p, q, pad_box(b, 2)), f"{sid} crosses {i}"
        for p in it["points"]:
            assert inside([p[0], p[1], p[0], p[1]], page), f"{sid} off page"
    if it["kind"] == "rect":
        assert inside(tboxes[skc[sid]["target"]], pad_box(it["bbox"], -3))
        for i, b in tboxes.items():
            if i != skc[sid]["target"]:
                assert not overlap(pad_box(b, 3), it["bbox"]), f"{sid} touches {i}"
# the arrow tip ends just right of the row it points at, not inside the table
tip = sketch_items["sk.arrow_lead"]["points"][-1]
assert r4[1] < tip[1] < r4[3] and 0 < tip[0] - r4[2] < 15

# characters: ASCII, Greek, and the primitive-drawn symbols only
def strings():
    for e in elements:
        if "text" in e:
            yield e["text"]
    for t in (act, sched):
        for h in t["header"]:
            yield h["text"]
        for r in t["rows"]:
            for c in r["cells"]:
                yield c["text"]
for s in strings():
    for ch in s:
        assert (32 <= ord(ch) < 127) or ch in GREEK or ch in SUPPORTED_SYMBOLS, \
            f"unsupported character {ch!r} in {s!r}"
        assert ch in ADVANCE, f"no width estimate for {ch!r}"

n_lines = len(text_ids) + 1 + len(act["rows"]) + 1 + len(sched["rows"])
assert 25 <= n_lines <= 40, n_lines

# ================================================================ write
os.makedirs(OUT, exist_ok=True)
for name, obj in (("content.json", content), ("layout.json", layout)):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
print(f"written {OUT}: {len(order)} writing_order entries, {n_lines} handwritten lines; checks OK")
