"""Hand-authored synthetic page (full adder). Emits content.json and layout.json."""
import json, os, sys

# Output directory; defaults to artifacts/page_0001 at the repo root.
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "artifacts", "page_0001")
os.makedirs(OUT, exist_ok=True)
W = "writer_0"
CHAR_W = 0.55  # glyph advance as fraction of text height (layout estimate)

# ---------------------------------------------------------------- content
title = {"id": "title", "type": "text", "role": "title",
         "text": "Full adder - design session"}
notes = [
    {"id": "note_1", "type": "text", "role": "bullet", "text": "- need 1-bit FA cell for 4-bit ripple carry"},
    {"id": "note_2", "type": "text", "role": "bullet", "text": "- in: A, B, Cin   out: S, Cout"},
    {"id": "note_3", "type": "text", "role": "bullet", "text": "- build w/ 2 XOR, 2 AND, 1 OR"},
]
header = ["A", "B", "Cin", "S", "Cout"]
rows = []
for i in range(8):
    a, b, c = (i >> 2) & 1, (i >> 1) & 1, i & 1
    s = a ^ b ^ c
    co = (a & b) | (c & (a ^ b))
    rows.append([a, b, c, s, co])
table = {
    "id": "tt_1", "type": "truth_table",
    "header": [{"id": f"tt_1.h{j}", "text": h} for j, h in enumerate(header)],
    "input_columns": [0, 1, 2], "output_columns": [3, 4],
    "row_order": "binary_counting_msb_first",
    "rows": [{"id": f"tt_1.r{i}",
              "cells": [{"id": f"tt_1.r{i}c{j}", "text": str(v)} for j, v in enumerate(r)]}
             for i, r in enumerate(rows)],
    "rules": [
        {"id": "tt_1.rule_header", "kind": "horizontal", "after_row": "header"},
        {"id": "tt_1.rule_io", "kind": "vertical", "after_column": 2},
    ],
}
math = [
    {"id": "math_1", "type": "math", "text": "S = A xor B xor Cin",
     "written": "S = A ⊕ B ⊕ Cin", "latex": r"S = A \oplus B \oplus C_{in}",
     "defines": "S"},
    {"id": "math_2", "type": "math", "text": "Cout = AB + A Cin + B Cin",
     "written": "Cout = AB + ACin + BCin", "latex": r"C_{out} = AB + AC_{in} + BC_{in}",
     "defines": "Cout"},
    {"id": "math_3", "type": "math", "text": "= AB + Cin (A xor B)",
     "written": "= AB + Cin(A ⊕ B)", "latex": r"= AB + C_{in}(A \oplus B)",
     "defines": "Cout", "continues": "math_2"},
]
gate_ports = {"xor2": ["a", "b"], "and2": ["a", "b"], "or2": ["a", "b"]}
components = [
    {"id": "pin_A", "type": "input_pin", "label": "A", "ports": ["y"]},
    {"id": "pin_B", "type": "input_pin", "label": "B", "ports": ["y"]},
    {"id": "pin_Cin", "type": "input_pin", "label": "Cin", "ports": ["y"]},
    {"id": "xor_1", "type": "xor2", "label": None, "ports": ["a", "b", "y"]},
    {"id": "and_1", "type": "and2", "label": None, "ports": ["a", "b", "y"]},
    {"id": "xor_2", "type": "xor2", "label": None, "ports": ["a", "b", "y"]},
    {"id": "and_2", "type": "and2", "label": None, "ports": ["a", "b", "y"]},
    {"id": "or_1", "type": "or2", "label": None, "ports": ["a", "b", "y"]},
    {"id": "pin_S", "type": "output_pin", "label": "S", "ports": ["a"]},
    {"id": "pin_Cout", "type": "output_pin", "label": "Cout", "ports": ["a"]},
]
wires = [  # point-to-point connections, grouped into nets
    {"id": "w_1", "net": "A", "from": "pin_A.y", "to": "xor_1.a"},
    {"id": "w_2", "net": "A", "from": "pin_A.y", "to": "and_1.a"},
    {"id": "w_3", "net": "B", "from": "pin_B.y", "to": "xor_1.b"},
    {"id": "w_4", "net": "B", "from": "pin_B.y", "to": "and_1.b"},
    {"id": "w_5", "net": "P", "from": "xor_1.y", "to": "xor_2.a"},
    {"id": "w_6", "net": "P", "from": "xor_1.y", "to": "and_2.a"},
    {"id": "w_7", "net": "Cin", "from": "pin_Cin.y", "to": "xor_2.b"},
    {"id": "w_8", "net": "Cin", "from": "pin_Cin.y", "to": "and_2.b"},
    {"id": "w_9", "net": "G", "from": "and_1.y", "to": "or_1.b"},
    {"id": "w_10", "net": "Q", "from": "and_2.y", "to": "or_1.a"},
    {"id": "w_11", "net": "S", "from": "xor_2.y", "to": "pin_S.a"},
    {"id": "w_12", "net": "Cout", "from": "or_1.y", "to": "pin_Cout.a"},
]
dlabels = [
    {"id": "lbl_caption", "type": "label", "text": "1-bit FA", "attached_to": None},
    {"id": "lbl_P", "type": "label", "text": "P", "attached_to": "net:P"},
    {"id": "lbl_G", "type": "label", "text": "G", "attached_to": "net:G"},
]
diagram = {"id": "dia_1", "type": "circuit", "components": components,
           "wires": wires, "labels": dlabels,
           "implements": {"S": "math_1", "Cout": "math_3"}}

content = {
    "schema": "ink_tokenizer.synthetic_page.content/v0",
    "page_id": "page_0001",
    "topic": "digital logic: 1-bit full adder",
    "provenance": {"generator": "hand-produced by LLM agent (Claude)", "automated": False},
    "elements": [title, *notes, table, *math, diagram],
}

# ---------------------------------------------------------------- layout
def tbox(x, y, text, h):
    return [x, y, round(x + len(text) * h * CHAR_W, 1), y + h]

el = {}
el["title"] = {"bbox": tbox(40, 30, title["text"], 30), "text_height": 30, "writer": W}
for k, n in enumerate(notes):
    el[n["id"]] = {"bbox": tbox(40, 85 + 38 * k, n["text"], 24), "text_height": 24, "writer": W}

# table grid
tx, ty, cw, rh, th = 40, 220, 60, 38, 22
col_x = [tx + cw * j for j in range(6)]           # column boundaries
row_y = [ty + rh * i for i in range(10)]          # header + 8 rows boundaries
cells = {}
def cell(idtag, j, i, text):
    x0, x1, y0, y1 = col_x[j], col_x[j + 1], row_y[i], row_y[i + 1]
    tw = len(text) * th * CHAR_W
    cx = x0 + (cw - tw) / 2
    cy = y0 + (rh - th) / 2
    cells[idtag] = {"cell_box": [x0, y0, x1, y1],
                    "text_bbox": [round(cx, 1), cy, round(cx + tw, 1), cy + th],
                    "row": i - 1 if i else "header", "col": j, "writer": W}
for j, h in enumerate(table["header"]):
    cell(h["id"], j, 0, h["text"])
for i, r in enumerate(table["rows"]):
    for j, c in enumerate(r["cells"]):
        cell(c["id"], j, i + 1, c["text"])
rules = {
    "tt_1.rule_header": {"polyline": [[col_x[0] - 5, row_y[1]], [col_x[-1] + 5, row_y[1]]], "writer": W},
    "tt_1.rule_io": {"polyline": [[col_x[3], row_y[0] - 5], [col_x[3], row_y[-1] + 5]], "writer": W},
}
el["tt_1"] = {"bbox": [col_x[0] - 5, row_y[0] - 5, col_x[-1] + 5, row_y[-1] + 5],
              "column_x": col_x, "row_y": row_y, "text_height": th,
              "cells": cells, "rules": rules, "writer": W}

for k, m in enumerate(math):
    x = 420 if k < 2 else 420 + 5 * 25 * CHAR_W  # continuation aligned under RHS
    el[m["id"]] = {"bbox": tbox(round(x, 1), 220 + 42 * k, m["written"], 25),
                   "text_height": 25, "writer": W}

# diagram geometry
GW, GH = 60, 50
def gate(x, y):
    return {"bbox": [x, y, x + GW, y + GH],
            "ports": {"a": [x, y + 10], "b": [x, y + 40], "y": [x + GW, y + GH / 2]}}
comp = {
    "xor_1": gate(580, 390), "and_1": gate(580, 510),
    "xor_2": gate(720, 405), "and_2": gate(720, 475), "or_1": gate(840, 515),
}
LH = 20  # pin label text height
def pin_in(x, y, label):
    lw = len(label) * LH * CHAR_W
    return {"bbox": [x - 4, y - 4, x + 4, y + 4], "ports": {"y": [x, y]},
            "label_bbox": [round(x - 10 - lw, 1), y - LH / 2, x - 10, y + LH / 2],
            "label_text_height": LH}
def pin_out(x, y, label):
    lw = len(label) * LH * CHAR_W
    return {"bbox": [x - 4, y - 4, x + 4, y + 4], "ports": {"a": [x, y]},
            "label_bbox": [x + 10, y - LH / 2, round(x + 10 + lw, 1), y + LH / 2],
            "label_text_height": LH}
comp["pin_A"] = pin_in(500, 400, "A")
comp["pin_B"] = pin_in(500, 430, "B")
comp["pin_Cin"] = pin_in(500, 630, "Cin")
comp["pin_S"] = pin_out(960, 430, "S")
comp["pin_Cout"] = pin_out(960, 540, "Cout")
for c in comp.values():
    c["writer"] = W

P = lambda ref: comp[ref.split(".")[0]]["ports"][ref.split(".")[1]]
poly = {
    "w_1": [P("pin_A.y"), P("xor_1.a")],
    "w_2": [P("pin_A.y"), [530, 400], [530, 520], P("and_1.a")],
    "w_3": [P("pin_B.y"), P("xor_1.b")],
    "w_4": [P("pin_B.y"), [550, 430], [550, 550], P("and_1.b")],
    "w_5": [P("xor_1.y"), P("xor_2.a")],
    "w_6": [P("xor_1.y"), [680, 415], [680, 485], P("and_2.a")],
    "w_7": [P("pin_Cin.y"), [700, 630], [700, 445], P("xor_2.b")],
    "w_8": [P("pin_Cin.y"), [700, 630], [700, 515], P("and_2.b")],
    "w_9": [P("and_1.y"), [810, 535], [810, 555], P("or_1.b")],
    "w_10": [P("and_2.y"), [820, 500], [820, 525], P("or_1.a")],
    "w_11": [P("xor_2.y"), P("pin_S.a")],
    "w_12": [P("or_1.y"), P("pin_Cout.a")],
}
wires_l = {k: {"polyline": v, "writer": W} for k, v in poly.items()}
junctions = [  # fan-out dots
    {"net": "A", "point": [530, 400], "writer": W}, {"net": "B", "point": [550, 430], "writer": W},
    {"net": "P", "point": [680, 415], "writer": W}, {"net": "Cin", "point": [700, 515], "writer": W},
]
labels_l = {
    "lbl_caption": {"bbox": tbox(480, 345, "1-bit FA", 24), "text_height": 24, "writer": W},
    "lbl_P": {"bbox": tbox(652, 393, "P", 18), "text_height": 18, "writer": W},
    "lbl_G": {"bbox": tbox(652, 513, "G", 18), "text_height": 18, "writer": W},
}
allx = [b for c in comp.values() for b in (c["bbox"][0], c["bbox"][2])] + \
       [b for c in comp.values() if "label_bbox" in c for b in (c["label_bbox"][0], c["label_bbox"][2])] + \
       [b for l in labels_l.values() for b in (l["bbox"][0], l["bbox"][2])]
ally = [b for c in comp.values() for b in (c["bbox"][1], c["bbox"][3])] + \
       [b for c in comp.values() if "label_bbox" in c for b in (c["label_bbox"][1], c["label_bbox"][3])] + \
       [b for l in labels_l.values() for b in (l["bbox"][1], l["bbox"][3])]
el["dia_1"] = {"bbox": [min(allx) - 5, min(ally) - 5, max(allx) + 5, max(ally) + 5],
               "components": comp, "wires": wires_l, "junctions": junctions,
               "labels": labels_l, "writer": W}

# writing order: title, notes, table header+rules, inputs row-wise, then
# output columns column-wise, then math, then diagram left-to-right.
order = ["title", "note_1", "note_2", "note_3"]
order += [h["id"] for h in table["header"]] + ["tt_1.rule_header", "tt_1.rule_io"]
for r in table["rows"]:
    order += [c["id"] for c in r["cells"][:3]]
for j in (3, 4):
    order += [r["cells"][j]["id"] for r in table["rows"]]
order += ["math_1", "math_2", "math_3", "lbl_caption",
          "pin_A", "pin_B", "pin_Cin", "xor_1", "and_1", "w_1", "w_2", "w_3", "w_4",
          "lbl_P", "lbl_G", "xor_2", "and_2", "w_5", "w_6", "w_7", "w_8",
          "or_1", "w_9", "w_10", "pin_S", "pin_Cout", "w_11", "w_12"]

layout = {
    "schema": "ink_tokenizer.synthetic_page.layout/v0",
    "page_id": "page_0001",
    "unit": "mm",
    "page": {"width": 1200, "height": 800, "origin": "top-left", "y_axis": "down",
             "surface": "whiteboard"},
    "writers": [{"id": W, "style": None}],
    "text_metrics": {"char_advance_per_height": CHAR_W,
                     "note": "bboxes are layout estimates; renderer may refit"},
    "elements": el,
    "writing_order": order,
}

for name, obj in (("content.json", content), ("layout.json", layout)):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
print("written")
