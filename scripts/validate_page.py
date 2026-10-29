"""Checks for synthetic pages: content, layout and rendered ink.

- check_content(content): the schema, ids and references, that every
  character can be drawn in every pool font and every equation typesets;
  truth tables count in binary, and agree with the expressions that define
  their output columns and with any circuit whose pins they name; numeric
  equations with a `check` evaluate equal on every side.
- check_layout(layout, content): items on the page, writing_order lists
  every item exactly once, writers and targets are consistent.
- check_ink(groups): rendered ink of a text or math item must not touch any
  other item's ink (closer than the pen width plus a margin), and must stay
  on the page. check_ink_clean(layout) runs it on clean renders in every
  pool font; check_inkml(path) on a rendered file.

Usage: uv run --extra render python scripts/validate_page.py page_dir
           [--layout FILE] [--inkml FILE ...]
With no --inkml, renders the layout clean in every pool font and checks
the ink. Exits non-zero if anything fails.
"""
import argparse
import cmath
import itertools
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

import layout_common as lc
from fonts import FONT_POOL

ID = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*")
GATE_FN = {"and2": lambda a, b: a & b, "or2": lambda a, b: a | b, "xor2": lambda a, b: a ^ b,
           "nand2": lambda a, b: 1 - (a & b), "nor2": lambda a, b: 1 - (a | b), "xnor2": lambda a, b: 1 - (a ^ b),
           "not": lambda a: 1 - a, "buf": lambda a: a}
GATE_PORTS = {k: ("a", "b") if k.endswith("2") else ("a",) for k in GATE_FN}
SCHEMATIC_TYPES = {"resistor", "capacitor", "inductor", "diode", "voltage_source", "current_source", "battery",
                   "ground", "terminal"}
NODE_SHAPES = {"terminator", "process", "decision", "io"}
TEXT_ROLES = {"title", "heading", "bullet", "subbullet", "note", "caption", "text"}
# Minimum distance between the centre lines of text ink and any other ink. The
# pen is 2 mm wide, so 2 mm means the strokes touch. Layouts are checked on
# clean ink with a margin. Noisy renders are only rejected for a clear
# overlap: an occasional touch (a descender grazing the next line) happens
# in real handwriting too.
CLEARANCE_CLEAN, MIN_CELLS_CLEAN = 2.5, 3
CLEARANCE_NOISY, MIN_CELLS_NOISY = 1.5, 10


# ------------------------------------------------------------------ content

def check_content(content):
    errs = []
    m = lc.pool_metrics()
    if content.get("schema") != "ink_tokenizer.synthetic_page.content/v1":
        return [f"schema must be ink_tokenizer.synthetic_page.content/v1, got {content.get('schema')!r}"]
    els = content.get("elements")
    if not isinstance(els, list) or not els:
        return ["elements must be a non-empty list"]
    ids = [e.get("id") for e in els]
    for i in ids:
        if not isinstance(i, str) or not ID.fullmatch(i):
            errs.append(f"bad element id {i!r} (letters, digits, _ and - only)")
    errs += [f"duplicate element id {i!r}" for i in {i for i in ids if ids.count(i) > 1}]
    by_id = {e.get("id"): e for e in els}

    def chars(where, s):
        if not isinstance(s, str) or not s.strip():
            errs.append(f"{where}: empty or missing text")
            return
        bad = sorted({ch for ch in s if not m.has(ch)})
        if bad:
            errs.append(f"{where}: characters not drawable: {''.join(bad)!r}")

    def label(where, s):
        if lc.is_math(s):
            typesets(where, s[1:-1])
        else:
            chars(where, s)

    def typesets(where, latex):
        import math_typeset
        try:
            math_typeset.typeset(latex, 20.0, m)
        except Exception as e:  # noqa: BLE001 - any failure means the LaTeX is unusable
            errs.append(f"{where}: LaTeX does not typeset: {e}")

    for e in els:
        t, eid = e.get("type"), e.get("id")
        if t == "text":
            if e.get("role") not in TEXT_ROLES:
                errs.append(f"{eid}: role must be one of {sorted(TEXT_ROLES)}")
            chars(eid, e.get("text"))
        elif t == "list":
            if e.get("style", "dash") not in ("dash", "numbered", "plain"):
                errs.append(f"{eid}: style must be dash, numbered or plain")
            if e.get("heading") is not None:
                chars(f"{eid}.heading", e["heading"])

            def walk(items, depth, where):
                if not isinstance(items, list) or not items:
                    errs.append(f"{where}: items must be a non-empty list")
                    return
                if depth > 2:
                    errs.append(f"{where}: lists nest at most 3 levels")
                for k, it in enumerate(items):
                    chars(f"{where}[{k}]", it.get("text"))
                    if it.get("items"):
                        walk(it["items"], depth + 1, f"{where}[{k}]")
            walk(e.get("items"), 0, eid)
        elif t == "table":
            errs += check_table(e, label)
        elif t == "math":
            if not e.get("latex"):
                errs.append(f"{eid}: math needs latex")
            else:
                typesets(eid, e["latex"])
            if e.get("continues") and e["continues"] not in by_id:
                errs.append(f"{eid}: continues unknown element {e['continues']!r}")
        elif t == "flowchart":
            errs += check_flowchart(e, label)
        elif t == "circuit":
            errs += check_circuit(e, label)
        elif t == "schematic":
            errs += check_schematic(e, label)
        elif t == "annotation":
            if e.get("kind") not in ("underline", "box", "circle", "arrow"):
                errs.append(f"{eid}: annotation kind must be underline, box, circle or arrow")
            for k in ("target",) if e.get("kind") != "arrow" else ("from", "to"):
                ref = e.get(k) or ""
                m_row = re.fullmatch(r"(.+)\.r(\d+)", ref)
                if m_row and by_id.get(m_row.group(1), {}).get("type") == "table":
                    if int(m_row.group(2)) >= len(by_id[m_row.group(1)].get("rows", [])):
                        errs.append(f"{eid}: {k} {ref!r}: no such row (rows count from 0, header excluded)")
                elif ref not in by_id and not any(ref == it.get("id") for x in els for it in iter_list_items(x)):
                    errs.append(f"{eid}: {k} refers to unknown id {ref!r}")
        else:
            errs.append(f"{eid}: unknown element type {t!r}")
    errs += check_semantics(els)
    return errs


def iter_list_items(e):
    def walk(items):
        for it in items or []:
            yield it
            yield from walk(it.get("items"))
    return walk(e.get("items")) if e.get("type") == "list" else iter(())


def check_table(e, label):
    errs, eid = [], e["id"]
    rows, header = e.get("rows"), e.get("header") or []
    if not isinstance(rows, list) or not rows or not all(isinstance(r, list) for r in rows):
        return [f"{eid}: rows must be a non-empty list of lists of strings"]
    ncol = max(len(r) for r in [header] + rows)
    for i, r in enumerate([header] + rows):
        for j, s in enumerate(r):
            if s:
                label(f"{eid}[{i}][{j}]", s)
    if e.get("kind") == "truth_table":
        ins, outs = e.get("input_columns", []), e.get("output_columns", [])
        if not ins or sorted(set(ins) | set(outs)) != list(range(ncol)) or set(ins) & set(outs):
            errs.append(f"{eid}: input_columns and output_columns must split the columns")
            return errs
        if any(len(r) != ncol for r in rows):
            errs.append(f"{eid}: every truth-table row needs all {ncol} cells")
            return errs
        if any(s not in ("0", "1", "x", "X", "-") for r in rows for s in r):
            errs.append(f"{eid}: truth-table cells must be 0, 1 or x")
        if e.get("row_order") == "binary_counting_msb_first":
            want = [list(map(str, bits)) for bits in itertools.product((0, 1), repeat=len(ins))]
            got = [[r[j] for j in ins] for r in rows]
            if got != want[:len(got)] or len(got) != len(want):
                errs.append(f"{eid}: input rows are not all {len(want)} rows in binary counting order")
    return errs


def check_flowchart(e, label):
    errs, eid = [], e["id"]
    nodes = e.get("nodes") or []
    ids = [n.get("id") for n in nodes]
    if not nodes or len(set(ids)) != len(ids):
        errs.append(f"{eid}: nodes must be non-empty with unique ids")
    for n in nodes:
        if n.get("shape") not in NODE_SHAPES:
            errs.append(f"{eid}.{n.get('id')}: shape must be one of {sorted(NODE_SHAPES)}")
        for k, line in enumerate((n.get("text") or "").split("\n")):
            label(f"{eid}.{n.get('id')}.text", line)
    for k, ed in enumerate(e.get("edges") or []):
        if ed.get("from") not in ids or ed.get("to") not in ids:
            errs.append(f"{eid}: edge {k} joins unknown nodes")
    for n in nodes:
        outs = [ed for ed in e.get("edges") or [] if ed.get("from") == n.get("id")]
        if n.get("shape") == "decision" and len(outs) < 2:
            errs.append(f"{eid}.{n.get('id')}: a decision needs at least two out-edges")
        if ed.get("label"):
            label(f"{eid}.edge{k}.label", ed["label"])
    return errs


def check_circuit(e, label):
    errs, eid = [], e["id"]
    comps = {c.get("id"): c for c in e.get("components") or []}
    ports = {}
    for cid, c in comps.items():
        t = c.get("type")
        if t == "input_pin":
            ports[cid] = ("y",)
        elif t == "output_pin":
            ports[cid] = ("a",)
        elif t in GATE_FN:
            ports[cid] = GATE_PORTS[t] + ("y",)
        else:
            errs.append(f"{eid}.{cid}: unknown component type {t!r}")
            continue
        if t.endswith("_pin"):
            label(f"{eid}.{cid}.label", c.get("label"))
    driven = {}
    for w in e.get("wires") or []:
        for end in ("from", "to"):
            cid, _, p = (w.get(end) or "").partition(".")
            if p not in ports.get(cid, ()):
                errs.append(f"{eid}.{w.get('id')}: {end} {w.get(end)!r} is not a port")
        if w.get("to") in driven:
            errs.append(f"{eid}: {w.get('to')} is driven twice")
        driven[w.get("to")] = w
    for cid, c in comps.items():
        for p in ports.get(cid, ()):
            if p != "y" and f"{cid}.{p}" not in driven:
                errs.append(f"{eid}: input {cid}.{p} is not connected")
    for lab in e.get("labels") or []:
        label(f"{eid}.{lab.get('id')}", lab.get("text"))
    return errs


def check_schematic(e, label):
    errs, eid = [], e["id"]
    comps = {c.get("id"): c for c in e.get("components") or []}
    seen = set()
    for cid, c in comps.items():
        if c.get("type") not in SCHEMATIC_TYPES:
            errs.append(f"{eid}.{cid}: unknown component type {c.get('type')!r}")
        at = c.get("at")
        if not (isinstance(at, list) and len(at) == 2 and all(isinstance(v, (int, float)) for v in at)):
            errs.append(f"{eid}.{cid}: needs at: [col, row]")
        if tuple(at or ()) in seen:
            errs.append(f"{eid}.{cid}: grid position {at} is taken")
        seen.add(tuple(at or ()))
        for k in ("label", "value"):
            if c.get(k):
                label(f"{eid}.{cid}.{k}", c[k])
    used = set()
    for n in e.get("nets") or []:
        if len(n.get("connects") or []) < 2:
            errs.append(f"{eid}.{n.get('id')}: a net connects at least two ports")
        for p in n.get("connects") or []:
            cid, _, port = p.partition(".")
            one = comps.get(cid, {}).get("type") in ("ground", "terminal")
            if cid not in comps or port not in (("a",) if one else ("a", "b")):
                errs.append(f"{eid}.{n.get('id')}: {p!r} is not a port")
            if p in used:
                errs.append(f"{eid}: port {p} is in two nets")
            used.add(p)
    return errs


# ------------------------------------------------------------------ semantics

def check_semantics(els):
    """Truth tables against their defining expressions and circuits; numeric checks."""
    errs = []
    by_id = {e["id"]: e for e in els}
    for tt in (e for e in els if e["type"] == "table" and e.get("kind") == "truth_table"):
        header = tt.get("header") or []
        ins, outs = tt.get("input_columns", []), tt.get("output_columns", [])
        if not header or len(header) <= max(ins + outs, default=0):
            continue
        names = [header[j] for j in ins]
        rows = [[int(r[j]) if r[j] in "01" else None for j in range(len(header))] for r in tt["rows"]]
        for e in els:
            if e["type"] != "math" or e.get("defines") not in [header[j] for j in outs] or not e.get("text"):
                continue
            col = header.index(e["defines"])
            parts = [p for p in e["text"].split("=")[1:] if p.strip()]
            for p in parts:
                try:
                    vals = [bool_eval(p, dict(zip(names, (r[j] for j in ins)))) for r in rows]
                except ValueError as ex:
                    errs.append(f"{e['id']}: cannot evaluate {p.strip()!r}: {ex}")
                    continue
                bad = [i for i, (v, r) in enumerate(zip(vals, rows)) if r[col] is not None and v != r[col]]
                if bad:
                    errs.append(f"{e['id']}: {p.strip()!r} disagrees with {tt['id']} column {e['defines']} "
                                f"in rows {bad}")
        for c in (e for e in els if e["type"] == "circuit"):
            errs += circuit_vs_table(c, tt, header, ins, outs, rows)
    for e in els:
        if e["type"] == "math" and e.get("check"):
            errs += numeric_check(e, by_id)
    return errs


def bool_eval(expr, env):
    """Evaluate a Boolean expression: + or | (or), xor / ^ / ⊕, * / & / · and
    juxtaposition (and), ' after a term or ~ ! before it (not), parentheses.
    Variable names are matched longest first, so AB means A and B."""
    names = sorted(env, key=len, reverse=True)
    toks, s, i = [], expr.replace("⊕", " xor ").replace("·", "*"), 0
    while i < len(s):
        ch = s[i]
        if ch.isspace():
            i += 1
        elif s.startswith("xor", i):
            toks.append("^")
            i += 3
        elif ch in "+|^*&()'~!01":
            toks.append({"|": "+", "&": "*", "!": "~"}.get(ch, ch))
            i += 1
        else:
            name = next((n for n in names if s.startswith(n, i)), None)
            if not name:
                raise ValueError(f"unknown name at {s[i:]!r}")
            toks.append(("v", name))
            i += len(name)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def take():
        nonlocal pos
        pos += 1
        return toks[pos - 1]

    def orx():
        v = xorx()
        while peek() == "+":
            take()
            v |= xorx()
        return v

    def xorx():
        v = andx()
        while peek() == "^":
            take()
            v ^= andx()
        return v

    def andx():
        v = unary()
        while peek() == "*" or isinstance(peek(), tuple) or peek() in ("(", "~", "0", "1"):
            if peek() == "*":
                take()
            v &= unary()
        return v

    def unary():
        if peek() == "~":
            take()
            return 1 - unary()
        t = take()
        if t == "(":
            v = orx()
            if take() != ")":
                raise ValueError("unbalanced parentheses")
        elif isinstance(t, tuple):
            v = env[t[1]]
        elif t in ("0", "1"):
            v = int(t)
        else:
            raise ValueError(f"unexpected {t!r}")
        while peek() == "'":
            take()
            v = 1 - v
        return v

    v = orx()
    if pos != len(toks):
        raise ValueError(f"trailing {toks[pos:]}")
    return v


def circuit_vs_table(c, tt, header, ins, outs, rows):
    comps = {x["id"]: x for x in c.get("components", [])}
    pins_in = {x["label"]: k for k, x in comps.items() if x["type"] == "input_pin"}
    pins_out = {x["label"]: k for k, x in comps.items() if x["type"] == "output_pin"}
    in_names, out_names = [header[j] for j in ins], [header[j] for j in outs]
    if set(pins_in) != set(in_names) or not set(pins_out) & set(out_names):
        return []  # this circuit isn't the table's
    src = {w["to"]: w["from"] for w in c.get("wires", [])}
    errs = []
    for r in rows:
        env = {pins_in[n]: r[j] for n, j in zip(in_names, ins)}

        def val(port, depth=0):
            cid, _, p = port.partition(".")
            if depth > 50:
                raise ValueError("combinational loop")
            t = comps[cid]["type"]
            if t == "input_pin":
                return env[cid]
            if t == "output_pin":
                return val(src[f"{cid}.a"], depth + 1)
            return GATE_FN[t](*(val(src[f"{cid}.{q}"], depth + 1) for q in GATE_PORTS[t]))
        for n in set(pins_out) & set(out_names):
            j = header.index(n)
            try:
                v = val(f"{pins_out[n]}.a")
            except (KeyError, ValueError) as ex:
                return [f"{c['id']}: cannot simulate: {ex}"]
            if r[j] is not None and v != r[j]:
                errs.append(f"{c['id']}: output {n} is {v} for inputs {[r[j] for j in ins]}, table says {r[j]}")
    return errs[:3]


def numeric_check(e, by_id):
    """`check: {vars: {name: number or expression}, rel_tol}`: every side of
    the plain-text form must evaluate to the same number. A continuation also
    includes the previous line's last side, and inherits its vars. A side
    that is just the defined name is skipped unless vars give it a value."""
    chain, cur = [], e
    while cur is not None:
        chain.append(cur)
        cur = by_id.get(cur.get("continues"))
    vars_ = {}
    for line in reversed(chain):
        vars_.update((line.get("check") or {}).get("vars", {}))
    # One dict as the globals, so lambdas in vars see the other vars.
    ns = {k: getattr(cmath, k) for k in ("sqrt", "exp", "log", "sin", "cos", "tan", "pi")}
    ns.update(abs=abs, j=1j, __builtins__={})
    try:
        for k, v in vars_.items():
            ns[k] = eval(v, ns) if isinstance(v, str) else v
    except Exception as ex:  # noqa: BLE001
        return [f"{e['id']}: cannot evaluate check vars: {ex}"]
    text = e["text"]
    if e.get("continues"):
        text = by_id[e["continues"]]["text"].split("=")[-1] + text
    sides = [s.strip() for s in text.split("=") if s.strip()]
    sides = [s for s in sides if s != e.get("defines") or s in vars_]
    try:
        vals = [complex(eval(s.replace("^", "**"), ns)) for s in sides]
    except Exception as ex:  # noqa: BLE001
        return [f"{e['id']}: cannot evaluate {text!r}: {ex}"]
    if len(vals) < 2:
        return [f"{e['id']}: check needs at least two sides to compare"]
    tol = e["check"].get("rel_tol", 1e-6)
    ref = vals[0]
    bad = [s for s, v in zip(sides, vals) if abs(v - ref) > tol * max(abs(ref), 1e-30)]
    return [f"{e['id']}: {bad} differ from {sides[0]!r} ({ref:.6g})"] if bad else []


def content_warnings(content):
    """Things that are valid but probably not what the writer meant: content
    that looks checkable but that no check covers."""
    warns = []
    els = content.get("elements", [])
    cols = {h for e in els if e.get("type") == "table" and e.get("kind") == "truth_table"
            for j, h in enumerate(e.get("header") or []) if j in e.get("output_columns", [])}
    ins = [set(e["header"][j] for j in e.get("input_columns", [])) for e in els
           if e.get("type") == "table" and e.get("kind") == "truth_table" and e.get("header")]
    for e in els:
        if e.get("type") == "math" and not e.get("check") and e.get("defines") not in cols:
            warns.append(f"{e['id']}: not checked (no numeric check, and defines no truth-table column)")
        if e.get("type") == "circuit":
            pins = {c.get("label") for c in e.get("components", []) if c.get("type") == "input_pin"}
            if not any(pins == i for i in ins):
                warns.append(f"{e['id']}: not simulated (input pin labels match no truth table's inputs)")
    return warns


# ------------------------------------------------------------------ layout

def check_layout(layout, content=None):
    errs = []
    items, order = layout["items"], layout["writing_order"]
    page = layout["page"]
    missing = set(items) - set(order)
    extra = [leaf for leaf in order if leaf not in items]
    dup = {leaf for leaf in order if order.count(leaf) > 1}
    if missing:
        errs.append(f"items not in writing_order: {sorted(missing)[:5]}")
    if extra or dup:
        errs.append(f"writing_order has unknown or repeated ids: {(extra + sorted(dup))[:5]}")
    writers = {w["id"] for w in layout["writers"]}
    for leaf, it in items.items():
        if it.get("writer") not in writers:
            errs.append(f"{leaf}: unknown writer {it.get('writer')!r}")
        x0, y0, x1, y1 = lc.item_bbox(it)
        if x0 < 0 or y0 < 0 or x1 > page["width"] or y1 > page["height"]:
            errs.append(f"{leaf}: off the page ({x0:.0f}, {y0:.0f}, {x1:.0f}, {y1:.0f})")
    for w in layout["writers"]:
        a = w.get("start_after")
        if a and (a not in items or items[a]["writer"] == w["id"]):
            errs.append(f"{w['id']}: start_after must be another writer's item")
        if a:
            first = next(i for i, leaf in enumerate(order) if items[leaf]["writer"] == w["id"])
            if order.index(a) > first:
                errs.append(f"{w['id']}: starts before its start_after item is written")
    # Text and math boxes must not overlap each other.
    texts = [(leaf, it) for leaf, it in items.items() if it["kind"] in ("text", "math")]
    for (la, a), (lb, b) in itertools.combinations(texts, 2):
        if lc.overlap(a["bbox"], b["bbox"], -0.5):
            errs.append(f"text boxes overlap: {la}, {lb}")
    pos = {leaf: i for i, leaf in enumerate(order)}
    for tg in layout.get("targets", []):
        ids = [tg["after"]["id"]] + [t["id"] for t in tg["target"]]
        if any(i not in items for i in ids) or any(pos[t["id"]] <= pos[tg["after"]["id"]] for t in tg["target"]):
            errs.append(f"target {tg['id']}: ids unknown or not after its context")
    return errs


# ------------------------------------------------------------------ ink

def densify(pts, step=0.5):
    out = [tuple(pts[0][:2])]
    for a, b in zip(pts, pts[1:]):
        n = max(1, math.ceil(math.dist(a[:2], b[:2]) / step))
        out += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(1, n + 1)]
    return out


def check_ink(groups, traces, page, tag="", clean=False):
    """groups: [(item id, item, [(label, [trace ids])])]; traces: {id: [(x, y, t)]}."""
    errs = []
    cell = 1.0
    clearance, min_cells = (CLEARANCE_CLEAN, MIN_CELLS_CLEAN) if clean else (CLEARANCE_NOISY, MIN_CELLS_NOISY)
    r = math.ceil(clearance / cell)
    occ = {}   # cell -> set of group indices with ink there
    for gi, (_, _, units) in enumerate(groups):
        for _, refs in units:
            for t in refs:
                pts = traces[t]
                for x, y in (densify(pts) if len(pts) > 1 else [p[:2] for p in pts]):
                    occ.setdefault((math.floor(x / cell), math.floor(y / cell)), set()).add(gi)
                    if not (-2 <= x <= page["width"] + 2 and -2 <= y <= page["height"] + 2):
                        errs.append(f"{tag}{groups[gi][0]}: ink off the page at ({x:.0f}, {y:.0f})")
    allowed = set()
    ids = [g[0] for g in groups]
    for gi, (_, item, _) in enumerate(groups):
        tgt = item.get("target")
        if item.get("role", "").startswith("annotation_") and tgt:
            allowed |= {(gi, gj) for gj, (_, it2, _) in enumerate(groups) if annotates(tgt, ids[gj], it2)}
        if item.get("role") == "underline":  # a title or heading underline and its text
            allowed |= {(gi, gj) for gj, (_, it2, _) in enumerate(groups) if it2.get("element") == item.get("element")}
    hits = {}
    for gi, (_, item, _) in enumerate(groups):
        if item["kind"] not in ("text", "math"):
            continue
        mine = [c for c, s in occ.items() if gi in s]
        for cx, cy in mine:
            near = set()
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        near |= occ.get((cx + dx, cy + dy), set())
            for gj in near - {gi}:
                if (gj, gi) in allowed or (gi, gj) in allowed:
                    continue
                key = (min(gi, gj), max(gi, gj))
                hits[key] = hits.get(key, 0) + 1
    for (a, b), n in sorted(hits.items(), key=lambda kv: -kv[1]):
        if n >= min_cells:
            errs.append(f"{tag}ink of {ids[a]} touches {ids[b]} ({n} cells)")
    return errs


def annotates(ref, leaf, item):
    """True if an annotation of `ref` (element, leaf, list item or table row
    <table>.r<i>) covers this item."""
    m = re.fullmatch(r"(.+)\.r(\d+)", ref)
    return (leaf in (ref, f"{ref}.marker") or item.get("element") == ref or item.get("line_of") == ref
            or (m is not None and item.get("element") == m.group(1) and str(item.get("row")) == m.group(2)))


def check_ink_clean(layout, fonts=FONT_POOL):
    """Render the layout clean in each font and check the ink."""
    import render_inkml
    errs = []
    for f in fonts:
        res = render_inkml.render(None, [f], 0.0, 0, layout=layout, write=False, quiet=True, device=False)
        errs += check_ink(res["groups"], res["traces"], layout["page"], tag=f"[{f}] ", clean=True)
    return errs


def read_inkml_groups(path):
    """(groups, traces, page) from an InkML file written by render_inkml.py."""
    ns = "{http://www.w3.org/2003/InkML}"
    xid = "{http://www.w3.org/XML/1998/namespace}id"
    root = ET.parse(path).getroot()
    traces = {}
    for tr in root.iter(f"{ns}trace"):
        traces[tr.get(xid)] = [tuple(float(v) for v in p.split()) for p in tr.text.split(",")]
    notes = {a.get("type"): a.text for a in root.findall(f"{ns}annotation")}
    groups = []
    for g in root.find(f"{ns}traceGroup").findall(f"{ns}traceGroup"):
        ann = {a.get("type"): a.text for a in g.findall(f"{ns}annotation")}
        units = []
        for u in g.findall(f"{ns}traceGroup"):
            units.append(({a.get("type"): a.text for a in u.findall(f"{ns}annotation")},
                          [v.get("traceDataRef").lstrip("#") for v in u.findall(f"{ns}traceView")]))
        item = {"kind": ann.get("kind"), "role": ann.get("role", ""), "element": ann.get("element")}
        groups.append((ann["element_id"], item, units))
    page = {"width": float(notes["page_width_mm"]), "height": float(notes["page_height_mm"])}
    return groups, traces, page


def check_inkml(path, layout=None):
    groups, traces, page = read_inkml_groups(path)
    if layout:  # restore annotation targets, which the InkML doesn't carry
        groups = [(eid, {**item, **{k: v for k, v in layout["items"].get(eid, {}).items()
                                    if k in ("target", "role", "element", "row", "line_of")}}, units)
                  for eid, item, units in groups]
    return check_ink(groups, traces, page)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("page_dir")
    ap.add_argument("--layout")
    ap.add_argument("--inkml", nargs="*")
    a = ap.parse_args()
    content = json.load(open(os.path.join(a.page_dir, "content.json")))
    layout = json.load(open(a.layout or os.path.join(a.page_dir, "layout.json")))
    for w in content_warnings(content):
        print(f"warning: {w}")
    report = {"content": check_content(content), "layout": check_layout(layout, content)}
    if a.inkml:
        for p in a.inkml:
            report[os.path.basename(p)] = check_inkml(p, layout)
    else:
        report["ink (clean, all pool fonts)"] = check_ink_clean(layout)
    bad = False
    for k, errs in report.items():
        print(f"{k}: {'OK' if not errs else f'{len(errs)} problem(s)'}")
        for e in errs[:20]:
            print(f"  - {e}")
        bad |= bool(errs)
    sys.exit(1 if bad else 0)
