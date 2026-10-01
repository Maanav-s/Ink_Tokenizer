"""Layout engine: content.json (v1, semantic, no coordinates) -> layout.json (v1).

The LLM writes only content (docs/content_prompt.md). This module decides
everything about the page, from a layout seed:

- sizes: text heights per role, line spacing, margins, indents, table cell
  padding, diagram spacing;
- placement: blocks are packed in reading order, below the previous block,
  beside it, or in a new column, without overlapping (text widths come from
  fonts.PoolMetrics, so boxes fit every font in the pool);
- presentation: list markers, heading and title underlines, table rule
  styles and cell alignment;
- the writing process: the order items are written in (e.g. tables by row,
  by column, or inputs before outputs), a second writer who works on part
  of the page at the same time, and crossed-out or overwritten mistakes.

Diagrams (flowchart, circuit, schematic) are laid out by diagram_layout.py
and math is typeset by math_typeset.py. Completion targets are added by
targets.py. The same (content, seed) always gives the same layout.

Usage: uv run --extra render python scripts/layout_engine.py page_dir
           [--seed N] [--out FILE] [--no-check]
Without --out, seed 0 writes <page_dir>/layout.json and other seeds write
<page_dir>/layouts/layout_s<N>.json.
"""
import argparse
import json
import math
import os
import random
import re

import layout_common as lc
import targets as targets_mod
from fonts import BASE

CONTENT_SCHEMA = "ink_tokenizer.synthetic_page.content/v1"
LAYOUT_SCHEMA = "ink_tokenizer.synthetic_page.layout/v1"
PAGE_W, PAGE_H = 1200.0, 800.0
TEXT_ROLES = ("title", "heading", "bullet", "subbullet", "note", "caption", "text")


class LayoutError(Exception):
    pass


# ------------------------------------------------------------------ style

def sample_style(rng):
    h = rng.uniform(19, 26)
    return {
        "h": h,
        "title": h * rng.uniform(1.2, 1.45),
        "heading": h * rng.uniform(1.0, 1.15),
        "subbullet": h * rng.uniform(0.82, 0.95),
        "note": h * rng.uniform(0.8, 0.95),
        "table": h * rng.uniform(0.85, 1.0),
        "diagram": h * rng.uniform(0.8, 0.95),
        "math": h * rng.uniform(1.0, 1.12),
        "line_gap": rng.uniform(1.55, 1.85),     # baseline to baseline, units of text height
        "block_gap": rng.uniform(0.9, 1.6),     # units of h
        "col_gap": rng.uniform(1.8, 3.5),       # units of h
        "margin_x": rng.uniform(25, 60), "margin_y": rng.uniform(18, 40),
        "indent": rng.uniform(1.1, 1.8),        # units of h per list level
        "dash": rng.choice(["-", "-", "-", "*"]),
        "numbered": rng.choice(["{}.", "{}.", "{})"]),
        "title_underline": rng.random() < 0.3,
        "title_center": rng.random() < 0.15,
        "heading_underline": rng.random() < 0.45,
        "table_style": rng.choice(["grid", "open", "open", "header"]),
        "table_rules": rng.choice(["after_header", "after_header", "first", "last"]),
        "table_left_align": rng.random() < 0.3,
        "row_h": rng.uniform(1.65, 1.95),       # row height, units of table text height
        "cell_pad": rng.uniform(0.6, 0.9),      # units of table text height
        "side_by_side": rng.uniform(0.2, 0.7),  # preference for placing a block beside the previous one
        "fill": rng.uniform(0.55, 0.95),        # how full a column gets before starting another
        "multi_writer": rng.random() < 0.35,
        "correction_rate": rng.choice([0.0, 0.03, 0.06, 0.1]),
        "scale": 1.0,
    }


def role_height(st, role):
    return st.get(role, st["h"]) if role in ("title", "heading", "subbullet", "note") else st["h"]


# ------------------------------------------------------------------ blocks

class Block:
    """A content block laid out in local coordinates (origin top-left)."""

    def __init__(self, bid, kind):
        self.id, self.kind = bid, kind
        self.ids = {bid}  # content ids inside (several for a heading kept with its block)
        self.items, self.order = {}, []
        self.w = self.h = 0.0
        self.meta = {}

    def add(self, leaf, item):
        if leaf in self.items:
            raise LayoutError(f"duplicate leaf id {leaf!r}")
        item.setdefault("element", self.id)
        self.items[leaf] = item
        self.order.append(leaf)
        return item

    def finish(self):
        """Shift items so the block's ink box starts at (0, 0)."""
        if not self.items:
            raise LayoutError(f"block {self.id!r} has nothing to write")
        x0, y0, x1, y1 = lc.union_bbox(lc.item_bbox(it) for it in self.items.values())
        for it in self.items.values():
            lc.translate(it, -x0, -y0)
        self.meta = {k: (v - x0 if k.endswith("_x") else v) for k, v in self.meta.items()}
        self.w, self.h = x1 - x0, y1 - y0
        return self


def keep_with_next(head, body, gap):
    """Stack a heading on top of the block it introduces, as one block."""
    b = Block(head.id, body.kind)
    b.ids = head.ids | body.ids
    for leaf in head.order:
        b.add(leaf, head.items[leaf])
    for leaf in body.order:
        b.add(leaf, lc.translate(body.items[leaf], 0, head.h + gap))
    b.meta = dict(body.meta)
    b.finish()
    return b


def wrap(text, h, max_w, metrics):
    """Split text at spaces into lines no wider than max_w (a single word may be wider)."""
    if metrics.width(text, h) <= max_w:
        return [text]
    lines, cur = [], ""
    for word in text.split(" "):
        cand = f"{cur} {word}" if cur else word
        if cur and metrics.width(cand, h) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = cand
    return lines + [cur]


def text_lines(block, leaf, text, role, x, y_top, h, st, metrics, max_w, bold=False, hang=0.0):
    """Add one text item per wrapped line; returns the bottom y. Continuation
    lines get leaf ids <leaf>.l<k> and are indented by `hang`."""
    lines = wrap(text, h, max_w, metrics) if max_w else [text]
    pitch = st["line_gap"] * h
    y = y_top + h
    for k, line in enumerate(lines):
        lid = leaf if len(lines) == 1 else f"{leaf}.l{k}"
        extra = {"role": role}
        if bold:
            extra["bold"] = True
        if len(lines) > 1:
            extra["line_of"] = leaf
        block.add(lid, lc.text_item(line, x + (hang if k else 0), y, h, metrics, **extra))
        y += pitch
    return y - pitch


def underline(block, leaf, target, rng):
    x0, _, x1, y1 = target["bbox"]
    h = target["text_height"]
    y = y1 + 0.15 * h
    block.add(f"{leaf}.underline", {"kind": "polyline", "role": "underline",
                                    "points": [[x0 - rng.uniform(0, 0.2) * h, y],
                                               [x1 + rng.uniform(0, 0.3) * h, y + rng.uniform(-0.05, 0.05) * h]]})


def build_text(blk, st, rng, metrics, max_w):
    role = blk.get("role", "text")
    h = role_height(st, role)
    b = Block(blk["id"], "text")
    text_lines(b, blk["id"], blk["text"], role, 0, 0, h, st, metrics, max_w, bold=role == "title")
    if (role == "title" and st["title_underline"]) or (role == "heading" and st["heading_underline"]):
        underline(b, blk["id"], b.items[b.order[-1]], rng)
    return b.finish()


def build_list(blk, st, rng, metrics, max_w):
    """A (nested) bullet or numbered list with an optional heading."""
    b = Block(blk["id"], "list")
    y = 0.0
    if blk.get("heading"):
        h = st["heading"]
        leaf = f"{blk['id']}.heading"
        y = text_lines(b, leaf, blk["heading"], "heading", 0, 0, h, st, metrics, max_w)
        if st["heading_underline"]:
            underline(b, leaf, b.items[b.order[-1]], rng)
        y += 0.55 * h
    style = blk.get("style", "dash")

    def walk(items, level, prefix):
        nonlocal y
        h = st["h"] if level == 0 else st["subbullet"]
        for k, it in enumerate(items):
            leaf = it.get("id") or f"{prefix}.{k + 1}"
            if style == "numbered" and level == 0:
                marker = st["numbered"].format(k + 1)
            elif style == "plain":
                marker = None
            else:
                marker = st["dash"]
            x = level * st["indent"] * st["h"]
            y += (st["line_gap"] - 1) * h
            if marker:
                mw = max(metrics.width(st["numbered"].format(n + 1), h) for n in range(len(items))) \
                    if style == "numbered" and level == 0 else metrics.width(marker, h)
                b.add(f"{leaf}.marker", lc.text_item(marker, x, y + h, h, metrics, role="list_marker",
                                                     list=blk["id"], level=level, index=k))
                x += mw + 0.35 * h
            role = "bullet" if level == 0 else "subbullet"
            y = text_lines(b, leaf, it["text"], role, x, y, h, st, metrics,
                           max_w - x if max_w else None)
            if it.get("items"):
                walk(it["items"], level + 1, leaf)

    walk(blk["items"], 0, blk["id"])
    return b.finish()


def build_table(blk, st, rng, metrics, max_w):
    """A table or truth table: header row, data rows and drawn rules."""
    tid = blk["id"]
    th = st["table"]
    header, rows = blk.get("header") or [], blk["rows"]
    ncol = max(len(r) for r in [header] + rows)
    grid = ([header] if header else []) + rows
    pad = st["cell_pad"] * th
    col_w = [max(lc.label_width(r[j], th, metrics) if j < len(r) and r[j] else 0 for r in grid) + 2 * pad
             for j in range(ncol)]
    col_w = [max(w, 1.5 * th) for w in col_w]
    long_col = [any(j < len(r) and len(r[j] or "") > 6 for r in rows) for j in range(ncol)]
    row_h = st["row_h"] * th * (1.15 if st["table_style"] == "grid" else 1.0)
    xs = [0.0]
    for w in col_w:
        xs.append(xs[-1] + w)
    W, H = xs[-1], row_h * len(grid)
    truth = blk.get("kind") == "truth_table"

    b = Block(tid, "table")
    cells = []  # (leaf, item, row, col); row "header" or data index
    for gi, r in enumerate(grid):
        row = "header" if header and gi == 0 else gi - (1 if header else 0)
        y_mid = (gi + 0.5) * row_h
        for j in range(ncol):
            s = r[j] if j < len(r) else ""
            if not s:
                continue
            left = (long_col[j] or st["table_left_align"]) and not truth
            x = xs[j] + pad if left else (xs[j] + xs[j + 1]) / 2
            leaf = f"{tid}.h{j}" if row == "header" else f"{tid}.r{row}c{j}"
            # Caps centred a little above the row's middle, leaving room for descenders.
            item = lc.label_item(s, x, y_mid + 0.5 * th, th, metrics, align="left" if left else "center",
                                 role="header_cell" if row == "header" else "cell", row=row, col=j)
            cells.append((leaf, item, row, j))

    # Rules, by style. Lines overhang the grid a little, as drawn by hand.
    over = lambda: rng.uniform(0.0, 0.35) * th
    hdr_y = row_h if header else None
    rules = []
    style = st["table_style"]
    if style == "grid":
        rules.append((f"{tid}.border", {"kind": "rect", "bbox": [0, 0, W, H]}))
        for gi in range(1, len(grid)):
            rules.append((f"{tid}.rule_h{gi}", {"kind": "polyline", "points": [[0, gi * row_h], [W, gi * row_h]]}))
        for j in range(1, ncol):
            rules.append((f"{tid}.rule_c{j - 1}", {"kind": "polyline", "points": [[xs[j], 0], [xs[j], H]]}))
    else:
        if hdr_y:
            rules.append((f"{tid}.rule_h1", {"kind": "polyline", "points": [[-over(), hdr_y], [W + over(), hdr_y]]}))
        seps = range(1, ncol) if style == "open" else []
        if truth:
            io = max(blk.get("input_columns", [0])) + 1
            seps = sorted(set(seps) | {io}) if io < ncol else seps
        for j in seps:
            rules.append((f"{tid}.rule_c{j - 1}",
                          {"kind": "polyline", "points": [[xs[j], -over()], [xs[j], H + over()]]}))
    for _, it in rules:
        it["role"] = "rule"

    # Writing order.
    hdr = [c for c in cells if c[2] == "header"]
    data = [c for c in cells if c[2] != "header"]
    if truth and rng.random() < 0.5:
        ins = set(blk.get("input_columns", []))
        data = ([c for c in data if c[3] in ins]
                + sorted([c for c in data if c[3] not in ins], key=lambda c: (c[3], c[2])))
        fill = "inputs_then_outputs"
    elif rng.random() < 0.2:
        data = sorted(data, key=lambda c: (c[3], c[2]))
        fill = "columns"
    else:
        fill = "rows"
    when = st["table_rules"]
    seq = []
    if when == "first":
        seq += rules
    seq += [(leaf, it) for leaf, it, _, _ in hdr]
    if when == "after_header":
        seq += rules
    seq += [(leaf, it) for leaf, it, _, _ in data]
    if when == "last":
        seq += rules
    for leaf, it in seq:
        b.add(leaf, it)
    b.meta = {"fill": fill, "rules": when, "style": style}
    return b.finish()


def build_math(blk, st, rng, metrics, max_w):
    import math_typeset
    h = st["math"]
    opts = {"frac_order": "bar_num_den" if rng.random() < 0.25 else "num_bar_den",
            "radical_bar": "after" if rng.random() < 0.3 else "joined"}
    b = Block(blk["id"], "math")
    item = lc.math_item(blk["latex"], 0, 0, h, metrics, options=opts, role="math")
    b.add(blk["id"], item)
    # x of the first top-level relation, to align continuation lines on it.
    eq = math_typeset.typeset(blk["latex"], h, metrics, opts)
    rel = next((a["x"] for a in eq.atoms if a["type"] == "char" and a["ch"] in "=≈≤≥<>"), None)
    b.finish()
    if rel is not None:
        b.meta["rel_x"] = rel
    return b


def build_diagram(blk, st, rng, metrics, max_w):
    import diagram_layout
    fn = {"flowchart": diagram_layout.layout_flowchart, "circuit": diagram_layout.layout_circuit,
          "schematic": diagram_layout.layout_schematic}[blk["type"]]
    res = fn(blk, rng, {"text_height": st["diagram"], "max_width": max_w,
                        "max_height": PAGE_H - 2 * st["margin_y"]})
    b = Block(blk["id"], blk["type"])
    for leaf in res["order"]:
        b.add(leaf, res["items"][leaf])
    return b.finish()


BUILDERS = {"text": build_text, "list": build_list, "table": build_table, "math": build_math,
            "flowchart": build_diagram, "circuit": build_diagram, "schematic": build_diagram}


# ------------------------------------------------------------------ packing

def pack(blocks, content_blocks, st, rng):
    """Place blocks in reading order. Returns {block id: (x, y)}."""
    mx, my = st["margin_x"], st["margin_y"]
    h = st["h"]
    gap, cgap = st["block_gap"] * h, st["col_gap"] * h
    placed = {}  # id -> (x, y, w, h)

    def free(x, y, w, hh):
        if x < mx * 0.5 or y < my * 0.5 or x + w > PAGE_W - mx * 0.5 or y + hh > PAGE_H - my * 0.5:
            return False
        return not any(lc.overlap([x, y, x + w, y + hh], [px, py, px + pw, py + ph], gap * 0.6)
                       for px, py, pw, ph in placed.values())

    by_id = {c["id"]: c for c in content_blocks}
    top = my
    prev = None
    col_x = mx
    # Spread the page over a few columns rather than one tall one: a column
    # stops (softly) at col_limit, chosen from the total height of the blocks.
    body = [b for b in blocks if by_id[b.id].get("role") != "title"]
    title_h = sum(b.h + gap * 1.2 for b in blocks if b not in body)
    avail = PAGE_H - 2 * my - title_h
    total = sum(b.h + gap for b in body)
    n_cols = max(1, math.ceil(total / (avail * st["fill"])))
    col_limit = max(max((b.h for b in body), default=0), total / n_cols * rng.uniform(1.0, 1.15))
    col_top = None
    remaining, acc = {}, 0.0  # height of each block and everything after it
    for b in reversed(body):
        acc += b.h + gap
        remaining[b.id] = acc
    for b in blocks:
        c = by_id[b.id]
        if c.get("role") == "title" and prev is None:
            x = (PAGE_W - b.w) / 2 if st["title_center"] else mx
            if not free(x, my, b.w, b.h):
                raise LayoutError("title does not fit")
            placed[b.id] = (x, my, b.w, b.h)
            top = my + b.h + gap * 1.2
            prev = b.id
            continue
        cands = []  # (starts a new column, x, y), in order of preference
        if prev is not None:
            px, py, pw, ph = placed[prev]
            cont = next((ob.id for ob in blocks if c.get("continues") in ob.ids), None)
            if cont in placed and "rel_x" in b.meta and "rel_x" in blocks_meta(blocks, cont):
                # A continuation line: its relation sits under the previous line's.
                cx = placed[cont][0] + blocks_meta(blocks, cont)["rel_x"] - b.meta["rel_x"]
                cands.append((False, cx, py + ph + 0.8 * gap))
            below = (False, col_x, py + ph + gap)
            beside = (True, px + pw + cgap, py)
            if ph > 4 * h and b.h < ph and rng.random() < st["side_by_side"]:
                cands += [beside, below]
            elif (col_top is not None and py + ph + gap + b.h - col_top > col_limit
                  and remaining[b.id] > 0.3 * col_limit):
                pass  # this column is full enough: start a new one (added below)
            else:
                cands += [below, beside]
        else:
            cands.append((False, col_x, top))
        # A new column starts right of the current one, as high as it is free
        # (a wide line at the top may push it down).
        right = max([px + pw for px, py, pw, ph in placed.values() if px + pw > col_x and py + ph > top]
                    or [mx - cgap])
        y_new = next((y for y in frange(top, PAGE_H - b.h, 5.0) if free(right + cgap, y, b.w, b.h)), top)
        cands.append((True, right + cgap, y_new))
        if prev is not None:
            cands.append(below)  # fall back to an over-full column
        pos = next(((new, x, y) for new, x, y in cands if free(x, y, b.w, b.h)), None)
        if pos is None:  # scan for any free spot, top to bottom, left to right
            step = 10.0
            pos = next(((True, x, y) for y in frange(top, PAGE_H - b.h, step)
                        for x in frange(mx, PAGE_W - b.w, step) if free(x, y, b.w, b.h)), None)
        if pos is None:
            raise LayoutError(f"block {b.id!r} ({b.w:.0f} x {b.h:.0f} mm) does not fit")
        new_col, x, y = pos
        if new_col or col_top is None:
            col_x = x  # later blocks go under this one
            col_top = y
        placed[b.id] = (x, y, b.w, b.h)
        prev = b.id
    return {k: (v[0], v[1]) for k, v in placed.items()}


def blocks_meta(blocks, bid):
    return next(b.meta for b in blocks if bid in b.ids)


def frange(a, b, step):
    x = a
    while x <= b:
        yield x
        x += step


# ------------------------------------------------------------------ writers

def assign_writers(layout, blocks, pos, st, rng):
    """Optionally hand the right-hand part of the page to a second writer who
    starts while the first is still writing, so their strokes interleave."""
    order = layout["writing_order"]
    for it in layout["items"].values():
        it["writer"] = "writer_0"
    layout["writers"] = [{"id": "writer_0"}]
    if not st["multi_writer"] or len(blocks) < 4:
        return
    right = [b for b in blocks[1:] if pos[b.id][0] > PAGE_W * 0.42]
    left = [b for b in blocks if b not in right]
    if not right or len(left) < 2:
        return
    w1 = {leaf for b in right for leaf in b.order}
    w0_order = [leaf for leaf in order if leaf not in w1]
    w1_order = [leaf for leaf in order if leaf in w1]
    k = max(0, min(len(w0_order) - 1, int(len(w0_order) * rng.uniform(0.15, 0.6))))
    after = w0_order[k]
    for leaf in w1:
        layout["items"][leaf]["writer"] = "writer_1"
    layout["writers"].append({"id": "writer_1", "start_after": after})
    layout["writing_order"] = w0_order[:k + 1] + w1_order + w0_order[k + 1:]


# ------------------------------------------------------------------ corrections

WORD = re.compile(r"[A-Za-z]{3,}")


def add_corrections(layout, st, rng, metrics):
    """Crossed-out and overwritten mistakes. A mistake is struck through or
    scribbled out and rewritten to its right (the item's box grows, if there
    is room), or a wrong digit is overwritten in place."""
    rate = st["correction_rate"]
    if not rate:
        return
    items = layout["items"]
    boxes = {k: lc.item_bbox(v) for k, v in items.items()}
    for leaf, it in items.items():
        if it["kind"] != "text" or it.get("role") in ("title", "list_marker") or rng.random() >= rate:
            continue
        text, h = it["text"], it["text_height"]
        digits = [i for i, ch in enumerate(text) if ch.isdigit()]
        if digits and rng.random() < 0.5:
            i = rng.choice(digits)
            wrong = rng.choice([d for d in "0123456789" if d != text[i]])
            style = "overwrite" if rng.random() < 0.6 else "strike"
        else:
            words = list(WORD.finditer(text))
            if not words:
                continue
            m = rng.choice(words)
            i, word = m.start(), m.group()
            wrong = typo(word, rng)
            style = rng.choice(["strike", "strike", "scribble"])
        corr = {"index": i, "wrong": wrong, "style": style}
        if style != "overwrite":
            extra = metrics.width(wrong, h) + 0.25 * h
            x0, y0, x1, y1 = it["bbox"]
            grown = [x0, y0, x1 + extra, y1]
            if grown[2] > PAGE_W - 10 or any(lc.overlap(grown, bb, 2.0) for k, bb in boxes.items()
                                             if k != leaf and not lc.overlap(it["bbox"], bb, 2.0)):
                continue
            it["bbox"] = grown
            boxes[leaf] = grown
        it["corrections"] = [corr]


def typo(word, rng):
    """A plausible misspelling or false start of word."""
    w = list(word)
    kind = rng.choice(["swap", "drop", "double", "prefix"])
    if kind == "swap" and len(w) > 3:
        i = rng.randrange(1, len(w) - 2)
        w[i], w[i + 1] = w[i + 1], w[i]
    elif kind == "drop":
        del w[rng.randrange(1, len(w))]
    elif kind == "double":
        i = rng.randrange(len(w))
        w.insert(i, w[i])
    else:
        return word[:rng.randint(1, max(1, len(word) - 2))]
    s = "".join(w)
    return s if s != word else word[:-1]


# ------------------------------------------------------------------ annotations

def resolve_box(layout, ref):
    """Box of a block id, a leaf id, or a table row id (<table>.r<i>)."""
    items = layout["items"]
    m = re.fullmatch(r"(.+)\.r(\d+)", ref)
    sel = [it for leaf, it in items.items()
           if leaf in (ref, f"{ref}.marker") or it.get("element") == ref or ref == it.get("line_of")]
    if not sel and m:
        sel = [it for it in items.values() if it.get("element") == m.group(1) and it.get("row") == int(m.group(2))]
    if not sel:
        raise LayoutError(f"annotation refers to unknown id {ref!r}")
    return lc.union_bbox(lc.item_bbox(it) for it in sel)


def add_annotations(layout, annotations, st, rng):
    """Underlines, boxes and arrows the content asks for, drawn right after
    their target or at the end of the page."""
    items, order = layout["items"], layout["writing_order"]
    for a in annotations:
        aid, kind = a["id"], a["kind"]
        if kind in ("underline", "box", "circle"):
            x0, y0, x1, y1 = resolve_box(layout, a["target"])
            others = [lc.item_bbox(it) for leaf, it in items.items() if not leaf_matches(it, leaf, a["target"])]
            # As much padding as the neighbours leave room for.
            for pad in [st["h"] * f for f in (0.35, 0.28, 0.22, 0.16, 0.11)]:
                if kind == "underline":
                    y = y1 + pad
                    item = {"kind": "polyline", "points": [[x0, y], [x1, y]]}
                elif kind == "box":
                    item = {"kind": "rect", "bbox": [x0 - pad, y0 - pad, x1 + pad, y1 + pad],
                            "corner_radius": rng.choice([0, 0, 0.3 * st["h"]])}
                else:
                    item = {"kind": "ellipse", "bbox": [x0 - 2.5 * pad, y0 - 1.5 * pad, x1 + 2.5 * pad, y1 + 1.5 * pad]}
                if not any(lc.overlap(lc.item_bbox(item), bb, 0.2 * st["h"] + 3.0) for bb in others):
                    break
            else:
                continue
            last = max(i for i, leaf in enumerate(order) if leaf_matches(items[leaf], leaf, a["target"]))
            at = last + 1 if rng.random() < 0.7 else len(order)
        elif kind == "arrow":
            sa, sb = resolve_box(layout, a["from"]), resolve_box(layout, a["to"])
            pts = arrow_between(sa, sb, st["h"])
            others = [lc.item_bbox(it) for leaf, it in items.items()
                      if not leaf_matches(it, leaf, a["from"]) and not leaf_matches(it, leaf, a["to"])]
            if pts is None or math.dist(pts[0], pts[-1]) > 0.35 * PAGE_W or any(segment_hits(p, q, bb) for p, q in zip(pts, pts[1:]) for bb in others):
                continue  # no clean path; leave the arrow out
            item = {"kind": "arrow", "points": pts, "head": 0.4 * st["h"]}
            at = len(order)
        else:
            raise LayoutError(f"unknown annotation kind {kind!r}")
        item.update(role=f"annotation_{kind}", element=aid)
        if kind != "arrow":
            item["target"] = a["target"]
        items[aid] = item
        order.insert(at, aid)


def leaf_matches(item, leaf, ref):
    m = re.fullmatch(r"(.+)\.r(\d+)", ref)
    return (leaf in (ref, f"{ref}.marker") or item.get("element") == ref or item.get("line_of") == ref
            or (m is not None and item.get("element") == m.group(1) and item.get("row") == int(m.group(2))))


def arrow_between(a, b, h):
    """A straight arrow between facing sides of boxes a and b, or None."""
    gap = 0.3 * h
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    if b[0] > a[2]:
        return [[a[2] + gap, ay], [b[0] - gap, by]]
    if b[2] < a[0]:
        return [[a[0] - gap, ay], [b[2] + gap, by]]
    if b[1] > a[3]:
        return [[ax, a[3] + gap], [bx, b[1] - gap]]
    if b[3] < a[1]:
        return [[ax, a[1] - gap], [bx, b[3] + gap]]
    return None


def segment_hits(p, q, box, n=40):
    return any(box[0] <= p[0] + (q[0] - p[0]) * k / n <= box[2] and box[1] <= p[1] + (q[1] - p[1]) * k / n <= box[3]
               for k in range(n + 1))


# ------------------------------------------------------------------ page

def layout_page(content, seed, metrics=None, check=None):
    """Lay out a content.json (v1) dict. `check(layout) -> list of problems` may
    reject a layout; the engine then retries with smaller text and new draws.
    Raises LayoutError if nothing works."""
    if content.get("schema") != CONTENT_SCHEMA:
        raise LayoutError(f"expected schema {CONTENT_SCHEMA}, got {content.get('schema')!r}")
    metrics = metrics or lc.pool_metrics()
    problems = []
    for attempt in range(8):
        rng = random.Random(f"{content['page_id']}:{seed}:{attempt}")
        st = sample_style(rng)
        scale = 0.92 ** max(0, attempt - 2)  # new draws first, then smaller text
        for k in ("h", "title", "heading", "subbullet", "note", "table", "diagram", "math"):
            st[k] *= scale
        try:
            layout = build(content, seed, st, rng, metrics)
        except LayoutError as e:
            problems = [str(e)]
            continue
        problems = check(layout) if check else []
        if not problems:
            layout["layout_attempt"] = attempt
            return layout
    raise LayoutError(f"no valid layout for {content['page_id']} seed {seed}: " + "; ".join(problems[:5]))


def build(content, seed, st, rng, metrics):
    blocks_c = [e for e in content["elements"] if e["type"] in BUILDERS]
    annotations = [e for e in content["elements"] if e["type"] == "annotation"]
    unknown = [e["type"] for e in content["elements"] if e["type"] not in BUILDERS and e["type"] != "annotation"]
    if unknown:
        raise LayoutError(f"unknown element types {unknown}")
    max_w = PAGE_W - 2 * st["margin_x"]
    blocks = []
    for k, c in enumerate(blocks_c):
        b = BUILDERS[c["type"]](c, st, rng, metrics, max_w)
        prev = blocks_c[k - 1] if k else None
        lead_in = prev and prev["type"] == "text" and (prev.get("role") == "heading" or
                                                       prev.get("role") != "title" and prev["text"].rstrip().endswith(":"))
        if lead_in and blocks[-1].id == prev["id"]:
            b = keep_with_next(blocks.pop(), b, 0.5 * st["h"])
        blocks.append(b)
    pos = pack(blocks, blocks_c, st, rng)
    layout = {"schema": LAYOUT_SCHEMA, "page_id": content["page_id"], "layout_seed": seed, "unit": "mm",
              "page": {"width": PAGE_W, "height": PAGE_H, "origin": "top-left", "y_axis": "down"},
              "style": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items()},
              "writers": [], "items": {}, "writing_order": []}
    for b in blocks:
        x, y = pos[b.id]
        for leaf in b.order:
            layout["items"][leaf] = lc.translate(b.items[leaf], x, y)
            layout["writing_order"].append(leaf)
        if b.meta:
            layout["style"].setdefault("blocks", {})[b.id] = {k: v for k, v in b.meta.items() if not k.endswith("_x")}
    add_annotations(layout, annotations, st, rng)
    add_corrections(layout, st, rng, metrics)
    assign_writers(layout, blocks, pos, st, rng)
    for it in layout["items"].values():
        lc.round_item(it)
    layout["targets"] = targets_mod.find_targets(content, layout)
    return layout


def default_path(page_dir, seed):
    if seed == 0:
        return os.path.join(page_dir, "layout.json")
    return os.path.join(page_dir, "layouts", f"layout_s{seed}.json")


def write_layout(page_dir, seed, out=None, check=True, quiet=False):
    import validate_page
    content = json.load(open(os.path.join(page_dir, "content.json")))
    errs = validate_page.check_content(content)
    if errs:
        raise LayoutError("content errors: " + "; ".join(errs))
    checker = (lambda lay: validate_page.check_layout(lay, content) + validate_page.check_ink_clean(lay)) \
        if check else None
    layout = layout_page(content, seed, check=checker)
    out = out or default_path(page_dir, seed)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(layout, f, indent=1, ensure_ascii=False)
    if not quiet:
        print(f"wrote {out}: {len(layout['items'])} items, {len(layout['writers'])} writer(s), "
              f"{len(layout['targets'])} targets, attempt {layout['layout_attempt']}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("page_dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--no-check", action="store_true", help="skip the layout and clean-ink checks")
    a = ap.parse_args()
    write_layout(a.page_dir, a.seed, a.out, check=not a.no_check)
