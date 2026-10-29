"""Structural completion targets for a laid-out page.

A target says: once the page has been written up to the end of item
`after` (in the layout's writing order), the items in `target` are a
completion the model could suggest. They come from the content's structure,
never from guessing its meaning (see the scope rules in AGENTS.md):

- truth_table_inputs: the input cells of a truth table in binary counting
  order, once at least two rows' worth of inputs are written. `next` is the
  run of input cells the writer writes next, `remaining` all of them.
- truth_table_outputs: output cells, once the expression that defines the
  column has been written on the page. Tier "arithmetic": evaluating it is
  simple arithmetic, which is tentatively in scope (AGENTS.md).
- table_rules: the remaining rules of a table once its first rule is drawn.
- header_sequence: the rest of a table header that counts up (Wk41, Wk42,
  ...), after two of its cells.
- list_marker: the next marker of a numbered list. Speculative: the writer
  may end the list, so the target only says what the marker would be.
- branch_label: the other label of a two-way decision (yes/no), once one is
  written.
- node_outline: a flowchart node's box, when its text was written first.

Targets refer to layout leaf ids, so they hold for any render of the layout;
render_inkml.py resolves them to times and trace ids per render (dropping
any that interleaved writers make invalid).
"""
import re

PAIRS = {"yes": "no", "no": "yes", "y": "n", "n": "y", "true": "false", "false": "true", "t": "f", "f": "t",
         "1": "0", "0": "1", "ok": "fail", "fail": "ok", "pass": "fail"}


def find_targets(content, layout):
    order = layout["writing_order"]
    items = layout["items"]
    pos = {leaf: i for i, leaf in enumerate(order)}
    out = []

    def add(kind, tier, element, after, target, **extra):
        target = [t for t in target if pos[t] > pos[after]]
        if target:
            out.append({"id": f"tgt_{len(out)}", "kind": kind, "tier": tier, "element": element,
                        "after": {"id": after}, "target": [{"id": t} for t in target], **extra})

    blocks = {e["id"]: e for e in content["elements"]}
    for blk in content["elements"]:
        kind = blk["type"]
        leaves = sorted((leaf for leaf, it in items.items() if it.get("element") == blk["id"]), key=pos.get)
        if kind == "table":
            truth_table_targets(blk, leaves, items, pos, order, blocks, add)
            rules = [leaf for leaf in leaves if items[leaf].get("role") == "rule"]
            if len(rules) >= 2:
                first = rules[0]
                run = contiguous(order, pos[first] + 1, lambda leaf: leaf in rules)
                add("table_rules", "structural", blk["id"], first, run)
            header = [leaf for leaf in leaves if items[leaf].get("role") == "header_cell"]
            header.sort(key=lambda leaf: items[leaf]["col"])
            if counts_up([items[leaf]["text"] for leaf in header if items[leaf]["kind"] == "text"]):
                written = sorted(header, key=pos.get)
                for k in range(1, len(written) - 1):
                    add("header_sequence", "structural", blk["id"], written[k], written[k + 1:])
        elif kind == "list" and blk.get("style") == "numbered":
            markers = sorted((leaf for leaf in leaves if items[leaf].get("role") == "list_marker"
                              and items[leaf].get("level") == 0), key=lambda leaf: items[leaf]["index"])
            for k in range(1, len(markers) - 1):
                # After item k's text (its last line), the marker of item k + 1.
                base = markers[k][:-len(".marker")]
                text_leaves = [leaf for leaf in leaves if (leaf == base or leaf.startswith(base + ".l"))
                               and pos[leaf] < pos[markers[k + 1]]]
                if text_leaves:
                    add("list_marker", "structural", blk["id"], max(text_leaves, key=pos.get), [markers[k + 1]],
                        speculative=True)
        elif kind == "flowchart":
            flowchart_targets(blk, leaves, items, pos, add)
    return out


def truth_table_targets(blk, leaves, items, pos, order, blocks, add):
    if blk.get("kind") != "truth_table" or blk.get("row_order") != "binary_counting_msb_first":
        return
    ins = set(blk.get("input_columns", []))
    cells = [leaf for leaf in leaves if items[leaf].get("role") == "cell"]
    inputs = [leaf for leaf in cells if items[leaf]["col"] in ins]
    n_in = len(ins)
    is_input = set(inputs).__contains__
    for i, leaf in enumerate(order[:-1]):
        nxt = order[i + 1]
        if not is_input(nxt) or is_input(leaf):
            continue
        written = sum(1 for c in inputs if pos[c] <= i)
        if written < 2 * n_in:
            continue
        run = contiguous(order, i + 1, is_input)
        rest = [c for c in inputs if pos[c] > i]
        add("truth_table_inputs", "structural", blk["id"], leaf, run, span="next")
        if len(rest) > len(run):
            add("truth_table_inputs", "structural", blk["id"], leaf, rest, span="remaining")
    # Output columns defined by an expression written earlier on the page.
    header = blk.get("header") or []
    for col in blk.get("output_columns", []):
        name = header[col] if col < len(header) else None
        defs = [e["id"] for e in blocks.values() if e["type"] == "math" and name and e.get("defines") == name]
        def_pos = [p for leaf, p in pos.items() if items[leaf].get("element") in defs]
        if not def_pos:
            continue
        col_cells = [c for c in cells if items[c]["col"] == col]
        is_col = set(col_cells).__contains__
        for i, leaf in enumerate(order[:-1]):
            if is_col(order[i + 1]) and not is_col(leaf) and i >= min(def_pos):
                add("truth_table_outputs", "arithmetic", blk["id"], leaf, contiguous(order, i + 1, is_col),
                    column=name)


def flowchart_targets(blk, leaves, items, pos, add):
    fid = blk["id"]
    out_edges = {}
    for k, e in enumerate(blk.get("edges", [])):
        out_edges.setdefault(e["from"], []).append(e.get("id") or f"e{k + 1}")
    by_node = {n["id"]: n for n in blk.get("nodes", [])}
    labels = {e.get("id") or f"e{k + 1}": (e.get("label") or "").strip().lower()
              for k, e in enumerate(blk.get("edges", []))}
    for node, edges in out_edges.items():
        if by_node.get(node, {}).get("shape") != "decision" or len(edges) != 2:
            continue
        a, b = edges
        if PAIRS.get(labels[a]) != labels[b]:
            continue
        la, lb = f"{fid}.{a}.label", f"{fid}.{b}.label"
        if la in pos and lb in pos:
            first, second = sorted((la, lb), key=pos.get)
            add("branch_label", "structural", fid, first, [second])
    for node in by_node:
        box = f"{fid}.{node}.box"
        texts = [leaf for leaf in leaves if leaf == f"{fid}.{node}.text" or leaf.startswith(f"{fid}.{node}.text.")]
        if box in pos and texts and max(pos[t] for t in texts) < pos[box]:
            add("node_outline", "structural", fid, max(texts, key=pos.get), [box])


def contiguous(order, start, pred):
    run = []
    for leaf in order[start:]:
        if not pred(leaf):
            break
        run.append(leaf)
    return run


def counts_up(texts):
    """True if texts are prefix + integer + suffix with a constant step (>= 3 cells)."""
    if len(texts) < 3:
        return False
    parts = [re.fullmatch(r"(\D*)(\d+)(\D*)", t) for t in texts]
    if not all(parts) or len({(m.group(1), m.group(3)) for m in parts}) != 1:
        return False
    nums = [int(m.group(2)) for m in parts]
    steps = {b - a for a, b in zip(nums, nums[1:])}
    return len(steps) == 1 and steps != {0}
