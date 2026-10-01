"""Checks a page's content.json (truth table, math, circuit) and layout.json (placement, overlaps, wiring)."""
import json, itertools, os, sys
# Page directory; defaults to artifacts/page_0001 at the repo root.
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "artifacts", "page_0001")
C = json.load(open(f"{D}/content.json")); L = json.load(open(f"{D}/layout.json"))
E = {e["id"]: e for e in C["elements"]}
errs = []

# ---- truth table vs reference function
tt = E["tt_1"]
hdr = [h["text"] for h in tt["header"]]
got = [[int(c["text"]) for c in r["cells"]] for r in tt["rows"]]
exp = [[a, b, c, (a + b + c) % 2, int(a + b + c >= 2)]
       for a, b, c in itertools.product((0, 1), repeat=3)]
if got != exp: errs.append("truth table mismatch vs arithmetic sum")
print("truth table:", "OK" if got == exp else "FAIL", f"({len(got)} rows, header {hdr})")

# ---- math expressions vs table
def ev(s, a, b, cin):
    s = s.split("=", 1)[1] if s.count("=") else s
    s = s.strip().lstrip("=").replace("Cin", "c").replace("xor", "^")
    # implicit AND: "AB", "A c", "c (A ^ B)"
    toks, out = s.replace("(", " ( ").replace(")", " ) ").replace("+", " | ").replace("^", " ^ ").split(), []
    expanded = []
    for t in toks:
        if t not in "()|^" and len(t) > 1: expanded += list(t)
        else: expanded.append(t)
    for t in expanded:
        if out and out[-1] not in ("(", "|", "^", "&") and t not in (")", "|", "^"):
            out.append("&")
        out.append(t)
    return eval(" ".join(out), {}, {"A": a, "B": b, "c": cin}) & 1
for mid, col in (("math_1", 3), ("math_2", 4), ("math_3", 4)):
    ok = all(ev(E[mid]["text"], *r[:3]) == r[col] for r in got)
    print(f"{mid} ({E[mid]['text']!r}) vs column {hdr[col]}:", "OK" if ok else "FAIL")
    if not ok: errs.append(mid)

# ---- circuit simulation vs table
dia = E["dia_1"]; comps = {c["id"]: c for c in dia["components"]}
drv = {}
for w in dia["wires"]:
    drv[w["to"]] = w["from"]
nets = {}
for w in dia["wires"]:
    nets.setdefault(w["net"], set()).add(w["from"])
for n, s in nets.items():
    if len(s) != 1: errs.append(f"net {n} has multiple drivers {s}")
def val(port, inp):
    cid, p = port.split(".")
    t = comps[cid]["type"]
    if t == "input_pin": return inp[comps[cid]["label"]]
    if t == "output_pin": return val(drv[port], inp)
    a, b = val(drv[f"{cid}.a"], inp), val(drv[f"{cid}.b"], inp)
    return {"xor2": a ^ b, "and2": a & b, "or2": a | b}[t]
ok = all([val("pin_S.a", dict(A=r[0], B=r[1], Cin=r[2])),
          val("pin_Cout.a", dict(A=r[0], B=r[1], Cin=r[2]))] == r[3:] for r in got)
print("circuit simulation vs table:", "OK" if ok else "FAIL")
if not ok: errs.append("circuit")
for c in comps.values():
    for p in c["ports"]:
        if comps[c["id"]]["type"] != "input_pin" and p in ("a", "b") and f"{c['id']}.{p}" not in drv:
            errs.append(f"unconnected input {c['id']}.{p}")

# ---- layout coverage
LE = L["elements"]; Wd, Ht = L["page"]["width"], L["page"]["height"]
ids = set(E)
sub = set()
for h in tt["header"]: sub.add(h["id"])
for r in tt["rows"]: sub |= {c["id"] for c in r["cells"]}
sub |= {r["id"] for r in tt["rules"]}
sub |= {c["id"] for c in dia["components"]} | {w["id"] for w in dia["wires"]} | {l["id"] for l in dia["labels"]}
laid = set(LE) | set(LE["tt_1"]["cells"]) | set(LE["tt_1"]["rules"]) | \
       set(LE["dia_1"]["components"]) | set(LE["dia_1"]["wires"]) | set(LE["dia_1"]["labels"])
if (ids | sub) != laid: errs.append(f"layout id mismatch {(ids|sub) ^ laid}")
wo = L["writing_order"]
expected_wo = (ids - {"tt_1", "dia_1"}) | sub
if len(wo) != len(set(wo)) or set(wo) != expected_wo:
    errs.append(f"writing_order mismatch {set(wo) ^ expected_wo}")
print(f"coverage: {len(laid)} laid-out ids, writing_order {len(wo)} entries, unique={len(wo)==len(set(wo))}")

# ---- writer field everywhere
def walk(o, path):
    if isinstance(o, dict):
        if "bbox" in o or "polyline" in o or "cell_box" in o:
            if o.get("writer") not in {w["id"] for w in L["writers"]}: errs.append(f"no writer at {path}")
        for k, v in o.items(): walk(v, f"{path}.{k}")
walk(LE, "elements")

# ---- bounds and overlaps
boxes = {}
for k in ("title", "note_1", "note_2", "note_3", "math_1", "math_2", "math_3"):
    boxes[k] = LE[k]["bbox"]
boxes["tt_1"] = LE["tt_1"]["bbox"]; boxes["dia_1"] = LE["dia_1"]["bbox"]
def inside(b): return 0 <= b[0] < b[2] <= Wd and 0 <= b[1] < b[3] <= Ht
def ov(a, b, m=0): return a[0] < b[2] + m and b[0] < a[2] + m and a[1] < b[3] + m and b[1] < a[3] + m
allb = dict(boxes)
for k, c in LE["tt_1"]["cells"].items(): allb[k + ":text"] = c["text_bbox"]
for k, c in LE["dia_1"]["components"].items():
    allb[k] = c["bbox"]
    if "label_bbox" in c: allb[k + ":label"] = c["label_bbox"]
for k, c in LE["dia_1"]["labels"].items(): allb[k] = c["bbox"]
for k, b in allb.items():
    if not inside(b): errs.append(f"{k} out of page {b}")
print("all boxes inside page:", all(inside(b) for b in allb.values()))
top = list(boxes.items()); bad = []
for (a, ba), (b, bb) in itertools.combinations(top, 2):
    if ov(ba, bb, 5): bad.append((a, b))   # require >=5mm clearance between blocks
dia_parts = [(k, b) for k, b in allb.items() if k in LE["dia_1"]["components"] or k in LE["dia_1"]["labels"] or k.endswith(":label")]
for (a, ba), (b, bb) in itertools.combinations(dia_parts, 2):
    if ov(ba, bb): bad.append((a, b))
cell_txt = [(k, b) for k, b in allb.items() if k.endswith(":text")]
for (a, ba), (b, bb) in itertools.combinations(cell_txt, 2):
    if ov(ba, bb): bad.append((a, b))
for k, c in LE["tt_1"]["cells"].items():
    t, cb = c["text_bbox"], c["cell_box"]
    if not (cb[0] <= t[0] and t[2] <= cb[2] and cb[1] <= t[1] and t[3] <= cb[3]): bad.append((k, "outside cell"))
for k, b in dia_parts:
    db = LE["dia_1"]["bbox"]
    if not (db[0] <= b[0] and b[2] <= db[2] and db[1] <= b[1] and b[3] <= db[3]): bad.append((k, "outside dia bbox"))
print("unintended overlaps:", bad or "none")
errs += [f"overlap {p}" for p in bad]

# ---- wires: endpoints on ports, orthogonal, no pass through gate bodies, no cross-net overlap
port = lambda ref: LE["dia_1"]["components"][ref.split(".")[0]]["ports"][ref.split(".")[1]]
segs = []
for w in dia["wires"]:
    pl = LE["dia_1"]["wires"][w["id"]]["polyline"]
    if pl[0] != port(w["from"]) or pl[-1] != port(w["to"]): errs.append(f"{w['id']} endpoints off-port")
    for p, q in zip(pl, pl[1:]):
        if p[0] != q[0] and p[1] != q[1]: errs.append(f"{w['id']} non-orthogonal")
        segs.append((w["net"], w["id"], p, q))
        for cid, c in LE["dia_1"]["components"].items():
            if cid in (w["from"].split(".")[0], w["to"].split(".")[0]) and c is not None and LE["dia_1"]["components"][cid].get("ports") and len(c["ports"]) == 1:
                continue  # wire legitimately starts/ends at this pin marker
            b = c["bbox"]
            # sample interior of segment, excluding endpoints
            for t in [i / 20 for i in range(1, 20)]:
                x, y = p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])
                if b[0] < x < b[2] and b[1] < y < b[3]:
                    errs.append(f"{w['id']} passes through {cid}"); break
def collinear_overlap(s, t):
    (p1, q1), (p2, q2) = s, t
    if p1[1] == q1[1] == p2[1] == q2[1]:
        return max(min(p1[0], q1[0]), min(p2[0], q2[0])) < min(max(p1[0], q1[0]), max(p2[0], q2[0]))
    if p1[0] == q1[0] == p2[0] == q2[0]:
        return max(min(p1[1], q1[1]), min(p2[1], q2[1])) < min(max(p1[1], q1[1]), max(p2[1], q2[1]))
    return False
for a, b in itertools.combinations(segs, 2):
    if a[0] != b[0] and collinear_overlap(a[2:], b[2:]):
        errs.append(f"nets {a[0]}/{b[0]} overlap ({a[1]},{b[1]})")
print("wire checks:", "OK" if not [e for e in errs if e.startswith("w_") or e.startswith("nets")] else "FAIL")
print("ERRORS:" if errs else "ALL CHECKS PASSED", *errs, sep="\n  ")
sys.exit(1 if errs else 0)
