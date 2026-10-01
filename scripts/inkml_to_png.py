"""Rasterize an InkML file to PNG for visual inspection.

No maintained InkML rendering library exists on PyPI, so this is a small
standalone reader (stdlib XML) plus Pillow. It reads <traceFormat> to find the
X, Y and T channels and draws every <trace> as a polyline. It does not handle
InkML's difference-encoded traces ("'" / '"' prefixes) or transforms; it
raises rather than drawing them wrong.

Usage:
  uv run --extra render python scripts/inkml_to_png.py page.inkml [out.png]
      [--px-per-unit 1.5] [--until SECONDS] [--color-groups] [--crop X0 Y0 X1 Y1]

--until draws only ink written up to that time, i.e. an in-progress page.
--color-groups colours each top-level <traceGroup> child differently.
--crop draws only that region (in InkML units), e.g. to zoom in with a higher
--px-per-unit.
"""
import argparse
import colorsys
import os
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw

NS = "{http://www.w3.org/2003/InkML}"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"


def read_inkml(path):
    root = ET.parse(path).getroot()
    fmt = root.find(f".//{NS}traceFormat")
    channels = [c.get("name") for c in fmt.findall(f"{NS}channel")] if fmt is not None else ["X", "Y"]
    ix, iy = channels.index("X"), channels.index("Y")
    it = channels.index("T") if "T" in channels else None

    traces = {}
    for i, tr in enumerate(root.iter(f"{NS}trace")):
        pts = []
        for chunk in (tr.text or "").split(","):
            vals = chunk.split()
            if not vals:
                continue
            if any(v[0] in "'\"!" for v in vals):
                raise NotImplementedError(f"difference-encoded trace {tr.get(XML_ID)}")
            v = [float(x) for x in vals]
            pts.append((v[ix], v[iy], v[it] if it is not None else None))
        traces[tr.get(XML_ID) or f"_{i}"] = pts

    groups = []  # list of trace-id lists, one per top-level group child
    for top in root.findall(f"{NS}traceGroup"):
        for g in top.findall(f"{NS}traceGroup") or [top]:
            groups.append([v.get("traceDataRef").lstrip("#") for v in g.iter(f"{NS}traceView")])

    notes = {a.get("type"): a.text for a in root.findall(f"{NS}annotation")}
    return traces, groups, notes


def render(path, out, px_per_unit, until, color_groups, line_width, crop=None):
    traces, groups, notes = read_inkml(path)
    if crop:
        x0, y0, x1, y1 = crop
    elif "page_width_mm" in notes:
        x0, y0, x1, y1 = 0, 0, float(notes["page_width_mm"]), float(notes["page_height_mm"])
    else:
        xs = [p[0] for t in traces.values() for p in t]
        ys = [p[1] for t in traces.values() for p in t]
        x0, y0, x1, y1 = min(xs) - 10, min(ys) - 10, max(xs) + 10, max(ys) + 10

    color = {tid: (20, 20, 20) for tid in traces}
    if color_groups:
        for i, refs in enumerate(groups):
            r, g, b = colorsys.hsv_to_rgb((i * 0.618) % 1, 0.8, 0.75)
            for tid in refs:
                color[tid] = (int(r * 255), int(g * 255), int(b * 255))

    img = Image.new("RGB", (round((x1 - x0) * px_per_unit), round((y1 - y0) * px_per_unit)), "white")
    draw = ImageDraw.Draw(img)
    drawn = 0
    for tid, pts in traces.items():
        if until is not None:
            pts = [p for p in pts if p[2] is not None and p[2] <= until]
        xy = [((x - x0) * px_per_unit, (y - y0) * px_per_unit) for x, y, _ in pts]
        if len(xy) == 1:
            xy = xy * 2
        if xy:
            draw.line(xy, fill=color[tid], width=line_width, joint="curve")
            drawn += 1
    img.save(out)
    print(f"wrote {out} ({img.width}x{img.height} px, {drawn}/{len(traces)} traces)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inkml")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--px-per-unit", type=float, default=1.5)
    ap.add_argument("--until", type=float, help="only draw ink with T <= this")
    ap.add_argument("--color-groups", action="store_true")
    ap.add_argument("--line-width", type=int, default=3)
    ap.add_argument("--crop", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    a = ap.parse_args()
    render(a.inkml, a.out or os.path.splitext(a.inkml)[0] + ".png",
           a.px_per_unit, a.until, a.color_groups, a.line_width, a.crop)
