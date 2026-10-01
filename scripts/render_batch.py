"""Lay out and render pages across layout seeds, fonts and noise seeds.

For each page directory (with a content.json):
1. each layout seed gets a layout from layout_engine.py (seed 0 is
   <page>/layout.json, others <page>/layouts/layout_s<N>.json), generated
   if missing or with --relayout;
2. each (layout, font, noise seed) is rendered with render_inkml.py. On a
   page with two writers, the second writer gets the next font in the pool;
3. every render's ink is checked (validate_page.check_ink) before it is
   written. Renders whose text touches other ink are rejected and listed in
   the manifest instead.

Output: <page>/renders/L<layout>_<fonts>_noise<L>_seed<N>.inkml (+ .png,
+ .targets.json when the layout has completion targets), and
<page>/renders/manifest.json.

Usage: uv run --extra render python scripts/render_batch.py [page_dir ...]
           [--layouts 0,1,2] [--fonts F1,F2,...] [--seeds 0] [--noise 0.6]
           [--fonts-per-layout K] [--relayout] [--clean] [--no-png]
With no page directories, every artifacts/page_*/ with a content.json is used.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkml_to_png  # noqa: E402
import layout_engine  # noqa: E402
import render_inkml  # noqa: E402
import validate_page  # noqa: E402
from fonts import FONT_POOL  # noqa: E402

# Noise level for batch renders. 0.6 is close to the first, milder noise
# version; 0.75 and 1.0 were judged too aggressive.
DEFAULT_NOISE = 0.6


def writer_fonts(font, n_writers, pool):
    """The given font for the first writer, then the next fonts in the pool."""
    i = pool.index(font) if font in pool else 0
    return [font] + [pool[(i + k) % len(pool)] for k in range(1, n_writers)]


def run(pages, layouts, fonts, seeds, noise, relayout=False, png=True, clean=False, per_layout=None):
    for page in pages:
        rdir = os.path.join(page, "renders")
        os.makedirs(rdir, exist_ok=True)
        if clean:  # drop earlier renders, so the directory matches the manifest
            for f in os.listdir(rdir):
                if f.endswith((".inkml", ".png", ".targets.json")) or f == "manifest.json":
                    os.remove(os.path.join(rdir, f))
        manifest = {"page": os.path.basename(os.path.normpath(page)), "noise": noise, "renders": [], "rejected": []}
        for ls in layouts:
            path = layout_engine.default_path(page, ls)
            if relayout or not os.path.exists(path):
                layout_engine.write_layout(page, ls)
            layout = json.load(open(path))
            # Either every font for every layout, or `per_layout` of them,
            # rotating through the list so each layout gets different ones.
            use = fonts if not per_layout else [fonts[(ls * per_layout + k) % len(fonts)] for k in range(per_layout)]
            for font in use:
                wf = writer_fonts(font, len(layout["writers"]), fonts)
                for seed in seeds:
                    name = f"L{ls}_{'+'.join(wf)}_noise{noise:g}_seed{seed}.inkml"
                    out = os.path.join(page, "renders", name)
                    res = render_inkml.render(page, wf, noise, seed, out=out, layout=layout)
                    problems = validate_page.check_ink(res["groups"], res["traces"], layout["page"])
                    if problems:
                        for p in (out, out[:-6] + ".targets.json"):
                            if os.path.exists(p):
                                os.remove(p)
                        manifest["rejected"].append({"file": name, "problems": problems[:10]})
                        print(f"rejected {name}: {problems[0]}")
                        continue
                    if png:
                        inkml_to_png.render(out, out[:-6] + ".png", 1.5, None, False, 3)
                    manifest["renders"].append({"file": name, "layout_seed": ls, "fonts": wf, "noise_seed": seed,
                                                "writers": len(layout["writers"])})
        with open(os.path.join(page, "renders", "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
        print(f"{page}: {len(manifest['renders'])} renders, {len(manifest['rejected'])} rejected")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pages", nargs="*")
    ap.add_argument("--layouts", default="0")
    ap.add_argument("--fonts", default=",".join(FONT_POOL))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--noise", type=float, default=DEFAULT_NOISE)
    ap.add_argument("--relayout", action="store_true", help="regenerate layouts that already exist")
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--clean", action="store_true", help="delete existing renders in renders/ first")
    ap.add_argument("--fonts-per-layout", type=int, help="render each layout in only this many fonts (rotating)")
    a = ap.parse_args()
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts")
    pages = a.pages or sorted(os.path.dirname(p) for p in glob.glob(os.path.join(root, "page_*", "content.json")))
    run(pages, [int(s) for s in a.layouts.split(",")], a.fonts.split(","), [int(s) for s in a.seeds.split(",")],
        a.noise, a.relayout, not a.no_png, a.clean, a.fonts_per_layout)
