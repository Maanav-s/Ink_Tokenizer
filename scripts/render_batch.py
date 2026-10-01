"""Render pages in several fonts and seeds, InkML plus PNG previews.

Usage: uv run --extra render python scripts/render_batch.py [page_dir ...]
           [--fonts F1,F2,...] [--seeds 0,1] [--noise 0.75] [--no-png]

With no page directories, every artifacts/page_*/ that has a layout.json is
rendered. Output goes to <page_dir>/renders/<font>_noise<L>_seed<N>.inkml
(and .png).
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkml_to_png  # noqa: E402
import render_inkml  # noqa: E402

DEFAULT_FONTS = ["futural", "cursive", "EMSReadability", "EMSReadabilityItalic",
                 "EMSTech", "EMSNixish", "EMSAllure", "EMSFelix"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pages", nargs="*")
    ap.add_argument("--fonts", default=",".join(DEFAULT_FONTS))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--noise", type=float, default=0.75)
    ap.add_argument("--no-png", action="store_true")
    a = ap.parse_args()
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts")
    pages = a.pages or sorted(os.path.dirname(p) for p in glob.glob(os.path.join(root, "page_*", "layout.json")))
    for page in pages:
        os.makedirs(os.path.join(page, "renders"), exist_ok=True)
        for font in a.fonts.split(","):
            for seed in (int(s) for s in a.seeds.split(",")):
                out = os.path.join(page, "renders", f"{font}_noise{a.noise:g}_seed{seed}.inkml")
                render_inkml.render(page, font, a.noise, seed, out)
                if not a.no_png:
                    inkml_to_png.render(out, out[:-6] + ".png", 1.5, None, False, 3)
