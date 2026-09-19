#!/usr/bin/env python3
"""Compare Canny symbol proposals + OCR on FC Offline pages.

The report does not call a proposal a diagram prediction. It only asks whether
geometry finds a labelled closed shape at IoU >= 0.3, and reports unmatched
proposals separately so high recall cannot hide a bad visible-suggestion rate.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ink_tokenizer.edge_ocr import (
    Box,
    contained_fraction,
    detect_symbol_proposals,
    greedy_matches,
    make_rapidocr,
    ocr_with_rapidocr,
    ocr_with_tesseract,
)


SHAPE_CATEGORIES = {"data", "decision", "process", "terminator"}


@dataclass(frozen=True)
class ReferencePage:
    shapes: list[Box]
    text: list[Box]


def parse_annotation(path: Path) -> ReferencePage:
    shapes: list[Box] = []
    text: list[Box] = []
    for symbol in ET.parse(path).getroot().findall("./symbols/symbol"):
        bounds = symbol.find("bounds")
        if bounds is None:
            continue
        category = symbol.attrib["name"]
        left, top = float(bounds.attrib["x"]), float(bounds.attrib["y"])
        box = Box(
            left,
            top,
            left + float(bounds.attrib["width"]),
            top + float(bounds.attrib["height"]),
            label=category,
        )
        if category in SHAPE_CATEGORIES:
            shapes.append(box)
        elif category == "text":
            text.append(Box(box.left, box.top, box.right, box.bottom, label=(symbol.findtext("textMeaning") or "").strip()))
    return ReferencePage(shapes, text)


def normalise(value: str) -> str:
    """Compare labels without claiming to understand their mathematical syntax."""
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def evaluate(proposals: list[Box], reference: ReferencePage, words: list[Box]) -> dict[str, object]:
    shape_matches = greedy_matches(proposals, reference.shapes)
    matched_proposals = {candidate for candidate, _, _ in shape_matches}
    matched_labels: set[int] = set()
    matched_words: set[int] = set()
    ocr_matches: list[dict[str, str]] = []
    for word_index, word in enumerate(words):
        value = normalise(word.label)
        if not value:
            continue
        for label_index, label in enumerate(reference.text):
            if contained_fraction(word, label) >= 0.35 and value in normalise(label.label):
                matched_labels.add(label_index)
                matched_words.add(word_index)
                ocr_matches.append({"word": word.label, "reference": label.label})
                break
    expected, total = len(reference.shapes), len(proposals)
    return {
        "shape_expected": expected,
        "shape_proposals": total,
        "shape_matched": len(shape_matches),
        "shape_recall": len(shape_matches) / expected if expected else None,
        "false_proposals": total - len(matched_proposals),
        "false_proposal_rate": (total - len(matched_proposals)) / total if total else None,
        "ocr_expected_labels": len(reference.text),
        "ocr_words": len(words),
        "ocr_matched_labels": len(matched_labels),
        "ocr_label_recall": len(matched_labels) / len(reference.text) if reference.text else None,
        "false_ocr_words": len(words) - len(matched_words),
        "false_ocr_word_rate": (len(words) - len(matched_words)) / len(words) if words else None,
        "shape_matches": [
            {"proposal": candidate, "reference": expected_shape, "iou": iou}
            for candidate, expected_shape, iou in shape_matches
        ],
        "ocr_matches": ocr_matches,
    }


def draw_box(image: np.ndarray, box: Box, colour: tuple[int, int, int], text: str, thickness: int) -> None:
    left, top, right, bottom = (round(value) for value in (box.left, box.top, box.right, box.bottom))
    cv2.rectangle(image, (left, top), (right, bottom), colour, thickness)
    cv2.putText(image, text, (left, max(16, top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)


def write_overlay(source: Path, target: Path, proposals: list[Box], words: list[Box], reference: ReferencePage) -> None:
    with Image.open(source) as image:
        canvas = np.asarray(image.convert("RGB")).copy()
    for box in reference.shapes:
        draw_box(canvas, box, (128, 128, 128), f"ref: {box.label}", 2)
    for box in proposals:
        draw_box(canvas, box, (32, 115, 255), "edge", 3)
    for word in words:
        draw_box(canvas, word, (0, 160, 80), f"{word.label} {word.score:.0f}", 2)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"could not write {target}")


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def total_metrics(pages: list[dict[str, object]]) -> dict[str, object]:
    keys = (
        "shape_expected", "shape_proposals", "shape_matched", "false_proposals",
        "ocr_expected_labels", "ocr_words", "ocr_matched_labels", "false_ocr_words",
    )
    summary = {key: sum(int(page["metrics"][key]) for page in pages) for key in keys}
    summary["shape_recall"] = ratio(summary["shape_matched"], summary["shape_expected"])
    summary["false_proposal_rate"] = ratio(summary["false_proposals"], summary["shape_proposals"])
    summary["ocr_label_recall"] = ratio(summary["ocr_matched_labels"], summary["ocr_expected_labels"])
    summary["false_ocr_word_rate"] = ratio(summary["false_ocr_words"], summary["ocr_words"])
    return summary


def rel_url(source: Path, output: Path) -> str:
    return Path(os.path.relpath(source.resolve(), output.resolve())).as_posix()


def write_html(target: Path, report: dict[str, object]) -> None:
    summary = report["summary"]
    pages = report["pages"]
    assert isinstance(summary, dict) and isinstance(pages, list)
    def percentage(value: object) -> str:
        return "n/a" if value is None else f"{float(value):.1%}"
    samples = []
    for page in pages:
        assert isinstance(page, dict)
        metrics = page["metrics"]
        assert isinstance(metrics, dict)
        samples.append(
            "<section class=sample>"
            f"<h2>{html.escape(str(page['id']))}</h2>"
            f"<p>Shapes: {metrics['shape_matched']}/{metrics['shape_expected']} matched; {metrics['false_proposals']}/{metrics['shape_proposals']} unmatched proposals. OCR labels: {metrics['ocr_matched_labels']}/{metrics['ocr_expected_labels']} matched; {metrics['false_ocr_words']}/{metrics['ocr_words']} unmatched words.</p>"
            "<div class=grid>"
            f"<figure><figcaption>Scanned input</figcaption><img loading=lazy src={html.escape(str(page['image_url']), quote=True)}></figure>"
            f"<figure><figcaption>Baseline overlay</figcaption><img loading=lazy src={html.escape(str(page['overlay_url']), quote=True)}></figure>"
            f"<figure><figcaption>Dataset registration/reference</figcaption><img loading=lazy src={html.escape(str(page['registration_url']), quote=True)}></figure>"
            "</div></section>"
        )
    target.write_text(
        "<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>FC Offline — edge + OCR baseline</title><style>"
        ":root{color-scheme:light dark;font-family:system-ui,sans-serif}body{max-width:1500px;margin:0 auto;padding:1.5rem}.note{max-width:80rem;color:#777}.sample{border-top:1px solid #8886;margin-top:2rem;padding-top:1rem}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem}figure{margin:0}figcaption{font-weight:600;margin-bottom:.4rem}img{background:white;width:100%;height:420px;object-fit:contain;border:1px solid #8886}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style></head><body>"
        "<h1>FC Offline — conventional edge + OCR baseline</h1>"
        "<p class=note>Blue boxes are Canny-contour symbol proposals; green boxes are OCR words; gray boxes are FC annotations. This measures layout detection and label recognition, not flowchart semantics or next-step suggestions. Unmatched detections are reported separately because that is the dangerous false-suggestion mode.</p>"
        f"<p><strong>Shape recall:</strong> {percentage(summary['shape_recall'])} ({summary['shape_matched']}/{summary['shape_expected']}); <strong>false-proposal rate:</strong> {percentage(summary['false_proposal_rate'])} ({summary['false_proposals']}/{summary['shape_proposals']}); <strong>OCR label recall:</strong> {percentage(summary['ocr_label_recall'])} ({summary['ocr_matched_labels']}/{summary['ocr_expected_labels']}); <strong>false OCR-word rate:</strong> {percentage(summary['false_ocr_word_rate'])} ({summary['false_ocr_words']}/{summary['ocr_words']}).</p>"
        + "".join(samples) + "</body></html>", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="FC_1.0_offline_extension directory")
    parser.add_argument("output", type=Path, help="report directory")
    parser.add_argument("--diagram", default="fc_001")
    parser.add_argument("--writers", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--ocr", choices=("rapidocr", "tesseract"), default="rapidocr")
    parser.add_argument("--tesseract", default="tesseract")
    parser.add_argument("--psm", type=int, default=11)
    parser.add_argument("--min-contour-score", type=float, default=0.55, help="minimum contour rectangularity; use 0 for the raw detector")
    parser.add_argument("--skip-ocr", action="store_true", help="run the symbol-only half when no OCR engine is available")
    args = parser.parse_args()
    database = args.dataset / "DB"
    if not database.is_dir():
        raise SystemExit(f"expected {database}; pass FC_1.0_offline_extension")
    if not args.skip_ocr and args.ocr == "tesseract" and shutil.which(args.tesseract) is None:
        raise SystemExit(f"could not find {args.tesseract!r}; install Tesseract or pass --tesseract PATH")
    rapidocr = make_rapidocr() if not args.skip_ocr and args.ocr == "rapidocr" else None
    pages: list[dict[str, object]] = []
    for writer in args.writers:
        sample_id = f"writer{writer:03d}_{args.diagram}"
        image_path = database / "images" / f"{sample_id}.png"
        registration_path = database / "registration" / f"{sample_id}.png"
        reference = parse_annotation(database / "annotation" / f"{sample_id}.xml")
        with Image.open(image_path) as image:
            proposals = detect_symbol_proposals(image)
        proposals = [box for box in proposals if box.score >= args.min_contour_score]
        if args.skip_ocr:
            words = []
        elif args.ocr == "rapidocr":
            assert rapidocr is not None
            words = ocr_with_rapidocr(image_path, rapidocr)
        else:
            words = ocr_with_tesseract(image_path, command=args.tesseract, psm=args.psm)
        overlay = args.output / f"{sample_id}.png"
        write_overlay(image_path, overlay, proposals, words, reference)
        metrics = evaluate(proposals, reference, words)
        pages.append({
            "id": sample_id,
            "image_url": rel_url(image_path, args.output),
            "registration_url": rel_url(registration_path, args.output),
            "overlay_url": overlay.name,
            "proposals": [box.to_dict() for box in proposals],
            "ocr_words": [box.to_dict() for box in words],
            "metrics": metrics,
        })
        print(f"{sample_id}: {len(proposals)} proposals, {len(words)} OCR words, {metrics['shape_matched']} shape matches")
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"pages": pages, "summary": total_metrics(pages)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_html(args.output / "index.html", report)
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.output / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
