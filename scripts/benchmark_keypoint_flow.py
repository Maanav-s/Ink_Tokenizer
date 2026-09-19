#!/usr/bin/env python3
"""Order image-derived pseudo-strokes by OCR-anchored flowchart layout.

The 3×3 Laplacian kernel produces edge contours and salient contour vertices.
Those are *not* recovered pen strokes: raster input does not contain pen-up
events or timing. This report instead tests whether OCR and conservative
flowchart layout can make their synthetic ordering useful and auditable.
"""

from __future__ import annotations

import argparse
import colorsys
import html
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ink_tokenizer.edge_ocr import Box, contained_fraction, detect_symbol_proposals, greedy_matches, make_rapidocr, ocr_with_rapidocr
from ink_tokenizer.keypoints import FlowNode, fragment_owner, infer_downward_links, laplacian_fragments, order_flow_nodes


SHAPE_CATEGORIES = {"data", "decision", "process", "terminator"}


def parse_reference(path: Path) -> tuple[list[tuple[str, Box]], list[tuple[str, str]]]:
    root = ET.parse(path).getroot()
    shapes: list[tuple[str, Box]] = []
    shape_ids: set[str] = set()
    for symbol in root.findall("./symbols/symbol"):
        if symbol.attrib["name"] not in SHAPE_CATEGORIES:
            continue
        bounds = symbol.find("bounds")
        if bounds is None:
            continue
        left = float(bounds.attrib["x"])
        top = float(bounds.attrib["y"])
        box = Box(left, top, left + float(bounds.attrib["width"]), top + float(bounds.attrib["height"]))
        symbol_id = symbol.attrib["id"]
        shapes.append((symbol_id, box))
        shape_ids.add(symbol_id)
    constraints: list[tuple[str, str]] = []
    for relation in root.findall("./relations/relation[@name='arrow_connection']"):
        views = relation.findall("symbolView")
        if len(views) < 3:
            continue
        source = views[1].attrib.get("symbolDataRef")
        target = views[2].attrib.get("symbolDataRef")
        if source in shape_ids and target in shape_ids:
            constraints.append((source, target))
    return shapes, constraints


def label_nodes(regions: list[Box], words: list[Box]) -> list[FlowNode]:
    nodes: list[FlowNode] = []
    for region in regions:
        labels = [
            word.label
            for word in words
            if contained_fraction(word, region) >= 0.50
        ]
        nodes.append(FlowNode(region, " ".join(labels)))
    return order_flow_nodes(nodes)


def order_fragments(fragments, nodes: list[FlowNode]) -> list[dict[str, object]]:
    assigned = []
    for fragment in fragments:
        owner = fragment_owner(fragment, nodes)
        assigned.append((owner, fragment))
    assigned.sort(
        key=lambda item: (
            item[0] if item[0] is not None else len(nodes),
            item[1].box.top,
            item[1].box.left,
        )
    )
    return [
        {
            "order": order,
            "node": owner,
            "box": fragment.box.to_dict(),
            "keypoints": fragment.keypoints,
        }
        for order, (owner, fragment) in enumerate(assigned)
    ]


def evaluate_node_order(nodes: list[FlowNode], references: list[tuple[str, Box]], constraints: list[tuple[str, str]]) -> dict[str, int | float | None]:
    matches = greedy_matches([node.box for node in nodes], [box for _, box in references])
    ranks = {references[reference_index][0]: node_index for node_index, reference_index, _ in matches}
    comparable = [(source, target) for source, target in constraints if source in ranks and target in ranks]
    ordered = sum(ranks[source] < ranks[target] for source, target in comparable)
    return {
        "reference_shapes": len(references),
        "detected_nodes": len(nodes),
        "matched_nodes": len(matches),
        "flow_constraints": len(constraints),
        "comparable_constraints": len(comparable),
        "forward_constraints": ordered,
        "constraint_order_accuracy": ordered / len(comparable) if comparable else None,
    }


def draw_overlay(source: Path, target: Path, nodes: list[FlowNode], links: list[tuple[int, int]], pseudo_strokes: list[dict[str, object]]) -> None:
    with Image.open(source) as image:
        canvas = np.asarray(image.convert("RGB")).copy()
    total = max(1, len(pseudo_strokes))
    for stroke in pseudo_strokes:
        points = np.asarray(stroke["keypoints"], dtype=np.int32).reshape(-1, 1, 2)
        red, green, blue = colorsys.hsv_to_rgb(float(stroke["order"]) / total, 0.85, 0.95)
        cv2.polylines(canvas, [points], False, (round(blue * 255), round(green * 255), round(red * 255)), 1, cv2.LINE_AA)
    for source_index, target_index in links:
        source_node, target_node = nodes[source_index], nodes[target_index]
        start = (round((source_node.box.left + source_node.box.right) / 2), round(source_node.box.bottom))
        end = (round((target_node.box.left + target_node.box.right) / 2), round(target_node.box.top))
        cv2.arrowedLine(canvas, start, end, (180, 60, 180), 2, cv2.LINE_AA, tipLength=0.03)
    for rank, node in enumerate(nodes, start=1):
        left, top, right, bottom = (round(value) for value in (node.box.left, node.box.top, node.box.right, node.box.bottom))
        cv2.rectangle(canvas, (left, top), (right, bottom), (180, 60, 180), 2)
        caption = f"{rank}: {node.label}" if node.label else str(rank)
        cv2.putText(canvas, caption[:32], (left, max(18, top - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 60, 180), 2, cv2.LINE_AA)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"could not write {target}")


def relative_url(source: Path, output: Path) -> str:
    return Path(os.path.relpath(source.resolve(), output.resolve())).as_posix()


def write_html(target: Path, report: dict[str, object]) -> None:
    pages = report["pages"]
    summary = report["summary"]
    assert isinstance(pages, list) and isinstance(summary, dict)
    samples = []
    for page in pages:
        assert isinstance(page, dict)
        metrics = page["metrics"]
        assert isinstance(metrics, dict)
        samples.append(
            "<section class=sample>"
            f"<h2>{html.escape(str(page['id']))}</h2>"
            f"<p>{page['fragments']} pseudo-strokes; {page['keypoints']} keypoints; {metrics['matched_nodes']}/{metrics['reference_shapes']} shape nodes matched; {metrics['forward_constraints']}/{metrics['comparable_constraints']} comparable flow constraints forward.</p>"
            "<div class=grid>"
            f"<figure><figcaption>Scanned input</figcaption><img loading=lazy src={html.escape(str(page['image_url']), quote=True)}></figure>"
            f"<figure><figcaption>Ordered pseudo-strokes</figcaption><img loading=lazy src={html.escape(str(page['overlay_url']), quote=True)}></figure>"
            f"<figure><figcaption>Dataset registration/reference</figcaption><img loading=lazy src={html.escape(str(page['registration_url']), quote=True)}></figure>"
            "</div></section>"
        )
    accuracy = summary["constraint_order_accuracy"]
    accuracy_text = "n/a" if accuracy is None else f"{float(accuracy):.1%}"
    target.write_text(
        "<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>FC Offline — simple-kernel pseudo-stroke order</title><style>"
        ":root{color-scheme:light dark;font-family:system-ui,sans-serif}body{max-width:1500px;margin:0 auto;padding:1.5rem}.note{max-width:80rem;color:#777}.sample{border-top:1px solid #8886;margin-top:2rem;padding-top:1rem}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem}figure{margin:0}figcaption{font-weight:600;margin-bottom:.4rem}img{background:white;width:100%;height:420px;object-fit:contain;border:1px solid #8886}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style></head><body>"
        "<h1>FC Offline — simple-kernel pseudo-stroke ordering</h1>"
        "<p class=note>A direct 3×3 Laplacian edge response supplies the colored contour keypoints. Purple boxes and arrows are a conservative OCR-labelled, top-to-bottom flow-layout heuristic. This does not reconstruct the writer’s pen order: offline scans contain no timestamps or pen-up events. Branches and loops remain ambiguous rather than being fabricated as a single execution trace.</p>"
        f"<p><strong>Flow constraints in layout order:</strong> {accuracy_text} ({summary['forward_constraints']}/{summary['comparable_constraints']}); <strong>pseudo-strokes:</strong> {summary['fragments']}; <strong>keypoints:</strong> {summary['keypoints']}.</p>"
        + "".join(samples) + "</body></html>", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="FC_1.0_offline_extension directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--diagram", default="fc_001")
    parser.add_argument("--writers", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--min-contour-score", type=float, default=0.30, help="layout-only contour score; not a visible-suggestion threshold")
    args = parser.parse_args()
    database = args.dataset / "DB"
    if not database.is_dir():
        raise SystemExit(f"expected {database}; pass FC_1.0_offline_extension")
    engine = make_rapidocr()
    pages: list[dict[str, object]] = []
    for writer in args.writers:
        sample_id = f"writer{writer:03d}_{args.diagram}"
        image_path = database / "images" / f"{sample_id}.png"
        registration_path = database / "registration" / f"{sample_id}.png"
        references, constraints = parse_reference(database / "annotation" / f"{sample_id}.xml")
        words = ocr_with_rapidocr(image_path, engine)
        with Image.open(image_path) as image:
            regions = [box for box in detect_symbol_proposals(image) if box.score >= args.min_contour_score]
            fragments, threshold = laplacian_fragments(image)
        nodes = label_nodes(regions, words)
        links = infer_downward_links(nodes)
        pseudo_strokes = order_fragments(fragments, nodes)
        metrics = evaluate_node_order(nodes, references, constraints)
        overlay = args.output / f"{sample_id}.png"
        draw_overlay(image_path, overlay, nodes, links, pseudo_strokes)
        pages.append({
            "id": sample_id,
            "image_url": relative_url(image_path, args.output),
            "registration_url": relative_url(registration_path, args.output),
            "overlay_url": overlay.name,
            "kernel": "[[0,-1,0],[-1,4,-1],[0,-1,0]]",
            "otsu_threshold": threshold,
            "fragments": len(pseudo_strokes),
            "keypoints": sum(len(stroke["keypoints"]) for stroke in pseudo_strokes),
            "nodes": [{"rank": rank, "box": node.box.to_dict(), "label": node.label} for rank, node in enumerate(nodes)],
            "downward_links": links,
            "pseudo_strokes": pseudo_strokes,
            "metrics": metrics,
        })
        print(f"{sample_id}: {len(pseudo_strokes)} pseudo-strokes, {len(nodes)} OCR-labelled nodes, {metrics['forward_constraints']}/{metrics['comparable_constraints']} flow constraints")
    count_keys = ("reference_shapes", "detected_nodes", "matched_nodes", "flow_constraints", "comparable_constraints", "forward_constraints")
    summary = {key: sum(int(page["metrics"][key]) for page in pages) for key in count_keys}
    summary["constraint_order_accuracy"] = summary["forward_constraints"] / summary["comparable_constraints"] if summary["comparable_constraints"] else None
    summary["fragments"] = sum(int(page["fragments"]) for page in pages)
    summary["keypoints"] = sum(int(page["keypoints"]) for page in pages)
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"pages": pages, "summary": summary}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_html(args.output / "index.html", report)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
