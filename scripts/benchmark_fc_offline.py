#!/usr/bin/env python3
"""Run InkSight on annotation-guided crops from the FC Offline dataset.

The InkSight model consumes one 224x224 crop at a time.  This script uses the
dataset's symbol boxes, suppresses text boxes already enclosed by shapes, and
maps every recovered stroke back into full-page coordinates.
"""

from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ink_tokenizer import LocalBackend
from ink_tokenizer.ink import Ink
from ink_tokenizer.preprocess import crop_and_pad


@dataclass(frozen=True)
class Region:
    page_index: int
    symbol_id: str
    category: str
    box: tuple[float, float, float, float]
    crop_box: tuple[float, float, float, float]


def parse_regions(
    annotation: Path,
    *,
    page_index: int,
    image_size: tuple[int, int],
    margin_ratio: float,
    min_margin: float,
) -> list[Region]:
    root = ET.parse(annotation).getroot()
    standalone_text_ids = {
        views[0].attrib["symbolDataRef"]
        for relation in root.findall("./relations/relation[@name='arrow_label']")
        if len(views := relation.findall("symbolView")) >= 1
    }

    image_width, image_height = image_size
    regions: list[Region] = []
    for symbol in root.findall("./symbols/symbol"):
        symbol_id = symbol.attrib["id"]
        category = symbol.attrib["name"]
        if category == "text" and symbol_id not in standalone_text_ids:
            continue
        bounds = symbol.find("bounds")
        if bounds is None:
            continue
        x = float(bounds.attrib["x"])
        y = float(bounds.attrib["y"])
        width = float(bounds.attrib["width"])
        height = float(bounds.attrib["height"])
        margin = max(min_margin, max(width, height) * margin_ratio)
        crop_box = (
            max(0.0, x - margin),
            max(0.0, y - margin),
            min(float(image_width), x + width + margin),
            min(float(image_height), y + height + margin),
        )
        regions.append(
            Region(
                page_index=page_index,
                symbol_id=symbol_id,
                category=category,
                box=(x, y, x + width, y + height),
                crop_box=crop_box,
            )
        )
    return regions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="FC_1.0_offline_extension path")
    parser.add_argument("output", type=Path)
    parser.add_argument("--diagram", default="fc_001")
    parser.add_argument("--writers", type=int, nargs="+", default=list(range(12)))
    parser.add_argument(
        "--symbol-id",
        action="append",
        help="only run these annotation symbol IDs (repeatable)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--margin-ratio", type=float, default=0.08)
    parser.add_argument("--min-margin", type=float, default=10.0)
    args = parser.parse_args()

    pages: list[Image.Image] = []
    page_paths: list[Path] = []
    annotation_paths: list[Path] = []
    regions: list[Region] = []
    crops = []

    try:
        for page_index, writer in enumerate(args.writers):
            stem = f"writer{writer:03d}_{args.diagram}"
            image_path = args.dataset / "DB" / "images" / f"{stem}.png"
            annotation_path = args.dataset / "DB" / "annotation" / f"{stem}.xml"
            page = Image.open(image_path)
            pages.append(page)
            page_paths.append(image_path)
            annotation_paths.append(annotation_path)
            page_regions = parse_regions(
                annotation_path,
                page_index=page_index,
                image_size=page.size,
                margin_ratio=args.margin_ratio,
                min_margin=args.min_margin,
            )
            if args.symbol_id:
                selected_ids = set(args.symbol_id)
                page_regions = [
                    region for region in page_regions if region.symbol_id in selected_ids
                ]
            regions.extend(page_regions)
            crops.extend(crop_and_pad(page, region.crop_box) for region in page_regions)

        started = time.perf_counter()
        with LocalBackend(batch_size=args.batch_size) as backend:
            results = backend.derender_crops(crops)
            gpu_devices = backend.gpus
        elapsed = time.perf_counter() - started
    finally:
        for page in pages:
            page.close()

    grouped: dict[int, list[tuple[Region, object]]] = defaultdict(list)
    for region, result in zip(regions, results):
        grouped[region.page_index].append((region, result))

    payload = []
    for page_index, image_path in enumerate(page_paths):
        merged = Ink()
        region_payload = []
        texts = []
        used_fallback = False
        for region, result in grouped[page_index]:
            merged.extend(result.ink)
            if result.text:
                texts.append(result.text)
            used_fallback = used_fallback or result.used_fallback
            region_payload.append(
                {
                    "symbol_id": region.symbol_id,
                    "category": region.category,
                    "box": region.box,
                    "crop_box": region.crop_box,
                    "text": result.text,
                    "strokes": len(result.ink),
                    "points": result.ink.num_points,
                    "used_fallback": result.used_fallback,
                }
            )
        payload.append(
            {
                "image": str(image_path),
                "annotation": str(annotation_paths[page_index]),
                "text": " ".join(texts),
                "strokes": len(merged),
                "points": merged.num_points,
                "used_fallback": used_fallback,
                "regions": region_payload,
                "ink": merged.to_dict(),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"wrote {args.output}: {len(payload)} pages, {len(regions)} regions, "
        f"{elapsed:.1f}s, GPUs={gpu_devices}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
