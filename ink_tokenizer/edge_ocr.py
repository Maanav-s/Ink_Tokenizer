"""OpenCV proposals and Tesseract TSV parsing for the comparison baseline.

This deliberately stays outside the product inference path. It tests whether
conventional vision is a better *layout* front end for a full handwritten
engineering page than squeezing that page into InkSight's 224×224 input.
"""

from __future__ import annotations

import csv
import io
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


@dataclass(frozen=True)
class Box:
    """An image-coordinate bounding box with an optional score or label."""

    left: float
    top: float
    right: float
    bottom: float
    score: float = 1.0
    label: str = ""

    @property
    def width(self) -> float:
        return max(0.0, self.right - self.left)

    @property
    def height(self) -> float:
        return max(0.0, self.bottom - self.top)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict[str, float | str]:
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
            "score": self.score,
            "label": self.label,
        }


def box_iou(first: Box, second: Box) -> float:
    """Return intersection-over-union for two boxes."""
    left, top = max(first.left, second.left), max(first.top, second.top)
    right, bottom = min(first.right, second.right), min(first.bottom, second.bottom)
    overlap = max(0.0, right - left) * max(0.0, bottom - top)
    union = first.area + second.area - overlap
    return overlap / union if union else 0.0


def contained_fraction(inner: Box, outer: Box) -> float:
    """Return the fraction of ``inner`` covered by ``outer``."""
    left, top = max(inner.left, outer.left), max(inner.top, outer.top)
    right, bottom = min(inner.right, outer.right), min(inner.bottom, outer.bottom)
    overlap = max(0.0, right - left) * max(0.0, bottom - top)
    return overlap / inner.area if inner.area else 0.0


def detect_symbol_proposals(
    image: Image.Image,
    *,
    min_width: int = 70,
    min_height: int = 45,
    max_page_fraction: float = 0.20,
) -> list[Box]:
    """Propose closed, diagram-sized regions from Canny contours.

    ``RETR_TREE`` preserves interior contours, so a closed flowchart shape can
    still be proposed even when a connector touches its outer edge. Arrows and
    tiny components are intentionally omitted: this is structural layout only.
    """
    cv2 = _cv2()
    import numpy as np

    original_width, original_height = image.size
    scale = min(1.0, 1600.0 / max(original_width, original_height))
    width, height = round(original_width * scale), round(original_height * scale)
    rgb = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = float(min_width * min_height) * scale * scale
    maximum_area = width * height * max_page_fraction
    candidates: list[Box] = []
    for contour in contours:
        x, y, candidate_width, candidate_height = cv2.boundingRect(contour)
        area = candidate_width * candidate_height
        if not minimum_area <= area <= maximum_area:
            continue
        aspect = candidate_width / candidate_height
        if not 0.35 <= aspect <= 6.0:
            continue
        rectangularity = float(cv2.contourArea(contour)) / area if area else 0.0
        # Solid blobs are normally text/noise; sparse giant contours are
        # normally connector chains rather than a closed symbol.
        if not 0.015 <= rectangularity <= 0.95:
            continue
        candidates.append(
            Box(
                x / scale,
                y / scale,
                (x + candidate_width) / scale,
                (y + candidate_height) / scale,
                score=rectangularity,
                label="edge-proposal",
            )
        )
    return non_maximum_suppression(candidates, threshold=0.65)


def non_maximum_suppression(boxes: Iterable[Box], *, threshold: float) -> list[Box]:
    """Keep one box from highly overlapping contour duplicates."""
    kept: list[Box] = []
    for candidate in sorted(boxes, key=lambda box: (box.score, box.area), reverse=True):
        if all(box_iou(candidate, existing) < threshold for existing in kept):
            kept.append(candidate)
    return sorted(kept, key=lambda box: (box.top, box.left))


def ocr_with_tesseract(
    image_path: str | Path, *, command: str = "tesseract", psm: int = 11
) -> list[Box]:
    """Run Tesseract sparse-text mode and return confident word boxes."""
    completed = subprocess.run(
        [command, str(image_path), "stdout", "--psm", str(psm), "tsv"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        raise RuntimeError(f"Tesseract failed: {detail}")
    return parse_tesseract_tsv(completed.stdout)


def make_rapidocr():
    """Construct the project-local ONNX OCR engine lazily."""
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:  # pragma: no cover - only hit without extra
        raise RuntimeError(
            "This experiment needs RapidOCR. Install it with "
            "`uv sync --extra experiments`."
        ) from exc
    return RapidOCR()


def ocr_with_rapidocr(image_path: str | Path, engine: object) -> list[Box]:
    """Convert RapidOCR's quadrilateral text detections into axis-aligned boxes."""
    output = engine(str(image_path))  # type: ignore[operator]
    words: list[Box] = []
    for polygon, text, confidence in zip(output.boxes, output.txts, output.scores):
        if not text or float(confidence) < 0.20:
            continue
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
        words.append(
            Box(min(xs), min(ys), max(xs), max(ys), float(confidence) * 100, text)
        )
    return words


def parse_tesseract_tsv(payload: str, *, min_confidence: float = 20.0) -> list[Box]:
    """Parse Tesseract TSV without requiring a Python OCR wrapper."""
    words: list[Box] = []
    for row in csv.DictReader(io.StringIO(payload), delimiter="\t"):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            confidence = float(row["conf"])
            left, top = float(row["left"]), float(row["top"])
            width, height = float(row["width"]), float(row["height"])
        except (KeyError, TypeError, ValueError):
            continue
        if confidence >= min_confidence and width > 0 and height > 0:
            words.append(Box(left, top, left + width, top + height, confidence, text))
    return words


def greedy_matches(
    candidates: Iterable[Box], references: Iterable[Box], *, threshold: float = 0.3
) -> list[tuple[int, int, float]]:
    """Return one-to-one candidate/reference matches, highest IoU first."""
    candidate_list, reference_list = list(candidates), list(references)
    scored = sorted(
        (
            (box_iou(candidate, reference), candidate_index, reference_index)
            for candidate_index, candidate in enumerate(candidate_list)
            for reference_index, reference in enumerate(reference_list)
        ),
        reverse=True,
    )
    used_candidates: set[int] = set()
    used_references: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, candidate_index, reference_index in scored:
        if score < threshold:
            break
        if candidate_index not in used_candidates and reference_index not in used_references:
            used_candidates.add(candidate_index)
            used_references.add(reference_index)
            matches.append((candidate_index, reference_index, score))
    return matches


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - only hit without extra
        raise RuntimeError(
            "This experiment needs OpenCV. Install it with "
            "`uv sync --extra experiments`."
        ) from exc
    return cv2
