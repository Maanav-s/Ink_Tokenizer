"""Small, portable SVG renderer for recovered ink.

This module intentionally has no TensorFlow dependency.  SVG is a useful
debugging artifact for InkSight output: it preserves source-image coordinates,
can be opened in a browser, and requires no notebook or desktop GUI.
"""

from __future__ import annotations

import base64
import json
import math
import mimetypes
from html import escape
from pathlib import Path
from typing import Any

from .ink import Ink


def load_ink_json(path: str | Path, index: int = 0) -> tuple[Ink, str | None]:
    """Load raw ``Ink`` JSON or an entry produced by ``derender -o``.

    The optional image path is returned when the input is a derender result;
    callers can use it as the SVG background if the original file still exists.
    """
    try:
        payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc

    image: str | None = None
    if isinstance(payload, list):
        if not 0 <= index < len(payload):
            raise ValueError(
                f"result index {index} is out of range for {len(payload)} result(s)"
            )
        payload = payload[index]

    if not isinstance(payload, dict):
        raise ValueError("expected an ink object or a list of derender results")
    if "ink" in payload:
        raw_ink = payload["ink"]
        candidate = payload.get("image")
        image = candidate if isinstance(candidate, str) else None
    else:
        raw_ink = payload

    if not isinstance(raw_ink, dict) or "strokes" not in raw_ink:
        raise ValueError("expected an object with an 'ink.strokes' or 'strokes' field")
    try:
        return Ink.from_dict(raw_ink), image
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid ink data: {exc}") from exc


def ink_to_svg(
    ink: Ink,
    *,
    background: str | Path | None = None,
    show_order: bool = False,
    stroke: str = "#0969da",
) -> str:
    """Render *ink* as a self-contained SVG string.

    When *background* is supplied, the image is embedded as a data URL and the
    ink is drawn in the image's coordinate system.  Without it, the drawing is
    framed tightly with a small margin.
    """
    points = [point for line in ink for point in line]
    for point in points:
        if (
            len(point) != 2
            or not all(isinstance(value, (int, float)) for value in point)
            or not all(math.isfinite(value) for value in point)
        ):
            raise ValueError("ink points must be finite numeric (x, y) pairs")

    background_element = ""
    if background is not None:
        image_path = Path(background)
        if not image_path.is_file():
            raise ValueError(f"no such background image: {image_path}")
        # Pillow is already a core dependency and handles image dimension
        # discovery without forcing TensorFlow or a GUI onto the visualizer.
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
        image_type = mimetypes.guess_type(image_path.name)[0] or "image/*"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        background_element = (
            f'<image href="data:{escape(image_type)};base64,{encoded}" '
            f'width="{width}" height="{height}" />'
        )
        view_box = f"0 0 {width} {height}"
    elif points:
        min_x = min(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_x = max(point[0] for point in points)
        max_y = max(point[1] for point in points)
        span = max(max_x - min_x, max_y - min_y, 1.0)
        margin = max(8.0, span * 0.05)
        width = math.ceil(max_x - min_x + 2 * margin)
        height = math.ceil(max_y - min_y + 2 * margin)
        view_box = (
            f"{_number(min_x - margin)} {_number(min_y - margin)} "
            f"{width} {height}"
        )
    else:
        width = height = 64
        view_box = "0 0 64 64"

    paths: list[str] = []
    order_markers: list[str] = []
    for number, line in enumerate(ink, start=1):
        if not line.points:
            continue
        if len(line) == 1:
            x, y = line[0]
            paths.append(f'<circle cx="{_number(x)}" cy="{_number(y)}" r="2" />')
        else:
            command = " ".join(
                ("M" if point_index == 0 else "L")
                + f" {_number(x)} {_number(y)}"
                for point_index, (x, y) in enumerate(line)
            )
            paths.append(f'<path d="{command}" />')
        if show_order:
            x, y = line[0]
            order_markers.append(
                f'<text x="{_number(x + 3)}" y="{_number(y - 3)}">{number}</text>'
            )

    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                '<svg xmlns="http://www.w3.org/2000/svg" '
                f'viewBox="{view_box}" width="{width}" height="{height}" '
                'role="img" aria-label="Recovered ink">'
            ),
            '<rect width="100%" height="100%" fill="white" />',
            background_element,
            f'<g fill="none" stroke="{escape(stroke)}" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round">',
            *paths,
            "</g>",
            '<g fill="#cf222e" font-family="sans-serif" font-size="10">',
            *order_markers,
            "</g>",
            "</svg>",
            "",
        ]
    )


def _number(value: float) -> str:
    """Use compact, locale-independent coordinate text."""
    return f"{value:.3f}".rstrip("0").rstrip(".")
