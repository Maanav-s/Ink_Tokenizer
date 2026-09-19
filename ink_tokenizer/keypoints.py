"""Simple-kernel ink fragments and conservative flow-layout ordering.

This is experiment support, not an ink derenderer. A raster has no pen-up
events or timestamps, so the returned fragments and their order are explicitly
*pseudo-strokes*: useful layout observations, never recovered handwriting
chronology.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable

from PIL import Image

from .edge_ocr import Box


@dataclass(frozen=True)
class InkFragment:
    """A contour-derived pseudo-stroke represented by salient edge keypoints."""

    box: Box
    keypoints: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class FlowNode:
    """A closed diagram region with zero or more OCR labels."""

    box: Box
    label: str = ""


def laplacian_fragments(
    image: Image.Image,
    *,
    min_arc_length: float = 50.0,
    min_box_area: float = 100.0,
    max_keypoints: int = 12,
) -> tuple[list[InkFragment], float]:
    """Get sparse contour keypoints from the direct 3×3 Laplacian response.

    The kernel is ``[[0,-1,0],[-1,4,-1],[0,-1,0]]``. It is intentionally much
    simpler than Canny: thresholded response contours are only image evidence,
    not attempts to infer an online pen path.
    """
    cv2 = _cv2()
    import numpy as np

    gray = np.asarray(image.convert("L"), dtype=np.int16)
    response = np.zeros_like(gray)
    response[1:-1, 1:-1] = abs(
        4 * gray[1:-1, 1:-1]
        - gray[:-2, 1:-1]
        - gray[2:, 1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
    )
    response = np.minimum(response, 255).astype(np.uint8)
    threshold, edges = cv2.threshold(
        response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    fragments: list[InkFragment] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width * height < min_box_area:
            continue
        perimeter = cv2.arcLength(contour, False)
        if perimeter < min_arc_length:
            continue
        approximation = cv2.approxPolyDP(
            contour, max(1.0, perimeter * 0.015), False
        )
        points = tuple((float(point[0][0]), float(point[0][1])) for point in approximation)
        points = _sample_points(points, max_keypoints)
        if len(points) < 2:
            continue
        fragments.append(InkFragment(Box(x, y, x + width, y + height), points))
    return sorted(fragments, key=lambda fragment: (fragment.box.top, fragment.box.left)), threshold


def order_flow_nodes(nodes: Iterable[FlowNode]) -> list[FlowNode]:
    """Return a deterministic top-to-bottom, main-lane-first layout order.

    This is deliberately conservative: a cycle or side branch is never
    rewritten into a fabricated linear execution trace. Nodes in the same
    vertical band prefer the median x-position (the primary flow lane), then
    fall back to left-to-right order.
    """
    node_list = list(nodes)
    if not node_list:
        return []
    lane_x = median((node.box.left + node.box.right) / 2 for node in node_list)
    return sorted(
        node_list,
        key=lambda node: (
            round(node.box.top / 48),
            abs((node.box.left + node.box.right) / 2 - lane_x),
            node.box.top,
            node.box.left,
        ),
    )


def infer_downward_links(nodes: Iterable[FlowNode]) -> list[tuple[int, int]]:
    """Link each node to its nearest plausible downward successor.

    Links are layout hints only; a flowchart branch or loop can yield multiple
    valid successors, so callers must not treat these as authoritative edges.
    """
    node_list = list(nodes)
    links: list[tuple[int, int]] = []
    for source_index, source in enumerate(node_list):
        source_center = (source.box.left + source.box.right) / 2
        candidates: list[tuple[float, int]] = []
        for target_index, target in enumerate(node_list):
            if target_index == source_index or target.box.top < source.box.bottom:
                continue
            target_center = (target.box.left + target.box.right) / 2
            vertical_gap = target.box.top - source.box.bottom
            horizontal_gap = abs(target_center - source_center)
            candidates.append((vertical_gap + 0.45 * horizontal_gap, target_index))
        if candidates:
            links.append((source_index, min(candidates)[1]))
    return links


def fragment_owner(fragment: InkFragment, nodes: list[FlowNode]) -> int | None:
    """Assign a fragment to the smallest flow node containing its midpoint."""
    center_x = (fragment.box.left + fragment.box.right) / 2
    center_y = (fragment.box.top + fragment.box.bottom) / 2
    candidates = [
        (node.box.area, index)
        for index, node in enumerate(nodes)
        if node.box.left <= center_x <= node.box.right
        and node.box.top <= center_y <= node.box.bottom
    ]
    return min(candidates)[1] if candidates else None


def _sample_points(
    points: tuple[tuple[float, float], ...], maximum: int
) -> tuple[tuple[float, float], ...]:
    if len(points) <= maximum:
        return points
    indexes = [round(index * (len(points) - 1) / (maximum - 1)) for index in range(maximum)]
    return tuple(points[index] for index in indexes)


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - only hit without extra
        raise RuntimeError(
            "This experiment needs OpenCV. Install it with "
            "`uv sync --extra experiments`."
        ) from exc
    return cv2
