"""Core ink data model.

Deliberately dependency-free: this is the representation the rest of the
project is built on, and it must stay importable anywhere (including on
Windows, where the InkSight inference stack cannot run).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Iterator, Sequence


@dataclass
class Stroke:
    """An ordered sequence of points making up one pen-down..pen-up stroke.

    Note there is no time axis. InkSight recovers stroke *order* and point
    *order*, but not timestamps or velocity -- see docs/inksight.md.
    """

    points: list[tuple[float, float]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self) -> Iterator[tuple[float, float]]:
        return iter(self.points)

    def __getitem__(self, index: int) -> tuple[float, float]:
        return self.points[index]

    @property
    def x(self) -> list[float]:
        return [p[0] for p in self.points]

    @property
    def y(self) -> list[float]:
        return [p[1] for p in self.points]

    def bbox(self) -> tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y); zeros for an empty stroke."""
        if not self.points:
            return (0.0, 0.0, 0.0, 0.0)
        xs, ys = self.x, self.y
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class Ink:
    """A collection of strokes in a single coordinate space."""

    strokes: list[Stroke] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.strokes)

    def __iter__(self) -> Iterator[Stroke]:
        return iter(self.strokes)

    def __getitem__(self, index: int) -> Stroke:
        return self.strokes[index]

    @property
    def num_points(self) -> int:
        return sum(len(s) for s in self.strokes)

    def bbox(self) -> tuple[float, float, float, float]:
        boxes = [s.bbox() for s in self.strokes if len(s)]
        if not boxes:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )

    def translated(self, dx: float, dy: float) -> "Ink":
        return Ink([Stroke([(x + dx, y + dy) for x, y in s]) for s in self.strokes])

    def scaled(self, sx: float, sy: float | None = None) -> "Ink":
        sy = sx if sy is None else sy
        return Ink([Stroke([(x * sx, y * sy) for x, y in s]) for s in self.strokes])

    def extend(self, other: "Ink") -> None:
        self.strokes.extend(other.strokes)

    # -- serialization -------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> "Ink":
        return cls(
            [Stroke([tuple(p) for p in s["points"]]) for s in data.get("strokes", [])]
        )

    @classmethod
    def from_json(cls, text: str) -> "Ink":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_point_lists(cls, strokes: Sequence[Sequence[tuple[float, float]]]) -> "Ink":
        return cls([Stroke([tuple(p) for p in s]) for s in strokes])
