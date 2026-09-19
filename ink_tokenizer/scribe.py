"""Fixed-vocabulary tokenization for online ink.

This is an independent implementation of the ScribeTokens representation
described by Douglass Wang: quantize an ink stream, decompose every movement
into Bresenham unit steps, and represent those steps with eight directions plus
explicit pen state. BPE may then compress *direction runs*; it never crosses a
pen-up/down boundary.

Unlike :mod:`ink_tokenizer.codec`, this module is not a wire codec for the
InkSight image model. It is a compact, OOV-free representation for the native
online-ink data that a future structural predictor can consume. It deliberately
has no ML or third-party dependencies.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from math import ceil, floor, isfinite
from typing import Iterable, Iterator, Sequence, TypeAlias

from .ink import Ink, Stroke


class ScribeToken(IntEnum):
    """The fixed base vocabulary (eight image-coordinate directions + pen state)."""

    DOWN = 0
    UP = 1
    N = 2
    NE = 3
    E = 4
    SE = 5
    S = 6
    SW = 7
    W = 8
    NW = 9


BASE_VOCABULARY: tuple[ScribeToken, ...] = tuple(ScribeToken)
DIRECTION_TOKENS: frozenset[ScribeToken] = frozenset(
    token for token in ScribeToken if token not in {ScribeToken.DOWN, ScribeToken.UP}
)

# Coordinates throughout Ink Tokenizer follow image convention: x grows right,
# y grows down. Therefore a positive y step is S, not N.
TOKEN_TO_DELTA: dict[ScribeToken, tuple[int, int]] = {
    ScribeToken.N: (0, -1),
    ScribeToken.NE: (1, -1),
    ScribeToken.E: (1, 0),
    ScribeToken.SE: (1, 1),
    ScribeToken.S: (0, 1),
    ScribeToken.SW: (-1, 1),
    ScribeToken.W: (-1, 0),
    ScribeToken.NW: (-1, -1),
}
DELTA_TO_TOKEN: dict[tuple[int, int], ScribeToken] = {
    delta: token for token, delta in TOKEN_TO_DELTA.items()
}

ScribeSymbol: TypeAlias = tuple[ScribeToken, ...]
CompressedScribeToken: TypeAlias = ScribeToken | ScribeSymbol


@dataclass(frozen=True)
class ScribeEncoding:
    """A base-token stream plus the geometry metadata omitted from the stream.

    The base alphabet intentionally has no absolute-position token. ``origin``
    anchors the first non-empty stroke, while ``step`` restores the chosen
    quantization scale. Keep this metadata with a training example; a model
    should receive position/context separately rather than infer it from a
    direction vocabulary.
    """

    tokens: tuple[ScribeToken, ...]
    origin: tuple[float, float] | None
    step: float

    def decode(self) -> Ink:
        """Reconstruct the quantized ink represented by this encoding."""
        return decode_scribe(self)


def _validate_step(step: float) -> None:
    if not isfinite(step) or step <= 0:
        raise ValueError("step must be a finite positive number")


def _nearest_integer(value: float) -> int:
    """Round half away from zero, consistently across Python versions."""
    return int(floor(value + 0.5)) if value >= 0 else int(ceil(value - 0.5))


def _quantize_point(
    point: tuple[float, float], origin: tuple[float, float], step: float
) -> tuple[int, int]:
    x, y = point
    if not isfinite(x) or not isfinite(y):
        raise ValueError(f"ink point must be finite, got {point!r}")
    return (
        _nearest_integer((x - origin[0]) / step),
        _nearest_integer((y - origin[1]) / step),
    )


def bresenham_steps(
    start: tuple[int, int], end: tuple[int, int]
) -> Iterator[ScribeToken]:
    """Yield the unit Freeman-chain steps from ``start`` to ``end``.

    This standard integer line walk includes neither endpoint as a token: each
    yielded direction advances the current position exactly once.
    """
    x, y = start
    target_x, target_y = end
    dx = abs(target_x - x)
    dy = abs(target_y - y)
    step_x = 1 if x < target_x else -1
    step_y = 1 if y < target_y else -1
    error = dx - dy

    while (x, y) != (target_x, target_y):
        previous = (x, y)
        doubled_error = 2 * error
        if doubled_error > -dy:
            error -= dy
            x += step_x
        if doubled_error < dx:
            error += dx
            y += step_y
        yield DELTA_TO_TOKEN[(x - previous[0], y - previous[1])]


def encode_scribe(
    ink: Ink, *, step: float = 1.0, max_tokens: int | None = None
) -> ScribeEncoding:
    """Encode online ink into the ten-token Scribe base vocabulary.

    Coordinates are rounded to a ``step`` grid relative to the first non-empty
    stroke. Pen-up travel between strokes is represented by direction tokens
    while the pen is up, preserving each stroke's start position on decode.
    ``max_tokens`` is a guard for callers accepting untrusted coordinates.
    """
    _validate_step(step)
    if max_tokens is not None and max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")

    nonempty_strokes = [stroke.points for stroke in ink if stroke.points]
    if not nonempty_strokes:
        return ScribeEncoding(tokens=(), origin=None, step=step)

    origin = tuple(float(value) for value in nonempty_strokes[0][0])
    quantized = [
        [_quantize_point(tuple(point), origin, step) for point in stroke]
        for stroke in nonempty_strokes
    ]

    tokens: list[ScribeToken] = []

    def append(token: ScribeToken) -> None:
        if max_tokens is not None and len(tokens) >= max_tokens:
            raise ValueError(f"Scribe stream exceeds max_tokens={max_tokens}")
        tokens.append(token)

    current = (0, 0)
    for stroke in quantized:
        for token in bresenham_steps(current, stroke[0]):
            append(token)
        current = stroke[0]
        append(ScribeToken.DOWN)
        for point in stroke[1:]:
            for token in bresenham_steps(current, point):
                append(token)
            current = point
        append(ScribeToken.UP)

    return ScribeEncoding(tokens=tuple(tokens), origin=origin, step=step)


def decode_scribe(encoding: ScribeEncoding) -> Ink:
    """Decode a :class:`ScribeEncoding` to its quantized geometry."""
    if encoding.origin is None:
        if encoding.tokens:
            raise ValueError("an encoding with tokens needs an origin")
        return Ink()
    return decode_scribe_tokens(encoding.tokens, origin=encoding.origin, step=encoding.step)


def decode_scribe_tokens(
    tokens: Sequence[ScribeToken | int],
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    step: float = 1.0,
) -> Ink:
    """Decode a base stream when geometry metadata is supplied separately."""
    _validate_step(step)
    if not isfinite(origin[0]) or not isfinite(origin[1]):
        raise ValueError("origin must contain finite coordinates")

    current = (0, 0)
    pen_down = False
    strokes: list[Stroke] = []

    def position() -> tuple[float, float]:
        return (origin[0] + current[0] * step, origin[1] + current[1] * step)

    for index, raw_token in enumerate(tokens):
        try:
            token = ScribeToken(raw_token)
        except ValueError as exc:
            raise ValueError(f"invalid Scribe token at index {index}: {raw_token!r}") from exc

        if token is ScribeToken.DOWN:
            if pen_down:
                raise ValueError(f"DOWN while already down at token index {index}")
            strokes.append(Stroke([position()]))
            pen_down = True
        elif token is ScribeToken.UP:
            if not pen_down:
                raise ValueError(f"UP while already up at token index {index}")
            pen_down = False
        else:
            dx, dy = TOKEN_TO_DELTA[token]
            current = (current[0] + dx, current[1] + dy)
            if pen_down:
                strokes[-1].points.append(position())

    if pen_down:
        raise ValueError("Scribe stream ended while the pen was down")
    return Ink(strokes)


def tokens_to_ids(tokens: Sequence[ScribeToken]) -> tuple[int, ...]:
    """Return stable base-vocabulary ids (0 through 9)."""
    return tuple(int(ScribeToken(token)) for token in tokens)


def ids_to_tokens(ids: Sequence[int]) -> tuple[ScribeToken, ...]:
    """Validate and restore base tokens from stable vocabulary ids."""
    result: list[ScribeToken] = []
    for index, token_id in enumerate(ids):
        try:
            result.append(ScribeToken(token_id))
        except ValueError as exc:
            raise ValueError(f"invalid Scribe token id at index {index}: {token_id!r}") from exc
    return tuple(result)


def _validate_symbol(symbol: ScribeSymbol) -> None:
    if not symbol or any(token not in DIRECTION_TOKENS for token in symbol):
        raise ValueError("BPE symbols must contain one or more direction tokens")


def _merge_symbols(
    symbols: Sequence[ScribeSymbol], left: ScribeSymbol, right: ScribeSymbol
) -> list[ScribeSymbol]:
    merged: list[ScribeSymbol] = []
    index = 0
    while index < len(symbols):
        if index + 1 < len(symbols) and symbols[index] == left and symbols[index + 1] == right:
            merged.append(left + right)
            index += 2
        else:
            merged.append(symbols[index])
            index += 1
    return merged


@dataclass(frozen=True)
class ScribeBPE:
    """A deterministic BPE model over direction runs of a Scribe stream.

    ``DOWN`` and ``UP`` retain their two base ids and act as hard boundaries.
    That preserves incremental pen state even when the path vocabulary is
    compressed. Composite symbols expand losslessly to base direction tokens.
    """

    merges: tuple[tuple[ScribeSymbol, ScribeSymbol], ...] = ()

    def __post_init__(self) -> None:
        known: set[ScribeSymbol] = {(token,) for token in DIRECTION_TOKENS}
        for left, right in self.merges:
            _validate_symbol(left)
            _validate_symbol(right)
            if left not in known or right not in known:
                raise ValueError("BPE merges must be ordered by construction")
            combined = left + right
            if combined in known:
                raise ValueError("BPE merge creates a duplicate symbol")
            known.add(combined)

    @property
    def vocabulary(self) -> tuple[CompressedScribeToken, ...]:
        """Base tokens followed by learned composite direction symbols."""
        return BASE_VOCABULARY + tuple(left + right for left, right in self.merges)

    @property
    def vocabulary_size(self) -> int:
        return len(self.vocabulary)

    def _encode_run(self, directions: Sequence[ScribeToken]) -> list[ScribeSymbol]:
        symbols: list[ScribeSymbol] = [(token,) for token in directions]
        for left, right in self.merges:
            symbols = _merge_symbols(symbols, left, right)
        return symbols

    def encode(self, tokens: Sequence[ScribeToken | int]) -> tuple[CompressedScribeToken, ...]:
        """Apply BPE without crossing pen-state boundaries."""
        encoded: list[CompressedScribeToken] = []
        direction_run: list[ScribeToken] = []

        def flush() -> None:
            if direction_run:
                encoded.extend(self._encode_run(direction_run))
                direction_run.clear()

        for index, raw_token in enumerate(tokens):
            try:
                token = ScribeToken(raw_token)
            except ValueError as exc:
                raise ValueError(f"invalid Scribe token at index {index}: {raw_token!r}") from exc
            if token in DIRECTION_TOKENS:
                direction_run.append(token)
            else:
                flush()
                encoded.append(token)
        flush()
        return tuple(encoded)

    @staticmethod
    def decode(tokens: Sequence[CompressedScribeToken]) -> tuple[ScribeToken, ...]:
        """Expand compressed tokens back to the ten-token base stream."""
        expanded: list[ScribeToken] = []
        for token in tokens:
            if isinstance(token, ScribeToken):
                expanded.append(token)
            else:
                _validate_symbol(token)
                expanded.extend(token)
        return tuple(expanded)

    def encode_ids(self, tokens: Sequence[ScribeToken | int]) -> tuple[int, ...]:
        """Encode to ids in the model's base-plus-learned vocabulary."""
        vocabulary_ids = {token: index for index, token in enumerate(self.vocabulary)}
        return tuple(vocabulary_ids[token] for token in self.encode(tokens))

    def decode_ids(self, ids: Sequence[int]) -> tuple[ScribeToken, ...]:
        """Restore base tokens from model-vocabulary ids."""
        vocabulary = self.vocabulary
        compressed: list[CompressedScribeToken] = []
        for index, token_id in enumerate(ids):
            if not 0 <= token_id < len(vocabulary):
                raise ValueError(f"invalid Scribe BPE id at index {index}: {token_id!r}")
            compressed.append(vocabulary[token_id])
        return self.decode(compressed)

    def to_dict(self) -> dict[str, object]:
        """Serialize learned merges without relying on a tokenizer library."""
        return {
            "version": 1,
            "merges": [
                [[int(token) for token in left], [int(token) for token in right]]
                for left, right in self.merges
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ScribeBPE":
        """Restore a model emitted by :meth:`to_dict`."""
        if data.get("version") != 1:
            raise ValueError("unsupported Scribe BPE serialization version")
        raw_merges = data.get("merges")
        if not isinstance(raw_merges, list):
            raise ValueError("Scribe BPE merges must be a list")
        merges: list[tuple[ScribeSymbol, ScribeSymbol]] = []
        for index, raw_pair in enumerate(raw_merges):
            if not isinstance(raw_pair, list) or len(raw_pair) != 2:
                raise ValueError(f"invalid BPE merge at index {index}")
            try:
                left = tuple(ScribeToken(token) for token in raw_pair[0])
                right = tuple(ScribeToken(token) for token in raw_pair[1])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid BPE symbols at index {index}") from exc
            merges.append((left, right))
        return cls(tuple(merges))

    @classmethod
    def fit(
        cls, streams: Iterable[Sequence[ScribeToken | int]], *, vocabulary_size: int
    ) -> "ScribeBPE":
        """Train deterministic pair merges up to vocabulary_size.

        Training is intentionally small and transparent for stage-one data work;
        it counts only adjacent direction symbols. Pen-state events are never
        merged, so no learned token spans an unknown or partial stroke boundary.
        """
        if vocabulary_size < len(BASE_VOCABULARY):
            raise ValueError(f"vocabulary_size must be at least {len(BASE_VOCABULARY)}")

        runs: list[list[ScribeSymbol]] = []
        for stream in streams:
            run: list[ScribeSymbol] = []
            for index, raw_token in enumerate(stream):
                try:
                    token = ScribeToken(raw_token)
                except ValueError as exc:
                    raise ValueError(f"invalid Scribe token at index {index}: {raw_token!r}") from exc
                if token in DIRECTION_TOKENS:
                    run.append((token,))
                elif run:
                    runs.append(run)
                    run = []
            if run:
                runs.append(run)

        merges: list[tuple[ScribeSymbol, ScribeSymbol]] = []
        while len(BASE_VOCABULARY) + len(merges) < vocabulary_size:
            pair_counts: Counter[tuple[ScribeSymbol, ScribeSymbol]] = Counter()
            for run in runs:
                pair_counts.update(zip(run, run[1:]))
            if not pair_counts:
                break
            best = min(
                pair_counts,
                key=lambda pair: (
                    -pair_counts[pair],
                    tuple(map(int, pair[0])),
                    tuple(map(int, pair[1])),
                ),
            )
            merges.append(best)
            runs = [_merge_symbols(run, *best) for run in runs]
        return cls(tuple(merges))
