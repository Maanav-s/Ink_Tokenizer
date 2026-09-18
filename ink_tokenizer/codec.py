"""InkSight ink-token codec.

`detokenize` and `text_to_tokens` are ported from google-research/inksight
(`colab.ipynb`, Apache-2.0) -- see NOTICE. Upstream ships them only inside a
notebook, so they are vendored here rather than depended on. They are pure
Python and pull in no TensorFlow, which is what lets this module run in the
main project environment while inference runs elsewhere.

`tokenize` is the inverse and is ours; upstream provides no encoder.

Token scheme
------------
Coordinates are quantized to a 225x225 integer grid over a 224x224 padded
crop, flattened into one vocabulary of 451 ids::

    x value v in [0, 224]  ->  token v
    y value v in [0, 224]  ->  token v + 225
    stroke separator       ->  token 450

In model output the ids appear as literal ``<ink_token_N>`` substrings,
interleaved with the recognized text.
"""

from __future__ import annotations

import re

from .ink import Ink, Stroke

COORDINATE_LENGTH = 224
NUM_TOKEN_PER_DIMENSION = COORDINATE_LENGTH + 1  # 225
VOCABULARY_SIZE = NUM_TOKEN_PER_DIMENSION * 2 + 1  # 451
START_TOKEN = NUM_TOKEN_PER_DIMENSION * 2  # 450, stroke separator

_INK_TOKEN_RE = re.compile(r"<ink_token_(\d+)>")
# mT5 sentinels. The recognizing prompt prefixes its answer with one
# (`<extra_id_0> Neat:`), which is vocabulary bookkeeping, not recognized text.
_SENTINEL_RE = re.compile(r"</?extra_id_\d+>|</s>|<pad>")


def text_to_tokens(text: str) -> list[int]:
    """Pull the ink token ids out of raw model output text."""
    return [int(tok) for tok in _INK_TOKEN_RE.findall(text)]


def strip_ink_tokens(text: str) -> str:
    """Recover just the recognized text from raw model output.

    With the "Recognize and derender." prompt the output interleaves the
    recognized text with ink tokens and opens with an mT5 sentinel; both are
    removed here. The untouched output is always kept as `DerenderResult.raw`.
    """
    cleaned = _SENTINEL_RE.sub("", _INK_TOKEN_RE.sub("", text))
    return " ".join(cleaned.split())


def detokenize(tokens: list[int]) -> Ink:
    """Decode ink token ids into an `Ink` in 224x224 crop space.

    Malformed pairs are skipped rather than raising -- model output is not
    guaranteed well-formed, and a dropped point is preferable to losing the
    whole sample. Out-of-range ids do raise, since those indicate the token
    stream was misparsed rather than merely truncated.
    """
    if any(t < 0 or t >= VOCABULARY_SIZE for t in tokens):
        raise ValueError(
            f"Ink token indices should be between 0 and {VOCABULARY_SIZE - 1}"
        )

    idx = 0
    res: list[list[tuple[int, int]]] = []
    current_stroke_tokens: list[tuple[int, int]] = []

    while idx < len(tokens):
        token = tokens[idx]
        if token == START_TOKEN:
            if current_stroke_tokens:
                res.append(current_stroke_tokens)
            current_stroke_tokens = []
            idx += 1
        elif idx + 1 < len(tokens) and tokens[idx + 1] != START_TOKEN:
            x = tokens[idx]
            y = tokens[idx + 1] - NUM_TOKEN_PER_DIMENSION
            if 0 <= x <= COORDINATE_LENGTH and 0 <= y <= COORDINATE_LENGTH:
                current_stroke_tokens.append((x, y))
            idx += 2
        else:
            # y is missing or is a separator; drop the dangling x.
            idx += 1

    if current_stroke_tokens:
        res.append(current_stroke_tokens)

    return Ink([Stroke([(float(x), float(y)) for x, y in stroke]) for stroke in res])


def tokenize(ink: Ink, *, clamp: bool = True) -> list[int]:
    """Encode an `Ink` in 224x224 crop space back into ink token ids.

    Inverse of `detokenize`. Coordinates are rounded to the integer grid;
    with `clamp` they are clipped into range, otherwise out-of-range points
    raise. Empty strokes are dropped, since a separator pair would decode
    back to nothing.
    """
    tokens: list[int] = []
    for stroke in ink:
        if not len(stroke):
            continue
        tokens.append(START_TOKEN)
        for x, y in stroke:
            xi, yi = round(x), round(y)
            if clamp:
                xi = min(max(xi, 0), COORDINATE_LENGTH)
                yi = min(max(yi, 0), COORDINATE_LENGTH)
            elif not (0 <= xi <= COORDINATE_LENGTH and 0 <= yi <= COORDINATE_LENGTH):
                raise ValueError(
                    f"Point ({x}, {y}) is outside the 0..{COORDINATE_LENGTH} grid"
                )
            tokens.extend((xi, yi + NUM_TOKEN_PER_DIMENSION))
    return tokens


def tokens_to_text(tokens: list[int]) -> str:
    """Render token ids back into the ``<ink_token_N>`` wire form."""
    return "".join(f"<ink_token_{t}>" for t in tokens)
