"""Codec tests.

These run anywhere -- no TensorFlow, no model, no service. That is the point
of keeping the codec on this side of the boundary.
"""

import pytest

from ink_tokenizer.codec import (
    COORDINATE_LENGTH,
    NUM_TOKEN_PER_DIMENSION,
    START_TOKEN,
    VOCABULARY_SIZE,
    detokenize,
    strip_ink_tokens,
    text_to_tokens,
    tokenize,
    tokens_to_text,
)
from ink_tokenizer.ink import Ink, Stroke


def test_vocabulary_constants():
    assert COORDINATE_LENGTH == 224
    assert NUM_TOKEN_PER_DIMENSION == 225
    assert START_TOKEN == 450
    assert VOCABULARY_SIZE == 451


def test_text_to_tokens_extracts_ids_in_order():
    text = "hello<ink_token_450><ink_token_10><ink_token_235>"
    assert text_to_tokens(text) == [450, 10, 235]


def test_strip_ink_tokens_recovers_recognized_text():
    text = "<ink_token_450>volt<ink_token_12><ink_token_300>age"
    assert strip_ink_tokens(text) == "voltage"


def test_strip_removes_mt5_sentinels():
    # Exactly what the recognizing prompt returned for upstream's word.jpg.
    assert strip_ink_tokens("<extra_id_0> Neat:<ink_token_450>") == "Neat:"
    assert strip_ink_tokens("<extra_id_0> V<ink_token_12>out</s>") == "Vout"


def test_strip_collapses_whitespace():
    assert strip_ink_tokens("<extra_id_0>  a<ink_token_1>   b  ") == "a b"


def test_detokenize_single_stroke():
    # y tokens are offset by 225: 235 -> y=10, 245 -> y=20
    ink = detokenize([START_TOKEN, 10, 235, 20, 245])
    assert len(ink) == 1
    assert ink[0].points == [(10.0, 10.0), (20.0, 20.0)]


def test_detokenize_splits_on_separator():
    ink = detokenize([START_TOKEN, 1, 226, START_TOKEN, 5, 230])
    assert [len(s) for s in ink] == [1, 1]
    assert ink[1].points == [(5.0, 5.0)]


def test_detokenize_drops_dangling_x():
    # Trailing x with no y, and an x immediately followed by a separator.
    ink = detokenize([START_TOKEN, 3, 228, 99])
    assert ink[0].points == [(3.0, 3.0)]
    ink2 = detokenize([START_TOKEN, 3, 228, 99, START_TOKEN, 4, 229])
    assert [s.points for s in ink2] == [[(3.0, 3.0)], [(4.0, 4.0)]]


def test_detokenize_skips_out_of_grid_pair():
    # x=10 paired with token 100 -> y = 100-225 = -125, out of range: dropped.
    ink = detokenize([START_TOKEN, 10, 100, 20, 245])
    assert ink[0].points == [(20.0, 20.0)]


def test_detokenize_rejects_out_of_vocabulary():
    with pytest.raises(ValueError):
        detokenize([START_TOKEN, 451])
    with pytest.raises(ValueError):
        detokenize([-1])


def test_empty_inputs():
    assert detokenize([]).strokes == []
    assert text_to_tokens("no ink here") == []
    assert tokenize(Ink()) == []


def test_roundtrip_tokenize_detokenize():
    ink = Ink.from_point_lists(
        [[(0, 0), (12, 200), (224, 224)], [(100, 3), (7, 99)]]
    )
    tokens = tokenize(ink)
    assert tokens[0] == START_TOKEN
    restored = detokenize(tokens)
    assert [s.points for s in restored] == [
        [(0.0, 0.0), (12.0, 200.0), (224.0, 224.0)],
        [(100.0, 3.0), (7.0, 99.0)],
    ]


def test_roundtrip_through_wire_text():
    ink = Ink.from_point_lists([[(5, 6), (7, 8)]])
    text = tokens_to_text(tokenize(ink))
    assert detokenize(text_to_tokens(text))[0].points == [(5.0, 6.0), (7.0, 8.0)]


def test_tokenize_quantizes_and_clamps():
    ink = Ink.from_point_lists([[(10.4, 10.6), (-5, 900)]])
    assert detokenize(tokenize(ink))[0].points == [(10.0, 11.0), (0.0, 224.0)]


def test_tokenize_strict_rejects_out_of_range():
    with pytest.raises(ValueError):
        tokenize(Ink.from_point_lists([[(300, 0)]]), clamp=False)


def test_tokenize_drops_empty_strokes():
    ink = Ink([Stroke([]), Stroke([(1, 1)]), Stroke([])])
    assert detokenize(tokenize(ink)).num_points == 1
    assert len(detokenize(tokenize(ink))) == 1
