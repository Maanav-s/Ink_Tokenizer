"""ScribeTokens-compatible online-ink codec tests."""

import pytest

from ink_tokenizer.ink import Ink
from ink_tokenizer.scribe import (
    BASE_VOCABULARY,
    ScribeBPE,
    ScribeToken,
    bresenham_steps,
    decode_scribe,
    decode_scribe_tokens,
    encode_scribe,
    ids_to_tokens,
    tokens_to_ids,
)


def test_fixed_base_vocabulary_and_bresenham_steps():
    assert len(BASE_VOCABULARY) == 10
    assert list(bresenham_steps((0, 0), (4, 2))) == [
        ScribeToken.E,
        ScribeToken.SE,
        ScribeToken.E,
        ScribeToken.SE,
    ]


def test_roundtrip_preserves_quantized_strokes_and_pen_up_travel():
    ink = Ink.from_point_lists(
        [
            [(10, 20), (14, 20), (14, 24)],
            [(20, 30), (18, 30)],
            [(7, 17)],
        ]
    )
    encoding = encode_scribe(ink, step=2)

    assert encoding.origin == (10.0, 20.0)
    assert encoding.tokens.count(ScribeToken.DOWN) == 3
    assert encoding.tokens.count(ScribeToken.UP) == 3
    assert [stroke.points for stroke in decode_scribe(encoding)] == [
        [(10.0, 20.0), (12.0, 20.0), (14.0, 20.0), (14.0, 22.0), (14.0, 24.0)],
        [(20.0, 30.0), (18.0, 30.0)],
        [(6.0, 16.0)],
    ]


def test_base_ids_and_malformed_state_streams_are_rejected():
    tokens = (ScribeToken.DOWN, ScribeToken.E, ScribeToken.UP)
    assert ids_to_tokens(tokens_to_ids(tokens)) == tokens
    with pytest.raises(ValueError, match="already up"):
        decode_scribe_tokens([ScribeToken.UP])
    with pytest.raises(ValueError, match="ended while"):
        decode_scribe_tokens([ScribeToken.DOWN])


def test_bpe_is_lossless_and_never_merges_across_pen_state():
    streams = [
        [ScribeToken.DOWN, ScribeToken.E, ScribeToken.E, ScribeToken.E, ScribeToken.UP],
        [ScribeToken.DOWN, ScribeToken.E, ScribeToken.E, ScribeToken.N, ScribeToken.UP],
        [ScribeToken.DOWN, ScribeToken.E, ScribeToken.E, ScribeToken.E, ScribeToken.UP],
    ]
    model = ScribeBPE.fit(streams, vocabulary_size=12)
    stream = tuple(streams[0])
    compressed = model.encode(stream)

    assert model.vocabulary_size == 12
    assert len(compressed) < len(stream)
    assert compressed[0] is ScribeToken.DOWN
    assert compressed[-1] is ScribeToken.UP
    assert model.decode(compressed) == stream
    assert model.decode_ids(model.encode_ids(stream)) == stream
    assert ScribeBPE.from_dict(model.to_dict()) == model
