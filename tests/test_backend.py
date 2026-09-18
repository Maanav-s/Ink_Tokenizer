"""Backend tests using a stub, so they need neither TensorFlow nor a model.

These lock down the part that is easy to get wrong and expensive to debug on
a GPU box: that decoded ink comes back in *source image* coordinates, not
padded-crop coordinates.
"""

import pytest
from PIL import Image

from ink_tokenizer.base import DerenderBackend
from ink_tokenizer.codec import tokenize, tokens_to_text
from ink_tokenizer.ink import Ink


class StubBackend(DerenderBackend):
    """Replays canned model output; records what it was asked to run."""

    def __init__(self, texts):
        self.texts = texts
        self.seen = []

    def _run(self, images, prompt, fallback):
        self.seen.append((len(images), prompt, fallback))
        for im in images:
            assert im.size == (224, 224), "backend must receive padded crops"
        return [{"text": t, "used_fallback": False} for t in self.texts]


def _wire(points):
    return tokens_to_text(tokenize(Ink.from_point_lists([points])))


def test_result_is_mapped_back_to_source_coordinates():
    # 448x224 source -> ratio 0.5, dy=56. A point at (112, 112) in crop space
    # is (224, 112) in the source image.
    backend = StubBackend([_wire([(112, 112)])])
    result = backend.derender_image(Image.new("RGB", (448, 224)))
    assert result.ink[0].points == [(224.0, 112.0)]


def test_boxes_come_back_in_page_space():
    backend = StubBackend([_wire([(0, 0)]), _wire([(0, 0)])])
    page = Image.new("RGB", (1000, 800))
    results = backend.derender_boxes(page, [(100, 200, 324, 424), (500, 600, 724, 824)])
    assert results[0].ink[0].points == [(100.0, 200.0)]
    assert results[1].ink[0].points == [(500.0, 600.0)]


def test_merge_flattens_into_one_ink():
    backend = StubBackend([_wire([(0, 0)]), _wire([(1, 1)])])
    page = Image.new("RGB", (1000, 800))
    results = backend.derender_boxes(page, [(0, 0, 224, 224), (10, 10, 234, 234)])
    assert len(backend.merge(results)) == 2


def test_recognized_text_is_separated_from_ink():
    backend = StubBackend(["vol" + _wire([(1, 2)]) + "tage"])
    result = backend.derender_image(Image.new("RGB", (224, 224)))
    assert result.text == "voltage"
    assert result.ink.num_points == 1


def test_empty_crop_list_short_circuits():
    backend = StubBackend([])
    assert backend.derender_crops([]) == []
    assert backend.seen == []


def test_prompt_and_fallback_are_forwarded():
    backend = StubBackend([_wire([(0, 0)])])
    backend.derender_image(
        Image.new("RGB", (224, 224)), prompt="Derender the ink.", fallback=False
    )
    assert backend.seen == [(1, "Derender the ink.", False)]


def test_local_backend_without_tensorflow_explains_itself():
    """On Windows this must fail with guidance, not an opaque ImportError."""
    from ink_tokenizer.local import LocalBackend

    try:
        LocalBackend(eager=False)
    except ImportError as exc:
        assert "tensorflow-text" in str(exc) or "inference" in str(exc)
    else:
        pytest.skip("TensorFlow is available here")
