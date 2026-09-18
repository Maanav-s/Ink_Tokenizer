"""Geometry tests: padding must be exactly invertible back to source coords."""

from PIL import Image

from ink_tokenizer.ink import Ink
from ink_tokenizer.preprocess import TARGET_SIZE, crop_and_pad, scale_and_pad


def test_scale_and_pad_produces_224_square():
    padded, _ = scale_and_pad(Image.new("RGB", (800, 100)))
    assert padded.size == (TARGET_SIZE, TARGET_SIZE)


def test_pad_transform_inverts_to_source_corners():
    src = Image.new("RGB", (448, 224))
    _, tf_ = scale_and_pad(src)
    # ratio 0.5, image becomes 224x112, centred vertically (dy=56).
    assert tf_.ratio == 0.5
    assert (tf_.dx, tf_.dy) == (0, 56)
    assert tf_.invert_point(0, 56) == (0.0, 0.0)
    assert tf_.invert_point(224, 168) == (448.0, 224.0)


def test_invert_roundtrips_an_ink():
    _, tf_ = scale_and_pad(Image.new("RGB", (448, 224)))
    ink = Ink.from_point_lists([[(0, 56), (112, 112)]])
    assert tf_.invert(ink)[0].points == [(0.0, 0.0), (224.0, 112.0)]


def test_crop_and_pad_maps_back_to_page_space():
    page = Image.new("RGB", (1000, 800))
    _, tf_ = crop_and_pad(page, (100, 200, 324, 424))  # 224x224 crop, ratio 1
    assert tf_.ratio == 1.0
    assert tf_.origin == (100.0, 200.0)
    assert tf_.invert_point(0, 0) == (100.0, 200.0)
    assert tf_.invert_point(224, 224) == (324.0, 424.0)


def test_corner_mean_padding_avoids_black_border():
    src = Image.new("RGB", (400, 100), (200, 190, 180))
    padded, _ = scale_and_pad(src, pad_black=False)
    assert padded.getpixel((2, 2)) == (200, 190, 180)
