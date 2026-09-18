"""Crop -> 224x224 padding, and the inverse mapping back to page coordinates.

This lives on the client side on purpose. The model only ever sees a padded
224x224 crop, so every coordinate the pipeline produces has to be mapped back
through this transform. Keeping it here (pure PIL, no TensorFlow) means the
geometry is unit-testable in the main environment instead of inside the
inference container.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from .ink import Ink, Stroke

TARGET_SIZE = 224


@dataclass(frozen=True)
class PadTransform:
    """Maps padded-crop coordinates back to the source image.

    `ratio` is the resize factor applied to the source crop; `dx`/`dy` are the
    padding offsets inside the 224x224 canvas; `origin` is the crop's top-left
    corner in the coordinate space you ultimately want back (e.g. the full
    page). Inverting is: undo padding, undo scale, then re-add the origin.
    """

    ratio: float
    dx: int
    dy: int
    origin: tuple[float, float] = (0.0, 0.0)

    def invert_point(self, x: float, y: float) -> tuple[float, float]:
        ox, oy = self.origin
        return ((x - self.dx) / self.ratio + ox, (y - self.dy) / self.ratio + oy)

    def invert(self, ink: Ink) -> Ink:
        return Ink(
            [Stroke([self.invert_point(x, y) for x, y in s]) for s in ink.strokes]
        )


def scale_and_pad(
    image: Image.Image,
    *,
    pad_black: bool = True,
    origin: tuple[float, float] = (0.0, 0.0),
) -> tuple[Image.Image, PadTransform]:
    """Fit `image` into a 224x224 canvas, preserving aspect ratio.

    Mirrors InkSight's `scale_and_pad`: pad with black by default, otherwise
    with the mean of the four corner pixels (which blends better on a photo of
    a whiteboard than a hard black border).
    """
    image = image.convert("RGB")
    ratio = min(TARGET_SIZE / image.width, TARGET_SIZE / image.height)
    resized = image.resize(
        (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
    )

    if pad_black:
        color: tuple[int, int, int] = (0, 0, 0)
    else:
        w, h = image.width - 1, image.height - 1
        corners = [
            image.getpixel((0, 0)),
            image.getpixel((w, 0)),
            image.getpixel((0, h)),
            image.getpixel((w, h)),
        ]
        color = tuple(round(sum(c[i] for c in corners) / 4) for i in range(3))

    canvas = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), color)
    dx = (TARGET_SIZE - resized.width) // 2
    dy = (TARGET_SIZE - resized.height) // 2
    canvas.paste(resized, (dx, dy))
    return canvas, PadTransform(ratio=ratio, dx=dx, dy=dy, origin=origin)


def crop_and_pad(
    page: Image.Image, box: tuple[float, float, float, float], **kwargs
) -> tuple[Image.Image, PadTransform]:
    """Crop `box` (left, top, right, bottom) from `page` and pad it to 224x224.

    The returned transform maps straight back to page coordinates, so inks
    from many crops can be merged into one page-level `Ink`.
    """
    left, top, right, bottom = box
    crop = page.crop((int(left), int(top), int(right), int(bottom)))
    return scale_and_pad(crop, origin=(float(left), float(top)), **kwargs)
