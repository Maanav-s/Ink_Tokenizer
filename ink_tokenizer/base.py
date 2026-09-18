"""Shared backend interface.

Two backends produce ink from images: `LocalBackend` (in-process TensorFlow,
Linux/macOS) and `InkSightClient` (HTTP, works anywhere). Everything except
the raw model call is shared and lives here, so switching backends never
changes the coordinates or tokens you get back.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from .codec import detokenize, strip_ink_tokens, text_to_tokens
from .ink import Ink
from .preprocess import PadTransform, crop_and_pad, scale_and_pad

PROMPT_RECOGNIZE_AND_DERENDER = "Recognize and derender."
PROMPT_DERENDER = "Derender the ink."


@dataclass
class DerenderResult:
    """One crop's worth of output.

    `ink` is in the source image's coordinate space (the pad transform has
    already been inverted). `text` is the recognized text, empty unless a
    recognizing prompt was used. `used_fallback` records whether the primary
    prompt returned no ink and the fallback prompt was retried.
    """

    ink: Ink
    text: str
    raw: str
    used_fallback: bool


class DerenderBackend:
    """Geometry + token decoding. Subclasses implement only `_run`."""

    def _run(
        self, images: list[Image.Image], prompt: str, fallback: bool
    ) -> list[dict]:
        """Return one ``{"text": str, "used_fallback": bool}`` per image.

        Images are padded 224x224 RGB.
        """
        raise NotImplementedError

    def derender_crops(
        self,
        crops: list[tuple[Image.Image, PadTransform]],
        *,
        prompt: str = PROMPT_RECOGNIZE_AND_DERENDER,
        fallback: bool = True,
    ) -> list[DerenderResult]:
        """Run already-padded 224x224 crops and map each result back."""
        if not crops:
            return []
        raw_results = self._run([c[0] for c in crops], prompt, fallback)
        out = []
        for (_, transform), res in zip(crops, raw_results):
            raw = res["text"]
            out.append(
                DerenderResult(
                    ink=transform.invert(detokenize(text_to_tokens(raw))),
                    text=strip_ink_tokens(raw),
                    raw=raw,
                    used_fallback=res.get("used_fallback", False),
                )
            )
        return out

    def derender_image(self, image: Image.Image, **kwargs) -> DerenderResult:
        """Run one word-level image. Returns ink in that image's coordinates."""
        return self.derender_crops([scale_and_pad(image)], **kwargs)[0]

    def derender_boxes(
        self,
        page: Image.Image,
        boxes: list[tuple[float, float, float, float]],
        **kwargs,
    ) -> list[DerenderResult]:
        """Run several crops of one page; every ink comes back in page space.

        Boxes must come from somewhere -- upstream uses an OCR word detector,
        which is a poor fit for math and diagrams. See docs/inksight.md.
        """
        return self.derender_crops([crop_and_pad(page, b) for b in boxes], **kwargs)

    @staticmethod
    def merge(results: list[DerenderResult]) -> Ink:
        """Flatten per-crop results into one page-level `Ink`."""
        merged = Ink()
        for r in results:
            merged.extend(r.ink)
        return merged

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()
