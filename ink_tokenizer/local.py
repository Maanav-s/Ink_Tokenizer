"""In-process backend. Linux/macOS only -- requires the `inference` extra.

Preferred over the HTTP client when you are on the machine that holds the
model: no serialization, no request overhead, and batches stay whole. Use
`InkSightClient` instead when the model lives somewhere else (for example
driving a lab box from a Windows laptop).
"""

from __future__ import annotations

from PIL import Image

from .base import DerenderBackend

_MISSING_TF = (
    "LocalBackend needs TensorFlow. Install the inference extra with "
    "`uv sync --extra inference`. That extra resolves to nothing on Windows, "
    "where tensorflow-text ships no wheels -- run the model on Linux and use "
    "InkSightClient to reach it over HTTP."
)


class LocalBackend(DerenderBackend):
    def __init__(
        self,
        model: str | None = None,
        *,
        batch_size: int = 32,
        eager: bool = True,
    ) -> None:
        try:
            from .tf_runtime import DEFAULT_MODEL, InkSightEngine
        except ImportError as exc:  # pragma: no cover - platform dependent
            raise ImportError(_MISSING_TF) from exc
        self.engine = InkSightEngine(model or DEFAULT_MODEL, eager=eager)
        self.batch_size = batch_size

    @property
    def gpus(self) -> list[str]:
        return self.engine.gpus

    def _run(
        self, images: list[Image.Image], prompt: str, fallback: bool
    ) -> list[dict]:
        return self.engine.run(
            images, prompt, fallback=fallback, batch_size=self.batch_size
        )
