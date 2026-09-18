"""HTTP client for the InkSight inference service.

Works on any platform, including Windows. Geometry and token decoding happen
here (inherited from `DerenderBackend`); the service only ever returns raw
model text.
"""

from __future__ import annotations

import io

import httpx
from PIL import Image

from .base import DerenderBackend

DEFAULT_ENDPOINT = "http://127.0.0.1:8000"


class InkSightClient(DerenderBackend):
    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        *,
        timeout: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)

    def health(self) -> dict:
        r = self._client.get(f"{self.endpoint}/health")
        r.raise_for_status()
        return r.json()

    def _run(
        self, images: list[Image.Image], prompt: str, fallback: bool
    ) -> list[dict]:
        files = []
        for i, img in enumerate(images):
            buf = io.BytesIO()
            img.save(buf, format="PNG")  # lossless on the wire; the service
            files.append(  # re-encodes to JPEG for the model
                ("images", (f"{i}.png", buf.getvalue(), "image/png"))
            )
        r = self._client.post(
            f"{self.endpoint}/derender",
            files=files,
            data={"prompt": prompt, "fallback": str(fallback).lower()},
        )
        r.raise_for_status()
        return r.json()["results"]

    def close(self) -> None:
        self._client.close()
