"""InkSight inference service: a thin HTTP shell around `InkSightEngine`.

The engine lives in `ink_tokenizer.tf_runtime`, so this service and the
in-process `LocalBackend` share one implementation of the model call. Only run
this when the caller is on a different machine (e.g. a Windows laptop); on the
box that holds the model, use LocalBackend and skip the HTTP hop entirely.

Contract: padded 224x224 images in, raw model text out. No geometry and no
token decoding happen here -- those live client-side in `ink_tokenizer`.
"""

from __future__ import annotations

import io
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from ink_tokenizer.tf_runtime import DEFAULT_MODEL, InkSightEngine, tf

MODEL = os.environ.get("INKSIGHT_MODEL", DEFAULT_MODEL)
MAX_BATCH = int(os.environ.get("INKSIGHT_MAX_BATCH", "32"))

app = FastAPI(title="InkSight inference", version="0.1.0")
_engine: InkSightEngine | None = None


def _get_engine() -> InkSightEngine:
    global _engine
    if _engine is None:
        _engine = InkSightEngine(MODEL)
    return _engine


@app.on_event("startup")
def _startup() -> None:
    # Load up front so the first real request is not charged for a 692 MB
    # download plus graph load. Set INKSIGHT_EAGER_LOAD=0 for fast restarts.
    if os.environ.get("INKSIGHT_EAGER_LOAD", "1") == "1":
        _get_engine()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL,
        "loaded": _engine is not None and _engine.loaded,
        "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
    }


@app.post("/derender")
async def derender(
    images: list[UploadFile] = File(...),
    prompt: str = Form("Recognize and derender."),
    fallback: bool = Form(True),
) -> dict:
    if not images:
        raise HTTPException(status_code=400, detail="no images provided")
    if len(images) > MAX_BATCH:
        raise HTTPException(
            status_code=413, detail=f"batch of {len(images)} exceeds {MAX_BATCH}"
        )

    crops = []
    for upload in images:
        try:
            crops.append(Image.open(io.BytesIO(await upload.read())).convert("RGB"))
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"unreadable image {upload.filename}: {exc}"
            ) from exc

    try:
        results = _get_engine().run(
            crops, prompt, fallback=fallback, batch_size=MAX_BATCH
        )
    except ValueError as exc:  # wrong crop size
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"results": results, "prompt": prompt}
