"""Ink Tokenizer -- structural autocomplete for handwritten engineering work.

Nothing imported here touches TensorFlow, so this package imports cleanly on
any platform. `LocalBackend` is resolved lazily (see `__getattr__`) because it
requires the `inference` extra, which is Linux/macOS only.
"""

from .base import (
    PROMPT_DERENDER,
    PROMPT_RECOGNIZE_AND_DERENDER,
    DerenderBackend,
    DerenderResult,
)
from .client import DEFAULT_ENDPOINT, InkSightClient
from .codec import (
    COORDINATE_LENGTH,
    START_TOKEN,
    VOCABULARY_SIZE,
    detokenize,
    strip_ink_tokens,
    text_to_tokens,
    tokenize,
    tokens_to_text,
)
from .ink import Ink, Stroke
from .preprocess import PadTransform, crop_and_pad, scale_and_pad
from .scribe import ScribeBPE, ScribeEncoding, ScribeToken, decode_scribe, encode_scribe

__all__ = [
    "Ink",
    "Stroke",
    "detokenize",
    "tokenize",
    "text_to_tokens",
    "tokens_to_text",
    "strip_ink_tokens",
    "COORDINATE_LENGTH",
    "START_TOKEN",
    "VOCABULARY_SIZE",
    "scale_and_pad",
    "crop_and_pad",
    "PadTransform",
    "DerenderBackend",
    "DerenderResult",
    "InkSightClient",
    "LocalBackend",
    "get_backend",
    "DEFAULT_ENDPOINT",
    "PROMPT_RECOGNIZE_AND_DERENDER",
    "PROMPT_DERENDER",
    "ScribeToken",
    "ScribeEncoding",
    "ScribeBPE",
    "encode_scribe",
    "decode_scribe",
]


def __getattr__(name: str):
    """Expose `LocalBackend` without importing TensorFlow at package import."""
    if name == "LocalBackend":
        from .local import LocalBackend

        return LocalBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_backend(endpoint: str | None = None, **kwargs) -> DerenderBackend:
    """Pick a backend: in-process if TensorFlow is available, else HTTP.

    Pass `endpoint` to force the HTTP client. With no endpoint this prefers
    `LocalBackend`, so the same script runs in-process on the lab machine and
    over the network from a laptop without a code change.
    """
    if endpoint is not None:
        return InkSightClient(endpoint, **kwargs)

    from .local import LocalBackend  # importable anywhere; TF loads on init

    try:
        return LocalBackend(**kwargs)
    except ImportError:
        # No TensorFlow here (e.g. Windows). Fall back to a local service.
        return InkSightClient(DEFAULT_ENDPOINT, **kwargs)
