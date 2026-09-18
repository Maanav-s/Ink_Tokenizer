"""The only module in this package that imports TensorFlow.

Importing it on a machine without TensorFlow raises ImportError, by design --
nothing else in `ink_tokenizer` pulls it in, so the codec and geometry stay
usable anywhere (notably Windows, where `tensorflow-text` has no wheels).

The XLA configuration is ported from google-research/inksight
`utils/tensorflow.py` (Apache-2.0, see NOTICE). The released checkpoint
predates TF 2.18's change to XLA GEMM lowering and autotuning; without these
flags, set *before* TensorFlow is first imported, TF >= 2.18 can silently
produce different numerics. That is why they are configured at module import,
above the TensorFlow import, and why nothing may import tensorflow earlier.
"""

from __future__ import annotations

import importlib
import os
import shlex

DEFAULT_MODEL = "Derendering/InkSight-Small-p"
FALLBACK_PROMPT = "Derender the ink."
TARGET_SIZE = 224

_INCOMPATIBLE_XLA_PASSES = (
    "custom-kernel-fusion-rewriter",
    "custom_kernel-fusion-autotuner",
)


def _configure_xla_flags() -> None:
    tokens = shlex.split(os.environ.get("XLA_FLAGS", ""))
    disabled_passes: list[str] = []
    preserved_flags: list[str] = []

    for token in tokens:
        if token.startswith("--xla_disable_hlo_passes="):
            disabled_passes.extend(token.partition("=")[2].split(","))
        elif not token.startswith("--xla_gpu_autotune_level="):
            preserved_flags.append(token)

    for pass_name in _INCOMPATIBLE_XLA_PASSES:
        if pass_name not in disabled_passes:
            disabled_passes.append(pass_name)

    preserved_flags.extend(
        (
            "--xla_gpu_autotune_level=0",
            "--xla_disable_hlo_passes=" + ",".join(filter(None, disabled_passes)),
        )
    )
    os.environ["XLA_FLAGS"] = shlex.join(preserved_flags)


_configure_xla_flags()

# Quieten TF's startup banner unless the user asked for it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# These modules must load after the XLA configuration.
tf = importlib.import_module("tensorflow")
tf_text = importlib.import_module("tensorflow_text")  # registers custom ops


def configure_gpu() -> list[str]:
    """Grow GPU memory on demand rather than reserving all of it up front.

    Matters on a shared lab machine, where claiming the whole card by default
    would evict everyone else.
    """
    names = []
    for gpu in tf.config.list_physical_devices("GPU"):
        names.append(gpu.name)
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # already initialized
    return names


class InkSightEngine:
    """Wraps the InkSight SavedModel. Not thread-safe."""

    def __init__(self, model: str = DEFAULT_MODEL, *, eager: bool = True) -> None:
        self.model = model
        self._signature = None
        self.gpus: list[str] = []
        if eager:
            self.load()

    def load(self):
        """Download (if needed) and load the SavedModel.

        Upstream's notebook uses `from_pretrained_keras`, which couples us to
        whichever Keras major version huggingface_hub expects. The HF repo
        root *is* a SavedModel directory, so loading it straight through TF
        avoids that coupling. A local directory path is accepted too, for
        air-gapped machines.
        """
        if self._signature is None:
            self.gpus = configure_gpu()
            path = self.model
            if not os.path.isdir(path):
                from huggingface_hub import snapshot_download

                path = snapshot_download(self.model)
            loaded = tf.saved_model.load(path)
            self._signature = loaded.signatures["serving_default"]
        return self._signature

    @property
    def loaded(self) -> bool:
        return self._signature is not None

    def _as_tensor(self, image):
        """PIL RGB image -> uint8 (224, 224, 3) tensor."""
        if image.size != (TARGET_SIZE, TARGET_SIZE):
            raise ValueError(
                f"expected a {TARGET_SIZE}x{TARGET_SIZE} crop, got {image.size}; "
                "pad with ink_tokenizer.preprocess.scale_and_pad"
            )
        raw = bytearray(image.convert("RGB").tobytes())
        return tf.reshape(
            tf.constant(raw, dtype=tf.uint8), (TARGET_SIZE, TARGET_SIZE, 3)
        )

    def infer(self, tensors: list, prompt: str) -> list[str]:
        """Run one batch of uint8 image tensors, returning raw output text.

        The serving signature takes JPEG bytes shaped (batch, 1), matching
        upstream's batch path.
        """
        sig = self.load()
        encoded = tf.stack([tf.reshape(tf.io.encode_jpeg(t), (1,)) for t in tensors])
        text = tf.constant([prompt] * len(tensors), dtype=tf.string)
        out = sig(**{"input_text": text, "image/encoded": encoded})
        return [out["output_0"].numpy()[i][0].decode() for i in range(len(tensors))]

    def run(
        self,
        images: list,
        prompt: str,
        fallback: bool = True,
        batch_size: int = 32,
    ) -> list[dict]:
        """Run PIL crops in batches, retrying empty results with the fallback.

        The recognizing prompt intermittently returns no ink at all;
        upstream's remedy is to retry those with the plain derender prompt.
        """
        tensors = [self._as_tensor(im) for im in images]
        texts: list[str] = []
        for i in range(0, len(tensors), batch_size):
            texts.extend(self.infer(tensors[i : i + batch_size], prompt))

        used_fallback = [False] * len(texts)
        if fallback and prompt != FALLBACK_PROMPT:
            retry = [i for i, t in enumerate(texts) if "<ink_token_" not in t]
            for i in range(0, len(retry), batch_size):
                chunk = retry[i : i + batch_size]
                for j, text in zip(
                    chunk, self.infer([tensors[k] for k in chunk], FALLBACK_PROMPT)
                ):
                    texts[j] = text
                    used_fallback[j] = True

        return [{"text": t, "used_fallback": f} for t, f in zip(texts, used_fallback)]
