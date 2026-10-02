"""Download the teacher's pretrained weights into models/teacher/.

The checkpoint comes from pytorch-handwriting-synthesis-toolkit (MIT), pinned to
one commit and checked against known hashes. models/ is gitignored.

Usage: python scripts/fetch_teacher.py
"""
import hashlib
import os
import urllib.request

from teacher_model import CHECKPOINT, default_checkpoint_dir

COMMIT = "899e9432dfe0998ba2ef31582d11c2d5f24a5185"
BASE = f"https://raw.githubusercontent.com/X-rayLaser/pytorch-handwriting-synthesis-toolkit/{COMMIT}/checkpoints"
SHA256 = {
    "model.pt": "29dabc00f41ee0a97aea5cbf29c2a6b1ae73cc4b83864babb4fa0bd7385be73b",
    "meta.json": "00c383051895af5a091743652597dc4a9c456c70d07ec7ca51a96fcad6e6f70c",
}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fetch(out_dir=None):
    out_dir = out_dir or default_checkpoint_dir()
    os.makedirs(out_dir, exist_ok=True)
    for name, digest in SHA256.items():
        path = os.path.join(out_dir, name)
        if os.path.exists(path) and sha256(path) == digest:
            continue
        urllib.request.urlretrieve(f"{BASE}/{CHECKPOINT}/{name}", path)
        if sha256(path) != digest:
            os.remove(path)
            raise RuntimeError(f"checksum mismatch for {name}")
    print(f"teacher checkpoint in {out_dir}")


if __name__ == "__main__":
    fetch()
