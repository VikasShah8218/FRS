"""InsightFace ``.bin`` verification packs (LFW, CFP-FP, AgeDB-30, ...).

These are the files every published face-recognition number is measured on,
distributed with the InsightFace training sets (``lfw.bin``, ``cfp_fp.bin``,
``agedb_30.bin`` ...). The format is a Python-2 pickle of::

    (bins, issame_list)

where ``bins`` is a list of encoded JPEG byte strings (``2 * num_pairs`` of
them, pair *i* being images ``2i`` and ``2i + 1``) and ``issame_list`` holds
one bool per pair. Every image is already aligned to the 112x112 ArcFace
template, so it flows through the eval transform with no cropping.

The loader returns the same ``(images, index_pairs, is_same)`` triple as
:func:`frs.eval.pairs.load_pair_images`, so
:func:`frs.eval.verification.evaluate_target` needs no changes.

Config::

    eval:
      targets:
        - {name: lfw, type: bin, path: /mnt/data/eval/lfw.bin}
        - {name: cfp_fp, type: bin, path: /mnt/data/eval/cfp_fp.bin, max_pairs: 1000}
"""

from __future__ import annotations

import io
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def read_bin(path: str | Path) -> tuple[list[bytes], list[bool]]:
    """Unpickle a ``.bin`` pack into ``(encoded_images, is_same_per_pair)``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"eval pack not found: {path}. InsightFace .bin packs ship with the "
            f"MS1MV3 / Glint360K training sets (see docs/AWS.md)."
        )
    with path.open("rb") as fh:
        try:
            payload = pickle.load(fh, encoding="bytes")
        except UnicodeDecodeError:
            fh.seek(0)
            payload = pickle.load(fh, encoding="latin1")

    if not (isinstance(payload, (tuple, list)) and len(payload) == 2):
        raise ValueError(f"{path}: expected a (bins, issame_list) pair")
    bins, issame = payload
    bins = list(bins)
    issame = [bool(x) for x in np.asarray(issame).reshape(-1).tolist()]
    if len(bins) != 2 * len(issame):
        raise ValueError(
            f"{path}: {len(bins)} images do not form {len(issame)} pairs "
            f"(expected {2 * len(issame)} images)"
        )
    # Some packs store numpy uint8 arrays instead of bytes; normalise.
    normalised: list[bytes] = []
    for blob in bins:
        if isinstance(blob, (bytes, bytearray)):
            normalised.append(bytes(blob))
        else:
            normalised.append(np.asarray(blob, dtype=np.uint8).tobytes())
    return normalised, issame


def decode_image(blob: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(blob)) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def load_bin_images(
    target: dict[str, Any], transform: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a ``.bin`` pack into ``(images, index_pairs, is_same)``.

    ``target.max_pairs`` (optional) truncates the protocol -- useful for smoke
    runs; never report a truncated number as the benchmark result.
    """
    bins, issame = read_bin(target["path"])

    max_pairs = int(target.get("max_pairs", 0) or 0)
    if max_pairs and max_pairs < len(issame):
        issame = issame[:max_pairs]
        bins = bins[: 2 * max_pairs]
        logger.warning(
            "%s: evaluating only the first %d pairs (max_pairs)", target.get("name"), max_pairs
        )

    stack = [transform(decode_image(blob)) for blob in bins]
    images = np.stack(stack).astype(np.float32)
    index_pairs = np.arange(len(bins), dtype=np.int64).reshape(-1, 2)
    is_same = np.asarray(issame, dtype=bool)
    return images, index_pairs, is_same
