"""The dataset adapter contract.

One adapter == one on-disk layout. Adapters know how to *enumerate* and *read*
samples; they know nothing about training, batching, class indices or PyTorch.

The contract that makes the whole framework pluggable
-----------------------------------------------------
``Sample.identity`` is a **raw string** -- never an integer class index.
:class:`~frs.data.class_map.ClassMap`, not the adapter, assigns indices. This
single decision is what lets a future dataset extend an already-trained
classifier head instead of forcing a restart from scratch: two adapters that
emit the same identity string refer to the same person, and a brand-new
identity string simply gets appended to the map.

Adding a new format
-------------------
1. Write ``frs/data/adapters/my_format.py`` with ``@ADAPTERS.register("my_format")``.
2. Add one import line to ``frs/data/adapters/__init__.py``.
3. Point YAML at it: ``data.adapter.type: my_format``.

No change to trainer.py, dataset.py or heads.py. See ``docs/ADAPTERS.md``.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class Sample:
    """One training image.

    Attributes
    ----------
    path:
        Absolute filesystem path, or an adapter-specific URI such as
        ``"rec://1234"`` for packed formats. Only the owning adapter's
        :meth:`DatasetAdapter.load_image` needs to understand it.
    identity:
        The raw identity key as a string (e.g. ``"7276470@N03_identity_3"``).
        Never an integer -- see the module docstring.
    meta:
        Optional per-sample extras (bbox, quality score, glasses flag, ...).
        Carried through untouched; adapters may use it however they like.
    """

    path: str
    identity: str
    meta: dict[str, Any] | None = field(default=None, compare=False)


class DatasetAdapter(ABC):
    """Base class for all dataset adapters."""

    #: Subclasses may override to declare the extensions they handle.
    default_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, prefix: str = "") -> None:
        """
        Parameters
        ----------
        prefix:
            Namespace prepended to every identity, e.g. ``"client_a/"``. Use it
            when merging multiple sources whose identity keys could collide but
            refer to *different* people. Leave empty when identity keys are
            globally meaningful.
        """
        self.prefix = prefix

    # ---------------------------------------------------------------- scanning

    @abstractmethod
    def scan(self) -> list[Sample]:
        """Enumerate every sample in the dataset.

        Called once per run; the result is cached to disk keyed by
        :attr:`cache_key`. Implementations should apply :meth:`_make_identity`
        so that ``prefix`` is honoured consistently.
        """

    @property
    @abstractmethod
    def cache_key(self) -> str:
        """A stable hash of this adapter's configuration.

        Changing any option that would change :meth:`scan`'s output must change
        this key, or a stale cache will be silently reused.
        """

    def _hash_config(self, **parts: Any) -> str:
        """Helper for building :attr:`cache_key` from arbitrary config values."""
        blob = json.dumps(
            {"adapter": type(self).__name__, "prefix": self.prefix, **parts},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def _make_identity(self, raw: str) -> str:
        return f"{self.prefix}{raw}"

    # ----------------------------------------------------------------- loading

    def load_image(self, sample: Sample) -> np.ndarray:
        """Return the image as an RGB ``HWC uint8`` array.

        The default reads ``sample.path`` with PIL. Packed formats (.rec, LMDB,
        WebDataset) override this and interpret their own URI scheme.
        """
        with Image.open(sample.path) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    # ---------------------------------------------------------------- reporting

    @staticmethod
    def summarize(samples: Iterable[Sample]) -> dict[str, Any]:
        """Dataset statistics used by logging and the training report."""
        counts = Counter(s.identity for s in samples)
        if not counts:
            return {
                "num_samples": 0,
                "num_identities": 0,
                "min_per_identity": 0,
                "max_per_identity": 0,
                "mean_per_identity": 0.0,
                "median_per_identity": 0.0,
            }
        per_id = np.array(sorted(counts.values()))
        return {
            "num_samples": int(per_id.sum()),
            "num_identities": len(counts),
            "min_per_identity": int(per_id.min()),
            "max_per_identity": int(per_id.max()),
            "mean_per_identity": float(per_id.mean()),
            "median_per_identity": float(np.median(per_id)),
            "identity_counts": counts,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(prefix={self.prefix!r})"


def filter_by_min_images(
    samples: list[Sample], min_images: int
) -> tuple[list[Sample], int]:
    """Drop identities with fewer than ``min_images`` samples.

    Identities with a single image contribute a classifier row that can never be
    discriminatively trained, and they inflate the class count for no benefit.
    Returns the filtered samples and the number of identities removed.
    """
    if min_images <= 1:
        return samples, 0
    counts = Counter(s.identity for s in samples)
    keep = {ident for ident, n in counts.items() if n >= min_images}
    dropped = len(counts) - len(keep)
    if dropped == 0:
        return samples, 0
    return [s for s in samples if s.identity in keep], dropped


def exclude_identities(
    samples: list[Sample], excluded: set[str]
) -> tuple[list[Sample], int]:
    """Remove every sample whose identity is in ``excluded``.

    This implements the held-out validation split. Excluding identities from the
    *training adapter* -- not merely from the pair list -- is what makes the
    reported verification accuracy meaningful rather than a memorisation score.
    """
    if not excluded:
        return samples, 0
    kept = [s for s in samples if s.identity not in excluded]
    return kept, len(samples) - len(kept)
