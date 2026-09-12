"""The *streaming* adapter contract, for datasets too large to enumerate.

Why a second contract
---------------------
:class:`~frs.data.adapters.base.DatasetAdapter` answers "what samples exist?"
with a list, and "give me sample *i*" with a random read. That is the right
shape for a folder of JPEGs or a ``.rec`` pack with an index. It is the wrong
shape for a 17-million-image dataset packed as gzip tar shards: gzip is not
seekable, and a Python list of 17M ``Sample`` objects costs several GB per
DataLoader worker.

A :class:`StreamingAdapter` instead answers two different questions:

1. :meth:`census` -- how many images of each identity exist? One sequential pass
   over the shards reading only the label entries, cached to disk. From this
   the :class:`~frs.data.class_map.ClassMap`, ``__len__`` and the report's
   dataset statistics are derived without ever listing samples.
2. :meth:`iter_raw` -- stream ``(encoded_image_bytes, raw_id)`` pairs for one
   worker of one rank for one epoch. Filtering, decoding and augmentation happen
   downstream in :class:`~frs.data.iterable_dataset.FaceIterableDataset`.

The one rule that matters is unchanged: **identities are strings**, produced by
:meth:`identity_of`, and the ``ClassMap`` -- not the adapter -- assigns indices.
That is what keeps a streamed checkpoint extensible and resumable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

CENSUS_FORMAT_VERSION = 1

EPOCH_MODES = ("natural", "resampled")
HANDLERS = ("warn", "reraise")


# ------------------------------------------------------------------- census


@dataclass
class Census:
    """Per-identity image counts for a streamed dataset.

    ``raw_ids`` are the dataset's own label values (integers for Glint360K),
    sorted ascending; ``counts[i]`` is the number of images of ``raw_ids[i]``.
    """

    raw_ids: np.ndarray
    counts: np.ndarray
    total: int
    shards: list[str]

    def __post_init__(self) -> None:
        self.raw_ids = np.asarray(self.raw_ids, dtype=np.int64)
        self.counts = np.asarray(self.counts, dtype=np.int64)
        if self.raw_ids.shape != self.counts.shape:
            raise ValueError("census raw_ids and counts differ in length")
        self.total = int(self.total)
        self.shards = list(self.shards)

    @property
    def num_identities(self) -> int:
        return int(self.raw_ids.size)

    @property
    def num_shards(self) -> int:
        return len(self.shards)

    def kept_mask(self, min_images: int) -> np.ndarray:
        """Boolean mask of identities with at least ``min_images`` images."""
        if min_images <= 1:
            return np.ones(self.raw_ids.shape, dtype=bool)
        return self.counts >= int(min_images)

    # --------------------------------------------------------- serialisation

    def to_json(self, path: str | os.PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": CENSUS_FORMAT_VERSION,
            "total": self.total,
            "num_identities": self.num_identities,
            "shards": self.shards,
            "raw_ids": self.raw_ids.tolist(),
            "counts": self.counts.tolist(),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        tmp.replace(path)

    @classmethod
    def from_json(cls, path: str | os.PathLike) -> "Census":
        with Path(path).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        version = int(data.get("format_version", 0))
        if version > CENSUS_FORMAT_VERSION:
            raise RuntimeError(
                f"census format v{version} is newer than this code supports"
            )
        return cls(
            raw_ids=np.asarray(data["raw_ids"], dtype=np.int64),
            counts=np.asarray(data["counts"], dtype=np.int64),
            total=int(data["total"]),
            shards=list(data.get("shards", [])),
        )

    @classmethod
    def merge(cls, parts: Iterable[tuple[np.ndarray, np.ndarray]], shards: Sequence[str]) -> "Census":
        """Combine per-shard ``(unique_ids, counts)`` pairs into one census."""
        acc: dict[int, int] = {}
        for ids, counts in parts:
            for raw, n in zip(ids.tolist(), counts.tolist()):
                acc[raw] = acc.get(raw, 0) + int(n)
        if not acc:
            raise RuntimeError("census found no labelled samples in any shard")
        raw_ids = np.fromiter(sorted(acc), dtype=np.int64, count=len(acc))
        counts = np.fromiter((acc[int(r)] for r in raw_ids), dtype=np.int64, count=len(acc))
        return cls(raw_ids=raw_ids, counts=counts, total=int(counts.sum()), shards=list(shards))

    def summary(self) -> dict[str, Any]:
        if self.num_identities == 0:
            return {
                "num_samples": 0,
                "num_identities": 0,
                "min_per_identity": 0,
                "max_per_identity": 0,
                "mean_per_identity": 0.0,
                "median_per_identity": 0.0,
            }
        return {
            "num_samples": self.total,
            "num_identities": self.num_identities,
            "min_per_identity": int(self.counts.min()),
            "max_per_identity": int(self.counts.max()),
            "mean_per_identity": float(self.counts.mean()),
            "median_per_identity": float(np.median(self.counts)),
        }


@dataclass
class FilterResult:
    """What survived ``min_images_per_identity`` and the exclusion file."""

    identities: list[str]      # sorted identity strings (the ClassMap order)
    raw_ids: np.ndarray        # raw id per identity, same order as ``identities``
    counts: np.ndarray         # images per identity, same order
    total: int                 # sum of counts
    dropped_min_images: int
    dropped_excluded: int


# ------------------------------------------------------------------ adapter


class StreamingAdapter(ABC):
    """Base class for adapters over sequential, non-indexable data.

    Parameters
    ----------
    prefix:
        Namespace prepended to every identity (see
        :class:`~frs.data.adapters.base.DatasetAdapter`).
    min_images_per_identity:
        Drop identities with fewer images. Applied from the census, so changing
        it never re-streams the data.
    exclude_identities_file:
        Text file of identities to hold out entirely (the validation split).
    epoch_mode:
        ``natural`` -- one pass over every shard per epoch (single process only).
        ``resampled`` -- shards drawn with replacement and the stream cut to
        exactly ``samples_per_epoch`` (required under DDP so every rank runs
        the same number of steps).
    samples_per_epoch:
        Length of a nominal epoch in ``resampled`` mode; ``None`` uses the
        census total after filtering.
    shuffle_buffer:
        Size of the per-worker reservoir that shuffles *raw* samples before
        decoding. ``0`` or ``1`` disables shuffling.
    handler:
        ``warn`` skips a corrupt sample or shard with a warning; ``reraise``
        stops the run.
    """

    def __init__(
        self,
        prefix: str = "",
        min_images_per_identity: int = 0,
        exclude_identities_file: str | None = None,
        epoch_mode: str = "natural",
        samples_per_epoch: int | None = None,
        shuffle_buffer: int = 2000,
        handler: str = "warn",
    ) -> None:
        if epoch_mode not in EPOCH_MODES:
            raise ValueError(f"epoch_mode must be one of {EPOCH_MODES}; got {epoch_mode!r}")
        if handler not in HANDLERS:
            raise ValueError(f"handler must be one of {HANDLERS}; got {handler!r}")
        self.prefix = prefix
        self.min_images_per_identity = int(min_images_per_identity)
        self.exclude_identities_file = exclude_identities_file
        self.epoch_mode = epoch_mode
        self.samples_per_epoch = None if samples_per_epoch is None else int(samples_per_epoch)
        self.shuffle_buffer = int(shuffle_buffer)
        self.handler = handler

    # ------------------------------------------------------------ contract

    @property
    @abstractmethod
    def cache_key(self) -> str:
        """Stable hash of everything that changes :meth:`census`'s output."""

    @abstractmethod
    def shard_list(self) -> list[str]:
        """Resolved, ordered list of shard paths or URLs."""

    @abstractmethod
    def census(self, workers: int = 4, progress: bool = True) -> Census:
        """Count images per identity with one pass over the label entries."""

    @abstractmethod
    def iter_raw(
        self,
        *,
        epoch: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        worker: int = 0,
        num_workers: int = 1,
        shuffle: bool = True,
    ) -> Iterator[tuple[bytes, int]]:
        """Stream ``(encoded_image, raw_id)`` for this worker's share of the epoch.

        In ``natural`` mode the stream ends after one pass over the shards
        assigned to this worker; in ``resampled`` mode it is infinite and the
        caller cuts it.
        """

    # ------------------------------------------------------------- helpers

    def _hash_config(self, **parts: Any) -> str:
        blob = json.dumps(
            {"adapter": type(self).__name__, "prefix": self.prefix, **parts},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def identity_of(self, raw_id: int) -> str:
        return f"{self.prefix}{int(raw_id)}"

    def decode(self, blob: bytes) -> np.ndarray:
        """Encoded image bytes -> RGB ``HWC uint8``."""
        import io

        with Image.open(io.BytesIO(blob)) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    def load_excluded(self) -> set[str]:
        if not self.exclude_identities_file:
            return set()
        path = Path(self.exclude_identities_file)
        if not path.is_file():
            raise FileNotFoundError(
                f"exclude_identities_file not found: {path}. Generate it with "
                f"scripts/scan_webdataset.py --holdout before training."
            )
        with path.open("r", encoding="utf-8") as fh:
            return {
                line.strip() for line in fh if line.strip() and not line.startswith("#")
            }

    def resolve_filters(self, census: Census) -> FilterResult:
        """Apply ``min_images_per_identity`` and the exclusion file to a census."""
        keep = census.kept_mask(self.min_images_per_identity)
        dropped_min = int((~keep).sum())

        excluded = self.load_excluded()
        dropped_excluded = 0
        if excluded:
            # The file may hold bare or prefixed identities; tolerate both.
            expanded = excluded | {f"{self.prefix}{e}" for e in excluded}
            identities_all = [self.identity_of(r) for r in census.raw_ids.tolist()]
            ex_mask = np.fromiter(
                (ident in expanded for ident in identities_all), dtype=bool, count=len(identities_all)
            )
            dropped_excluded = int((keep & ex_mask).sum())
            if not ex_mask.any():
                logger.warning(
                    "exclude_identities_file listed %d identities but none appear in "
                    "the census of the configured shards -- check the adapter prefix "
                    "(%r) and that the file matches this dataset",
                    len(excluded),
                    self.prefix,
                )
            keep &= ~ex_mask

        raw_kept = census.raw_ids[keep]
        counts_kept = census.counts[keep]
        if raw_kept.size == 0:
            raise RuntimeError(
                "no identities survive the filters "
                f"(min_images_per_identity={self.min_images_per_identity}, "
                f"excluded={len(excluded)}). Lower min_images_per_identity or "
                "configure more shards."
            )

        identities = [self.identity_of(r) for r in raw_kept.tolist()]
        # ClassMap order is lexicographic (ClassMap.from_samples sorts strings);
        # keep raw ids / counts aligned with that order.
        order = sorted(range(len(identities)), key=lambda i: identities[i])
        identities = [identities[i] for i in order]
        raw_kept = raw_kept[order]
        counts_kept = counts_kept[order]
        return FilterResult(
            identities=identities,
            raw_ids=raw_kept,
            counts=counts_kept,
            total=int(counts_kept.sum()),
            dropped_min_images=dropped_min,
            dropped_excluded=dropped_excluded,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(prefix={self.prefix!r}, epoch_mode={self.epoch_mode!r})"


# ---------------------------------------------------------------- utilities


def cached_census(
    adapter: StreamingAdapter,
    cache_dir: str | os.PathLike | None,
    workers: int = 4,
    explicit_path: str | os.PathLike | None = None,
    force: bool = False,
) -> Census:
    """Run ``adapter.census()`` once and memoise it as JSON.

    Mirrors :func:`frs.data.dataset._cached_scan`: the cache is keyed by the
    adapter's ``cache_key`` and a corrupt cache is ignored rather than fatal.
    """
    if explicit_path is not None:
        cache_path: Path | None = Path(explicit_path)
    elif cache_dir:
        cache_path = Path(cache_dir) / f"census_{adapter.cache_key}.json"
    else:
        cache_path = None

    if cache_path is not None and cache_path.is_file() and not force:
        try:
            census = Census.from_json(cache_path)
            logger.info(
                "Loaded census from %s: %s images, %s identities, %d shards",
                cache_path, f"{census.total:,}", f"{census.num_identities:,}", census.num_shards,
            )
            return census
        except Exception as exc:  # never fatal
            logger.warning("Ignoring unreadable census cache %s: %s", cache_path, exc)

    census = adapter.census(workers=workers)
    if cache_path is not None:
        census.to_json(cache_path)
        logger.info("Cached census to %s", cache_path)
    return census


def buffered_shuffle(items: Iterable[Any], bufsize: int, seed: str | int) -> Iterator[Any]:
    """Reservoir shuffle: hold ``bufsize`` items and emit a random one each step.

    Perfect shuffling of a 17M-sample stream is impossible without an index;
    this is the standard approximation. It is applied to *raw bytes*, before
    decoding, so the buffer costs ~8 KB per item rather than ~150 KB.
    """
    if bufsize <= 1:
        yield from items
        return
    rng = random.Random(seed)
    buf: list[Any] = []
    for item in items:
        buf.append(item)
        if len(buf) >= bufsize:
            i = rng.randrange(len(buf))
            buf[i], buf[-1] = buf[-1], buf[i]
            yield buf.pop()
    rng.shuffle(buf)
    yield from buf
