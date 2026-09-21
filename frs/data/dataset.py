"""The torch Dataset that ties an adapter, a ClassMap and a transform together.

The adapter supplies string identities; the ClassMap turns them into the integer
targets the loss needs. Scan results are cached to disk so a 5.8M-image scan is
paid once rather than every run.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..registry import ADAPTERS
from .adapters.base import DatasetAdapter, Sample
from .adapters.streaming import StreamingAdapter
from .class_map import ClassMap
from .iterable_dataset import FaceIterableDataset, InMemoryDataset  # noqa: F401  re-export
from .transforms import FaceTransform

logger = logging.getLogger(__name__)


def _cached_scan(adapter: DatasetAdapter, cache_dir: str | None) -> list[Sample]:
    """Run ``adapter.scan()``, memoised on disk by the adapter's cache key."""
    if not cache_dir:
        return adapter.scan()

    cache_path = Path(cache_dir) / f"scan_{adapter.cache_key}.pkl"
    if cache_path.is_file():
        try:
            with cache_path.open("rb") as fh:
                samples = pickle.load(fh)
            logger.info("Loaded %d samples from scan cache %s", len(samples), cache_path)
            return samples
        except Exception as exc:  # corrupt cache must never be fatal
            logger.warning("Ignoring unreadable scan cache %s: %s", cache_path, exc)

    samples = adapter.scan()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(samples, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache_path)
    logger.info("Cached %d samples to %s", len(samples), cache_path)
    return samples


class FaceTrainDataset(Dataset):
    """Yields ``(image_tensor, class_index)`` pairs.

    Parameters
    ----------
    adapter:
        Any :class:`DatasetAdapter`. Supplies paths and string identities.
    class_map:
        Identity -> index mapping. If omitted, one is built from the samples.
        Pass an existing map when resuming or extending so indices stay stable.
    transform:
        Preprocessing pipeline.
    cache_dir:
        Where to memoise scan results. ``None`` disables caching.
    strict_labels:
        If True (default), a sample whose identity is not in the class map is an
        error. Set False to silently drop such samples instead.
    """

    def __init__(
        self,
        adapter: DatasetAdapter,
        class_map: ClassMap | None = None,
        transform: FaceTransform | None = None,
        cache_dir: str | None = None,
        strict_labels: bool = True,
    ) -> None:
        self.adapter = adapter
        self.transform = transform
        self.samples: list[Sample] = _cached_scan(adapter, cache_dir)
        if not self.samples:
            raise RuntimeError(f"{adapter!r} produced no samples")

        self.class_map = class_map or ClassMap.from_samples(self.samples)

        unknown = {
            s.identity for s in self.samples if s.identity not in self.class_map
        }
        if unknown:
            if strict_labels:
                raise KeyError(
                    f"{len(unknown)} identities are not in the class map, e.g. "
                    f"{sorted(unknown)[:3]}. Call ClassMap.extend() (and extend the "
                    f"head) before training on new identities."
                )
            self.samples = [s for s in self.samples if s.identity in self.class_map]
            logger.warning("Dropped %d samples with unknown identities", len(unknown))

        # Precompute targets once; per-item dict lookups on 5.8M samples add up.
        self.targets = np.fromiter(
            (self.class_map.index_of(s.identity) for s in self.samples),
            dtype=np.int64,
            count=len(self.samples),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        try:
            img = self.adapter.load_image(sample)
        except Exception as exc:
            raise RuntimeError(f"failed to load {sample.path}: {exc}") from exc

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = img.astype(np.float32).transpose(2, 0, 1) / 255.0

        return torch.from_numpy(np.ascontiguousarray(img)), int(self.targets[index])

    # ------------------------------------------------------------- introspection

    @property
    def num_classes(self) -> int:
        return self.class_map.num_classes

    @property
    def identities(self) -> list[str]:
        """Sorted identity strings present in this dataset."""
        return sorted({s.identity for s in self.samples})

    def reindex(self, class_map: ClassMap, strict: bool = True) -> None:
        """Recompute ``targets`` against a (typically extended) class map."""
        if not strict:
            self.samples = [s for s in self.samples if s.identity in class_map]
        self.class_map = class_map
        self.targets = np.fromiter(
            (class_map.index_of(s.identity) for s in self.samples),
            dtype=np.int64,
            count=len(self.samples),
        )

    def summary(self) -> dict[str, Any]:
        stats = DatasetAdapter.summarize(self.samples)
        stats.pop("identity_counts", None)
        stats["num_classes"] = self.num_classes
        return stats

    def report_stats(self) -> dict[str, Any]:
        """``summary()`` plus the per-identity histogram data for the report."""
        stats = self.summary()
        stats["identity_counts"] = self.identity_counts().tolist()
        return stats

    def identity_counts(self) -> np.ndarray:
        """Images per class index -- used by the balanced sampler and the report."""
        return np.bincount(self.targets, minlength=self.num_classes)

    def __repr__(self) -> str:
        return (
            f"FaceTrainDataset(n={len(self)}, classes={self.num_classes}, "
            f"adapter={type(self.adapter).__name__})"
        )


def build_dataset(
    data_cfg: Any,
    transform: FaceTransform | None = None,
    class_map: ClassMap | None = None,
    strict_labels: bool = True,
    *,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
) -> "FaceTrainDataset | FaceIterableDataset":
    """Build adapter + dataset from the ``data`` block of a config.

    Map-style adapters (folders, ``.rec`` packs) give a
    :class:`FaceTrainDataset`; streaming adapters (WebDataset shards) give a
    :class:`FaceIterableDataset`. Both yield ``(image_tensor, class_index)``.
    """
    adapter = ADAPTERS.build(dict(data_cfg["adapter"]))
    if isinstance(adapter, StreamingAdapter):
        return FaceIterableDataset(
            adapter=adapter,
            class_map=class_map,
            transform=transform,
            cache_dir=data_cfg.get("scan_cache"),
            strict_labels=strict_labels,
            batch_size=int(data_cfg.get("batch_size", 64)),
            num_workers=int(data_cfg.get("num_workers", 4)),
            rank=rank,
            world_size=world_size,
            seed=seed,
            census_workers=int(data_cfg.get("census_workers", 4)),
        )
    return FaceTrainDataset(
        adapter=adapter,
        class_map=class_map,
        transform=transform,
        cache_dir=data_cfg.get("scan_cache"),
        strict_labels=strict_labels,
    )
