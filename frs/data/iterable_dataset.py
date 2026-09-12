"""The torch ``IterableDataset`` for streaming adapters.

Counterpart of :class:`~frs.data.dataset.FaceTrainDataset` for data that cannot
be indexed. Same outputs -- ``(image_tensor, class_index)`` -- and the same
:class:`~frs.data.class_map.ClassMap` contract, built from a cached census
instead of a sample list.

How an epoch works
------------------
Every DataLoader worker calls :meth:`__iter__` once per epoch. The worker asks
the adapter for its share of the shards (``rank``/``worker`` partition), streams
raw ``(jpeg_bytes, raw_id)`` pairs, drops ids the ``ClassMap`` does not know
(held-out or too-rare identities), decodes, augments and yields.

Two things that are easy to get wrong:

* **``set_epoch`` and persistent workers.** Workers receive a *pickled copy* of
  this object when they start, so an epoch counter changed in the main process
  is invisible to workers that persist across epochs. ``scripts/train.py``
  therefore forces ``persistent_workers=False`` for streaming datasets; a
  respawn costs a few seconds per epoch.
* **What gets pickled.** Only the adapter config, a ~1.4 MB ``int32`` lookup
  table and the transform. The census itself (``raw_ids`` + ``counts``) is
  dropped in ``__getstate__`` -- workers never need it.
"""

from __future__ import annotations

import logging
from itertools import islice
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .adapters.streaming import Census, StreamingAdapter, cached_census
from .class_map import ClassMap
from .transforms import FaceTransform

logger = logging.getLogger(__name__)


class InMemoryDataset(Dataset):
    """A handful of decoded samples held in RAM (used by ``--overfit``)."""

    def __init__(
        self, items: list[tuple[np.ndarray, int]], transform: FaceTransform | None = None
    ) -> None:
        self.items = items
        self.transform = transform
        self.targets = np.asarray([label for _, label in items], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        img, label = self.items[index]
        if self.transform is not None:
            arr = self.transform(img)
        else:
            arr = img.astype(np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(np.ascontiguousarray(arr)), int(label)


class FaceIterableDataset(IterableDataset):
    """Yields ``(image_tensor, class_index)`` from a :class:`StreamingAdapter`.

    Parameters
    ----------
    adapter:
        Any :class:`StreamingAdapter`.
    class_map:
        Existing identity -> index map (resume / extension). If omitted, one is
        built from the census after filtering.
    transform:
        Preprocessing pipeline applied to the decoded ``HWC uint8`` image.
    cache_dir:
        Where the census JSON is memoised.
    strict_labels:
        With a supplied ``class_map``: raise if the census holds identities the
        map does not (True), or silently drop them (False).
    batch_size, num_workers, rank, world_size:
        Needed to make ``__len__`` exact in ``resampled`` mode, where every
        worker of every rank must yield the same number of full batches.
    seed:
        Base seed mixed with the epoch, rank and worker ids.
    """

    def __init__(
        self,
        adapter: StreamingAdapter,
        class_map: ClassMap | None = None,
        transform: FaceTransform | None = None,
        cache_dir: str | None = None,
        strict_labels: bool = True,
        batch_size: int = 64,
        num_workers: int = 0,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        census_workers: int = 4,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.transform = transform
        self.batch_size = int(batch_size)
        self.num_workers = max(1, int(num_workers))
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.seed = int(seed)
        self.epoch = 0

        if adapter.epoch_mode == "natural" and self.world_size > 1:
            raise ValueError(
                "epoch_mode: natural cannot split shards evenly across ranks. Set "
                "data.adapter.epoch_mode: resampled for distributed training."
            )

        self._census: Census | None = cached_census(
            adapter, cache_dir, workers=census_workers,
            explicit_path=getattr(adapter, "census_file", None),
        )
        filtered = adapter.resolve_filters(self._census)
        self._filtered_identities = filtered.identities
        self._filtered_raw_ids = filtered.raw_ids
        self._filtered_counts = filtered.counts
        self.kept_total = filtered.total
        self.total_before_filter = self._census.total
        self.num_shards = self._census.num_shards
        # Size the lookup by the census (not the kept subset) so every raw id
        # the shards can produce has an entry; anything beyond it is -1 too.
        self._max_raw_id = int(self._census.raw_ids.max()) if self._census.raw_ids.size else 0
        if filtered.dropped_min_images or filtered.dropped_excluded:
            logger.info(
                "Filters: dropped %d identities below min_images=%d and %d held-out; "
                "%s images of %s identities remain",
                filtered.dropped_min_images, adapter.min_images_per_identity,
                filtered.dropped_excluded, f"{filtered.total:,}", f"{len(filtered.identities):,}",
            )

        if class_map is None:
            self.class_map = ClassMap(filtered.identities, default_source="stream")
        else:
            self.class_map = class_map
        self.cls_lookup: np.ndarray = np.zeros(0, dtype=np.int32)
        self.reindex(self.class_map, strict=strict_labels)

    # ------------------------------------------------------------ class map

    def reindex(self, class_map: ClassMap, strict: bool = True) -> None:
        """(Re)build the raw id -> class index table against ``class_map``."""
        unknown = [i for i in self._filtered_identities if i not in class_map]
        if unknown:
            if strict:
                raise KeyError(
                    f"{len(unknown)} identities are not in the class map, e.g. "
                    f"{sorted(unknown)[:3]}. Call ClassMap.extend() (and extend the "
                    f"head) before training on new identities."
                )
            logger.warning("Dropping %d identities unknown to the class map", len(unknown))
        unknown_set = set(unknown)

        lookup = np.full(self._max_raw_id + 1, -1, dtype=np.int32)
        for ident, raw in zip(self._filtered_identities, self._filtered_raw_ids.tolist()):
            if ident in unknown_set:
                continue
            lookup[raw] = class_map.index_of(ident)
        self.class_map = class_map
        self.cls_lookup = lookup
        self._kept_mask = np.fromiter(
            (i not in unknown_set for i in self._filtered_identities),
            dtype=bool, count=len(self._filtered_identities),
        )
        self.kept_total = int(self._filtered_counts[self._kept_mask].sum())

    @property
    def identities(self) -> list[str]:
        """Identity strings present in this dataset (after filtering)."""
        return [i for i, k in zip(self._filtered_identities, self._kept_mask) if k]

    @property
    def num_classes(self) -> int:
        return self.class_map.num_classes

    @property
    def census(self) -> Census | None:
        return self._census

    # ------------------------------------------------------------- epochs

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def samples_per_epoch(self) -> int:
        if self.adapter.epoch_mode == "resampled" and self.adapter.samples_per_epoch:
            return int(self.adapter.samples_per_epoch)
        return self.kept_total

    def _per_worker_quota(self) -> int:
        """Samples each worker of this rank yields in ``resampled`` mode.

        Rounded down to a multiple of ``batch_size`` so that, with
        ``drop_last=True``, every worker of every rank produces exactly the same
        number of full batches -- the property DDP needs to not deadlock.
        """
        per_rank = self.samples_per_epoch // self.world_size
        per_worker = per_rank // self.num_workers
        return (per_worker // self.batch_size) * self.batch_size

    def __len__(self) -> int:
        """Samples this *rank* yields per epoch.

        Exact in ``resampled`` mode. In ``natural`` mode it is the filtered
        census total (exact for one process; with several workers each drops
        its own partial batch, so the loader may run a few batches short).
        """
        if self.adapter.epoch_mode == "resampled":
            return self._per_worker_quota() * self.num_workers
        return self.kept_total // self.world_size

    # ---------------------------------------------------------- iteration

    def _raw_stream(
        self, worker: int, num_workers: int, epoch: int | None = None, shuffle: bool = True
    ) -> Iterator[tuple[bytes, int]]:
        return self.adapter.iter_raw(
            epoch=self.epoch if epoch is None else epoch,
            seed=self.seed,
            rank=self.rank,
            world_size=self.world_size,
            worker=worker,
            num_workers=num_workers,
            shuffle=shuffle,
        )

    def _class_of(self, raw_id: int) -> int:
        if 0 <= raw_id < self.cls_lookup.size:
            return int(self.cls_lookup[raw_id])
        return -1

    def _kept_samples(
        self, worker: int, num_workers: int, epoch: int | None = None, shuffle: bool = True
    ) -> Iterator[tuple[bytes, int]]:
        """Raw samples whose identity is in the class map, as ``(bytes, class_index)``."""
        for blob, raw_id in self._raw_stream(worker, num_workers, epoch, shuffle):
            index = self._class_of(raw_id)
            if index >= 0:
                yield blob, index

    def __iter__(self) -> Iterator[tuple[torch.Tensor, int]]:
        info = get_worker_info()
        worker, num_workers = (0, 1) if info is None else (info.id, info.num_workers)
        if num_workers != self.num_workers and self.adapter.epoch_mode == "resampled":
            logger.warning(
                "DataLoader has %d workers but the dataset was sized for %d; "
                "__len__ will not match the yielded count", num_workers, self.num_workers,
            )

        stream = self._kept_samples(worker, num_workers)
        if self.adapter.epoch_mode == "resampled":
            stream = islice(stream, self._per_worker_quota())

        for blob, index in stream:
            try:
                img = self.adapter.decode(blob)
            except Exception as exc:
                if self.adapter.handler == "reraise":
                    raise
                logger.warning("Skipping undecodable image (class %d): %s", index, exc)
                continue
            if self.transform is not None:
                arr = self.transform(img)
            else:
                arr = img.astype(np.float32).transpose(2, 0, 1) / 255.0
            yield torch.from_numpy(np.ascontiguousarray(arr)), index

    # -------------------------------------------------- bounded extraction

    def take(self, n: int) -> InMemoryDataset:
        """First ``n`` kept samples, decoded, as a small map-style dataset.

        Single process, natural order, no shuffle -- used by ``--overfit``.
        """
        items: list[tuple[np.ndarray, int]] = []
        for blob, index in self._kept_samples(0, 1, epoch=0, shuffle=False):
            items.append((self.adapter.decode(blob), index))
            if len(items) >= n:
                break
        if not items:
            raise RuntimeError("streaming dataset yielded no samples")
        return InMemoryDataset(items, transform=self.transform)

    def collect_by_class(
        self, class_indices: set[int], per_class: int, max_images: int = 200_000
    ) -> dict[int, list[np.ndarray]]:
        """Decoded images for the requested classes, up to ``per_class`` each.

        Scans at most ``max_images`` kept samples in natural order. Used to seed
        new class prototypes when extending a checkpoint.
        """
        wanted = set(int(i) for i in class_indices)
        out: dict[int, list[np.ndarray]] = {i: [] for i in wanted}
        remaining = set(wanted)
        scanned = 0
        for blob, index in self._kept_samples(0, 1, epoch=0, shuffle=False):
            scanned += 1
            if index in remaining:
                out[index].append(self.adapter.decode(blob))
                if len(out[index]) >= per_class:
                    remaining.discard(index)
                    if not remaining:
                        break
            if scanned >= max_images:
                break
        return out

    # ------------------------------------------------------- introspection

    def identity_counts(self) -> np.ndarray:
        """Images per class index (zeros for classes absent from this dataset)."""
        counts = np.zeros(self.num_classes, dtype=np.int64)
        for ident, n, kept in zip(
            self._filtered_identities, self._filtered_counts.tolist(), self._kept_mask
        ):
            if kept:
                counts[self.class_map.index_of(ident)] = n
        return counts

    def summary(self) -> dict[str, Any]:
        counts = self._filtered_counts[self._kept_mask]
        stats: dict[str, Any] = {
            "num_samples": int(counts.sum()) if counts.size else 0,
            "num_identities": int(counts.size),
            "min_per_identity": int(counts.min()) if counts.size else 0,
            "max_per_identity": int(counts.max()) if counts.size else 0,
            "mean_per_identity": float(counts.mean()) if counts.size else 0.0,
            "median_per_identity": float(np.median(counts)) if counts.size else 0.0,
            "num_classes": self.num_classes,
            "num_shards": self.num_shards,
            "num_samples_before_filter": self.total_before_filter,
            "epoch_mode": self.adapter.epoch_mode,
            "samples_per_epoch": self.samples_per_epoch,
        }
        return stats

    def report_stats(self) -> dict[str, Any]:
        stats = self.summary()
        stats["identity_counts"] = self._filtered_counts[self._kept_mask].tolist()
        return stats

    # ------------------------------------------------------------ pickling

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Workers only need the lookup table; the census arrays are dead weight.
        state["_census"] = None
        state["_filtered_identities"] = []
        state["_filtered_raw_ids"] = np.zeros(0, dtype=np.int64)
        state["_filtered_counts"] = np.zeros(0, dtype=np.int64)
        state["_kept_mask"] = np.zeros(0, dtype=bool)
        return state

    def __repr__(self) -> str:
        return (
            f"FaceIterableDataset(n={self.kept_total}, classes={self.num_classes}, "
            f"shards={self.num_shards}, mode={self.adapter.epoch_mode}, "
            f"adapter={type(self.adapter).__name__})"
        )
