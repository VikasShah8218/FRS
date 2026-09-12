"""WebDataset tar shards -- the format Glint360K ships in on HuggingFace.

    glint360k-0000.tar.gz
      011413985.cls   b"205241"          <- ASCII integer class id
      011413985.jpg   <112x112 JPEG>
      014786152.cls   ...

Each sample is the pair of members sharing a key; ``.cls`` holds the label and
``.jpg`` the aligned face. The shards are globally shuffled by the publisher,
so a subset of shards holds a random slice of *every* identity -- use
``min_images_per_identity`` to keep the subset trainable.

Why the tar reading is done here rather than with ``webdataset``'s pipeline
--------------------------------------------------------------------------
``webdataset.gopen`` parses ``D:/data/x.tar.gz`` as a URL with scheme ``d`` and
refuses it, and its ``https`` reader shells out to ``curl``. Streaming a tar
sequentially is ~40 lines of :mod:`tarfile`, so this module owns it: local
files open directly, ``http(s)`` streams through :mod:`urllib`, and the same
code path runs on Windows, Linux and in spawned DataLoader workers. The
``webdataset`` package is used only for brace expansion of shard patterns.

Config
------
    data:
      adapter:
        type: webdataset
        shards: "D:/data/glint360k/glint360k-{0000..0001}.tar.gz"   # pattern | list | directory | https pattern
        image_key: jpg
        label_key: cls
        min_images_per_identity: 2
        exclude_identities_file: data/splits/glint360k_val_identities.txt
        epoch_mode: natural          # natural | resampled (DDP requires resampled)
        samples_per_epoch: null      # resampled only
        shuffle_buffer: 2000
        handler: warn                # warn | reraise
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import random
import tarfile
import urllib.request
from pathlib import Path
from typing import IO, Any, Iterable, Iterator

import numpy as np

from ...registry import ADAPTERS
from .streaming import Census, StreamingAdapter, buffered_shuffle

logger = logging.getLogger(__name__)

_REMOTE_SCHEMES = ("http://", "https://")
_TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz")
_USER_AGENT = "essi-frs/1.0 (+https://github.com/)"


# --------------------------------------------------------------- shard I/O


def expand_shards(spec: str | list[str]) -> list[str]:
    """Turn a pattern / list / directory into an ordered list of shard paths.

    * ``"x-{0000..0009}.tar.gz"`` -- brace expansion (local path or URL)
    * ``["a.tar", "b-{00..01}.tar.gz"]`` -- each entry expanded, concatenated
    * ``"/data/glint360k"`` (a directory) -- every ``*.tar`` / ``*.tar.gz`` inside, sorted
    """
    from webdataset.shardlists import expand_urls

    specs = [spec] if isinstance(spec, str) else list(spec)
    out: list[str] = []
    for item in specs:
        item = str(item)
        if not item.startswith(_REMOTE_SCHEMES):
            # braceexpand treats '\' as an escape; Windows paths must use '/'.
            item = item.replace("\\", "/")
        if not item.startswith(_REMOTE_SCHEMES) and Path(item).is_dir():
            found = sorted(
                str(p) for p in Path(item).iterdir()
                if p.is_file() and p.name.lower().endswith(_TAR_SUFFIXES)
            )
            if not found:
                raise FileNotFoundError(f"no .tar / .tar.gz shards found in directory {item}")
            out.extend(found)
        else:
            out.extend(expand_urls(item))
    if not out:
        raise ValueError(f"shard spec {spec!r} expanded to nothing")
    return out


def open_shard(url: str, timeout: float = 60.0) -> IO[bytes]:
    """Open a shard for sequential reading (local path or http(s) URL)."""
    if url.startswith(_REMOTE_SCHEMES):
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310
    return open(url, "rb")


def _split_member(name: str) -> tuple[str, str] | None:
    """``"dir/011413985.jpg"`` -> ``("dir/011413985", "jpg")``."""
    base, dot, ext = name.rpartition(".")
    if not dot or not base:
        return None
    return base, ext.lower()


def iter_tar_samples(
    url: str,
    keys: tuple[str, ...],
    handler: str = "warn",
) -> Iterator[dict[str, Any]]:
    """Yield ``{"__key__": ..., key: bytes, ...}`` for each complete sample.

    Members are grouped by their basename exactly as ``webdataset`` does; a
    sample is emitted once its key changes or the tar ends. Only members whose
    extension is in ``keys`` are read from the stream.
    """
    wanted = set(keys)
    try:
        with open_shard(url) as fh, tarfile.open(fileobj=fh, mode="r|*") as tf:
            current_key: str | None = None
            current: dict[str, Any] = {}
            for member in tf:
                if not member.isfile():
                    continue
                split = _split_member(member.name)
                if split is None:
                    continue
                base, ext = split
                if ext not in wanted:
                    continue
                if base != current_key:
                    if current_key is not None and wanted <= current.keys():
                        current["__key__"] = current_key
                        yield current
                    current_key, current = base, {}
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue
                current[ext] = extracted.read()
            if current_key is not None and wanted <= current.keys():
                current["__key__"] = current_key
                yield current
    except Exception as exc:
        if handler == "reraise":
            raise
        logger.warning("Skipping the rest of shard %s after error: %s", url, exc)


# ------------------------------------------------------------------ census


def _census_one_shard(args: tuple[str, str, str]) -> tuple[str, np.ndarray, np.ndarray]:
    """Pool worker: unique labels and counts for one shard (reads only ``.cls``)."""
    url, label_key, handler = args
    ids: list[int] = []
    for sample in iter_tar_samples(url, (label_key,), handler=handler):
        try:
            ids.append(int(sample[label_key]))
        except (TypeError, ValueError):
            if handler == "reraise":
                raise
    if not ids:
        return url, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    uniq, counts = np.unique(np.asarray(ids, dtype=np.int64), return_counts=True)
    return url, uniq, counts.astype(np.int64)


# ----------------------------------------------------------------- adapter


@ADAPTERS.register("webdataset")
class WebDatasetAdapter(StreamingAdapter):
    """Stream ``(image, label)`` samples from WebDataset-style tar shards.

    Parameters
    ----------
    shards:
        Brace pattern, list of patterns, directory, or ``https://`` pattern.
    image_key, label_key:
        Member extensions holding the image bytes and the label.
    Other parameters are documented on :class:`StreamingAdapter`.
    """

    def __init__(
        self,
        shards: str | list[str] | None = None,
        root: str | None = None,
        image_key: str = "jpg",
        label_key: str = "cls",
        prefix: str = "",
        min_images_per_identity: int = 0,
        exclude_identities_file: str | None = None,
        epoch_mode: str = "natural",
        samples_per_epoch: int | None = None,
        shuffle_buffer: int = 2000,
        shard_shuffle: bool = True,
        handler: str = "warn",
        census_file: str | None = None,
    ) -> None:
        super().__init__(
            prefix=prefix,
            min_images_per_identity=min_images_per_identity,
            exclude_identities_file=exclude_identities_file,
            epoch_mode=epoch_mode,
            samples_per_epoch=samples_per_epoch,
            shuffle_buffer=shuffle_buffer,
            handler=handler,
        )
        # ``root`` (inherited from base.yaml for every adapter) doubles as a
        # shard directory when ``shards`` is not given.
        if shards is None and root is None:
            raise ValueError("webdataset adapter needs `shards` (pattern/list/dir) or `root` (dir)")
        self.shards_spec: str | list[str] = shards if shards is not None else str(root)
        self.image_key = str(image_key).lower()
        self.label_key = str(label_key).lower()
        self.shard_shuffle = bool(shard_shuffle)
        self.census_file = census_file
        self._shards: list[str] | None = None

    # ------------------------------------------------------------ shards

    def shard_list(self) -> list[str]:
        if self._shards is None:
            shards = expand_shards(self.shards_spec)
            missing = [
                s for s in shards
                if not s.startswith(_REMOTE_SCHEMES) and not Path(s).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} of {len(shards)} shards are missing, e.g. "
                    f"{missing[0]}. Download them with scripts/download_glint360k.py "
                    f"or narrow data.adapter.shards to what is present."
                )
            self._shards = shards
        return self._shards

    @property
    def cache_key(self) -> str:
        # Filters are deliberately NOT part of the key: they are applied to the
        # census after the fact, so changing them never re-streams the shards.
        return self._hash_config(
            shards=self.shard_list(),
            image_key=self.image_key,
            label_key=self.label_key,
        )

    # ------------------------------------------------------------ census

    def census(self, workers: int = 4, progress: bool = True) -> Census:
        shards = self.shard_list()
        logger.info("Census: reading labels from %d shards with %d workers", len(shards), workers)
        jobs = [(url, self.label_key, self.handler) for url in shards]
        parts: list[tuple[np.ndarray, np.ndarray]] = []

        def _consume(results: Iterable[tuple[str, np.ndarray, np.ndarray]]) -> None:
            for i, (url, ids, counts) in enumerate(results, start=1):
                if ids.size == 0:
                    logger.warning("Shard yielded no labels: %s", url)
                parts.append((ids, counts))
                if progress and (i % 25 == 0 or i == len(shards)):
                    logger.info("  census %d/%d shards", i, len(shards))

        if workers <= 1 or len(shards) == 1:
            _consume(map(_census_one_shard, jobs))
        else:
            ctx = multiprocessing.get_context("spawn" if os.name == "nt" else None)
            with ctx.Pool(min(workers, len(shards))) as pool:
                _consume(pool.imap_unordered(_census_one_shard, jobs))

        census = Census.merge(parts, shards)
        logger.info(
            "Census: %s images, %s identities across %d shards",
            f"{census.total:,}", f"{census.num_identities:,}", census.num_shards,
        )
        return census

    # ----------------------------------------------------------- streaming

    def _shard_order(self, epoch: int, seed: int) -> list[str]:
        order = list(self.shard_list())
        if self.shard_shuffle:
            random.Random(f"shards:{seed}:{epoch}").shuffle(order)
        return order

    def _iter_shards(
        self, *, epoch: int, seed: int, rank: int, world_size: int, worker: int, num_workers: int
    ) -> Iterator[str]:
        if self.epoch_mode == "natural":
            # Deterministic partition: rank first, then worker within the rank.
            mine = self._shard_order(epoch, seed)[rank::world_size][worker::num_workers]
            yield from mine
            return
        # resampled: every worker draws shards with replacement, forever.
        rng = random.Random(f"resample:{seed}:{epoch}:{rank}:{worker}")
        shards = self.shard_list()
        while True:
            yield shards[rng.randrange(len(shards))]

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
        keys = (self.image_key, self.label_key)

        def raw() -> Iterator[tuple[bytes, int]]:
            for url in self._iter_shards(
                epoch=epoch, seed=seed, rank=rank, world_size=world_size,
                worker=worker, num_workers=num_workers,
            ):
                for sample in iter_tar_samples(url, keys, handler=self.handler):
                    try:
                        yield sample[self.image_key], int(sample[self.label_key])
                    except (TypeError, ValueError) as exc:
                        if self.handler == "reraise":
                            raise
                        logger.warning("Bad label in %s key %s: %s", url, sample.get("__key__"), exc)

        stream: Iterator[tuple[bytes, int]] = raw()
        if shuffle and self.shuffle_buffer > 1:
            stream = buffered_shuffle(
                stream, self.shuffle_buffer, seed=f"buf:{seed}:{epoch}:{rank}:{worker}"
            )
        return stream

    def __repr__(self) -> str:
        return (
            f"WebDatasetAdapter(shards={self.shards_spec!r}, prefix={self.prefix!r}, "
            f"epoch_mode={self.epoch_mode!r})"
        )
