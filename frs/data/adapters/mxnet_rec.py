"""MXNet RecordIO (.rec/.idx) reader -- the format used by MS1MV3, Glint360K,
WebFace4M and most other large public face datasets.

Why a hand-rolled reader
------------------------
The obvious approach is ``import mxnet``, but mxnet has no Python 3.13 wheel and
is effectively unmaintained. The RecordIO container is a simple length-prefixed
binary format, so reading it directly costs ~60 lines and removes a dead
dependency entirely. JPEG decoding is delegated to PIL.

Format
------
``train.idx`` is a text file of ``record_id\\tbyte_offset`` pairs.
``train.rec`` is a sequence of records, each::

    uint32  magic   (0xced7230a)
    uint32  lrecord (2-bit continuation flag << 29 | length)
    bytes   payload, padded to a 4-byte boundary

The payload is an ``IRHeader`` followed by the encoded image::

    uint32  flag        (number of label floats; 0 or 1 means a scalar label)
    float32 label[...]  (flag==0 -> exactly one float)
    uint64  id
    uint64  id2

The first record (index 0) of an InsightFace pack is a *header record* whose
label holds ``[first_image_idx, last_image_idx]`` rather than a class id.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from ...registry import ADAPTERS
from .base import DatasetAdapter, Sample, filter_by_min_images

_MAGIC = 0xCED7230A
_IR_HEADER = struct.Struct("<If2Q")  # flag, label(1 float), id, id2


def _decode_record(blob: bytes) -> tuple[np.ndarray, bytes]:
    """Split one record payload into its label array and image bytes."""
    flag, label0, _id, _id2 = _IR_HEADER.unpack_from(blob, 0)
    offset = _IR_HEADER.size
    if flag > 0:
        # Multi-value label: `flag` floats follow the header, and label0 was the
        # first of them (already consumed as part of the fixed header layout).
        labels = np.frombuffer(blob, dtype="<f4", count=flag, offset=offset).copy()
        offset += 4 * flag
    else:
        labels = np.asarray([label0], dtype="<f4")
    return labels, blob[offset:]


class _RecordIOReader:
    """Random-access reader over a .rec/.idx pair.

    File handles are opened lazily and kept per-instance, which keeps the object
    picklable so DataLoader workers on Windows (spawn) can each open their own.
    """

    def __init__(self, rec_path: Path, idx_path: Path) -> None:
        self.rec_path = rec_path
        self.idx_path = idx_path
        self._offsets: dict[int, int] = {}
        self._fh: io.BufferedReader | None = None
        self._load_index()

    def _load_index(self) -> None:
        with self.idx_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                key, _, value = line.partition("\t")
                self._offsets[int(key)] = int(value)

    @property
    def keys(self) -> list[int]:
        return sorted(self._offsets)

    def _handle(self) -> io.BufferedReader:
        if self._fh is None:
            self._fh = self.rec_path.open("rb")
        return self._fh

    def read(self, key: int) -> tuple[np.ndarray, bytes]:
        fh = self._handle()
        fh.seek(self._offsets[key])
        magic, lrecord = struct.unpack("<II", fh.read(8))
        if magic != _MAGIC:
            raise ValueError(
                f"{self.rec_path}: bad record magic 0x{magic:08x} at key {key} "
                f"-- the .idx and .rec files may not match"
            )
        length = lrecord & ((1 << 29) - 1)
        return _decode_record(fh.read(length))

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_fh"] = None  # file handles do not survive pickling
        return state

    def __del__(self) -> None:
        if getattr(self, "_fh", None) is not None:
            try:
                self._fh.close()
            except Exception:
                pass


@ADAPTERS.register("mxnet_rec")
class MXNetRecAdapter(DatasetAdapter):
    """Read an InsightFace-style ``train.rec`` / ``train.idx`` pack.

    Parameters
    ----------
    root:
        Directory containing ``train.rec`` and ``train.idx``.
    rec_name, idx_name:
        Override the filenames if they differ.
    min_images_per_identity:
        Drop sparse identities.
    prefix:
        Identity namespace for multi-source merges.

    Notes
    -----
    ``scan()`` reads the label of every record, which for a 5.8M-image pack takes
    a couple of minutes on first run. The result is cached by the dataset layer,
    so subsequent runs are instant.
    """

    def __init__(
        self,
        root: str,
        rec_name: str = "train.rec",
        idx_name: str = "train.idx",
        min_images_per_identity: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.root = Path(root)
        self.rec_path = self.root / rec_name
        self.idx_path = self.root / idx_name
        self.min_images_per_identity = int(min_images_per_identity)
        self._reader: _RecordIOReader | None = None

    @property
    def cache_key(self) -> str:
        stat = self.rec_path.stat() if self.rec_path.is_file() else None
        return self._hash_config(
            rec=str(self.rec_path.resolve()),
            size=stat.st_size if stat else None,
            mtime=stat.st_mtime_ns if stat else None,
            min_images=self.min_images_per_identity,
        )

    def _get_reader(self) -> _RecordIOReader:
        if self._reader is None:
            for p in (self.rec_path, self.idx_path):
                if not p.is_file():
                    raise FileNotFoundError(f"RecordIO file not found: {p}")
            self._reader = _RecordIOReader(self.rec_path, self.idx_path)
        return self._reader

    def scan(self) -> list[Sample]:
        reader = self._get_reader()
        keys = reader.keys
        if not keys:
            raise RuntimeError(f"{self.idx_path} lists no records")

        # InsightFace packs put a header record at key 0 whose label is
        # [first_idx, last_idx] rather than a class id. Detect and skip it.
        data_keys = keys
        head_labels, _ = reader.read(keys[0])
        if head_labels.size >= 2 and head_labels[0] > 1:
            first, last = int(head_labels[0]), int(head_labels[1])
            data_keys = [k for k in keys if first <= k < last]

        samples: list[Sample] = []
        for key in data_keys:
            labels, _ = reader.read(key)
            class_id = int(labels[0])
            samples.append(
                Sample(
                    path=f"rec://{key}",
                    identity=self._make_identity(str(class_id)),
                )
            )

        samples, _ = filter_by_min_images(samples, self.min_images_per_identity)
        return samples

    def load_image(self, sample: Sample) -> np.ndarray:
        if not sample.path.startswith("rec://"):
            return super().load_image(sample)
        key = int(sample.path[len("rec://") :])
        _, encoded = self._get_reader().read(key)
        with Image.open(io.BytesIO(encoded)) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    def __repr__(self) -> str:
        return f"MXNetRecAdapter(root={str(self.root)!r}, prefix={self.prefix!r})"
