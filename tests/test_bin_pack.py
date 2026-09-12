"""InsightFace ``.bin`` verification packs and the eval-target dispatcher."""

from __future__ import annotations

import io
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frs.eval.bin_pack import load_bin_images, read_bin  # noqa: E402
from frs.eval.pairs import load_eval_images  # noqa: E402


def _jpeg(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


def _identity_transform(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32).transpose(2, 0, 1) / 255.0


@pytest.fixture
def bin_file(tmp_path):
    bins = [_jpeg(i) for i in range(4)]
    path = tmp_path / "lfw.bin"
    with path.open("wb") as fh:
        pickle.dump((bins, [True, False]), fh, protocol=2)
    return path


def test_read_bin(bin_file):
    bins, issame = read_bin(bin_file)
    assert len(bins) == 4 and issame == [True, False]
    assert all(isinstance(b, bytes) for b in bins)


def test_load_bin_images_shapes_and_pairs(bin_file):
    target = {"name": "lfw", "type": "bin", "path": str(bin_file)}
    images, pairs, is_same = load_bin_images(target, _identity_transform)
    assert images.shape == (4, 3, 112, 112) and images.dtype == np.float32
    assert pairs.tolist() == [[0, 1], [2, 3]]
    assert is_same.tolist() == [True, False]


def test_max_pairs_truncates(bin_file):
    target = {"name": "lfw", "type": "bin", "path": str(bin_file), "max_pairs": 1}
    images, pairs, is_same = load_bin_images(target, _identity_transform)
    assert images.shape[0] == 2 and pairs.tolist() == [[0, 1]] and is_same.tolist() == [True]


def test_dispatch(bin_file, tmp_path):
    images, _, _ = load_eval_images(
        {"name": "lfw", "type": "bin", "path": str(bin_file)}, _identity_transform
    )
    assert images.shape[0] == 4
    with pytest.raises(ValueError, match="unknown type"):
        load_eval_images({"name": "x", "type": "lmdb", "path": "y"}, _identity_transform)
    with pytest.raises(FileNotFoundError):
        load_eval_images({"name": "x", "type": "bin", "path": str(tmp_path / "no.bin")}, _identity_transform)


def test_malformed_pack_rejected(tmp_path):
    path = tmp_path / "bad.bin"
    with path.open("wb") as fh:
        pickle.dump(([b"a", b"b", b"c"], [True]), fh, protocol=2)
    with pytest.raises(ValueError, match="do not form"):
        read_bin(path)
