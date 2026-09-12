"""Streaming (WebDataset) adapter, census, filters and the iterable dataset.

Every test builds tiny synthetic ``.tar.gz`` shards with :mod:`tarfile` in
``tmp_path`` -- exactly the member layout Glint360K uses (``<key>.cls`` +
``<key>.jpg``) -- so nothing here needs the real data.
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader  # noqa: E402

from frs.config import Config  # noqa: E402
from frs.data.adapters.streaming import Census, buffered_shuffle, cached_census  # noqa: E402
from frs.data.adapters.webdataset import (  # noqa: E402
    WebDatasetAdapter,
    expand_shards,
    iter_tar_samples,
)
from frs.data.class_map import ClassMap  # noqa: E402
from frs.data.dataset import build_dataset  # noqa: E402
from frs.data.iterable_dataset import FaceIterableDataset, InMemoryDataset  # noqa: E402
from frs.eval.pairs import load_pair_images, read_pair_file  # noqa: E402
from frs.registry import ADAPTERS  # noqa: E402

# --------------------------------------------------------------- fixtures


def _jpeg(seed: int, size: int = 112) -> bytes:
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


def write_shards(
    root: Path, spec: dict[str, list[tuple[str, int]]], gz: bool = True
) -> list[Path]:
    """``spec = {"shard-0000": [("k1", 7), ("k2", 3)], ...}`` -> tar files."""
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, samples in spec.items():
        path = root / (f"{name}.tar.gz" if gz else f"{name}.tar")
        with tarfile.open(path, "w:gz" if gz else "w") as tf:
            for i, (key, cls) in enumerate(samples):
                for ext, payload in ((".cls", str(cls).encode()), (".jpg", _jpeg(hash(key) % 1000))):
                    info = tarfile.TarInfo(name=f"{key}{ext}")
                    info.size = len(payload)
                    tf.addfile(info, io.BytesIO(payload))
        paths.append(path)
    return paths


# identities: 1 -> 3 images, 2 -> 2 images, 3 -> 1 image, 4 -> 2 images (across shards)
SPEC = {
    "s-0000": [("a", 1), ("b", 2), ("c", 3), ("d", 1)],
    "s-0001": [("e", 4), ("f", 1), ("g", 2), ("h", 4)],
}
EXPECTED_COUNTS = {1: 3, 2: 2, 3: 1, 4: 2}


@pytest.fixture
def shards(tmp_path):
    write_shards(tmp_path / "shards", SPEC)
    return (tmp_path / "shards" / "s-{0000..0001}.tar.gz").as_posix()


def _identity_transform(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32).transpose(2, 0, 1) / 255.0


# --------------------------------------------------------------- basics


def test_registered():
    assert "webdataset" in ADAPTERS.names()


def test_expand_shards_pattern_list_and_directory(tmp_path):
    paths = write_shards(tmp_path / "d", {"x-0000": [("a", 1)], "x-0001": [("b", 1)]})
    pattern = (tmp_path / "d" / "x-{0000..0001}.tar.gz").as_posix()
    assert expand_shards(pattern) == [p.as_posix() for p in paths]
    assert expand_shards([pattern]) == [p.as_posix() for p in paths]
    assert expand_shards(str(tmp_path / "d")) == [str(p) for p in paths]
    # Windows-style backslashes must not be eaten as escapes
    assert expand_shards(pattern.replace("/", "\\")) == [p.as_posix() for p in paths]


def test_missing_shard_is_a_clear_error(tmp_path):
    adapter = WebDatasetAdapter(shards=(tmp_path / "nope-{00..01}.tar.gz").as_posix())
    with pytest.raises(FileNotFoundError, match="download_glint360k"):
        adapter.shard_list()


def test_iter_tar_samples_groups_members(shards):
    url = expand_shards(shards)[0]
    samples = list(iter_tar_samples(url, ("jpg", "cls")))
    assert [s["__key__"] for s in samples] == ["a", "b", "c", "d"]
    assert samples[0]["cls"] == b"1" and samples[0]["jpg"][:2] == b"\xff\xd8"
    # reading only labels skips the image bytes entirely
    labels_only = list(iter_tar_samples(url, ("cls",)))
    assert all("jpg" not in s for s in labels_only)


# --------------------------------------------------------------- census


def test_census_counts_and_json_round_trip(shards, tmp_path):
    adapter = WebDatasetAdapter(shards=shards)
    census = adapter.census(workers=1)
    assert dict(zip(census.raw_ids.tolist(), census.counts.tolist())) == EXPECTED_COUNTS
    assert census.total == 8 and census.num_shards == 2

    path = tmp_path / "census.json"
    census.to_json(path)
    loaded = Census.from_json(path)
    assert loaded.total == census.total
    assert loaded.raw_ids.tolist() == census.raw_ids.tolist()
    assert json.loads(path.read_text())["format_version"] >= 1


def test_census_cache_reused_and_key_ignores_filters(shards, tmp_path):
    cache = tmp_path / "cache"
    a = WebDatasetAdapter(shards=shards, min_images_per_identity=1)
    b = WebDatasetAdapter(shards=shards, min_images_per_identity=2)
    assert a.cache_key == b.cache_key  # filters are applied after the census
    cached_census(a, cache, workers=1)
    files = list(cache.glob("census_*.json"))
    assert len(files) == 1
    mtime = files[0].stat().st_mtime_ns
    cached_census(b, cache, workers=1)
    assert files[0].stat().st_mtime_ns == mtime  # not rebuilt

    c = WebDatasetAdapter(shards=expand_shards(shards)[:1])
    assert c.cache_key != a.cache_key


# ----------------------------------------------------------- filters/map


def test_filters_and_class_map(shards, tmp_path):
    exclude = tmp_path / "val.txt"
    exclude.write_text("# held out\n4\n")  # bare form, no prefix
    adapter = WebDatasetAdapter(
        shards=shards, min_images_per_identity=2, exclude_identities_file=str(exclude), prefix="g/"
    )
    ds = FaceIterableDataset(adapter, transform=_identity_transform, cache_dir=None)
    # 3 dropped (one image), 4 excluded -> identities 1 and 2 remain
    assert ds.class_map.index_to_identity == ["g/1", "g/2"]
    assert ds.kept_total == 5
    assert ds.cls_lookup[1] == 0 and ds.cls_lookup[2] == 1
    assert ds.cls_lookup[3] == -1 and ds.cls_lookup[4] == -1
    assert ds.identities == ["g/1", "g/2"]
    assert ds.identity_counts().tolist() == [3, 2]
    stats = ds.summary()
    assert stats["num_samples"] == 5 and stats["num_identities"] == 2
    assert stats["num_samples_before_filter"] == 8


def test_strict_labels_with_supplied_map(shards):
    adapter = WebDatasetAdapter(shards=shards)
    small = ClassMap(["1", "2"])
    with pytest.raises(KeyError, match="not in the class map"):
        FaceIterableDataset(adapter, class_map=small, cache_dir=None, strict_labels=True)
    ds = FaceIterableDataset(adapter, class_map=small, cache_dir=None, strict_labels=False)
    assert ds.identities == ["1", "2"] and ds.kept_total == 5
    labels = sorted(y for _, y in ds)
    assert labels == [0, 0, 0, 1, 1]


# ------------------------------------------------------------- iteration


def test_natural_epoch_yields_every_kept_sample(shards):
    adapter = WebDatasetAdapter(shards=shards, min_images_per_identity=2, shuffle_buffer=4)
    ds = FaceIterableDataset(adapter, transform=_identity_transform, cache_dir=None)
    items = list(ds)
    assert len(items) == len(ds) == 7
    x, y = items[0]
    assert x.shape == (3, 112, 112) and x.dtype == torch.float32 and isinstance(y, int)
    counts = np.bincount([y for _, y in items], minlength=3).tolist()
    assert counts == [3, 2, 2]  # identities 1, 2, 4


def test_resampled_epoch_is_exact_and_deterministic(shards):
    adapter = WebDatasetAdapter(
        shards=shards, epoch_mode="resampled", samples_per_epoch=40, shuffle_buffer=8
    )
    ds = FaceIterableDataset(adapter, transform=_identity_transform, cache_dir=None, batch_size=4, num_workers=1)
    assert len(ds) == 40
    loader = DataLoader(ds, batch_size=4, drop_last=True)
    assert len(loader) == 10 and sum(1 for _ in loader) == 10

    ds.set_epoch(1)
    a = [y for _, y in ds]
    ds.set_epoch(1)
    b = [y for _, y in ds]
    ds.set_epoch(2)
    c = [y for _, y in ds]
    assert a == b and a != c and len(a) == 40


def test_natural_mode_refuses_ddp(shards):
    adapter = WebDatasetAdapter(shards=shards)
    with pytest.raises(ValueError, match="resampled"):
        FaceIterableDataset(adapter, cache_dir=None, world_size=2, rank=0)


def test_multi_worker_loader_matches_len(shards):
    adapter = WebDatasetAdapter(
        shards=shards, epoch_mode="resampled", samples_per_epoch=48, shuffle_buffer=2
    )
    ds = FaceIterableDataset(adapter, transform=_identity_transform, cache_dir=None, batch_size=4, num_workers=2)
    loader = DataLoader(ds, batch_size=4, drop_last=True, num_workers=2, persistent_workers=False)
    assert sum(1 for _ in loader) == len(loader) == 12


def test_build_dataset_dispatch_and_take(shards, tmp_path):
    cfg = Config._wrap({
        "adapter": {"type": "webdataset", "shards": shards, "root": None, "min_images_per_identity": 2},
        "batch_size": 4, "num_workers": 0, "scan_cache": str(tmp_path / "cache"),
    })
    ds = build_dataset(cfg, transform=_identity_transform)
    assert isinstance(ds, FaceIterableDataset)
    mem = ds.take(3)
    assert isinstance(mem, InMemoryDataset) and len(mem) == 3
    assert mem.targets.shape == (3,) and mem[0][0].shape == (3, 112, 112)
    got = ds.collect_by_class({0}, per_class=2)
    assert len(got[0]) == 2 and got[0][0].shape == (112, 112, 3)


def test_pickling_drops_census(shards):
    import pickle

    adapter = WebDatasetAdapter(shards=shards)
    ds = FaceIterableDataset(adapter, cache_dir=None)
    clone = pickle.loads(pickle.dumps(ds))
    assert clone.census is None and clone.cls_lookup.tolist() == ds.cls_lookup.tolist()
    assert len(list(clone)) == len(list(ds))


def test_buffered_shuffle_is_a_permutation():
    items = list(range(50))
    out = list(buffered_shuffle(iter(items), 8, seed="x"))
    assert sorted(out) == items and out != items
    assert list(buffered_shuffle(iter(items), 1, seed="x")) == items


# ------------------------------------------------------------ scan script


def test_scan_script_census_and_holdout(shards, tmp_path, monkeypatch):
    from scripts import scan_webdataset

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "experiment: {name: t, output_dir: out, seed: 1}\n"
        "data:\n"
        f"  adapter: {{type: webdataset, shards: '{shards}', min_images_per_identity: 2}}\n"
        f"  scan_cache: '{(tmp_path / 'cache').as_posix()}'\n"
        "  input_size: [112, 112]\n  batch_size: 4\n"
        "model: {backbone: {arch: ir_18}, head: {type: arcface}}\n"
        "optim: {type: sgd, lr: 0.1}\nscheduler: {type: polylr}\ntrain: {epochs: 1}\n"
    )
    ids_out = tmp_path / "val_ids.txt"
    pairs_out = tmp_path / "val_pairs.txt"
    hold_out = tmp_path / "holdout"
    monkeypatch.setattr(sys, "argv", [
        "scan", "--config", str(cfg_path), "--workers", "1",
        "--holdout", "2", "--holdout-min-images", "2", "--holdout-max-images", "5",
        "--holdout-out", str(hold_out), "--identities-out", str(ids_out),
        "--pairs-out", str(pairs_out), "--num-positive", "3", "--num-negative", "3",
    ])
    assert scan_webdataset.main() == 0

    assert list((tmp_path / "cache").glob("census_*.json"))
    held = [l for l in ids_out.read_text().splitlines() if l and not l.startswith("#")]
    assert len(held) == 2 and all(h in {"1", "2", "4"} for h in held)
    assert {d.name for d in hold_out.iterdir() if d.is_dir()} == set(held)
    pairs = read_pair_file(pairs_out)
    assert pairs and all(Path(a).is_file() and Path(b).is_file() for a, b, _ in pairs)
    images, index_pairs, is_same = load_pair_images({"pair_file": str(pairs_out)}, _identity_transform)
    assert images.shape[1:] == (3, 112, 112) and len(index_pairs) == len(is_same) == len(pairs)

    # refuses to overwrite the protocol silently
    monkeypatch.setattr(sys, "argv", sys.argv)
    assert scan_webdataset.main() == 1
