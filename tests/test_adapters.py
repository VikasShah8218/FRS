"""Dataset adapter behaviour, including the real MeGlass counts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frs.data.adapters.base import DatasetAdapter, exclude_identities, filter_by_min_images  # noqa: E402
from frs.data.adapters.csv_manifest import CsvManifestAdapter  # noqa: E402
from frs.data.adapters.flat_regex import DEFAULT_PATTERN, FlatRegexAdapter  # noqa: E402
from frs.data.adapters.folder_per_identity import FolderPerIdentityAdapter  # noqa: E402
from frs.registry import ADAPTERS  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MEGLASS = REPO / "MeGlass_120x120"


def write_image(path: Path, size=(120, 120)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    ).save(path)


# ----------------------------------------------------------------- registry


def test_all_adapters_registered():
    assert set(ADAPTERS.names()) >= {
        "flat_regex", "folder_per_identity", "csv_manifest", "mxnet_rec", "webdataset"
    }


def test_registry_build_and_unknown_type():
    with pytest.raises(KeyError, match="unknown type"):
        ADAPTERS.build({"type": "does_not_exist"})


# --------------------------------------------------------------- flat_regex


def test_meglass_identity_pattern_handles_at_in_user_id():
    """The Flickr id itself contains '@', so identity is everything before the LAST one."""
    import re

    rx = re.compile(DEFAULT_PATTERN)
    stem = "7276470@N03_identity_3@8733080112_2"
    assert rx.match(stem).group("identity") == "7276470@N03_identity_3"


def test_flat_regex_on_synthetic_tree(tmp_path):
    for stem in [
        "111@N01_identity_0@900_0",
        "111@N01_identity_0@901_0",
        "111@N01_identity_1@902_0",
        "222@N02_identity_0@903_0",
    ]:
        write_image(tmp_path / f"{stem}.jpg")

    samples = FlatRegexAdapter(root=str(tmp_path)).scan()
    assert len(samples) == 4
    stats = DatasetAdapter.summarize(samples)
    assert stats["num_identities"] == 3


def test_flat_regex_requires_identity_group(tmp_path):
    with pytest.raises(ValueError, match="named group"):
        FlatRegexAdapter(root=str(tmp_path), pattern=r"^(.+)@")


def test_prefix_namespaces_identities(tmp_path):
    write_image(tmp_path / "a@N01_identity_0@1_0.jpg")
    samples = FlatRegexAdapter(root=str(tmp_path), prefix="client_a/").scan()
    assert samples[0].identity.startswith("client_a/")


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        FlatRegexAdapter(root=str(tmp_path / "nope")).scan()


def test_cache_key_changes_with_config(tmp_path):
    a = FlatRegexAdapter(root=str(tmp_path))
    b = FlatRegexAdapter(root=str(tmp_path), min_images_per_identity=5)
    assert a.cache_key != b.cache_key


# ------------------------------------------------------- folder_per_identity


def test_folder_per_identity(tmp_path):
    for identity, n in (("alice", 3), ("bob", 2)):
        for i in range(n):
            write_image(tmp_path / identity / f"{i}.jpg")

    samples = FolderPerIdentityAdapter(root=str(tmp_path)).scan()
    assert len(samples) == 5
    assert {s.identity for s in samples} == {"alice", "bob"}


def test_folder_per_identity_nested(tmp_path):
    write_image(tmp_path / "alice" / "session1" / "a.jpg")
    write_image(tmp_path / "alice" / "session2" / "b.jpg")
    samples = FolderPerIdentityAdapter(root=str(tmp_path), recursive=True).scan()
    assert len(samples) == 2
    assert all(s.identity == "alice" for s in samples)


# ------------------------------------------------------------ csv_manifest


def test_csv_manifest(tmp_path):
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        write_image(tmp_path / "images" / name)
    manifest = tmp_path / "labels.csv"
    manifest.write_text(
        "path,identity\nimages/a.jpg,alice\nimages/b.jpg,alice\nimages/c.jpg,bob\n",
        encoding="utf-8",
    )

    samples = CsvManifestAdapter(manifest=str(manifest), root=str(tmp_path)).scan()
    assert len(samples) == 3
    assert DatasetAdapter.summarize(samples)["num_identities"] == 2


def test_csv_manifest_reports_missing_files(tmp_path):
    manifest = tmp_path / "labels.csv"
    manifest.write_text("path,identity\nghost.jpg,alice\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="do not exist"):
        CsvManifestAdapter(
            manifest=str(manifest), root=str(tmp_path), verify_exists=True
        ).scan()


# -------------------------------------------------------------- filtering


def test_filter_by_min_images():
    from frs.data.adapters.base import Sample

    samples = (
        [Sample(f"/a{i}.jpg", "rare") for i in range(2)]
        + [Sample(f"/b{i}.jpg", "common") for i in range(10)]
    )
    kept, dropped = filter_by_min_images(samples, 5)
    assert dropped == 1
    assert {s.identity for s in kept} == {"common"}


def test_exclude_identities():
    from frs.data.adapters.base import Sample

    samples = [Sample("/a.jpg", "alice"), Sample("/b.jpg", "bob")]
    kept, removed = exclude_identities(samples, {"alice"})
    assert removed == 1 and kept[0].identity == "bob"


# ----------------------------------------------------- real dataset (opt-in)


@pytest.mark.skipif(not MEGLASS.is_dir(), reason="MeGlass dataset not present")
def test_meglass_real_counts():
    """Guards the exact verified numbers for the shipped dataset."""
    samples = FlatRegexAdapter(root=str(MEGLASS)).scan()
    stats = DatasetAdapter.summarize(samples)
    assert stats["num_samples"] == 47917
    assert stats["num_identities"] == 1710
    assert stats["min_per_identity"] == 4
    assert stats["max_per_identity"] == 578


@pytest.mark.skipif(not MEGLASS.is_dir(), reason="MeGlass dataset not present")
def test_meglass_validation_split_is_excluded():
    """The held-out identities must not appear in the training set."""
    split = REPO / "data" / "splits" / "meglass_val_identities.txt"
    if not split.is_file():
        pytest.skip("run scripts/build_meglass_pairs.py first")

    excluded = {
        line.strip()
        for line in split.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    samples = FlatRegexAdapter(
        root=str(MEGLASS), exclude_identities_file=str(split)
    ).scan()

    present = {s.identity for s in samples}
    assert not (present & excluded), "validation identities leaked into training"
    assert DatasetAdapter.summarize(samples)["num_identities"] == 1710 - len(excluded)
