"""ClassMap invariants.

These guard the property that makes incremental training possible: an
identity's class index, once assigned, never changes. If these tests fail, every
previously-trained head row silently refers to the wrong person.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frs.data.adapters.base import Sample  # noqa: E402
from frs.data.class_map import ClassMap  # noqa: E402


def make_samples(identities):
    return [Sample(path=f"/tmp/{i}.jpg", identity=ident) for i, ident in enumerate(identities)]


def test_from_samples_is_sorted_and_deterministic():
    a = ClassMap.from_samples(make_samples(["charlie", "alice", "bob", "alice"]))
    b = ClassMap.from_samples(make_samples(["bob", "alice", "charlie"]))
    assert a.index_to_identity == ["alice", "bob", "charlie"]
    # Same identity set in a different order must produce the same mapping,
    # otherwise a re-scan would invalidate a trained head.
    assert a.identity_to_index == b.identity_to_index


def test_extend_preserves_existing_indices():
    cm = ClassMap.from_samples(make_samples(["alice", "bob"]))
    original = dict(cm.identity_to_index)

    result = cm.extend(["carol", "dave", "alice"])

    for identity, index in original.items():
        assert cm.identity_to_index[identity] == index, "existing index changed"
    assert result.old_num_classes == 2
    assert result.new_num_classes == 4
    assert result.added == ["carol", "dave"]
    assert result.reused == ["alice"]


def test_new_indices_are_contiguous_and_appended():
    cm = ClassMap.from_samples(make_samples(["a", "b", "c"]))
    cm.extend(["x", "y"])
    assert cm.identity_to_index["x"] == 3
    assert cm.identity_to_index["y"] == 4
    assert cm.num_classes == 5


def test_extend_is_idempotent():
    cm = ClassMap.from_samples(make_samples(["a", "b"]))
    cm.extend(["c"])
    snapshot = dict(cm.identity_to_index)
    result = cm.extend(["c"])
    assert not result.changed
    assert cm.identity_to_index == snapshot


def test_extend_deduplicates_within_one_call():
    cm = ClassMap.from_samples(make_samples(["a"]))
    result = cm.extend(["b", "b", "c"])
    assert result.added == ["b", "c"]
    assert cm.num_classes == 3


def test_roundtrip_serialisation():
    cm = ClassMap.from_samples(make_samples(["a", "b", "c"]), source="ds1")
    cm.extend(["d"], source="ds2")

    restored = ClassMap.from_dict(cm.to_dict())
    assert restored.index_to_identity == cm.index_to_identity
    assert restored.identity_to_index == cm.identity_to_index
    assert restored.source_tags == cm.source_tags
    assert restored.counts_by_source() == {"ds1": 3, "ds2": 1}


def test_corrupt_mapping_is_rejected():
    data = {
        "index_to_identity": ["a", "b"],
        "identity_to_index": {"a": 0, "b": 5},  # inconsistent
        "source_tags": ["s", "s"],
    }
    with pytest.raises(ValueError, match="inconsistent"):
        ClassMap.from_dict(data)


def test_duplicate_identities_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        ClassMap(["a", "b", "a"])


def test_unknown_identity_raises_helpful_error():
    cm = ClassMap.from_samples(make_samples(["a"]))
    with pytest.raises(KeyError, match="not in the ClassMap"):
        cm.index_of("zzz")


def test_source_tags_track_provenance():
    cm = ClassMap.from_samples(make_samples(["a", "b"]), source="meglass")
    cm.extend(["c"], source="client_a")
    assert cm.source_of(0) == "meglass"
    assert cm.source_of(2) == "client_a"
