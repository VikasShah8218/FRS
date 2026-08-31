"""Identity string <-> contiguous class index, with safe extension.

Why this exists
---------------
The classifier head is a matrix of shape ``(num_classes, embedding_size)`` whose
row *i* is the learned prototype for class *i*. If class indices were reassigned
between training runs, every learned row would suddenly refer to a different
person and the head would be worthless. :class:`ClassMap` is the durable record
that prevents that: it lives in the checkpoint alongside the weights.

Invariants (enforced by ``tests/test_classmap.py``)
---------------------------------------------------
1. An identity's index is assigned once and **never** changes.
2. New identities always receive indices in ``[old_num_classes, new_num_classes)``,
   so the existing head rows stay valid and only new rows are appended.
3. First creation sorts identities, so re-scanning the same folder reproduces
   exactly the same mapping.
4. Identical identity strings from different sources are treated as the *same*
   person. If that is wrong for a given merge, namespace them with the adapter's
   ``prefix`` option.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .adapters.base import Sample


@dataclass
class ExtensionResult:
    """What :meth:`ClassMap.extend` did -- logged loudly and worth asserting on."""

    old_num_classes: int
    new_num_classes: int
    added: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)

    @property
    def num_added(self) -> int:
        return len(self.added)

    @property
    def changed(self) -> bool:
        return self.new_num_classes != self.old_num_classes

    def __str__(self) -> str:
        return (
            f"ClassMap extended: {self.old_num_classes} -> {self.new_num_classes} "
            f"classes ({self.num_added} new, {len(self.reused)} already known)"
        )


class ClassMap:
    """A bidirectional, append-only identity <-> index mapping."""

    def __init__(
        self,
        identities: Sequence[str] | None = None,
        source_tags: Sequence[str] | None = None,
        default_source: str = "unknown",
    ) -> None:
        self.index_to_identity: list[str] = list(identities or [])
        self.identity_to_index: dict[str, int] = {
            ident: i for i, ident in enumerate(self.index_to_identity)
        }
        if len(self.identity_to_index) != len(self.index_to_identity):
            dupes = [
                ident
                for ident, n in Counter(self.index_to_identity).items()
                if n > 1
            ]
            raise ValueError(f"duplicate identities in ClassMap: {dupes[:5]}")

        self.default_source = default_source
        if source_tags is None:
            self.source_tags: list[str] = [default_source] * len(self.index_to_identity)
        else:
            if len(source_tags) != len(self.index_to_identity):
                raise ValueError(
                    f"source_tags length {len(source_tags)} != "
                    f"num identities {len(self.index_to_identity)}"
                )
            self.source_tags = list(source_tags)

    # ------------------------------------------------------------ construction

    @classmethod
    def from_samples(
        cls,
        samples: Iterable[Sample],
        sort: bool = True,
        source: str = "unknown",
    ) -> "ClassMap":
        """Build a fresh map from scanned samples.

        ``sort=True`` (the default) makes the mapping deterministic across runs
        and machines -- important because the checkpoint's head rows are only
        meaningful relative to this ordering.
        """
        identities = {s.identity for s in samples}
        ordered = sorted(identities) if sort else list(dict.fromkeys(
            s.identity for s in samples
        ))
        return cls(ordered, [source] * len(ordered), default_source=source)

    # --------------------------------------------------------------- accessors

    def __len__(self) -> int:
        return len(self.index_to_identity)

    def __contains__(self, identity: object) -> bool:
        return identity in self.identity_to_index

    @property
    def num_classes(self) -> int:
        return len(self.index_to_identity)

    def index_of(self, identity: str) -> int:
        try:
            return self.identity_to_index[identity]
        except KeyError as exc:
            raise KeyError(
                f"identity {identity!r} is not in the ClassMap "
                f"({self.num_classes} known). Call extend() first, or check the "
                f"adapter's prefix setting."
            ) from exc

    def identity_of(self, index: int) -> str:
        try:
            return self.index_to_identity[index]
        except IndexError as exc:
            raise IndexError(
                f"class index {index} out of range (num_classes={self.num_classes})"
            ) from exc

    def source_of(self, index: int) -> str:
        return self.source_tags[index]

    def counts_by_source(self) -> dict[str, int]:
        return dict(Counter(self.source_tags))

    # --------------------------------------------------------------- extension

    def extend(
        self, new_identities: Iterable[str], source: str | None = None
    ) -> ExtensionResult:
        """Append previously unseen identities, preserving all existing indices.

        This is the mechanism behind incremental training: after calling this,
        pass the result to :func:`frs.engine.checkpoint.extend_head_and_optimizer`
        to grow the classifier weight matrix and the optimizer state to match.
        """
        source = source or self.default_source
        old = self.num_classes
        added: list[str] = []
        reused: list[str] = []

        for identity in new_identities:
            if identity in self.identity_to_index:
                reused.append(identity)
                continue
            if identity in added:  # duplicate within this same call
                continue
            added.append(identity)

        for identity in added:
            self.identity_to_index[identity] = len(self.index_to_identity)
            self.index_to_identity.append(identity)
            self.source_tags.append(source)

        return ExtensionResult(
            old_num_classes=old,
            new_num_classes=self.num_classes,
            added=added,
            reused=reused,
        )

    def extend_from_samples(
        self, samples: Iterable[Sample], source: str | None = None
    ) -> ExtensionResult:
        return self.extend(sorted({s.identity for s in samples}), source=source)

    # ----------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, Any]:
        # index_to_identity alone is sufficient (identity_to_index is derivable),
        # but both are stored so a checkpoint can be inspected without code.
        return {
            "index_to_identity": list(self.index_to_identity),
            "identity_to_index": dict(self.identity_to_index),
            "source_tags": list(self.source_tags),
            "default_source": self.default_source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClassMap":
        obj = cls(
            data["index_to_identity"],
            data.get("source_tags"),
            default_source=data.get("default_source", "unknown"),
        )
        # Trust the stored forward map if present -- it is the authority on the
        # index assignment the checkpoint's head rows were trained against.
        stored = data.get("identity_to_index")
        if stored and dict(stored) != obj.identity_to_index:
            raise ValueError(
                "ClassMap is inconsistent: identity_to_index does not match "
                "index_to_identity. Refusing to load a corrupt mapping."
            )
        return obj

    def save(self, path: str) -> None:
        import json

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "ClassMap":
        import json

        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def __repr__(self) -> str:
        return f"ClassMap(num_classes={self.num_classes}, sources={self.counts_by_source()})"
