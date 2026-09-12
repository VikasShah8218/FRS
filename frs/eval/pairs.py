"""Verification pair lists: building them, and loading them for evaluation.

MeGlass ships no official pair protocol, so this module builds one. The rule
that makes the resulting number meaningful:

    **Validation identities must be held out of training entirely.**

If a pair is drawn from an identity the model trained on, a high score proves
memorisation, not verification ability. ``scripts/build_meglass_pairs.py`` writes
the held-out identity list, and the training adapter's ``exclude_identities_file``
removes exactly those identities from the training set.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..data.adapters.base import Sample

logger = logging.getLogger(__name__)


def split_identities(
    samples: Sequence[Sample],
    num_val_identities: int,
    seed: int = 42,
    min_images: int = 4,
) -> tuple[list[str], list[str]]:
    """Partition identities into (train, val) sets.

    Validation identities are sampled only from those with at least
    ``min_images`` images, so every held-out identity can contribute genuine
    positive pairs. Sorting before sampling makes the split reproducible across
    machines regardless of filesystem ordering.
    """
    by_identity: dict[str, int] = defaultdict(int)
    for sample in samples:
        by_identity[sample.identity] += 1

    eligible = sorted(i for i, n in by_identity.items() if n >= min_images)
    if len(eligible) < num_val_identities:
        raise ValueError(
            f"only {len(eligible)} identities have >= {min_images} images, "
            f"cannot hold out {num_val_identities}"
        )

    rng = np.random.default_rng(seed)
    val = sorted(rng.choice(eligible, size=num_val_identities, replace=False).tolist())
    val_set = set(val)
    train = sorted(i for i in by_identity if i not in val_set)
    return train, val


def build_pairs(
    samples: Sequence[Sample],
    identities: Sequence[str],
    num_positive: int = 3000,
    num_negative: int = 3000,
    seed: int = 42,
) -> tuple[list[tuple[str, str, bool]], dict[str, Any]]:
    """Generate balanced positive/negative verification pairs.

    Positive pairs are two distinct images of the same identity; negative pairs
    are images of two different identities. Both are sampled without repetition
    of the exact same pair.
    """
    wanted = set(identities)
    by_identity: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        if sample.identity in wanted:
            by_identity[sample.identity].append(sample.path)

    usable = {i: sorted(p) for i, p in by_identity.items() if len(p) >= 2}
    if len(usable) < 2:
        raise ValueError(
            f"need >= 2 identities with >= 2 images each; got {len(usable)}"
        )

    rng = np.random.default_rng(seed)
    ident_list = sorted(usable)

    # --- positives ---
    positives: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = num_positive * 50
    while len(positives) < num_positive and attempts < max_attempts:
        attempts += 1
        identity = ident_list[int(rng.integers(len(ident_list)))]
        paths = usable[identity]
        i, j = rng.choice(len(paths), size=2, replace=False)
        positives.add(tuple(sorted((paths[int(i)], paths[int(j)]))))

    # --- negatives ---
    negatives: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = num_negative * 50
    while len(negatives) < num_negative and attempts < max_attempts:
        attempts += 1
        a, b = rng.choice(len(ident_list), size=2, replace=False)
        pa = usable[ident_list[int(a)]]
        pb = usable[ident_list[int(b)]]
        left = pa[int(rng.integers(len(pa)))]
        right = pb[int(rng.integers(len(pb)))]
        negatives.add(tuple(sorted((left, right))))

    if len(positives) < num_positive or len(negatives) < num_negative:
        logger.warning(
            "Requested %d/%d pairs but could only build %d positive and %d negative "
            "(the identity pool is small)",
            num_positive,
            num_negative,
            len(positives),
            len(negatives),
        )

    pairs = [(a, b, True) for a, b in sorted(positives)]
    pairs += [(a, b, False) for a, b in sorted(negatives)]

    # Interleave so every fold of the 10-fold protocol sees both classes.
    order = rng.permutation(len(pairs))
    pairs = [pairs[int(i)] for i in order]

    stats = {
        "num_pairs": len(pairs),
        "num_positive": len(positives),
        "num_negative": len(negatives),
        "num_identities": len(usable),
    }
    return pairs, stats


def write_pair_file(path: str | Path, pairs: Sequence[tuple[str, str, bool]]) -> None:
    """Write ``path_a<TAB>path_b<TAB>0|1``, one pair per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# path_a\tpath_b\tis_same\n")
        for a, b, same in pairs:
            fh.write(f"{a}\t{b}\t{int(same)}\n")


def read_pair_file(path: str | Path) -> list[tuple[str, str, bool]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"pair file not found: {path}. Generate it with "
            f"scripts/build_meglass_pairs.py"
        )
    pairs: list[tuple[str, str, bool]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path}: malformed line: {line!r}")
            pairs.append((parts[0], parts[1], bool(int(parts[2]))))
    return pairs


def write_identity_file(path: str | Path, identities: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# {len(identities)} held-out validation identities\n")
        fh.write("# These MUST be excluded from training via the adapter's\n")
        fh.write("# exclude_identities_file option, or the reported accuracy\n")
        fh.write("# measures memorisation rather than verification.\n")
        for identity in identities:
            fh.write(f"{identity}\n")


def load_pair_images(
    target: dict[str, Any], transform: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a pair file into ``(images, index_pairs, is_same)``.

    Each unique path is loaded and preprocessed exactly once; the pair array
    indexes into that stack, which roughly halves both I/O and forward passes.
    """
    from PIL import Image

    pairs = read_pair_file(target["pair_file"])

    unique: dict[str, int] = {}
    for a, b, _ in pairs:
        unique.setdefault(a, len(unique))
        unique.setdefault(b, len(unique))

    ordered = sorted(unique, key=lambda p: unique[p])
    stack: list[np.ndarray] = []
    for path in ordered:
        with Image.open(path) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        stack.append(transform(arr))

    images = np.stack(stack).astype(np.float32)
    index_pairs = np.array([[unique[a], unique[b]] for a, b, _ in pairs], dtype=np.int64)
    is_same = np.array([same for _, _, same in pairs], dtype=bool)
    return images, index_pairs, is_same


EVAL_TARGET_TYPES = ("pairs", "bin")


def load_eval_images(
    target: dict[str, Any], transform: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch on ``target["type"]``: ``pairs`` (text pair file) or ``bin``.

    Both loaders return ``(images, index_pairs, is_same)`` so
    :func:`frs.eval.verification.evaluate_target` is format-agnostic.
    """
    kind = str(target.get("type", "pairs")).lower()
    if kind == "pairs":
        return load_pair_images(target, transform)
    if kind == "bin":
        from .bin_pack import load_bin_images

        return load_bin_images(target, transform)
    raise ValueError(
        f"eval target {target.get('name')!r}: unknown type {kind!r}; "
        f"expected one of {EVAL_TARGET_TYPES}"
    )
