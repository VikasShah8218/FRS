"""Build the identity-disjoint validation split and verification pair list.

**Run this before the first training run.** It writes two files:

``data/splits/meglass_val_identities.txt``
    The held-out identities. The training config's ``exclude_identities_file``
    points at this, so these people never appear in training.

``data/pairs/meglass_val_pairs.txt``
    Positive and negative pairs drawn only from those held-out identities.

Why the split must come first
-----------------------------
If validation pairs come from identities the model trained on, a high score
proves the model memorised those faces -- it says nothing about whether it can
verify a stranger. Holding the identities out of *training*, not merely out of
the pair list, is the only way to get an honest number.

Usage
-----
    python -m scripts.build_meglass_pairs
    python -m scripts.build_meglass_pairs --num-val-identities 150 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frs.data.adapters.base import DatasetAdapter  # noqa: E402
from frs.data.adapters.flat_regex import DEFAULT_PATTERN, FlatRegexAdapter  # noqa: E402
from frs.eval.pairs import (  # noqa: E402
    build_pairs,
    split_identities,
    write_identity_file,
    write_pair_file,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="MeGlass_120x120", help="dataset directory")
    p.add_argument("--pattern", default=DEFAULT_PATTERN, help="identity regex")
    p.add_argument(
        "--num-val-identities",
        type=int,
        default=300,
        help="identities held out of training entirely (default: 300 of 1710)",
    )
    p.add_argument("--num-positive", type=int, default=3000)
    p.add_argument("--num-negative", type=int, default=3000)
    p.add_argument(
        "--min-images",
        type=int,
        default=4,
        help="minimum images for an identity to be eligible for validation",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--identities-out", default="data/splits/meglass_val_identities.txt")
    p.add_argument("--pairs-out", default="data/pairs/meglass_val_pairs.txt")
    p.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing split files (this invalidates prior runs' metrics)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    for path in (args.identities_out, args.pairs_out):
        if Path(path).exists() and not args.force:
            print(
                f"ERROR: {path} already exists.\n"
                f"Regenerating the split would make previously reported accuracies "
                f"incomparable (and could leak identities the model has trained on).\n"
                f"Pass --force if you really intend to replace it.",
                file=sys.stderr,
            )
            return 1

    print(f"Scanning {args.root} ...")
    adapter = FlatRegexAdapter(root=args.root, pattern=args.pattern)
    samples = adapter.scan()
    stats = DatasetAdapter.summarize(samples)
    print(
        f"  {stats['num_samples']:,} images, {stats['num_identities']:,} identities "
        f"({stats['min_per_identity']}-{stats['max_per_identity']} per identity, "
        f"mean {stats['mean_per_identity']:.1f})"
    )

    train_ids, val_ids = split_identities(
        samples,
        num_val_identities=args.num_val_identities,
        seed=args.seed,
        min_images=args.min_images,
    )
    val_set = set(val_ids)
    n_train_imgs = sum(1 for s in samples if s.identity not in val_set)
    n_val_imgs = len(samples) - n_train_imgs

    print(
        f"\nSplit (seed={args.seed}):\n"
        f"  train : {len(train_ids):,} identities, {n_train_imgs:,} images\n"
        f"  val   : {len(val_ids):,} identities, {n_val_imgs:,} images (held out)"
    )

    pairs, pair_stats = build_pairs(
        samples,
        identities=val_ids,
        num_positive=args.num_positive,
        num_negative=args.num_negative,
        seed=args.seed,
    )
    print(
        f"\nPairs: {pair_stats['num_pairs']:,} total "
        f"({pair_stats['num_positive']:,} positive, {pair_stats['num_negative']:,} negative) "
        f"from {pair_stats['num_identities']:,} identities"
    )

    write_identity_file(args.identities_out, val_ids)
    write_pair_file(args.pairs_out, pairs)
    print(f"\nWrote {args.identities_out}\nWrote {args.pairs_out}")
    print(
        "\nThe training config's data.adapter.exclude_identities_file already "
        "points at the identity file, so these people are now excluded from "
        "training. You can start training."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
