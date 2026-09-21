"""Census a streaming (WebDataset) dataset and build its held-out validation split.

**Run this before the first training run on a streaming dataset.**

    # 1. census only: count images per identity, cache it, print a summary
    python -m scripts.scan_webdataset --config configs/essi_fr_v1_local.yaml

    # 2. also hold out identities for an honest verification benchmark
    python -m scripts.scan_webdataset --config configs/essi_fr_v1_local.yaml \\
        --holdout 100 --holdout-min-images 2 \\
        --holdout-out data/glint360k_val \\
        --identities-out data/splits/glint360k_val_identities.txt \\
        --pairs-out data/pairs/glint360k_val_pairs.txt

What it writes
--------------
``<scan_cache>/census_<key>.json``
    Per-identity image counts for the configured shards. Training loads this
    instead of re-streaming; it is keyed by the shard list, so adding shards
    triggers a new census automatically.

``--identities-out`` (with ``--holdout``)
    The held-out identities. The training config's
    ``data.adapter.exclude_identities_file`` points here, so these people never
    appear in training.

``--holdout-out/<identity>/<key>.jpg`` and ``--pairs-out``
    The held-out images, extracted from the shards as plain JPEG files, and a
    balanced positive/negative pair list over them. The ordinary
    ``type: pairs`` eval target then works unchanged.

Why the split must come first
-----------------------------
If validation pairs come from identities the model trained on, a high score
proves memorisation. Holding identities out of *training*, not merely out of
the pair list, is the only way to get an honest number. (The same rule as
``scripts/build_meglass_pairs.py``.)

Cost
----
The census reads every ``.cls`` entry of every configured shard: seconds for
two shards, ~15-30 minutes for all 1,385 Glint360K shards from local NVMe with
``--workers 16``. Hold-out extraction is a second pass of the same cost.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from frs.config import load_config  # noqa: E402
from frs.data.adapters.folder_per_identity import FolderPerIdentityAdapter  # noqa: E402
from frs.data.adapters.streaming import StreamingAdapter, cached_census  # noqa: E402
from frs.data.adapters.webdataset import WebDatasetAdapter, iter_tar_samples  # noqa: E402
from frs.eval.pairs import build_pairs, write_identity_file, write_pair_file  # noqa: E402
from frs.registry import ADAPTERS  # noqa: E402
from frs.utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger("scan")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True, help="training config with a webdataset adapter")
    p.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    p.add_argument("--force", action="store_true", help="redo the census / overwrite split files")

    g = p.add_argument_group("hold-out split")
    g.add_argument("--holdout", type=int, default=0, help="number of identities to hold out")
    g.add_argument("--holdout-min-images", type=int, default=4, help="eligibility threshold")
    g.add_argument(
        "--holdout-max-images", type=int, default=20,
        help="cap on extracted images per held-out identity",
    )
    g.add_argument("--holdout-out", default="data/glint360k_val", help="folder for extracted JPEGs")
    g.add_argument("--identities-out", default="data/splits/glint360k_val_identities.txt")
    g.add_argument("--pairs-out", default="data/pairs/glint360k_val_pairs.txt")
    g.add_argument("--num-positive", type=int, default=3000)
    g.add_argument("--num-negative", type=int, default=3000)
    g.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ----------------------------------------------------------------- extraction


def _extract_shard(args: tuple) -> tuple[str, int]:
    """Pool worker: write every sample of a wanted identity to ``out/<identity>/``."""
    url, image_key, label_key, wanted, out_dir, prefix, handler = args
    out_dir = Path(out_dir)
    written = 0
    for sample in iter_tar_samples(url, (image_key, label_key), handler=handler):
        try:
            raw = int(sample[label_key])
        except (TypeError, ValueError):
            continue
        if raw not in wanted:
            continue
        ident_dir = out_dir / f"{prefix}{raw}".replace("/", "_")
        ident_dir.mkdir(parents=True, exist_ok=True)
        key = Path(sample["__key__"]).name
        (ident_dir / f"{key}.{image_key}").write_bytes(sample[image_key])
        written += 1
    return url, written


def extract_holdout(
    adapter: WebDatasetAdapter,
    wanted_raw: set[int],
    out_dir: Path,
    max_images: int,
    workers: int,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (url, adapter.image_key, adapter.label_key, frozenset(wanted_raw),
         str(out_dir), adapter.prefix, adapter.handler)
        for url in adapter.shard_list()
    ]
    total = 0
    if workers <= 1 or len(jobs) == 1:
        results = map(_extract_shard, jobs)
    else:
        ctx = multiprocessing.get_context("spawn" if os.name == "nt" else None)
        pool = ctx.Pool(min(workers, len(jobs)))
        results = pool.imap_unordered(_extract_shard, jobs)
    for i, (_, n) in enumerate(results, start=1):
        total += n
        if i % 25 == 0 or i == len(jobs):
            logger.info("  extracted %d/%d shards (%d images)", i, len(jobs), total)
    if workers > 1 and len(jobs) > 1:
        pool.close()
        pool.join()

    # Enforce the per-identity cap (workers cannot coordinate across shards).
    trimmed = 0
    for ident_dir in out_dir.iterdir():
        if not ident_dir.is_dir():
            continue
        files = sorted(ident_dir.iterdir())
        for extra in files[max_images:]:
            extra.unlink()
            trimmed += 1
    if trimmed:
        logger.info("  trimmed %d images above the %d-per-identity cap", trimmed, max_images)
    return total - trimmed


# ----------------------------------------------------------------------- main


def main() -> int:
    args = parse_args()
    setup_logging()

    cfg = load_config(args.config)
    adapter = ADAPTERS.build(dict(cfg.data.adapter))
    if not isinstance(adapter, StreamingAdapter):
        print(
            f"data.adapter.type={cfg.data.adapter.type!r} is a map-style adapter; "
            f"this script is for streaming (webdataset) datasets.",
            file=sys.stderr,
        )
        return 1

    if args.holdout:
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

    # ---------------------------------------------------------------- census
    shards = adapter.shard_list()
    print(f"Census over {len(shards)} shard(s) with {args.workers} workers ...")
    census = cached_census(
        adapter, cfg.data.get("scan_cache"), workers=args.workers,
        explicit_path=getattr(adapter, "census_file", None), force=args.force,
    )
    s = census.summary()
    print(
        f"  {s['num_samples']:,} images, {s['num_identities']:,} identities "
        f"({s['min_per_identity']}-{s['max_per_identity']} per identity, "
        f"mean {s['mean_per_identity']:.1f}, median {s['median_per_identity']:.0f})"
    )
    keep = census.kept_mask(adapter.min_images_per_identity)
    print(
        f"  with min_images_per_identity={adapter.min_images_per_identity}: "
        f"{int(census.counts[keep].sum()):,} images of {int(keep.sum()):,} identities remain"
    )

    if not args.holdout:
        return 0

    # --------------------------------------------------------------- hold-out
    eligible = census.raw_ids[census.counts >= args.holdout_min_images]
    if eligible.size < args.holdout:
        print(
            f"ERROR: only {eligible.size} identities have >= {args.holdout_min_images} images "
            f"in the configured shards; cannot hold out {args.holdout}. Lower "
            f"--holdout / --holdout-min-images or configure more shards.",
            file=sys.stderr,
        )
        return 1
    rng = np.random.default_rng(args.seed)
    chosen = np.sort(rng.choice(eligible, size=args.holdout, replace=False))
    identities = sorted(adapter.identity_of(int(r)) for r in chosen.tolist())
    write_identity_file(args.identities_out, identities)
    print(
        f"\nHeld out {len(identities)} identities (seed={args.seed}, "
        f">= {args.holdout_min_images} images each) -> {args.identities_out}"
    )

    holdout_dir = Path(args.holdout_out)
    if holdout_dir.exists() and args.force:
        import shutil

        shutil.rmtree(holdout_dir)
    print(f"Extracting their images to {holdout_dir} ...")
    if not isinstance(adapter, WebDatasetAdapter):
        print("hold-out extraction is implemented for the webdataset adapter only", file=sys.stderr)
        return 1
    n_images = extract_holdout(
        adapter, set(int(r) for r in chosen.tolist()), holdout_dir,
        args.holdout_max_images, args.workers,
    )
    print(f"  {n_images:,} images written")

    samples = FolderPerIdentityAdapter(root=str(holdout_dir), prefix=adapter.prefix).scan()
    pairs, stats = build_pairs(
        samples,
        identities=sorted({s.identity for s in samples}),
        num_positive=args.num_positive,
        num_negative=args.num_negative,
        seed=args.seed,
    )
    write_pair_file(args.pairs_out, pairs)
    print(
        f"\nPairs: {stats['num_pairs']:,} total ({stats['num_positive']:,} positive, "
        f"{stats['num_negative']:,} negative) from {stats['num_identities']:,} identities "
        f"-> {args.pairs_out}"
    )
    print(
        "\nMake sure the training config has:\n"
        f"  data.adapter.exclude_identities_file: {args.identities_out}\n"
        f"  eval.targets: [{{name: glint_val, type: pairs, pair_file: {args.pairs_out}}}]\n"
        "You can start training."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
