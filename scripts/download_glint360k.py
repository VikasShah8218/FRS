"""Download Glint360K WebDataset shards from HuggingFace.

Standalone on purpose: this file imports nothing from ``frs`` so it can run in
a separate terminal / process (even a different environment) while the
training code is untouched. Training then points ``data.adapter.shards`` at
whatever has landed.

    # two shards (~190 MB) for a local plumbing run
    python scripts/download_glint360k.py --out D:/data/glint360k --shards 0-1

    # everything (1,385 shards, ~130 GB) on the AWS box
    python scripts/download_glint360k.py --out /mnt/data/glint360k --shards all --workers 8

    # a specific selection
    python scripts/download_glint360k.py --out D:/data/glint360k --shards 0-9,100,200-210

Downloads are resumable: a shard that is already complete is skipped, and a
partially transferred one continues from where it stopped. Pass ``--verify``
to gzip-read each shard end to end afterwards (slow, but catches truncation).

Dataset: https://huggingface.co/datasets/gaunernst/glint360k-wds-gz
  17,091,657 images / 360,232 identities / 1,385 shards of ~94 MB
  Glint360K (An et al., Partial FC, arXiv:2010.05222), aligned 112x112 by InsightFace.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ID = "gaunernst/glint360k-wds-gz"
NUM_SHARDS = 1385
SHARD_MB = 94.0
SHARD_NAME = "glint360k-{:04d}.tar.gz"


def parse_shards(spec: str, total: int = NUM_SHARDS) -> list[int]:
    """``"0-1"``, ``"0-9,100,200-210"`` or ``"all"`` -> sorted shard indices."""
    if spec.strip().lower() == "all":
        return list(range(total))
    chosen: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            lo_i, hi_i = int(lo), int(hi)
            if lo_i > hi_i:
                raise ValueError(f"bad range {part!r}")
            chosen.update(range(lo_i, hi_i + 1))
        else:
            chosen.add(int(part))
    bad = [i for i in chosen if not 0 <= i < total]
    if bad:
        raise ValueError(f"shard indices out of range 0..{total - 1}: {sorted(bad)[:5]}")
    if not chosen:
        raise ValueError("no shards selected")
    return sorted(chosen)


def brace_pattern(indices: list[int], out_dir: Path) -> str:
    """Config-ready pattern for a contiguous selection, or a list otherwise."""
    if indices == list(range(indices[0], indices[-1] + 1)):
        return f"{out_dir.as_posix()}/glint360k-{{{indices[0]:04d}..{indices[-1]:04d}}}.tar.gz"
    return "[" + ", ".join(f'"{(out_dir / SHARD_NAME.format(i)).as_posix()}"' for i in indices) + "]"


def verify_gzip(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as fh:
            while fh.read(1 << 20):
                pass
        return True
    except Exception:
        return False


def download_one(index: int, out_dir: Path, revision: str, force: bool) -> tuple[int, str, float]:
    from huggingface_hub import hf_hub_download

    name = SHARD_NAME.format(index)
    dest = out_dir / name
    if dest.is_file() and not force:
        return index, "present", dest.stat().st_size / 1e6
    t0 = time.time()
    hf_hub_download(
        repo_id=REPO_ID,
        filename=name,
        repo_type="dataset",
        revision=revision,
        local_dir=str(out_dir),
        force_download=force,
    )
    return index, f"downloaded in {time.time() - t0:.0f}s", dest.stat().st_size / 1e6


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", required=True, help="directory to place the shards in")
    parser.add_argument("--shards", default="0-1", help='e.g. "0-1", "0-49,100", or "all"')
    parser.add_argument("--workers", type=int, default=2, help="parallel downloads")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--force", action="store_true", help="re-download present shards")
    parser.add_argument("--verify", action="store_true", help="gzip-read every shard afterwards")
    parser.add_argument("--dry-run", action="store_true", help="list what would be fetched")
    args = parser.parse_args()

    try:
        indices = parse_shards(args.shards)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    est_gb = len(indices) * SHARD_MB / 1024
    free_gb = shutil.disk_usage(out_dir).free / 1024**3
    print(
        f"{len(indices)} shard(s) from {REPO_ID} -> {out_dir}\n"
        f"  estimated size {est_gb:.1f} GB, free space {free_gb:.1f} GB"
    )
    if est_gb > free_gb * 0.95:
        print("error: not enough free disk for this selection", file=sys.stderr)
        return 2

    if args.dry_run:
        for i in indices:
            print("  ", SHARD_NAME.format(i))
        print("\nConfig pattern:\n  data.adapter.shards:", brace_pattern(indices, out_dir))
        return 0

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("error: pip install huggingface_hub", file=sys.stderr)
        return 2

    failures: list[tuple[int, str]] = []
    total_mb = 0.0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(download_one, i, out_dir, args.revision, args.force): i for i in indices
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                _, status, size_mb = future.result()
                total_mb += size_mb
                print(f"  {SHARD_NAME.format(index)}  {size_mb:6.1f} MB  {status}")
            except Exception as exc:  # keep going; report at the end
                failures.append((index, str(exc)))
                print(f"  {SHARD_NAME.format(index)}  FAILED: {exc}", file=sys.stderr)

    if args.verify:
        print("\nVerifying gzip integrity ...")
        for i in indices:
            path = out_dir / SHARD_NAME.format(i)
            if path.is_file() and not verify_gzip(path):
                failures.append((i, "gzip verification failed (truncated?)"))
                print(f"  {path.name}: CORRUPT -- delete it and re-run", file=sys.stderr)

    print(f"\n{len(indices) - len(failures)}/{len(indices)} shards ready, {total_mb / 1024:.2f} GB")
    if failures:
        print("Failures:", file=sys.stderr)
        for i, msg in failures:
            print(f"  {SHARD_NAME.format(i)}: {msg}", file=sys.stderr)
        print("Re-run the same command to resume.", file=sys.stderr)
        return 1

    print("\nPoint the training config at them:")
    print("  data:\n    adapter:\n      type: webdataset")
    print(f"      shards: \"{brace_pattern(indices, out_dir)}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
