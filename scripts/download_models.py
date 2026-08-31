"""Fetch pretrained models needed by the alignment pipeline.

    python -m scripts.download_models              # SCRFD face detector
    python -m scripts.download_models --list

Only the face detector is required, and only for aligning *new* datasets --
MeGlass is already cropped, so training works without it.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODELS = {
    "scrfd": {
        "url": (
            "https://huggingface.co/public-data/insightface/resolve/main/"
            "models/buffalo_l/det_10g.onnx"
        ),
        "dest": "models/det_10g.onnx",
        "size_mb": 16.9,
        "description": "SCRFD-10G face detector (RetinaFace family), ONNX",
    },
}


def download(url: str, dest: Path, expected_mb: float | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}\n         -> {dest}")

    def progress(count: int, block_size: int, total: int) -> None:
        if total > 0:
            done = count * block_size
            pct = min(100.0, done * 100.0 / total)
            sys.stdout.write(
                f"\r  {pct:5.1f}%  {done / 1e6:7.1f} / {total / 1e6:.1f} MB"
            )
            sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=progress)
    print()

    actual_mb = tmp.stat().st_size / 1e6
    if expected_mb and abs(actual_mb - expected_mb) > expected_mb * 0.2:
        tmp.unlink()
        raise RuntimeError(
            f"downloaded size {actual_mb:.1f} MB differs from the expected "
            f"{expected_mb:.1f} MB -- the download was likely truncated"
        )

    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    tmp.replace(dest)
    print(f"  saved {actual_mb:.1f} MB, sha256 {digest[:16]}...")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=["scrfd"], choices=[*MODELS, []])
    parser.add_argument("--list", action="store_true", help="list available models")
    parser.add_argument("--force", action="store_true", help="re-download if present")
    args = parser.parse_args()

    if args.list:
        for name, spec in MODELS.items():
            print(f"{name:<10} {spec['size_mb']:>6.1f} MB  {spec['description']}")
        return 0

    for name in args.models or ["scrfd"]:
        spec = MODELS[name]
        dest = Path(spec["dest"])
        if dest.is_file() and not args.force:
            print(f"{name}: already present at {dest} (use --force to re-download)")
            continue
        download(spec["url"], dest, spec.get("size_mb"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
