"""Offline face alignment for a dataset.

    # Check the resize policy before committing to a long run (10 seconds):
    python -m scripts.align_dataset --probe --src MeGlass_120x120

    # Align a raw dataset to 112x112 ArcFace-canonical crops:
    python -m scripts.align_dataset --src raw_photos --dst aligned_112

MeGlass does not need this -- it ships pre-cropped, and the training config's
``center_crop_112`` policy handles the 120 -> 112 step. This script is for the
*next* dataset: raw photos, CCTV frames, enrolment captures.

The --probe mode
----------------
Renders a montage of sample images under each resize policy with the canonical
landmark template overlaid. The eye markers should land on the eyes. If they
sit noticeably off, the policy is wrong and every downstream number suffers --
worth ten seconds before a multi-hour run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from frs.align.pipeline import align_dataset  # noqa: E402
from frs.align.warp import draw_template_overlay  # noqa: E402
from frs.data.transforms import apply_resize_policy  # noqa: E402
from frs.utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger("align")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--src", required=True, help="source image directory")
    p.add_argument("--dst", help="output directory (required unless --probe)")
    p.add_argument("--model", default="models/det_10g.onnx", help="SCRFD onnx path")
    p.add_argument("--image-size", type=int, default=112)
    p.add_argument(
        "--select",
        default="largest",
        choices=["largest", "center", "highest_score"],
        help="which face to keep when several are detected",
    )
    p.add_argument("--min-face-size", type=int, default=0, help="reject faces smaller than N px")
    p.add_argument(
        "--fallback",
        choices=["center_crop"],
        default=None,
        help="what to do when no face is found (default: record a failure)",
    )
    p.add_argument("--no-resume", action="store_true", help="re-align existing outputs")
    p.add_argument("--limit", type=int, default=0, help="process at most N images")
    p.add_argument("--cpu", action="store_true", help="force CPU inference")
    p.add_argument(
        "--probe",
        action="store_true",
        help="render a resize-policy comparison montage and exit",
    )
    p.add_argument("--probe-out", default="probe_alignment.png")
    return p.parse_args()


def find_images(root: Path, limit: int = 0) -> list[Path]:
    paths = [
        p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return paths[:limit] if limit else paths


def run_probe(src: Path, out: Path, image_size: int = 112, n: int = 5) -> int:
    """Montage: source, center_crop_112 and resize_112, each with the template."""
    paths = find_images(src, limit=200)
    if not paths:
        print(f"No images found in {src}", file=sys.stderr)
        return 1

    step = max(1, len(paths) // n)
    chosen = paths[::step][:n]
    policies = ["center_crop_112", "resize_112"]

    cell = image_size
    rows, cols = len(policies) + 1, len(chosen)
    canvas = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)

    for col, path in enumerate(chosen):
        with Image.open(path) as img:
            original = np.asarray(img.convert("RGB"), dtype=np.uint8)

        # Row 0: the source, plainly resized for display only.
        display = np.asarray(
            Image.fromarray(original).resize((cell, cell), Image.NEAREST)
        )
        canvas[0:cell, col * cell : (col + 1) * cell] = display

        for row, policy in enumerate(policies, start=1):
            processed = apply_resize_policy(original, policy, image_size)
            canvas[
                row * cell : (row + 1) * cell, col * cell : (col + 1) * cell
            ] = draw_template_overlay(processed, image_size)

    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out)
    print(
        f"Wrote {out}\n"
        f"  row 1: source images (as-is)\n"
        f"  row 2: center_crop_112  <- the default policy\n"
        f"  row 3: resize_112\n\n"
        f"Red markers are the canonical eye positions, green the nose, blue the\n"
        f"mouth corners. Pick the policy whose markers land on the actual\n"
        f"features. If neither does, the images need real alignment: run this\n"
        f"script without --probe."
    )
    return 0


def main() -> int:
    args = parse_args()
    setup_logging()

    src = Path(args.src)
    if not src.is_dir():
        print(f"Source directory not found: {src}", file=sys.stderr)
        return 1

    if args.probe:
        return run_probe(src, Path(args.probe_out), args.image_size)

    if not args.dst:
        print("--dst is required unless --probe is given", file=sys.stderr)
        return 1

    from frs.align.detector import SCRFDDetector

    providers = (
        ("CPUExecutionProvider",)
        if args.cpu
        else ("CUDAExecutionProvider", "CPUExecutionProvider")
    )
    try:
        detector = SCRFDDetector(args.model, providers=providers)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    paths = find_images(src, args.limit)
    logger.info("Found %d images under %s", len(paths), src)
    if not paths:
        return 1

    counts = align_dataset(
        detector=detector,
        src_paths=paths,
        src_root=src,
        dst_root=Path(args.dst),
        image_size=args.image_size,
        select=args.select,
        min_face_size=args.min_face_size,
        fallback=args.fallback,
        resume=not args.no_resume,
    )

    print(
        f"\nAligned dataset written to {args.dst}\n"
        f"  ok {counts['ok']:,} | no-face {counts['no_face']:,} | "
        f"error {counts['error']:,} | skipped {counts['skipped']:,}\n\n"
        f"Point your training config at it:\n"
        f"  data.adapter.root: {args.dst}\n"
        f"  data.resize_policy: center_crop_112   # already 112x112, so this is a no-op"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
