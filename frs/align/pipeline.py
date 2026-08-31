"""Offline dataset alignment: detect -> select -> warp -> save.

Offline, not on-the-fly
-----------------------
Running a detector inside the DataLoader would re-detect the same faces on every
epoch -- 24x the work for identical output, while stealing GPU from training.
Aligning once to a new folder makes training loaders trivially fast and lets the
alignment be inspected before committing hours to a run.

The pipeline is resumable (it skips images whose output already exists) and
records every decision in ``manifest.jsonl``, with failures in ``failed.jsonl``
rather than silently vanishing -- a dataset that quietly loses 8% of its images
to failed detection is a bug you want to see.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .warp import warp_face

logger = logging.getLogger(__name__)


@dataclass
class AlignResult:
    src: str
    dst: str | None
    status: str  # ok | no_face | error | skipped
    score: float | None = None
    bbox: list[float] | None = None
    kps: list[list[float]] | None = None
    message: str | None = None


def align_image(
    detector: Any,
    src_path: Path,
    dst_path: Path,
    image_size: int = 112,
    select: str = "largest",
    min_face_size: int = 0,
    fallback: str | None = None,
) -> AlignResult:
    """Align one image and write the result.

    ``fallback`` controls what happens when no face is detected:
    ``None`` records a failure, ``"center_crop"`` writes a centre crop instead
    (useful for datasets like MeGlass that are already tightly cropped, where a
    detection failure usually means the face fills the frame).
    """
    try:
        with Image.open(src_path) as img:
            image = np.asarray(img.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        return AlignResult(str(src_path), None, "error", message=f"read failed: {exc}")

    try:
        detection = detector.detect_one(image, select=select)
    except Exception as exc:
        return AlignResult(str(src_path), None, "error", message=f"detect failed: {exc}")

    if detection is None:
        if fallback == "center_crop":
            aligned = _center_crop(image, image_size)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(aligned).save(dst_path, quality=95)
            return AlignResult(
                str(src_path), str(dst_path), "ok", message="fallback: center_crop"
            )
        return AlignResult(str(src_path), None, "no_face")

    box, kps = detection
    face_size = min(box[2] - box[0], box[3] - box[1])
    if min_face_size and face_size < min_face_size:
        return AlignResult(
            str(src_path),
            None,
            "no_face",
            message=f"face too small: {face_size:.0f}px < {min_face_size}px",
        )

    try:
        aligned = warp_face(image, kps, image_size)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(aligned).save(dst_path, quality=95)
    except Exception as exc:
        return AlignResult(str(src_path), None, "error", message=f"warp failed: {exc}")

    return AlignResult(
        src=str(src_path),
        dst=str(dst_path),
        status="ok",
        score=float(box[4]),
        bbox=[float(v) for v in box[:4]],
        kps=[[float(x), float(y)] for x, y in kps],
    )


def _center_crop(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h < size or w < size:
        return np.asarray(Image.fromarray(image).resize((size, size), Image.BILINEAR))
    top, left = (h - size) // 2, (w - size) // 2
    return image[top : top + size, left : left + size]


def align_dataset(
    detector: Any,
    src_paths: Iterable[Path],
    src_root: Path,
    dst_root: Path,
    image_size: int = 112,
    select: str = "largest",
    min_face_size: int = 0,
    fallback: str | None = None,
    resume: bool = True,
    progress: bool = True,
) -> dict[str, int]:
    """Align a whole dataset, preserving the source directory structure.

    Filenames are preserved exactly, so an adapter configured for the source
    folder works unchanged on the aligned output.
    """
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    manifest_path = dst_root / "manifest.jsonl"
    failed_path = dst_root / "failed.jsonl"

    src_paths = list(src_paths)
    counts = {"ok": 0, "no_face": 0, "error": 0, "skipped": 0}

    iterator: Iterable[Path] = src_paths
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(src_paths, desc="aligning", unit="img")
        except ImportError:
            pass

    with manifest_path.open("a", encoding="utf-8") as manifest, failed_path.open(
        "a", encoding="utf-8"
    ) as failures:
        for src in iterator:
            relative = Path(src).relative_to(src_root)
            dst = dst_root / relative

            if resume and dst.is_file():
                counts["skipped"] += 1
                continue

            result = align_image(
                detector, Path(src), dst, image_size, select, min_face_size, fallback
            )
            counts[result.status] = counts.get(result.status, 0) + 1

            line = json.dumps(asdict(result))
            manifest.write(line + "\n")
            if result.status != "ok":
                failures.write(line + "\n")

    total = sum(counts.values())
    logger.info(
        "Alignment done: %d ok, %d no-face, %d error, %d skipped (of %d)",
        counts["ok"], counts["no_face"], counts["error"], counts["skipped"], total,
    )
    processed = counts["ok"] + counts["no_face"] + counts["error"]
    if processed:
        fail_rate = (counts["no_face"] + counts["error"]) / processed
        if fail_rate > 0.05:
            logger.warning(
                "%.1f%% of images failed to align -- inspect %s before training. "
                "A high rate usually means the images are already tight crops "
                "(try --fallback center_crop) or the detector input size is too small.",
                fail_rate * 100,
                failed_path,
            )
    return counts
