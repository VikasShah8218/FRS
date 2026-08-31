"""Image preprocessing and augmentation.

Implemented with numpy + PIL rather than torchvision transforms so the resize
policy is explicit and auditable -- the 120 -> 112 decision below materially
affects accuracy and deserves to be readable.

The 120 -> 112 problem
----------------------
MeGlass ships as 120x120 crops, but IResNet expects 112x112 and the ArcFace
canonical landmark template is defined at 112x112. Three policies:

``center_crop_112`` (default)
    Crop a 4px border. **Preserves face scale and eye positions exactly**, so the
    landmarks stay where the canonical template expects them.
``resize_112``
    Bilinear 120 -> 112. Shrinks the face by 6.7%, moving the eyes inward from
    the template positions. Cheap, slightly lossy in alignment terms.
``realign``
    Re-detect landmarks and warp. Most correct in principle, but detection on an
    already-tight 120px crop is unreliable, so it is opt-in only.

Verify the choice with ``python -m scripts.align_dataset --probe`` before a long
run: it renders the canonical template over both policies so the eye line can be
checked by eye. A 10-minute check that protects a multi-hour run.
"""

from __future__ import annotations

import random
from typing import Any, Sequence

import numpy as np
from PIL import Image

RESIZE_POLICIES = ("center_crop_112", "resize_112", "pad_112", "realign")


# --------------------------------------------------------------------- geometry


def apply_resize_policy(
    img: np.ndarray, policy: str, target: int = 112
) -> np.ndarray:
    """Bring an arbitrary-size RGB image to ``target x target``."""
    h, w = img.shape[:2]
    if h == target and w == target:
        return img

    if policy == "center_crop_112":
        # Only crop when the source is larger; otherwise fall back to a resize
        # so the policy never silently upscales by padding.
        if h >= target and w >= target:
            top, left = (h - target) // 2, (w - target) // 2
            return img[top : top + target, left : left + target]
        return _pil_resize(img, target)

    if policy in ("resize_112", "realign"):
        # 'realign' is handled offline by scripts/align_dataset.py; by the time
        # images reach the loader they are already 112x112, so a resize is the
        # correct no-op-ish fallback.
        return _pil_resize(img, target)

    if policy == "pad_112":
        scale = target / max(h, w)
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
        resized = np.asarray(
            Image.fromarray(img).resize((nw, nh), Image.BILINEAR), dtype=np.uint8
        )
        out = np.zeros((target, target, 3), dtype=np.uint8)
        top, left = (target - nh) // 2, (target - nw) // 2
        out[top : top + nh, left : left + nw] = resized
        return out

    raise ValueError(f"unknown resize policy {policy!r}; expected one of {RESIZE_POLICIES}")


def _pil_resize(img: np.ndarray, target: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(img).resize((target, target), Image.BILINEAR), dtype=np.uint8
    )


# ----------------------------------------------------------------- augmentation


def _rand() -> float:
    return random.random()


def horizontal_flip(img: np.ndarray, p: float) -> np.ndarray:
    """Mirror the image. The single most valuable face-recognition augmentation."""
    if p > 0 and _rand() < p:
        return np.ascontiguousarray(img[:, ::-1])
    return img


def color_jitter(
    img: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
    p: float = 0.0,
) -> np.ndarray:
    """Photometric jitter, applied in a random channel order.

    Models illumination and camera differences between enrolment and query
    images, which is exactly the domain shift a deployed FRS faces.
    """
    if p <= 0 or _rand() >= p:
        return img

    out = img.astype(np.float32)
    ops = []
    if brightness > 0:
        ops.append(("b", random.uniform(max(0.0, 1 - brightness), 1 + brightness)))
    if contrast > 0:
        ops.append(("c", random.uniform(max(0.0, 1 - contrast), 1 + contrast)))
    if saturation > 0:
        ops.append(("s", random.uniform(max(0.0, 1 - saturation), 1 + saturation)))
    random.shuffle(ops)

    for kind, factor in ops:
        if kind == "b":
            out *= factor
        elif kind == "c":
            mean = out.mean()
            out = (out - mean) * factor + mean
        elif kind == "s":
            gray = out @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            out = (out - gray[..., None]) * factor + gray[..., None]

    if hue > 0:  # rare in face recognition; kept for completeness
        shift = random.uniform(-hue, hue) * 255.0
        out[..., 0] += shift
        out[..., 2] -= shift

    return np.clip(out, 0, 255).astype(np.uint8)


def random_grayscale(img: np.ndarray, p: float) -> np.ndarray:
    """Occasionally drop colour -- helps with IR/monochrome cameras."""
    if p <= 0 or _rand() >= p:
        return img
    gray = (img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32))
    return np.repeat(gray[..., None], 3, axis=2).clip(0, 255).astype(np.uint8)


def random_erasing(
    img: np.ndarray,
    p: float = 0.0,
    scale: Sequence[float] = (0.02, 0.15),
    ratio: Sequence[float] = (0.3, 3.3),
    value: str | int = "random",
) -> np.ndarray:
    """Occlude a random rectangle.

    Simulates masks, hands, glare and crops. Applied *after* normalisation
    would be equivalent; doing it on uint8 keeps this module tensor-free.
    """
    if p <= 0 or _rand() >= p:
        return img

    h, w = img.shape[:2]
    area = h * w
    for _ in range(10):  # retry until a valid rectangle is found
        target_area = area * random.uniform(*scale)
        aspect = random.uniform(*ratio)
        eh = int(round((target_area * aspect) ** 0.5))
        ew = int(round((target_area / aspect) ** 0.5))
        if eh < h and ew < w:
            top = random.randint(0, h - eh)
            left = random.randint(0, w - ew)
            out = img.copy()
            if value == "random":
                out[top : top + eh, left : left + ew] = np.random.randint(
                    0, 256, (eh, ew, 3), dtype=np.uint8
                )
            else:
                out[top : top + eh, left : left + ew] = int(value)
            return out
    return img


# -------------------------------------------------------------------- pipelines


class FaceTransform:
    """Callable pipeline: raw RGB uint8 HWC -> normalised float32 CHW.

    Returns a plain numpy array; the Dataset converts it to a tensor. Keeping
    torch out of this module makes the transforms unit-testable without a GPU
    and importable before torch is installed.
    """

    def __init__(
        self,
        input_size: int = 112,
        resize_policy: str = "center_crop_112",
        mean: Sequence[float] = (0.5, 0.5, 0.5),
        std: Sequence[float] = (0.5, 0.5, 0.5),
        train: bool = True,
        augment: dict[str, Any] | None = None,
    ) -> None:
        if resize_policy not in RESIZE_POLICIES:
            raise ValueError(
                f"resize_policy must be one of {RESIZE_POLICIES}, got {resize_policy!r}"
            )
        self.input_size = int(input_size)
        self.resize_policy = resize_policy
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        self.train = bool(train)
        self.augment = dict(augment or {})

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = apply_resize_policy(img, self.resize_policy, self.input_size)

        if self.train and self.augment:
            aug = self.augment
            img = horizontal_flip(img, float(aug.get("horizontal_flip", 0.0) or 0.0))
            if aug.get("color_jitter"):
                img = color_jitter(img, **aug["color_jitter"])
            img = random_grayscale(img, float(aug.get("grayscale", 0.0) or 0.0))
            if aug.get("random_erasing"):
                img = random_erasing(img, **aug["random_erasing"])

        # HWC uint8 -> CHW float32, scaled to roughly [-1, 1]
        arr = img.astype(np.float32).transpose(2, 0, 1) / 255.0
        return (arr - self.mean) / self.std

    def __repr__(self) -> str:
        return (
            f"FaceTransform(size={self.input_size}, policy={self.resize_policy}, "
            f"train={self.train})"
        )


def build_transforms(data_cfg: Any) -> tuple[FaceTransform, FaceTransform]:
    """Construct the (train, eval) transform pair from a data config block."""
    size = data_cfg["input_size"][0]
    policy = data_cfg.get("resize_policy", "center_crop_112")
    mean = data_cfg.get("mean", (0.5, 0.5, 0.5))
    std = data_cfg.get("std", (0.5, 0.5, 0.5))
    augment = data_cfg.get("augment", {}) or {}

    train_tf = FaceTransform(size, policy, mean, std, train=True, augment=augment)
    eval_tf = FaceTransform(size, policy, mean, std, train=False, augment=None)
    return train_tf, eval_tf
