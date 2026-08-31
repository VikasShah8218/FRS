"""Similarity-transform face alignment to the ArcFace canonical template.

Why a *similarity* transform, not affine
----------------------------------------
An affine transform has 6 degrees of freedom and can shear. Given 5 noisy
landmarks it will happily squash a face to make them fit the template exactly,
distorting the very geometry that identifies the person. A similarity transform
(rotation + uniform scale + translation, 4 DOF) cannot shear, so it corrects
pose and scale while preserving facial proportions. This costs a measurable
amount of accuracy if you get it wrong, and it is a one-word mistake
(``estimateAffine2D`` vs ``estimateAffinePartial2D``).

The transform is solved with Umeyama's method, implemented here in numpy so
OpenCV is optional.
"""

from __future__ import annotations

import numpy as np

#: The ArcFace/InsightFace canonical 5-point template at 112x112.
#: Order: left eye, right eye, nose tip, left mouth corner, right mouth corner.
#: (Left/right are from the *viewer's* perspective, matching detector output.)
ARCFACE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def get_reference_landmarks(image_size: int = 112) -> np.ndarray:
    """The canonical template scaled to ``image_size``.

    The template is defined at 112x112; 96x112 crops and other sizes scale
    proportionally.
    """
    if image_size == 112:
        return ARCFACE_112.copy()
    return ARCFACE_112 * (image_size / 112.0)


def umeyama(src: np.ndarray, dst: np.ndarray, estimate_scale: bool = True) -> np.ndarray:
    """Least-squares similarity transform mapping ``src`` onto ``dst``.

    Implements Umeyama (1991), "Least-squares estimation of transformation
    parameters between two point patterns".

    Returns
    -------
    np.ndarray
        A 3x3 homogeneous matrix; the top 2x3 block is what ``warpAffine`` wants.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    num, dim = src.shape

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = (dst_demean.T @ src_demean) / num
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)
    U, S, Vt = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)

    if rank == 0:
        return np.full((dim + 1, dim + 1), np.nan)
    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(Vt) > 0:
            T[:dim, :dim] = U @ Vt
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ Vt
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ Vt

    scale = 1.0
    if estimate_scale:
        var_src = src_demean.var(axis=0).sum()
        if var_src > 0:
            scale = (S @ d) / var_src

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean)
    T[:dim, :dim] *= scale
    return T


def estimate_alignment_matrix(
    landmarks: np.ndarray, image_size: int = 112
) -> np.ndarray:
    """2x3 affine matrix warping ``landmarks`` onto the canonical template."""
    landmarks = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    reference = get_reference_landmarks(image_size)
    return umeyama(landmarks, reference, estimate_scale=True)[:2, :].astype(np.float32)


def warp_face(
    image: np.ndarray, landmarks: np.ndarray, image_size: int = 112
) -> np.ndarray:
    """Align a face crop to the canonical template.

    Uses ``cv2.warpAffine`` when OpenCV is available, otherwise a numpy inverse
    mapping with bilinear interpolation (slower, but keeps OpenCV optional).
    """
    matrix = estimate_alignment_matrix(landmarks, image_size)
    try:
        import cv2

        return cv2.warpAffine(
            image, matrix, (image_size, image_size), borderValue=0.0
        )
    except ImportError:
        return _warp_affine_numpy(image, matrix, image_size)


def _warp_affine_numpy(
    image: np.ndarray, matrix: np.ndarray, size: int
) -> np.ndarray:
    """Bilinear inverse-mapped affine warp, OpenCV-free."""
    full = np.vstack([matrix, [0, 0, 1]]).astype(np.float64)
    inverse = np.linalg.inv(full)

    ys, xs = np.mgrid[0:size, 0:size]
    ones = np.ones_like(xs)
    dst = np.stack([xs.ravel(), ys.ravel(), ones.ravel()])
    src = inverse @ dst
    sx, sy = src[0].reshape(size, size), src[1].reshape(size, size)

    h, w = image.shape[:2]
    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = sx - x0, sy - y0

    valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    x0c, y0c = np.clip(x0, 0, w - 1), np.clip(y0, 0, h - 1)
    x1c, y1c = np.clip(x1, 0, w - 1), np.clip(y1, 0, h - 1)

    img = image.astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, None]

    out = (
        img[y0c, x0c] * ((1 - wx) * (1 - wy))[..., None]
        + img[y0c, x1c] * (wx * (1 - wy))[..., None]
        + img[y1c, x0c] * ((1 - wx) * wy)[..., None]
        + img[y1c, x1c] * (wx * wy)[..., None]
    )
    out[~valid] = 0
    return np.clip(out, 0, 255).astype(np.uint8).squeeze()


def draw_template_overlay(image: np.ndarray, image_size: int = 112) -> np.ndarray:
    """Mark the canonical landmark positions -- used by ``align_dataset --probe``.

    Overlaying the template on an aligned crop makes misalignment visible at a
    glance: the eye markers should sit on the eyes.
    """
    out = image.copy()
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    reference = get_reference_landmarks(image_size).astype(int)
    colors = [
        (255, 64, 64), (255, 64, 64), (64, 255, 64), (64, 128, 255), (64, 128, 255)
    ]
    h, w = out.shape[:2]
    for (x, y), color in zip(reference, colors):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                py, px = y + dy, x + dx
                if 0 <= py < h and 0 <= px < w:
                    out[py, px] = color
    return out
