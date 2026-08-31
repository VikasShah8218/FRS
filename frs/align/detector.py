"""RetinaFace-family face detection (SCRFD) via raw ONNX Runtime.

Why not ``insightface``
-----------------------
The ``insightface`` package has no Python 3.13 wheel and its Cython extensions
fail to build on 3.13. Since SCRFD ships as a plain ONNX model and onnxruntime
is already a dependency, running the model directly removes the broken package
entirely -- at the cost of ~120 lines of post-processing, which is well
specified and stable.

SCRFD is the current RetinaFace-lineage detector from InsightFace: a
single-stage anchor-based detector with three FPN levels (strides 8/16/32), two
anchors per location, predicting box distances and 5 facial landmarks per
anchor. ``det_10g.onnx`` is the standard 16.9 MB variant.

Alternative backends can be registered under :data:`frs.registry.DETECTORS` and
selected with ``detector.type`` in config -- e.g. ``facexlib`` (pure PyTorch
RetinaFace) if ONNX ever becomes inconvenient.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..registry import DETECTORS

logger = logging.getLogger(__name__)

#: Direct download for the standard SCRFD model (no insightface install needed).
SCRFD_URL = (
    "https://huggingface.co/public-data/insightface/resolve/main/"
    "models/buffalo_l/det_10g.onnx"
)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.4) -> list[int]:
    """Greedy non-maximum suppression. Returns kept indices, highest score first."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= threshold]
    return keep


def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Anchor centres + per-side distances -> (x1, y1, x2, y2)."""
    return np.stack(
        [
            points[:, 0] - distance[:, 0],
            points[:, 1] - distance[:, 1],
            points[:, 0] + distance[:, 2],
            points[:, 1] + distance[:, 3],
        ],
        axis=-1,
    )


def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Anchor centres + landmark offsets -> (N, 5, 2)."""
    preds = [
        points[:, i % 2] + distance[:, i] for i in range(distance.shape[1])
    ]
    return np.stack(preds, axis=-1).reshape(len(points), -1, 2)


@DETECTORS.register("scrfd")
class SCRFDDetector:
    """SCRFD face detector returning boxes and 5-point landmarks.

    Parameters
    ----------
    model_path:
        Path to ``det_10g.onnx``. Fetch it with ``scripts/download_models.py``.
    providers:
        ONNX Runtime execution providers, tried in order.
    input_size:
        Network input resolution. Larger finds smaller faces at more cost.
    conf_threshold, nms_threshold:
        Detection score cutoff and NMS IoU.
    """

    def __init__(
        self,
        model_path: str,
        providers: tuple[str, ...] | list[str] = (
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ),
        input_size: tuple[int, int] = (640, 640),
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ) -> None:
        import onnxruntime as ort

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"SCRFD model not found: {path}\n"
                f"Download it with: python -m scripts.download_models"
            )

        available = set(ort.get_available_providers())
        chosen = [p for p in providers if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(path), providers=chosen)
        logger.info("SCRFD loaded (%s) from %s", chosen[0], path.name)

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_size = tuple(input_size)
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)

        # det_10g has 3 FPN levels x (score, bbox, kps) = 9 outputs, 2 anchors each.
        self.fmc = 3
        self.feat_strides = [8, 16, 32]
        self.num_anchors = 2
        self.use_kps = len(self.output_names) >= 9
        self._anchor_cache: dict[tuple, np.ndarray] = {}

    def _anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        if key not in self._anchor_cache:
            ys, xs = np.mgrid[:height, :width][::-1]
            centers = np.stack([ys, xs], axis=-1).astype(np.float32) * stride
            centers = centers.reshape(-1, 2)
            if self.num_anchors > 1:
                centers = np.stack([centers] * self.num_anchors, axis=1).reshape(-1, 2)
            self._anchor_cache[key] = centers
        return self._anchor_cache[key]

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Letterbox to the network input, preserving aspect ratio."""
        target_w, target_h = self.input_size
        h, w = image.shape[:2]
        scale = min(target_h / h, target_w / w)
        nh, nw = int(h * scale), int(w * scale)

        try:
            import cv2

            resized = cv2.resize(image, (nw, nh))
        except ImportError:
            from PIL import Image

            resized = np.asarray(Image.fromarray(image).resize((nw, nh)))

        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:nh, :nw] = resized

        # SCRFD expects BGR, mean 127.5, scale 1/128.
        blob = canvas[:, :, ::-1].astype(np.float32)
        blob = (blob - 127.5) / 128.0
        return blob.transpose(2, 0, 1)[None], scale

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Detect faces in an RGB uint8 image.

        Returns
        -------
        (boxes, landmarks)
            ``boxes`` is ``(N, 5)`` -- x1, y1, x2, y2, score -- sorted by
            descending area. ``landmarks`` is ``(N, 5, 2)`` in image pixels.
        """
        blob, scale = self._preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        all_scores, all_boxes, all_kps = [], [], []
        for level, stride in enumerate(self.feat_strides):
            scores = outputs[level].reshape(-1)
            bbox_preds = outputs[level + self.fmc].reshape(-1, 4) * stride
            kps_preds = (
                outputs[level + self.fmc * 2].reshape(-1, 10) * stride
                if self.use_kps
                else None
            )

            height = self.input_size[1] // stride
            width = self.input_size[0] // stride
            centers = self._anchor_centers(height, width, stride)

            keep = np.where(scores >= self.conf_threshold)[0]
            if len(keep) == 0:
                continue

            all_scores.append(scores[keep])
            all_boxes.append(distance2bbox(centers, bbox_preds)[keep])
            if kps_preds is not None:
                all_kps.append(distance2kps(centers, kps_preds)[keep])

        if not all_boxes:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)

        scores = np.concatenate(all_scores)
        boxes = np.concatenate(all_boxes) / scale
        kps = np.concatenate(all_kps) / scale if all_kps else np.zeros((len(boxes), 5, 2))

        keep = nms(boxes, scores, self.nms_threshold)
        boxes, scores, kps = boxes[keep], scores[keep], kps[keep]

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = np.argsort(-areas)
        return (
            np.hstack([boxes, scores[:, None]])[order].astype(np.float32),
            kps[order].astype(np.float32),
        )

    def detect_one(
        self, image: np.ndarray, select: str = "largest"
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Detect and pick a single face.

        ``select``: ``largest`` (default -- the subject of a portrait),
        ``center`` (closest to the frame centre), or ``highest_score``.
        """
        boxes, kps = self.detect(image)
        if len(boxes) == 0:
            return None
        if select == "largest":
            index = 0  # detect() already sorts by area
        elif select == "highest_score":
            index = int(np.argmax(boxes[:, 4]))
        elif select == "center":
            h, w = image.shape[:2]
            cx, cy = w / 2, h / 2
            centers = np.stack(
                [(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2],
                axis=1,
            )
            index = int(np.argmin(((centers - [cx, cy]) ** 2).sum(axis=1)))
        else:
            raise ValueError(
                f"unknown select mode {select!r}; expected largest, center or highest_score"
            )
        return boxes[index], kps[index]


def build_detector(cfg: dict) -> object:
    """Construct a detector from a ``detector`` config block."""
    return DETECTORS.build(dict(cfg))
