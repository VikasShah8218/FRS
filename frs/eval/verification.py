"""Face verification evaluation using the standard LFW 10-fold protocol.

Why this protocol specifically
------------------------------
Training accuracy on margin logits tells you the model is fitting; it says
nothing about whether two *unseen* photos of the same person will match. That is
what a deployed FRS actually does, so it is what we measure:

1. Embed every image (optionally averaging the embedding of its mirror -- a free
   ~0.3% because a face and its mirror should map to the same identity).
2. Score each pair by cosine similarity.
3. **10-fold cross-validation:** choose the accept threshold on 9 folds, apply it
   to the held-out fold, and report the mean. Picking one threshold on all the
   data would leak the test labels into the metric.

Also reported is TAR@FAR -- the true accept rate at a fixed false accept rate.
For access control this matters far more than raw accuracy: it answers "if I
tolerate one impostor in a thousand, how many genuine users get in?"

Critical: the model **must** be in eval mode. The backbone ends in
``BatchNorm1d(512, affine=False)``, so a model left in train mode normalises by
batch statistics and produces silently wrong embeddings. :func:`extract_embeddings`
asserts this rather than trusting the caller.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    images: np.ndarray | torch.Tensor,
    batch_size: int = 128,
    device: torch.device | str = "cuda",
    flip_test: bool = True,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> np.ndarray:
    """Embed a stack of preprocessed images -> ``(N, D)`` L2-normalised float32.

    Parameters
    ----------
    images:
        ``(N, C, H, W)`` already normalised the same way as training.
    flip_test:
        Average the embedding of each image and its horizontal mirror. Standard
        practice; small but free accuracy.
    """
    if model.training:
        raise RuntimeError(
            "extract_embeddings requires eval mode: the backbone's final "
            "BatchNorm1d would otherwise normalise by batch statistics and "
            "produce wrong embeddings. Call model.eval() first."
        )

    device = torch.device(device)
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)

    out: list[np.ndarray] = []
    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size].to(device, non_blocking=True).float()

        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp and device.type == "cuda"
        ):
            emb, _ = model(batch)
            if flip_test:
                emb_flip, _ = model(torch.flip(batch, dims=[3]))
                emb = emb + emb_flip

        emb = torch.nn.functional.normalize(emb.float(), dim=1)
        out.append(emb.cpu().numpy())

    return np.concatenate(out, axis=0)


def cosine_scores(embeddings: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Cosine similarity for each ``(i, j)`` index pair. Inputs are normalised."""
    left = embeddings[pairs[:, 0]]
    right = embeddings[pairs[:, 1]]
    return np.sum(left * right, axis=1)


def _best_threshold(
    scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray
) -> float:
    """Threshold maximising accuracy on the given (training-fold) data."""
    accuracies = [
        np.mean((scores > t) == labels) for t in thresholds
    ]
    return float(thresholds[int(np.argmax(accuracies))])


def evaluate_pairs(
    embeddings: np.ndarray,
    pairs: np.ndarray,
    is_same: np.ndarray,
    n_folds: int = 10,
    far_targets: Sequence[float] = (1e-2, 1e-3, 1e-4),
) -> dict[str, Any]:
    """Run the 10-fold verification protocol.

    Returns accuracy mean/std, the mean chosen threshold, ROC points, AUC and
    TAR at each requested FAR.
    """
    scores = cosine_scores(embeddings, pairs)
    labels = np.asarray(is_same).astype(bool)
    n = len(labels)
    if n < n_folds:
        raise ValueError(f"need at least {n_folds} pairs, got {n}")

    thresholds = np.arange(-1.0, 1.0, 0.001)
    indices = np.arange(n)
    fold_size = n // n_folds

    accuracies, chosen = [], []
    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else n
        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        threshold = _best_threshold(scores[train_idx], labels[train_idx], thresholds)
        accuracies.append(float(np.mean((scores[test_idx] > threshold) == labels[test_idx])))
        chosen.append(threshold)

    roc = _roc_curve(scores, labels)
    tar_at_far = {
        f"tar@far={target:g}": _tar_at_far(roc["fpr"], roc["tpr"], target)
        for target in far_targets
    }

    return {
        "accuracy": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "threshold": float(np.mean(chosen)),
        "fold_accuracies": accuracies,
        "auc": roc["auc"],
        "roc": {"fpr": roc["fpr"].tolist(), "tpr": roc["tpr"].tolist()},
        "num_pairs": int(n),
        "num_positive": int(labels.sum()),
        **tar_at_far,
    }


def _roc_curve(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """ROC by sweeping the threshold. Hand-rolled to avoid a sklearn dependency."""
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    tps = np.cumsum(sorted_labels)
    fps = np.cumsum(~sorted_labels)
    n_pos = max(1, int(labels.sum()))
    n_neg = max(1, int((~labels).sum()))

    tpr = np.concatenate([[0.0], tps / n_pos, [1.0]])
    fpr = np.concatenate([[0.0], fps / n_neg, [1.0]])
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(
        np.trapz(tpr, fpr)
    )
    return {"fpr": fpr, "tpr": tpr, "auc": auc}


def _tar_at_far(fpr: np.ndarray, tpr: np.ndarray, target_far: float) -> float:
    """True accept rate at the largest FAR not exceeding ``target_far``."""
    valid = np.where(fpr <= target_far)[0]
    return float(tpr[valid[-1]]) if len(valid) else 0.0


@torch.no_grad()
def evaluate_target(
    model: torch.nn.Module,
    target: dict[str, Any],
    load_images: Callable[[dict], tuple[np.ndarray, np.ndarray, np.ndarray]],
    device: torch.device | str = "cuda",
    batch_size: int = 128,
    flip_test: bool = True,
    amp: bool = False,
) -> dict[str, Any]:
    """Evaluate one benchmark described by an ``eval.targets`` entry."""
    was_training = model.training
    model.eval()
    try:
        images, pairs, is_same = load_images(target)
        embeddings = extract_embeddings(
            model,
            images,
            batch_size=batch_size,
            device=device,
            flip_test=flip_test,
            amp=amp,
        )
        result = evaluate_pairs(
            embeddings,
            pairs,
            is_same,
            n_folds=int(target.get("n_folds", 10)),
        )
        result["name"] = target["name"]
        return result
    finally:
        if was_training:
            model.train()


def format_results(results: dict[str, dict[str, Any]]) -> str:
    """Render evaluation results as a log-friendly table."""
    if not results:
        return "(no evaluation targets)"
    header = (
        f"{'benchmark':<18} {'accuracy':>10} {'std':>8} {'thresh':>8} "
        f"{'AUC':>7} {'TAR@1e-3':>10}"
    )
    lines = [header, "-" * len(header)]
    for name, res in results.items():
        lines.append(
            f"{name:<18} "
            f"{res.get('accuracy', 0) * 100:>9.3f}% "
            f"{res.get('accuracy_std', 0) * 100:>7.3f} "
            f"{res.get('threshold', 0):>8.4f} "
            f"{res.get('auc', 0):>7.4f} "
            f"{res.get('tar@far=0.001', 0) * 100:>9.2f}%"
        )
    return "\n".join(lines)
