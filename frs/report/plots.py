"""Training plots, rendered to base64 PNGs for embedding in a single HTML file.

Everything degrades gracefully: if matplotlib is missing, each function returns
``None`` and the report simply omits that figure rather than failing.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")  # no display needed; must be set before pyplot
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except Exception:  # pragma: no cover
    HAVE_MPL = False


_PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]


def _figure_to_base64(fig: Any) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _style(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="600", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def line_plot(
    series: dict[str, tuple[Sequence[float], Sequence[float]]],
    title: str,
    xlabel: str = "epoch",
    ylabel: str = "",
    logy: bool = False,
) -> str | None:
    """One or more (x, y) series on shared axes."""
    if not HAVE_MPL or not series:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, (label, (xs, ys)) in enumerate(series.items()):
        if not xs:
            continue
        ax.plot(xs, ys, label=label, color=_PALETTE[i % len(_PALETTE)], linewidth=1.8)
    if logy:
        ax.set_yscale("log")
    _style(ax, title, xlabel, ylabel)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=9)
    return _figure_to_base64(fig)


def roc_plot(curves: dict[str, dict[str, Sequence[float]]], title: str = "ROC") -> str | None:
    """ROC curves on a log-x axis -- the region that matters for access control.

    A linear FPR axis hides everything below 1e-2, which is exactly the operating
    region a real verification system runs in.
    """
    if not HAVE_MPL or not curves:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for i, (name, curve) in enumerate(curves.items()):
        fpr, tpr = curve.get("fpr", []), curve.get("tpr", [])
        if not len(fpr):
            continue
        auc = curve.get("auc")
        label = f"{name}" + (f" (AUC {auc:.4f})" if auc is not None else "")
        ax.plot(fpr, tpr, label=label, color=_PALETTE[i % len(_PALETTE)], linewidth=1.8)

    ax.plot([1e-6, 1], [1e-6, 1], "k--", alpha=0.3, linewidth=1, label="chance")
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1.0)
    ax.set_ylim(0.0, 1.02)
    _style(ax, title, "False Accept Rate (log)", "True Accept Rate")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    return _figure_to_base64(fig)


def histogram(
    values: Sequence[float], title: str, xlabel: str, bins: int = 50, logy: bool = False
) -> str | None:
    if not HAVE_MPL or values is None or not len(values):
        return None
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(values, bins=bins, color=_PALETTE[0], alpha=0.85, edgecolor="white", linewidth=0.4)
    if logy:
        ax.set_yscale("log")
    _style(ax, title, xlabel, "count")
    return _figure_to_base64(fig)


def build_training_figures(history: Any, dataset_stats: dict[str, Any]) -> dict[str, str]:
    """Assemble every figure the report shows."""
    figures: dict[str, str] = {}
    if not HAVE_MPL:
        logger.warning("matplotlib unavailable -- report will have no figures")
        return figures

    def add(key: str, image: str | None) -> None:
        if image:
            figures[key] = image

    add("loss", line_plot({"train loss": history.series("train_loss")},
                          "Training loss", ylabel="cross-entropy"))
    add("lr", line_plot({"learning rate": history.series("lr")},
                        "Learning rate schedule", ylabel="lr"))
    add("train_acc", line_plot({"top-1 (margin logits)": history.series("train_acc1")},
                               "Training accuracy", ylabel="%"))
    add("feature_norm", line_plot({"mean |embedding|": history.series("feature_norm")},
                                  "Feature norm", ylabel="L2 norm"))
    add("grad_norm", line_plot({"grad norm": history.series("grad_norm")},
                               "Gradient norm (pre-clip)", ylabel="norm"))
    add("throughput", line_plot({"images/sec": history.series("images_per_sec")},
                                "Throughput", ylabel="img/s"))

    # Every eval target that produced an accuracy series.
    eval_series = {}
    known = {
        "epoch", "train_loss", "train_acc1", "feature_norm", "grad_norm", "lr",
        "epoch_time_sec", "images_per_sec", "data_time_frac",
    }
    for record in history.to_list():
        for key in record:
            if key not in known and isinstance(record.get(key), (int, float)):
                eval_series.setdefault(key, history.series(key))
    if eval_series:
        add("eval_accuracy", line_plot(eval_series, "Verification accuracy", ylabel="accuracy"))

    counts = dataset_stats.get("identity_counts")
    if counts is not None:
        add(
            "identity_distribution",
            histogram(
                list(counts.values()) if hasattr(counts, "values") else list(counts),
                "Images per identity",
                "images",
                bins=60,
                logy=True,
            ),
        )
    return figures
