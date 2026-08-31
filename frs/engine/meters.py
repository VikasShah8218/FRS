"""Metric accumulators and throughput/memory probes.

These feed both TensorBoard and the final report. The less obvious ones:

``data_time_frac``
    Fraction of wall-clock spent waiting for the DataLoader rather than
    computing. If this is above ~0.1 the GPU is starving and ``num_workers``
    (or the storage backend) is the bottleneck -- on AWS with 4M small JPEGs on
    EBS this is the number that reveals it.

``feature_norm``
    Mean L2 norm of un-normalised embeddings. It should rise steadily through
    training. A collapse toward zero means the model is degenerating; for
    AdaFace it is also literally an input to the loss.

``grad_norm``
    Total gradient norm *before* clipping. Spikes precede divergence, so this is
    the earliest warning signal available.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import torch


class AverageMeter:
    """Running mean plus a windowed mean for smooth logging."""

    def __init__(self, window: int = 50) -> None:
        self.window = int(window)
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0
        self.recent: deque[float] = deque(maxlen=self.window)
        self.last = 0.0

    def update(self, value: float, n: int = 1) -> None:
        value = float(value)
        self.last = value
        self.total += value * n
        self.count += n
        self.recent.append(value)

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def smooth(self) -> float:
        return sum(self.recent) / len(self.recent) if self.recent else 0.0

    def __format__(self, spec: str) -> str:
        return format(self.smooth, spec or ".4f")


class ThroughputMeter:
    """Images/second and the data-loading stall fraction."""

    def __init__(self, window: int = 50) -> None:
        self.window = int(window)
        self.reset()

    def reset(self) -> None:
        self._step_times: deque[float] = deque(maxlen=self.window)
        self._data_times: deque[float] = deque(maxlen=self.window)
        self._batch_sizes: deque[int] = deque(maxlen=self.window)
        self._t_start = time.perf_counter()
        self._t_data_start = time.perf_counter()
        self.total_images = 0

    def start_data(self) -> None:
        self._t_data_start = time.perf_counter()

    def end_data(self) -> None:
        self._data_time = time.perf_counter() - self._t_data_start

    def step(self, batch_size: int) -> None:
        now = time.perf_counter()
        self._step_times.append(now - self._t_start)
        self._data_times.append(getattr(self, "_data_time", 0.0))
        self._batch_sizes.append(int(batch_size))
        self.total_images += int(batch_size)
        self._t_start = now

    @property
    def images_per_sec(self) -> float:
        total_time = sum(self._step_times)
        return sum(self._batch_sizes) / total_time if total_time > 0 else 0.0

    @property
    def data_time_frac(self) -> float:
        total = sum(self._step_times)
        return sum(self._data_times) / total if total > 0 else 0.0

    @property
    def sec_per_step(self) -> float:
        return (
            sum(self._step_times) / len(self._step_times) if self._step_times else 0.0
        )


def gpu_memory_stats(device: torch.device | None = None) -> dict[str, float]:
    """Allocated / reserved / peak GPU memory in GiB.

    ``reserved`` is what actually limits you: the caching allocator holds onto
    freed blocks, so on a 4 GB card reserved can hit the ceiling while allocated
    looks comfortable. That gap is fragmentation.
    """
    if not torch.cuda.is_available():
        return {}
    gib = 1024.0 ** 3
    return {
        "alloc_gb": torch.cuda.memory_allocated(device) / gib,
        "reserved_gb": torch.cuda.memory_reserved(device) / gib,
        "peak_alloc_gb": torch.cuda.max_memory_allocated(device) / gib,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / gib,
    }


@torch.no_grad()
def topk_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, ks: tuple[int, ...] = (1,)
) -> list[float]:
    """Top-k accuracy on the margin logits.

    Note this is measured on *margin-penalised* logits, so it reads lower than a
    plain classifier's accuracy on the same model. It is a training-progress
    signal, not a verification metric -- judge the model by ``eval/`` numbers.
    """
    maxk = min(max(ks), logits.size(1))
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    correct = pred.eq(labels.view(-1, 1).expand_as(pred))
    batch = labels.size(0)
    return [
        correct[:, : min(k, maxk)].reshape(-1).float().sum().item() * 100.0 / batch
        for k in ks
    ]


class MetricHistory:
    """Per-epoch metric history, serialised into the checkpoint and the report."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def append(self, epoch: int, **metrics: Any) -> None:
        self.records.append({"epoch": int(epoch), **metrics})

    def series(self, key: str) -> tuple[list[int], list[float]]:
        xs, ys = [], []
        for rec in self.records:
            if key in rec and rec[key] is not None:
                xs.append(rec["epoch"])
                ys.append(float(rec[key]))
        return xs, ys

    def best(self, key: str, mode: str = "max") -> dict[str, Any] | None:
        candidates = [r for r in self.records if r.get(key) is not None]
        if not candidates:
            return None
        pick = max if mode == "max" else min
        return pick(candidates, key=lambda r: float(r[key]))

    def to_list(self) -> list[dict[str, Any]]:
        return list(self.records)

    @classmethod
    def from_list(cls, records: list[dict[str, Any]]) -> "MetricHistory":
        obj = cls()
        obj.records = list(records or [])
        return obj

    def __len__(self) -> int:
        return len(self.records)
