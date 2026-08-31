"""Samplers for long-tailed identity distributions.

MeGlass ranges from 4 to 578 images per identity -- a 144:1 imbalance. Under
uniform random sampling the rarest identities receive only ~4 gradient updates
per epoch, so their classifier rows barely move.

Recommendation
--------------
**Default to ``random``.** Margin-softmax losses (ArcFace/AdaFace) already
handle moderate imbalance well because every class row is updated by the softmax
denominator on *every* step, not just when its own samples appear. Aggressively
oversampling a 4-image identity mostly teaches the model to memorise those four
images.

If rare-identity accuracy is measurably poor, try ``sqrt_frequency`` before
reaching for full ``balanced_identity`` -- it is a much gentler correction.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler


class BalancedIdentitySampler(Sampler[int]):
    """P identities per batch x K instances each (the PK sampler).

    Standard for metric-learning losses that need multiple same-identity samples
    in a batch (triplet, circle, contrastive). For margin-softmax it is usually
    unnecessary, but it is here for when the loss changes.

    Parameters
    ----------
    targets:
        Class index per sample.
    batch_size:
        Must be divisible by ``instances_per_identity``.
    instances_per_identity:
        K -- images drawn per selected identity.
    seed:
        Base seed; combined with the epoch set via :meth:`set_epoch`.
    """

    def __init__(
        self,
        targets: np.ndarray,
        batch_size: int,
        instances_per_identity: int = 4,
        seed: int = 0,
    ) -> None:
        if batch_size % instances_per_identity != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by "
                f"instances_per_identity ({instances_per_identity})"
            )
        self.targets = np.asarray(targets)
        self.batch_size = int(batch_size)
        self.k = int(instances_per_identity)
        self.p = self.batch_size // self.k
        self.seed = int(seed)
        self.epoch = 0

        self.index_by_class: dict[int, np.ndarray] = {
            int(c): np.where(self.targets == c)[0]
            for c in np.unique(self.targets)
        }
        self.classes = np.array(sorted(self.index_by_class))
        if len(self.classes) < self.p:
            raise ValueError(
                f"need at least {self.p} identities for batch_size={batch_size} "
                f"with K={self.k}, but only {len(self.classes)} exist"
            )
        self.num_batches = len(self.targets) // self.batch_size
        self._length = self.num_batches * self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order: list[int] = []
        for _ in range(self.num_batches):
            chosen = rng.choice(self.classes, size=self.p, replace=False)
            for cls in chosen:
                pool = self.index_by_class[int(cls)]
                # replace=True when an identity has fewer than K images
                picks = rng.choice(pool, size=self.k, replace=len(pool) < self.k)
                order.extend(int(i) for i in picks)
        return iter(order)


def build_sampler(
    sampler_cfg: dict | None,
    targets: np.ndarray,
    batch_size: int,
    seed: int = 0,
) -> Sampler | None:
    """Construct a sampler from config. ``None`` means plain shuffling."""
    if not sampler_cfg:
        return None
    kind = sampler_cfg.get("type", "random")

    if kind == "random":
        return None

    if kind == "balanced_identity":
        return BalancedIdentitySampler(
            targets=targets,
            batch_size=batch_size,
            instances_per_identity=int(sampler_cfg.get("instances_per_identity", 4)),
            seed=seed,
        )

    if kind in ("sqrt_frequency", "inverse_frequency"):
        counts = np.bincount(targets, minlength=int(targets.max()) + 1).astype(
            np.float64
        )
        counts[counts == 0] = 1.0
        power = 0.5 if kind == "sqrt_frequency" else 1.0
        class_weight = 1.0 / np.power(counts, power)
        weights = class_weight[targets]
        return WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(targets),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )

    raise ValueError(
        f"unknown sampler type {kind!r}; expected random, balanced_identity, "
        f"sqrt_frequency or inverse_frequency"
    )
