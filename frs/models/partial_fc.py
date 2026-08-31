"""Partial-FC: sampled-softmax for very large class counts.

The problem
-----------
The classifier weight is ``num_classes x 512``, and SGD momentum doubles it:

    1,710 classes (MeGlass)   ->     3.5 MB  (+3.5 MB momentum)  trivial
       93 K classes (MS1MV3)  ->     190 MB  (+190 MB)           fine
      360 K classes (WebFace4M) ->   737 MB  (+737 MB)           tight on 24 GB
        2 M classes (WebFace42M) ->  4.1 GB  (+4.1 GB)           impossible

Above a few hundred thousand identities the classifier, not the backbone,
dominates memory -- and the ``B x num_classes`` logit matrix compounds it.

The fix
-------
For any given batch, only the classes actually present contribute positive
gradient; the rest merely appear in the softmax denominator. Partial-FC keeps
every positive class and samples a fraction of the negatives, computing the
margin and softmax over that subset. At ``sample_rate=0.1`` this is a 10x
reduction in classifier memory and compute for a negligible accuracy cost.

Design
------
This is a **wrapper around any** :class:`~frs.models.heads.MarginHead`, not a
head in its own right, so the ArcFace/CosFace/AdaFace margin maths is written
exactly once and cannot drift between the full and sampled paths.

    head = PartialFC(ArcFaceHead(512, 2_000_000), sample_rate=0.1)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .heads import MarginHead

logger = logging.getLogger(__name__)

#: Above this many classes the full FC starts to hurt on a 24 GB card.
AUTO_ENABLE_THRESHOLD = 300_000


class PartialFC(nn.Module):
    """Sampled-softmax wrapper around a margin head.

    Parameters
    ----------
    head:
        Any :class:`MarginHead`. Its ``weight`` is the full class prototype
        matrix; this wrapper slices it per step.
    sample_rate:
        Fraction of classes to keep per step, in (0, 1]. ``1.0`` disables
        sampling and is equivalent to the bare head.
    """

    def __init__(self, head: MarginHead, sample_rate: float = 0.1) -> None:
        super().__init__()
        if not 0.0 < sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be in (0, 1]; got {sample_rate}")
        self.head = head
        self.sample_rate = float(sample_rate)
        self.num_classes = head.num_classes
        self.embedding_size = head.embedding_size
        self.num_sampled = max(1, int(self.num_classes * self.sample_rate))

        logger.info(
            "Partial-FC: sampling %d of %d classes per step (%.0f%%)",
            self.num_sampled, self.num_classes, self.sample_rate * 100,
        )

    @property
    def weight(self) -> nn.Parameter:
        """Expose the wrapped head's weight so checkpointing sees one tensor."""
        return self.head.weight

    def _sample_classes(self, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Choose the class subset for this step.

        Every class present in the batch is kept (they carry the positive
        gradient); the remainder of the budget is filled with uniformly sampled
        negatives.

        Returns
        -------
        (class_indices, remapped_labels)
            ``class_indices`` indexes into the full weight matrix;
            ``remapped_labels`` are the batch labels expressed in the subset's
            own index space.
        """
        positives = torch.unique(labels)
        n_extra = self.num_sampled - positives.numel()

        if n_extra <= 0:
            selected = positives
        else:
            # Sample from all classes and drop collisions with the positives.
            # Oversampling by 2x makes a short draw vanishingly unlikely, and
            # the subset only needs to be *approximately* the target size.
            candidates = torch.randint(
                0, self.num_classes, (n_extra * 2,), device=labels.device
            )
            mask = ~torch.isin(candidates, positives)
            negatives = torch.unique(candidates[mask])[:n_extra]
            selected = torch.cat([positives, negatives])

        selected, _ = torch.sort(selected)

        # Map original label -> position within `selected`.
        lookup = torch.full(
            (self.num_classes,), -1, dtype=torch.long, device=labels.device
        )
        lookup[selected] = torch.arange(selected.numel(), device=labels.device)
        remapped = lookup[labels]

        if (remapped < 0).any():  # cannot happen; guards a silent corruption
            raise RuntimeError("Partial-FC dropped a positive class from the subset")

        return selected, remapped

    def forward(
        self,
        embeddings: torch.Tensor,
        norms: torch.Tensor | None,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute margin logits over the sampled class subset.

        Returns
        -------
        (logits, remapped_labels)
            **Note the two-value return.** Cross-entropy must be computed against
            ``remapped_labels``, not the original labels, because the logits span
            the subset rather than all classes.
        """
        if self.sample_rate >= 1.0:
            return self.head(embeddings, norms, labels), labels

        selected, remapped = self._sample_classes(labels)

        # Temporarily view the head as if it had only the sampled classes. The
        # sliced weight keeps its graph connection, so gradient flows back into
        # exactly the sampled rows of the full parameter and no others.
        full_weight = self.head.weight
        full_count = self.head.num_classes
        try:
            self.head.weight = full_weight[selected]  # type: ignore[assignment]
            self.head.num_classes = selected.numel()
            logits = self.head(embeddings, norms, remapped)
        finally:
            self.head.weight = full_weight  # type: ignore[assignment]
            self.head.num_classes = full_count

        return logits, remapped

    def extra_repr(self) -> str:
        return (
            f"sample_rate={self.sample_rate}, num_sampled={self.num_sampled}, "
            f"num_classes={self.num_classes}"
        )


def maybe_wrap_partial_fc(
    head: MarginHead, partial_fc_cfg: dict | None, num_classes: int
) -> nn.Module:
    """Apply Partial-FC according to config.

    ``enabled: auto`` turns it on above :data:`AUTO_ENABLE_THRESHOLD` classes.
    """
    if not partial_fc_cfg:
        return head

    enabled = partial_fc_cfg.get("enabled", "auto")
    if enabled == "auto":
        enabled = num_classes > AUTO_ENABLE_THRESHOLD
        if enabled:
            logger.info(
                "Partial-FC auto-enabled: %d classes exceeds the %d threshold",
                num_classes, AUTO_ENABLE_THRESHOLD,
            )
    if not enabled:
        return head

    return PartialFC(head, sample_rate=float(partial_fc_cfg.get("sample_rate", 0.1)))
