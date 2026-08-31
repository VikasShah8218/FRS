"""Margin-softmax classifier heads: ArcFace, CosFace and AdaFace.

All three share one interface so the trainer never branches on loss type:

    logits = head(embeddings, norms, labels)
    loss   = F.cross_entropy(logits, labels)

Heads return **logits**, not a loss. That keeps label smoothing, class weighting
and Partial-FC orthogonal to the margin maths.

The common idea
---------------
A plain linear classifier learns ``W`` such that ``W_y . x`` is large for the
correct class. Margin softmax first *normalises* both ``W`` and ``x`` so the
logit is exactly ``cos(theta)`` between the sample and its class prototype, then
makes the correct class harder by subtracting a margin before the softmax. The
result is embeddings that are compact within an identity and well separated
between identities -- which is what a verification system actually needs, since
at test time you compare two embeddings by cosine similarity and never use this
head at all.

Numerical stability
-------------------
The single most important line in this file is the clamp in
:meth:`MarginHead._cosine`. ``d/dx acos(x) = -1/sqrt(1-x^2)`` is infinite at
``|x| = 1``, and a freshly initialised head in fp16 *will* produce ``cos = 1.0``
for some sample. Without the clamp you get NaN loss within the first few
hundred steps. See ``tests/test_heads.py`` for the adversarial test.

The second is that these heads must run in **fp32 even under AMP** -- fp16
``acos`` near the domain edge loses catastrophic precision. The trainer enforces
this with ``autocast(enabled=False)``; see ``frs/engine/trainer.py``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import HEADS

#: Keeps cos strictly inside (-1, 1) so acos and sqrt stay differentiable.
EPS = 1e-7


class MarginHead(nn.Module):
    """Base class holding the shared normalised-cosine computation.

    Parameters
    ----------
    embedding_size:
        Dimensionality of the backbone output (512 for these IResNets).
    num_classes:
        Number of identities. Row *i* of :attr:`weight` is class *i*'s prototype
        -- which is why the ClassMap must never reassign indices.
    scale:
        The ``s`` hyperparameter. Cosines live in [-1, 1], far too small a range
        for cross-entropy to produce useful gradients, so logits are multiplied
        by ``s`` (64 is standard for 512-d embeddings).
    """

    def __init__(
        self, embedding_size: int, num_classes: int, scale: float = 64.0
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        self.embedding_size = int(embedding_size)
        self.num_classes = int(num_classes)
        self.scale = float(scale)

        self.weight = nn.Parameter(torch.empty(self.num_classes, self.embedding_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # std=0.01 is the ArcFace reference init. Xavier/Kaiming are wrong here:
        # the rows are normalised before use, so only their direction matters and
        # a small isotropic init spreads prototypes evenly on the hypersphere.
        nn.init.normal_(self.weight, std=0.01)

    def _cosine(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between every embedding and every class prototype.

        ``embeddings`` is expected to be L2-normalised already (the IResNet
        backbone does this), but it is re-normalised defensively so the head is
        correct with any backbone.
        """
        x = F.normalize(embeddings, dim=1)
        w = F.normalize(self.weight, dim=1)
        return F.linear(x, w).clamp(-1.0 + EPS, 1.0 - EPS)

    def forward(
        self,
        embeddings: torch.Tensor,
        norms: torch.Tensor | None,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return (
            f"embedding_size={self.embedding_size}, num_classes={self.num_classes}, "
            f"scale={self.scale}"
        )


@HEADS.register("arcface")
class ArcFaceHead(MarginHead):
    """Additive **angular** margin: ``cos(theta_y + m)``.

    The margin is applied to the angle, so it is uniform across the hypersphere
    -- this is what makes ArcFace's decision boundary geodesically consistent
    and why it outperformed CosFace on most benchmarks.

    Two implementation details that matter:

    1. **No ``acos`` on the default path.** ``cos(theta + m)`` is expanded with
       the angle-addition identity into ``cos_y*cos(m) - sin_y*sin(m)``, which is
       algebraically identical, avoids the ill-conditioned ``acos``, and is what
       the reference CUDA kernel does. ``use_acos=True`` selects the literal
       formulation; ``tests/test_heads.py`` asserts the two agree to 1e-5.

    2. **The theta + m > pi guard.** ``cos`` is not monotonic past pi, so for
       samples already pointing away from their prototype the margin would
       *reduce* the loss -- exactly backwards. ArcFace handles this with a linear
       extension below the threshold; ``easy_margin`` instead just skips the
       margin there. The linear extension is correct and is the default;
       ``easy_margin`` is a stability crutch for early training.
    """

    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        scale: float = 64.0,
        m: float = 0.5,
        easy_margin: bool = False,
        use_acos: bool = False,
    ) -> None:
        super().__init__(embedding_size, num_classes, scale)
        self.m = float(m)
        self.easy_margin = bool(easy_margin)
        self.use_acos = bool(use_acos)

        # Constants for the theta + m > pi branch, precomputed as buffers so they
        # move with .to(device) and land in the checkpoint.
        self.register_buffer("cos_m", torch.tensor(math.cos(self.m)), persistent=False)
        self.register_buffer("sin_m", torch.tensor(math.sin(self.m)), persistent=False)
        # theta > pi - m  <=>  cos_y < cos(pi - m)
        self.register_buffer(
            "threshold", torch.tensor(math.cos(math.pi - self.m)), persistent=False
        )
        # The linear extension's offset: sin(pi - m) * m
        self.register_buffer(
            "mm", torch.tensor(math.sin(math.pi - self.m) * self.m), persistent=False
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        norms: torch.Tensor | None,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        cos = self._cosine(embeddings)
        idx = torch.arange(cos.size(0), device=cos.device)
        cos_y = cos[idx, labels]

        if self.use_acos:
            theta = torch.acos(cos_y)
            target = torch.cos(theta + self.m)
        else:
            sin_y = torch.sqrt((1.0 - cos_y.pow(2)).clamp_min(0.0))
            target = cos_y * self.cos_m - sin_y * self.sin_m

        if self.easy_margin:
            target = torch.where(cos_y > 0, target, cos_y)
        else:
            target = torch.where(cos_y > self.threshold, target, cos_y - self.mm)

        logits = cos.clone()
        logits[idx, labels] = target
        return logits * self.scale

    def extra_repr(self) -> str:
        return (
            f"{super().extra_repr()}, m={self.m}, easy_margin={self.easy_margin}"
        )


@HEADS.register("cosface")
class CosFaceHead(MarginHead):
    """Additive **cosine** margin: ``cos(theta_y) - m``.

    Simpler and unconditionally stable -- there is no ``acos``, no domain edge
    and no monotonicity problem. Keep it as the diagnostic baseline: if CosFace
    trains cleanly but ArcFace produces NaN, the bug is in the angular branch,
    not in the data or the backbone.
    """

    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        scale: float = 64.0,
        m: float = 0.35,
    ) -> None:
        super().__init__(embedding_size, num_classes, scale)
        self.m = float(m)

    def forward(
        self,
        embeddings: torch.Tensor,
        norms: torch.Tensor | None,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        cos = self._cosine(embeddings)
        idx = torch.arange(cos.size(0), device=cos.device)
        logits = cos.clone()
        logits[idx, labels] = cos[idx, labels] - self.m
        return logits * self.scale

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, m={self.m}"


@HEADS.register("adaface")
class AdaFaceHead(MarginHead):
    """Quality-adaptive margin, scaled by the embedding's feature norm.

    The insight: the L2 norm of an un-normalised face embedding correlates
    strongly with image quality. A blurry, occluded or badly-posed face produces
    a low norm. Applying a large margin to such a sample forces the model to fit
    an image whose identity information is genuinely degraded -- it is being
    asked to memorise noise.

    AdaFace therefore scales the margin by the normalised feature norm:

    * high norm (clean image)  -> larger margin, emphasise this sample
    * low norm  (poor image)   -> smaller margin, de-emphasise it

    This is why the IResNet ``forward`` returns ``(embedding, norm)``: the norm
    is not a diagnostic, it is an input to the loss.

    Three details implementations routinely get wrong
    -------------------------------------------------
    * ``batch_std`` is initialised to **100**, not 1. That makes ``margin_scaler``
      approximately 0 for the first ~100 steps, so training begins ArcFace-like
      and eases into quality adaptivity as the EMA converges. It is deliberate.
    * The EMA buffers are part of ``state_dict`` and therefore the checkpoint.
      Losing them on resume silently changes the loss landscape.
    * The norm statistics are **detached** -- no gradient flows through the
      quality estimate, only through the embedding direction.
    """

    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        scale: float = 64.0,
        m: float = 0.4,
        h: float = 0.333,
        t_alpha: float = 0.01,
        eps: float = 1e-3,
    ) -> None:
        super().__init__(embedding_size, num_classes, scale)
        self.m = float(m)
        self.h = float(h)
        self.t_alpha = float(t_alpha)
        self.eps = float(eps)

        # Persistent: these must survive save/load or resume changes behaviour.
        self.register_buffer("batch_mean", torch.ones(1) * 20.0)
        self.register_buffer("batch_std", torch.ones(1) * 100.0)

    def forward(
        self,
        embeddings: torch.Tensor,
        norms: torch.Tensor | None,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if norms is None:
            raise ValueError(
                "AdaFace requires feature norms. The IResNet backbone returns "
                "(embedding, norm) -- pass the second value through."
            )

        cos = self._cosine(embeddings)
        idx = torch.arange(cos.size(0), device=cos.device)

        safe_norms = norms.clone().detach().view(-1).clamp(0.001, 100.0)

        # --- EMA of the batch norm statistics -------------------------------
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = (
                safe_norms.std().detach()
                if safe_norms.numel() > 1
                else torch.zeros((), device=safe_norms.device)
            )
            self.batch_mean.mul_(1.0 - self.t_alpha).add_(self.t_alpha * mean)
            self.batch_std.mul_(1.0 - self.t_alpha).add_(self.t_alpha * std)

        # Standardise the norm, then squash to [-1, 1]. h controls how much of
        # the norm distribution maps into the usable margin range.
        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps)
        margin_scaler = (margin_scaler * self.h).clamp(-1.0, 1.0)

        cos_y = cos[idx, labels]

        # --- g_angle: angular margin, negated so high norm => larger margin ---
        theta = torch.acos(cos_y)
        g_angle = -self.m * margin_scaler
        theta_m = (theta + g_angle).clamp(self.eps, math.pi - self.eps)
        cos_y = torch.cos(theta_m)

        # --- g_add: additive cosine margin in [0, 2m] ------------------------
        g_add = self.m + (self.m * margin_scaler)

        logits = cos.clone()
        logits[idx, labels] = cos_y - g_add
        return logits * self.scale

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, m={self.m}, h={self.h}, t_alpha={self.t_alpha}"


def build_head(head_cfg: dict, embedding_size: int, num_classes: int) -> MarginHead:
    """Construct a margin head from the ``model.head`` config block.

    ``partial_fc`` is consumed here rather than passed to the head: it is a
    wrapper around any head, not a head option.
    """
    cfg = {k: v for k, v in dict(head_cfg).items() if k != "partial_fc"}
    if "scale" in cfg:  # YAML calls it 'scale'; heads take 'scale' too
        cfg["scale"] = float(cfg["scale"])
    return HEADS.build(
        cfg, embedding_size=embedding_size, num_classes=num_classes
    )
