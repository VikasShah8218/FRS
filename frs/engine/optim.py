"""Optimiser, parameter grouping and learning-rate schedules.

Two things here are easy to get wrong and cost real accuracy:

1. **Weight decay must skip normalisation and bias parameters.** Decaying a
   BatchNorm scale pulls it toward zero, which fights the layer's purpose;
   published ablations put the cost at roughly 0.5-1% verification accuracy.
   The filter is ``param.ndim <= 1``, *not* a name match -- PReLU's learnable
   parameter is called ``weight`` and is 1-D, so name-based filtering misses it.

2. **Warmup is not optional.** At the start of training the classifier
   prototypes are random, so a sizeable fraction of samples sit at
   ``theta > pi - m`` where the ArcFace margin hits its guard branch. A high LR
   there produces the divergence people blame on "ArcFace being unstable".
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def build_param_groups(
    modules: Iterable[torch.nn.Module],
    weight_decay: float,
    no_wd_on_bn_and_bias: bool = True,
    head_lr_multiplier: float = 1.0,
    head_modules: Iterable[torch.nn.Module] = (),
) -> list[dict[str, Any]]:
    """Split parameters into decayed / non-decayed (and optionally head) groups."""
    head_params = {id(p) for m in head_modules for p in m.parameters()}

    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    head_decay: list[torch.nn.Parameter] = []
    head_no_decay: list[torch.nn.Parameter] = []

    for module in modules:
        for param in module.parameters():
            if not param.requires_grad:
                continue
            # ndim <= 1 catches BatchNorm weight/bias, all biases, and PReLU.
            skip_wd = no_wd_on_bn_and_bias and param.ndim <= 1
            is_head = id(param) in head_params
            if is_head:
                (head_no_decay if skip_wd else head_decay).append(param)
            else:
                (no_decay if skip_wd else decay).append(param)

    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay, "lr_scale": 1.0})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0, "lr_scale": 1.0})
    if head_decay:
        groups.append(
            {
                "params": head_decay,
                "weight_decay": weight_decay,
                "lr_scale": head_lr_multiplier,
            }
        )
    if head_no_decay:
        groups.append(
            {"params": head_no_decay, "weight_decay": 0.0, "lr_scale": head_lr_multiplier}
        )
    if not groups:
        raise ValueError("no trainable parameters found")
    return groups


def build_optimizer(
    optim_cfg: Any,
    backbone: torch.nn.Module,
    head: torch.nn.Module,
) -> Optimizer:
    """Construct the optimiser described by the ``optim`` config block.

    SGD with momentum is the default and the right choice: on margin-softmax
    face recognition it consistently beats AdamW, which tends to underfit the
    classifier by 1-2%. AdamW is offered for fine-tuning and for architectures
    where SGD is fussy.
    """
    kind = str(optim_cfg.get("type", "sgd")).lower()
    lr = float(optim_cfg["lr"])
    weight_decay = float(optim_cfg.get("weight_decay", 5e-4))

    groups = build_param_groups(
        modules=[backbone, head],
        weight_decay=weight_decay,
        no_wd_on_bn_and_bias=bool(optim_cfg.get("no_wd_on_bn_and_bias", True)),
        head_lr_multiplier=float(optim_cfg.get("head_lr_multiplier", 1.0)),
        head_modules=[head],
    )
    for g in groups:
        g["lr"] = lr * g.get("lr_scale", 1.0)

    if kind == "sgd":
        return torch.optim.SGD(
            groups,
            lr=lr,
            momentum=float(optim_cfg.get("momentum", 0.9)),
            nesterov=bool(optim_cfg.get("nesterov", False)),
        )
    if kind == "adamw":
        return torch.optim.AdamW(
            groups,
            lr=lr,
            betas=tuple(optim_cfg.get("betas", (0.9, 0.999))),
            eps=float(optim_cfg.get("eps", 1e-8)),
        )
    if kind == "adam":
        return torch.optim.Adam(
            groups, lr=lr, betas=tuple(optim_cfg.get("betas", (0.9, 0.999)))
        )
    raise ValueError(f"unknown optimiser type {kind!r}; expected sgd, adamw or adam")


class WarmupWrapper(LRScheduler):
    """Linear warmup followed by any per-step base schedule.

    Stepping is **per optimiser step**, not per epoch, so the warmup is smooth
    even when an epoch is only a few hundred steps (as it is on MeGlass).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        mode: str = "poly",
        power: float = 2.0,
        min_lr: float = 0.0,
        warmup_start_factor: float = 0.1,
        milestones: list[int] | None = None,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.mode = mode
        self.power = float(power)
        self.min_lr = float(min_lr)
        self.warmup_start_factor = float(warmup_start_factor)
        self.milestones = sorted(milestones or [])
        self.gamma = float(gamma)
        super().__init__(optimizer, last_epoch)

    def _factor(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            alpha = step / max(1, self.warmup_steps)
            return self.warmup_start_factor + (1.0 - self.warmup_start_factor) * alpha

        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)

        if self.mode == "poly":
            return (1.0 - progress) ** self.power
        if self.mode == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if self.mode in ("step", "multistep"):
            n = sum(1 for m in self.milestones if step >= m)
            return self.gamma ** n
        if self.mode == "constant":
            return 1.0
        raise ValueError(f"unknown scheduler mode {self.mode!r}")

    def get_lr(self) -> list[float]:  # type: ignore[override]
        factor = self._factor(self.last_epoch)
        return [
            self.min_lr + (base - self.min_lr) * factor for base in self.base_lrs
        ]


def build_scheduler(
    sched_cfg: Any,
    optimizer: Optimizer,
    steps_per_epoch: int,
    epochs: int,
) -> WarmupWrapper:
    """Construct the LR schedule.

    PolyLR (power 2) is the default: it is what InsightFace uses, it decays to
    exactly zero at the final step, and unlike step decay it needs no milestone
    tuning when the dataset size changes.
    """
    kind = str(sched_cfg.get("type", "polylr")).lower()
    mode = {
        "polylr": "poly",
        "poly": "poly",
        "cosine": "cosine",
        "step": "step",
        "multistep": "multistep",
        "constant": "constant",
    }.get(kind)
    if mode is None:
        raise ValueError(
            f"unknown scheduler type {kind!r}; expected polylr, cosine, step, "
            f"multistep or constant"
        )

    total_steps = steps_per_epoch * epochs
    warmup_steps = int(round(float(sched_cfg.get("warmup_epochs", 0)) * steps_per_epoch))
    milestones = [
        int(m * steps_per_epoch) for m in sched_cfg.get("milestones", []) or []
    ]

    return WarmupWrapper(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        mode=mode,
        power=float(sched_cfg.get("power", 2.0)),
        min_lr=float(sched_cfg.get("min_lr", 0.0)),
        warmup_start_factor=float(sched_cfg.get("warmup_start_factor", 0.1)),
        milestones=milestones,
        gamma=float(sched_cfg.get("gamma", 0.1)),
    )
