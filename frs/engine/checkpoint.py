"""Full-state checkpointing, resume, and incremental class extension.

What is saved
-------------
Everything needed to continue training as if it never stopped:

* backbone weights + architecture spec
* margin head weights **and buffers** (AdaFace's EMA norm statistics live here;
  losing them silently changes the loss landscape on resume)
* optimizer state (momentum / Adam moments)
* LR scheduler state
* AMP GradScaler state
* epoch and global step
* RNG state for python, numpy and torch (CPU + all CUDA devices)
* the **ClassMap** -- identity string to class index
* metric history and the fully-resolved config, for provenance

Why the ClassMap matters
------------------------
Head row *i* is the learned prototype for class *i*. Without a durable record of
which identity *i* refers to, a later run cannot reuse those rows -- it would be
starting from zero no matter how many weights it loaded. Storing the map is what
turns "load some weights" into genuine incremental training.

Extending to new identities
---------------------------
:func:`extend_head_and_optimizer` grows the classifier for new identities while
preserving every existing row. It also grows the **optimizer state**, which is
the step people forget: SGD's momentum buffer for ``head.weight`` has shape
``(old_C, D)`` and raises a shape-mismatch RuntimeError on the first step after
extension if left untouched.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data.class_map import ClassMap
from ..models.partial_fc import unwrap_partial_fc

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT_VERSION = 1


# --------------------------------------------------------------------- RNG state


def capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG streams.

    Note: DataLoader workers are reseeded per epoch from ``base_seed + epoch``,
    so augmentation order after a resume differs from an uninterrupted run even
    though the model state is bit-identical. That is expected -- do not chase it.
    """
    if not state:
        return
    try:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.set_rng_state(_as_byte_tensor(state["torch"]))
        if "torch_cuda" in state and torch.cuda.is_available():
            saved = state["torch_cuda"]
            if len(saved) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all([_as_byte_tensor(s) for s in saved])
            else:
                logger.warning(
                    "Skipping CUDA RNG restore: checkpoint has %d device states, "
                    "this machine has %d",
                    len(saved),
                    torch.cuda.device_count(),
                )
    except Exception as exc:  # never let RNG restore kill a resume
        logger.warning("Could not fully restore RNG state: %s", exc)


def _as_byte_tensor(value: Any) -> torch.Tensor:
    t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return t.cpu().to(torch.uint8)


# ------------------------------------------------------------------ save / load


@dataclass
class ResumeState:
    """What a resume recovered."""

    epoch: int = 0
    global_step: int = 0
    best_metrics: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    class_map: ClassMap | None = None
    extended: bool = False
    checkpoint_path: str | None = None


def save_checkpoint(
    path: str | os.PathLike,
    *,
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    class_map: ClassMap | None = None,
    epoch: int = 0,
    global_step: int = 0,
    config: dict | None = None,
    metrics: dict | None = None,
    history: list[dict] | None = None,
    save_rng: bool = True,
    extra: dict | None = None,
) -> str:
    """Write a complete checkpoint atomically.

    Atomic write (temp file then ``os.replace``) matters more than it sounds:
    a crash or spot-instance reclaim mid-write would otherwise leave a truncated
    ``last.pt`` and destroy the run's only recovery point.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Always describe and store the *inner* margin head. Partial-FC is a
    # training-time wrapper; recording ``sample_rate`` is provenance only.
    inner_head = unwrap_partial_fc(unwrap_model(head))
    partial_fc_rate = getattr(unwrap_model(head), "sample_rate", None)
    backbone_inner = unwrap_model(backbone)

    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "backbone": {
            "arch": getattr(backbone_inner, "arch", "unknown"),
            "input_size": list(getattr(backbone_inner, "input_size", (112, 112))),
            "embedding_size": int(getattr(backbone_inner, "embedding_size", 512)),
            "state_dict": _cpu_state_dict(backbone),
        },
        "head": {
            "type": type(inner_head).__name__,
            "num_classes": int(getattr(inner_head, "num_classes", 0)),
            "embedding_size": int(getattr(inner_head, "embedding_size", 512)),
            "partial_fc": partial_fc_rate,
            "state_dict": _cpu_state_dict(head),
        },
        "metrics": metrics or {},
        "history": history or [],
        "config": config or {},
    }

    if optimizer is not None:
        payload["optimizer"] = {
            "type": type(optimizer).__name__,
            "state_dict": optimizer.state_dict(),
        }
    if scheduler is not None:
        payload["scheduler"] = {
            "type": type(scheduler).__name__,
            "state_dict": scheduler.state_dict(),
        }
    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        payload["scaler"] = scaler.state_dict()
    if class_map is not None:
        payload["class_map"] = class_map.to_dict()
    if save_rng:
        payload["rng"] = capture_rng_state()
    if extra:
        payload["extra"] = extra

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    logger.info("Saved checkpoint: %s (epoch %d, step %d)", path, epoch, global_step)
    return str(path)


def save_backbone_only(
    path: str | os.PathLike,
    backbone: torch.nn.Module,
    model_name: str | None = None,
) -> str:
    """Write a deployment artifact: backbone weights, no head, no optimizer.

    Inference only ever needs embeddings, so shipping the margin head (and its
    per-class rows) wastes space and leaks the training identity list.

    This is the **only** file that should leave the building. It carries the
    weights, the input contract and a model name -- and deliberately nothing
    about who was in the training set, what loss shaped the embedding space, or
    what hyperparameters were used. ``best.pt`` and ``last.pt`` carry all of
    that (they must, to resume and to extend), so they stay internal.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    inner = unwrap_model(backbone)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_name": model_name or "unnamed",
        "arch": getattr(inner, "arch", "unknown"),
        "input_size": list(getattr(inner, "input_size", (112, 112))),
        "embedding_size": int(getattr(inner, "embedding_size", 512)),
        "state_dict": _cpu_state_dict(backbone),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return str(path)


def unwrap_model(module: torch.nn.Module) -> torch.nn.Module:
    """Strip DDP and ``torch.compile`` wrappers in whichever order they were applied."""
    while True:
        if hasattr(module, "_orig_mod"):  # torch.compile
            module = module._orig_mod
        elif hasattr(module, "module") and isinstance(
            getattr(module, "module"), torch.nn.Module
        ) and type(module).__name__ in ("DistributedDataParallel", "DataParallel"):
            module = module.module
        else:
            return module


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    inner = unwrap_partial_fc(unwrap_model(module))  # Partial-FC -> bare margin head
    return {k: v.detach().cpu() for k, v in inner.state_dict().items()}


def load_checkpoint(
    path: str | os.PathLike,
    *,
    backbone: torch.nn.Module | None = None,
    head: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    class_map: ClassMap | None = None,
    map_location: str = "cpu",
    strict: bool = True,
    restore_rng: bool = True,
) -> ResumeState:
    """Restore a checkpoint into the given objects.

    If ``class_map`` is supplied and contains more identities than the
    checkpoint, the head and optimizer state are extended automatically before
    the weights are loaded, and the fact is logged loudly.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    version = ckpt.get("format_version", 0)
    if version > CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError(
            f"checkpoint format v{version} is newer than this code supports "
            f"(v{CHECKPOINT_FORMAT_VERSION}). Update the codebase."
        )

    state = ResumeState(
        epoch=int(ckpt.get("epoch", 0)),
        global_step=int(ckpt.get("global_step", 0)),
        best_metrics=dict(ckpt.get("metrics", {}) or {}),
        history=list(ckpt.get("history", []) or []),
        checkpoint_path=str(path),
    )

    ckpt_map = (
        ClassMap.from_dict(ckpt["class_map"]) if ckpt.get("class_map") else None
    )
    state.class_map = ckpt_map

    # Load into the bare modules: the checkpoint stores inner-module tensors.
    if backbone is not None:
        backbone = unwrap_model(backbone)
    if head is not None:
        head = unwrap_partial_fc(unwrap_model(head))

    head_state = dict(ckpt.get("head", {}).get("state_dict", {}))
    old_classes = int(ckpt.get("head", {}).get("num_classes", 0))
    optim_state = ckpt.get("optimizer", {}).get("state_dict")

    # --- incremental extension --------------------------------------------
    if class_map is not None and ckpt_map is not None:
        _assert_prefix_compatible(ckpt_map, class_map)
        if class_map.num_classes > ckpt_map.num_classes:
            logger.warning(
                "Extending head: checkpoint has %d classes, current dataset has %d "
                "(+%d new identities)",
                ckpt_map.num_classes,
                class_map.num_classes,
                class_map.num_classes - ckpt_map.num_classes,
            )
            head_state, optim_state = extend_head_and_optimizer(
                head_state=head_state,
                optimizer_state=optim_state,
                old_num_classes=ckpt_map.num_classes,
                new_num_classes=class_map.num_classes,
                head=head,
            )
            state.extended = True
            state.class_map = class_map
        elif class_map.num_classes < ckpt_map.num_classes:
            logger.warning(
                "Current dataset has FEWER identities (%d) than the checkpoint (%d). "
                "Keeping the checkpoint's class map so existing head rows stay valid.",
                class_map.num_classes,
                ckpt_map.num_classes,
            )

    if backbone is not None and ckpt.get("backbone"):
        result = backbone.load_state_dict(
            ckpt["backbone"]["state_dict"], strict=strict
        )
        _log_load_result("backbone", result)

    if head is not None and head_state:
        result = head.load_state_dict(head_state, strict=strict)
        _log_load_result("head", result)

    if optimizer is not None and optim_state is not None:
        try:
            optimizer.load_state_dict(optim_state)
        except ValueError as exc:
            logger.warning(
                "Could not restore optimizer state (%s). Continuing with a fresh "
                "optimizer -- momentum will rebuild within a few hundred steps.",
                exc,
            )

    if scheduler is not None and ckpt.get("scheduler"):
        try:
            scheduler.load_state_dict(ckpt["scheduler"]["state_dict"])
        except Exception as exc:
            logger.warning("Could not restore scheduler state: %s", exc)

    if scaler is not None and ckpt.get("scaler"):
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as exc:
            logger.warning("Could not restore GradScaler state: %s", exc)

    if restore_rng and ckpt.get("rng") and not state.extended:
        # After an extension the data order legitimately changes, so restoring
        # the old RNG stream would be misleading rather than helpful.
        restore_rng_state(ckpt["rng"])

    logger.info(
        "Resumed from %s at epoch %d, step %d%s",
        path,
        state.epoch,
        state.global_step,
        " (head extended)" if state.extended else "",
    )
    return state


def _log_load_result(name: str, result: Any) -> None:
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    if missing:
        logger.warning("%s: %d missing keys, e.g. %s", name, len(missing), missing[:3])
    if unexpected:
        logger.warning(
            "%s: %d unexpected keys, e.g. %s", name, len(unexpected), unexpected[:3]
        )


def _assert_prefix_compatible(old: ClassMap, new: ClassMap) -> None:
    """The new map must be an append-only extension of the old one.

    If index *i* means a different person than it did when the head row was
    trained, every learned prototype is silently wrong. Better to fail loudly.
    """
    n = min(old.num_classes, new.num_classes)
    for i in range(n):
        if old.index_to_identity[i] != new.index_to_identity[i]:
            raise ValueError(
                f"class map mismatch at index {i}: checkpoint has "
                f"{old.index_to_identity[i]!r} but the current dataset has "
                f"{new.index_to_identity[i]!r}. The head's learned rows would be "
                f"meaningless. Build the new map by calling extend() on the "
                f"checkpoint's map (scripts/extend_classmap.py) rather than "
                f"rebuilding it from scratch."
            )


# ----------------------------------------------------------------- extension


def extend_head_and_optimizer(
    head_state: dict[str, torch.Tensor],
    optimizer_state: dict | None,
    old_num_classes: int,
    new_num_classes: int,
    head: torch.nn.Module | None = None,
    init: str = "normal",
    new_prototypes: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict | None]:
    """Grow the classifier weight and its optimizer state to ``new_num_classes``.

    Parameters
    ----------
    init:
        ``"normal"`` -- new rows drawn from N(0, 0.01), matching the original init.
        ``"mean_embedding"`` -- new rows supplied via ``new_prototypes``, computed
        by averaging normalised embeddings of each new identity. This starts new
        classes near their true angular position, which removes the loss spike
        that random init causes and roughly halves fine-tuning time.
    new_prototypes:
        ``(new_num_classes - old_num_classes, D)`` tensor, required when
        ``init="mean_embedding"``.

    Returns
    -------
    (head_state, optimizer_state)
        Both grown in place-compatible form. Existing rows are bit-identical.
    """
    if new_num_classes <= old_num_classes:
        return head_state, optimizer_state

    if "weight" not in head_state:
        raise KeyError(
            f"head state dict has no 'weight' tensor; keys are {list(head_state)}"
        )

    w_old = head_state["weight"]
    n_new = new_num_classes - old_num_classes
    dim = w_old.shape[1]

    w_new = torch.empty(new_num_classes, dim, dtype=w_old.dtype)
    if init == "mean_embedding":
        if new_prototypes is None:
            raise ValueError("init='mean_embedding' requires new_prototypes")
        if tuple(new_prototypes.shape) != (n_new, dim):
            raise ValueError(
                f"new_prototypes must be {(n_new, dim)}, got {tuple(new_prototypes.shape)}"
            )
        w_new[old_num_classes:] = new_prototypes.to(w_old.dtype)
    elif init == "normal":
        torch.nn.init.normal_(w_new[old_num_classes:], std=0.01)
    else:
        raise ValueError(f"unknown init {init!r}; expected 'normal' or 'mean_embedding'")

    w_new[:old_num_classes] = w_old  # preserve every learned prototype
    head_state["weight"] = w_new

    if head is not None and tuple(head.weight.shape) != (new_num_classes, dim):
        # Resize the live module so load_state_dict does not complain. Only
        # when needed: replacing the Parameter object would orphan it from an
        # optimizer that was already built around it.
        head.weight = torch.nn.Parameter(
            torch.empty(new_num_classes, dim, device=head.weight.device,
                        dtype=head.weight.dtype)
        )
    if head is not None:
        head.num_classes = new_num_classes

    optimizer_state = _extend_optimizer_state(
        optimizer_state, old_num_classes, new_num_classes, dim
    )
    logger.info(
        "Head extended %d -> %d classes (%d new rows, init=%s)",
        old_num_classes,
        new_num_classes,
        n_new,
        init,
    )
    return head_state, optimizer_state


def _extend_optimizer_state(
    optimizer_state: dict | None,
    old_num_classes: int,
    new_num_classes: int,
    dim: int,
) -> dict | None:
    """Zero-pad any optimizer buffer shaped like the old head weight.

    This is the step that is almost always forgotten. SGD stores
    ``momentum_buffer`` and AdamW stores ``exp_avg``/``exp_avg_sq``, each the same
    shape as the parameter. After extending the weight to ``(new_C, D)`` the
    stale ``(old_C, D)`` buffers raise a RuntimeError on the very next step.
    """
    if not optimizer_state or "state" not in optimizer_state:
        return optimizer_state

    patched = 0
    for param_state in optimizer_state["state"].values():
        for key, value in list(param_state.items()):
            if (
                isinstance(value, torch.Tensor)
                and value.ndim == 2
                and value.shape == (old_num_classes, dim)
            ):
                grown = torch.zeros(
                    new_num_classes, dim, dtype=value.dtype, device=value.device
                )
                grown[:old_num_classes] = value
                param_state[key] = grown
                patched += 1

    if patched:
        logger.info("Extended %d optimizer buffers to %d rows", patched, new_num_classes)
    return optimizer_state


# ------------------------------------------------------------------ utilities


def peek_class_map(path: str | os.PathLike) -> ClassMap | None:
    """Read only the ClassMap out of a checkpoint (before any model is built).

    ``scripts/train.py`` uses this to build the dataset *against the
    checkpoint's index assignment* and append new identities to it, instead of
    rebuilding a fresh map that would not line up with the trained head rows.
    """
    path = Path(path)
    if not path.is_file():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    data = ckpt.get("class_map")
    return ClassMap.from_dict(data) if data else None


def find_latest_checkpoint(output_dir: str | os.PathLike) -> str | None:
    """Locate ``last.pt`` for ``resume: auto``."""
    candidate = Path(output_dir) / "checkpoints" / "last.pt"
    return str(candidate) if candidate.is_file() else None


def prune_checkpoints(checkpoint_dir: str | os.PathLike, keep_last_n: int) -> None:
    """Delete old per-epoch checkpoints, keeping ``last.pt`` and ``best.pt``."""
    if keep_last_n <= 0:
        return
    directory = Path(checkpoint_dir)
    if not directory.is_dir():
        return
    epoch_ckpts = sorted(
        directory.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for stale in epoch_ckpts[keep_last_n:]:
        try:
            stale.unlink()
            logger.debug("Pruned old checkpoint %s", stale)
        except OSError as exc:
            logger.warning("Could not delete %s: %s", stale, exc)


def inspect_checkpoint(path: str | os.PathLike) -> dict[str, Any]:
    """Summarise a checkpoint without loading it into modules."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    class_map = ckpt.get("class_map") or {}
    return {
        "path": str(path),
        "format_version": ckpt.get("format_version"),
        "created_at": ckpt.get("created_at"),
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "backbone_arch": ckpt.get("backbone", {}).get("arch"),
        "head_type": ckpt.get("head", {}).get("type"),
        "num_classes": ckpt.get("head", {}).get("num_classes"),
        "num_identities_in_map": len(class_map.get("index_to_identity", [])),
        "has_optimizer": "optimizer" in ckpt,
        "has_scheduler": "scheduler" in ckpt,
        "has_scaler": "scaler" in ckpt,
        "has_rng": "rng" in ckpt,
        "metrics": ckpt.get("metrics", {}),
    }
