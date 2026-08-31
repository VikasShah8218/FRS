"""Distributed (DDP) helpers.

Single-GPU training uses none of this -- every function degrades to a sensible
no-op when ``torchrun`` did not set the environment. Launch multi-GPU with:

    torchrun --nproc_per_node=4 -m scripts.train --config configs/aws_ir100_adaface.yaml

Two things to remember when scaling out:

* **LR scales with the TOTAL batch** across all GPUs, not per-GPU batch.
* Leave ``SyncBatchNorm`` off when the per-GPU batch is >= 64 -- the local
  statistics are already good and syncing costs throughput.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed() -> tuple[int, int, int]:
    """Initialise the process group if launched under torchrun.

    Returns ``(rank, world_size, local_rank)``; ``(0, 1, 0)`` when running
    single-process.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # NCCL on Linux; gloo is the only option on Windows.
    backend = "nccl" if dist.is_nccl_available() and torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    logger.info(
        "Distributed: rank %d/%d (local %d), backend %s",
        rank, world_size, local_rank, backend,
    )
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def all_reduce_mean(value: torch.Tensor | float) -> float:
    """Average a scalar across ranks (for logging a global metric)."""
    if not is_distributed():
        return float(value)
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / get_world_size())


def wrap_ddp(model: torch.nn.Module, local_rank: int, **kwargs: Any) -> torch.nn.Module:
    """Wrap a model in DistributedDataParallel when running distributed.

    ``find_unused_parameters`` defaults to False: every parameter in this
    architecture receives gradient on every step, and enabling the search costs
    a measurable amount of throughput for nothing.
    """
    if not is_distributed():
        return model
    kwargs.setdefault("find_unused_parameters", False)
    device_ids = [local_rank] if torch.cuda.is_available() else None
    return torch.nn.parallel.DistributedDataParallel(
        model, device_ids=device_ids, output_device=local_rank if device_ids else None,
        **kwargs,
    )


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Get the underlying module out of a DDP wrapper (for checkpointing)."""
    return getattr(model, "module", model)
