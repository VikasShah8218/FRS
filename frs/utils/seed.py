"""Reproducibility helpers.

``deterministic=True`` trades roughly 10-20% throughput for bit-reproducible
runs. Use it when chasing a bug; leave it off for real training, where
``cudnn.benchmark`` picks faster convolution algorithms.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_all(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG the training loop touches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required for deterministic cuBLAS GEMMs; must be set before first use.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        # Lets cuDNN benchmark conv algorithms on first sight of each shape.
        # Worth ~15% here because our input shape never varies.
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker a distinct, reproducible seed.

    Without this, workers forked/spawned from the same parent can generate
    identical augmentation streams, silently reducing effective augmentation.
    """
    base = torch.initial_seed() % (2**32)
    seed = (base + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
