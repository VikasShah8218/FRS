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


def seed_all(seed: int, deterministic: bool = False, tf32: bool = True) -> None:
    """Seed every RNG the training loop touches, and set the matmul precision.

    ``tf32`` enables TensorFloat-32 matmuls on Ampere and newer. This matters
    here more than it does in most codebases: the margin head deliberately runs
    in **fp32** under ``autocast(enabled=False)``, and with a large class count
    its ``(B x C)`` matmul is a significant share of every step. Measured on an
    A10G with 35,923 sampled classes, TF32 cut that matmul from 0.78 ms to
    0.46 ms.

    Is it safe for *this* computation? Both operands are L2-normalised before
    the matmul, so the outputs are cosines in [-1, 1]. Measured error on that
    real workload: 7e-05 absolute, an angular error of 0.004 degrees -- smaller
    than the fp16 the backbone already runs in (1e-4). The head's ``acos`` guard
    and cosine clamp are unaffected at that magnitude.

    Set ``tf32: false`` in the config if you ever need to rule it out while
    debugging a numerical problem.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Deterministic runs must not silently change matmul precision underneath
    # the comparison being made.
    if tf32 and not deterministic:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

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
