"""Training entry point.

    python -m scripts.train --config configs/meglass_ir50_adaface.yaml
    python -m scripts.train --config configs/... --set train.epochs=2 data.batch_size=32
    python -m scripts.train --config configs/... --overfit 100

IMPORTANT (Windows): every top-level statement lives under ``if __name__ ==
"__main__"``. DataLoader workers on Windows use ``spawn``, which re-imports this
module in each worker -- without the guard that recursion spawns processes
forever.
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from frs.config import load_config  # noqa: E402
from frs.data.class_map import ClassMap  # noqa: E402
from frs.data.dataset import build_dataset  # noqa: E402
from frs.data.sampler import build_sampler  # noqa: E402
from frs.data.transforms import build_transforms  # noqa: E402
from frs.engine.checkpoint import find_latest_checkpoint, load_checkpoint  # noqa: E402
from frs.engine.optim import build_optimizer, build_scheduler  # noqa: E402
from frs.engine.trainer import Trainer  # noqa: E402
from frs.eval.pairs import load_pair_images  # noqa: E402
from frs.eval.verification import evaluate_target  # noqa: E402
from frs.models.backbones.iresnet import (  # noqa: E402
    build_backbone,
    count_parameters,
    load_pretrained_backbone,
)
from frs.models.heads import build_head  # noqa: E402
from frs.utils.logging import log_config_banner, setup_logging  # noqa: E402
from frs.utils.seed import seed_all, worker_init_fn  # noqa: E402

logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a face recognition model")
    p.add_argument("--config", required=True, help="path to a YAML config")
    p.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="dot-path config overrides, e.g. train.epochs=2 optim.lr=0.01",
    )
    p.add_argument(
        "--overfit",
        type=int,
        default=0,
        metavar="N",
        help="sanity check: train on only N samples, expecting loss -> 0",
    )
    p.add_argument("--resume", default=None, help="checkpoint path (overrides config)")
    p.add_argument("--no-eval", action="store_true", help="skip verification eval")
    return p.parse_args()


def build_eval_fn(cfg, device, eval_transform):
    """Return a callable that evaluates every configured benchmark."""
    targets = cfg.get("eval", {}).get("targets", []) or []
    if not targets:
        return None

    def run_eval(model):
        results = {}
        for target in targets:
            try:
                results[target["name"]] = evaluate_target(
                    model,
                    dict(target),
                    load_images=partial(load_pair_images, transform=eval_transform),
                    device=device,
                    batch_size=int(cfg.eval.get("batch_size", 128)),
                    flip_test=bool(cfg.eval.get("flip_test", True)),
                    amp=bool(cfg.train.get("amp", False)),
                )
            except FileNotFoundError as exc:
                logger.warning("Skipping eval target %s: %s", target["name"], exc)
            except Exception as exc:
                logger.exception("Eval target %s failed: %s", target["name"], exc)
        return results

    return run_eval


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.set)

    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    seed_all(int(cfg.experiment.seed), bool(cfg.experiment.get("deterministic", False)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning(
            "CUDA is not available -- training on CPU will be impractically slow."
        )
    else:
        props = torch.cuda.get_device_properties(0)
        logger.info(
            "GPU: %s, %.1f GiB, compute %d.%d",
            props.name,
            props.total_memory / 1024**3,
            props.major,
            props.minor,
        )
        # On a small card, allocator fragmentation is the difference between a
        # batch size working and OOM-ing several hundred steps in.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # ---------------------------------------------------------------- data
    train_tf, eval_tf = build_transforms(cfg.data)
    dataset = build_dataset(cfg.data, transform=train_tf)
    stats = dataset.summary()
    logger.info(
        "Dataset: %(num_samples)s images, %(num_identities)s identities "
        "(%(min_per_identity)s-%(max_per_identity)s per identity, "
        "mean %(mean_per_identity).1f)",
        stats,
    )

    if args.overfit:
        n = min(args.overfit, len(dataset))
        logger.warning("OVERFIT MODE: training on %d samples only", n)
        indices = list(range(n))
        train_view: object = Subset(dataset, indices)
        targets = dataset.targets[indices]
    else:
        train_view = dataset
        targets = dataset.targets

    class_map = dataset.class_map
    sampler = build_sampler(
        cfg.data.get("sampler"),
        targets,
        int(cfg.data.batch_size),
        seed=int(cfg.experiment.seed),
    )

    num_workers = int(cfg.data.get("num_workers", 4))
    loader = DataLoader(
        train_view,
        batch_size=int(cfg.data.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(cfg.data.get("pin_memory", True)) and device.type == "cuda",
        drop_last=bool(cfg.data.get("drop_last", True)) and not args.overfit,
        persistent_workers=bool(cfg.data.get("persistent_workers", True)) and num_workers > 0,
        prefetch_factor=int(cfg.data.get("prefetch_factor", 4)) if num_workers > 0 else None,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )
    if len(loader) == 0:
        raise RuntimeError(
            f"DataLoader is empty: {len(train_view)} samples with batch_size="
            f"{cfg.data.batch_size} and drop_last=True. Lower the batch size."
        )

    # --------------------------------------------------------------- model
    backbone = build_backbone(
        cfg.model.backbone.arch, tuple(cfg.data.input_size)
    ).to(device)

    if cfg.model.backbone.get("pretrained"):
        load_pretrained_backbone(backbone, cfg.model.backbone.pretrained, strict=False)
    if cfg.model.backbone.get("channels_last", False) and device.type == "cuda":
        backbone = backbone.to(memory_format=torch.channels_last)
    if cfg.model.backbone.get("freeze_backbone", False):
        for param in backbone.parameters():
            param.requires_grad = False
        logger.warning("Backbone is FROZEN -- only the head will train")

    head = build_head(
        cfg.model.head,
        embedding_size=int(cfg.model.backbone.get("embedding_size", 512)),
        num_classes=class_map.num_classes,
    ).to(device)

    trainable, total = count_parameters(backbone)
    logger.info(
        "Model: %s (%.1fM params) + %s (%d classes)",
        cfg.model.backbone.arch,
        total / 1e6,
        type(head).__name__,
        class_map.num_classes,
    )

    # ---------------------------------------------------------------- optim
    optimizer = build_optimizer(cfg.optim, backbone, head)
    scheduler = build_scheduler(
        cfg.scheduler, optimizer, steps_per_epoch=len(loader), epochs=int(cfg.train.epochs)
    )

    # --------------------------------------------------------------- resume
    start_epoch, global_step = 0, 0
    history, best_metrics = [], {}
    resume_path = args.resume or cfg.train.get("resume")
    if resume_path == "auto":
        resume_path = find_latest_checkpoint(output_dir)
    if resume_path:
        state = load_checkpoint(
            resume_path,
            backbone=backbone,
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            class_map=class_map,
            map_location=str(device),
        )
        start_epoch, global_step = state.epoch, state.global_step
        history, best_metrics = state.history, state.best_metrics
        if state.class_map is not None:
            class_map = state.class_map

    if cfg.train.get("torch_compile", False):
        if platform.system() == "Windows":
            logger.warning(
                "torch.compile is disabled on Windows (needs an MSVC toolchain "
                "and gains little here). Enable it on Linux/AWS."
            )
        else:
            backbone = torch.compile(backbone, mode="max-autotune")
            logger.info("torch.compile enabled")

    log_config_banner(
        logger,
        cfg,
        extra={
            "device": str(device),
            "classes": class_map.num_classes,
            "steps/ep": len(loader),
        },
    )

    # ---------------------------------------------------------------- train
    eval_fn = None if (args.no_eval or args.overfit) else build_eval_fn(cfg, device, eval_tf)

    trainer = Trainer(
        cfg=cfg,
        backbone=backbone,
        head=head,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        class_map=class_map,
        device=device,
        eval_fn=eval_fn,
        start_epoch=start_epoch,
        global_step=global_step,
        history=history,
        best_metrics=best_metrics,
    )
    history_obj = trainer.train()

    # --------------------------------------------------------------- report
    if cfg.get("report", {}).get("enabled", True) and not args.overfit:
        try:
            from frs.report.render import write_report

            paths = write_report(
                output_dir=cfg.report.output,
                cfg=cfg,
                history=history_obj,
                dataset_stats=stats,
                best_metrics=trainer.best_metrics,
                formats=cfg.report.get("formats", ["html", "md"]),
            )
            for path in paths:
                logger.info("Report written: %s", path)
        except Exception as exc:
            logger.exception("Report generation failed: %s", exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
