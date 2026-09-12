"""Training entry point.

    python -m scripts.train --config configs/meglass_ir50_adaface.yaml
    python -m scripts.train --config configs/glint360k_ir50_adaface_local.yaml
    python -m scripts.train --config configs/... --set train.epochs=2 data.batch_size=32
    python -m scripts.train --config configs/... --overfit 100

    # multi-GPU (Linux / AWS)
    torchrun --nproc_per_node=4 -m scripts.train --config configs/glint360k_ir100_adaface_aws.yaml

Two kinds of dataset flow through the same loop:

* **map-style** (MeGlass folders, ``.rec`` packs): scanned once, indexed,
  shuffled by a sampler;
* **streaming** (Glint360K WebDataset shards): never enumerated, each worker
  streams its share of the shards every epoch. Samplers do not apply, and
  ``persistent_workers`` is forced off so the epoch counter reaches the workers.

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

import torch  # noqa: E402
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset, Subset  # noqa: E402

from frs.config import load_config  # noqa: E402
from frs.data.dataset import build_dataset  # noqa: E402
from frs.data.sampler import build_sampler  # noqa: E402
from frs.data.transforms import build_transforms  # noqa: E402
from frs.data.class_map import ClassMap  # noqa: E402
from frs.engine.checkpoint import (  # noqa: E402
    find_latest_checkpoint,
    load_checkpoint,
    peek_class_map,
)
from frs.engine.optim import build_optimizer, build_scheduler  # noqa: E402
from frs.engine.trainer import Trainer  # noqa: E402
from frs.eval.pairs import load_eval_images  # noqa: E402
from frs.eval.verification import evaluate_target  # noqa: E402
from frs.models.backbones.iresnet import (  # noqa: E402
    build_backbone,
    count_parameters,
    load_pretrained_backbone,
)
from frs.models.heads import build_head  # noqa: E402
from frs.models.partial_fc import maybe_wrap_partial_fc  # noqa: E402
from frs.utils.distributed import (  # noqa: E402
    barrier,
    cleanup_distributed,
    setup_distributed,
    wrap_ddp,
)
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
                    load_images=partial(load_eval_images, transform=eval_transform),
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


def build_loader(cfg, args, dataset, device, rank: int, world_size: int, seed: int):
    """DataLoader for either dataset kind (and for ``--overfit``)."""
    is_iterable = isinstance(dataset, IterableDataset)
    batch_size = int(cfg.data.batch_size)
    num_workers = int(cfg.data.get("num_workers", 4))
    sampler_cfg = cfg.data.get("sampler")
    sampler_type = (sampler_cfg or {}).get("type", "random")

    # ------------------------------------------------------------ overfit
    if args.overfit:
        n = args.overfit
        if is_iterable:
            train_view = dataset.take(n)  # small in-memory map-style dataset
        else:
            train_view = Subset(dataset, list(range(min(n, len(dataset)))))
        logger.warning("OVERFIT MODE: training on %d samples only", len(train_view))
        if int(cfg.train.get("grad_accum_steps", 1)) > 1:
            # With a handful of batches per epoch, accumulation could leave the
            # optimizer stepping once per epoch (or never). The point of this
            # mode is to watch the loss fall, so step on every batch.
            cfg.train.grad_accum_steps = 1
            logger.warning("OVERFIT MODE: grad_accum_steps forced to 1")
        sampler = (
            DistributedSampler(train_view, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
            if world_size > 1
            else None
        )
        return DataLoader(
            train_view,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=0,
            drop_last=False,
            pin_memory=device.type == "cuda",
        )

    # ---------------------------------------------------------- streaming
    if is_iterable:
        if sampler_type != "random":
            raise ValueError(
                f"data.sampler.type={sampler_type!r} is not supported for streaming "
                f"datasets; the shard stream is the shuffle. Use random."
            )
        if bool(cfg.data.get("persistent_workers", True)) and num_workers > 0:
            logger.warning(
                "persistent_workers is forced OFF for streaming datasets so that "
                "each epoch's shard order reaches the workers (costs a few seconds "
                "of worker start-up per epoch)."
            )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=None,
            num_workers=num_workers,
            pin_memory=bool(cfg.data.get("pin_memory", True)) and device.type == "cuda",
            drop_last=True,
            persistent_workers=False,
            prefetch_factor=int(cfg.data.get("prefetch_factor", 4)) if num_workers > 0 else None,
            worker_init_fn=worker_init_fn if num_workers > 0 else None,
        )

    # ---------------------------------------------------------- map-style
    if world_size > 1:
        if sampler_type != "random":
            raise ValueError(
                f"data.sampler.type={sampler_type!r} is not supported under DDP; use random"
            )
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed,
            drop_last=bool(cfg.data.get("drop_last", True)),
        )
    else:
        sampler = build_sampler(sampler_cfg, dataset.targets, batch_size, seed=seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(cfg.data.get("pin_memory", True)) and device.type == "cuda",
        drop_last=bool(cfg.data.get("drop_last", True)),
        persistent_workers=bool(cfg.data.get("persistent_workers", True)) and num_workers > 0,
        prefetch_factor=int(cfg.data.get("prefetch_factor", 4)) if num_workers > 0 else None,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )


def run(args: argparse.Namespace, rank: int, world_size: int, local_rank: int) -> int:
    cfg = load_config(args.config, overrides=args.set)
    is_main = rank == 0
    seed = int(cfg.experiment.seed)

    if args.overfit:
        # Keep the sanity run's checkpoints, logs and TensorBoard out of the
        # real experiment folder -- otherwise the next real run would
        # `resume: auto` from an overfit checkpoint.
        cfg.experiment.output_dir = str(Path(cfg.experiment.output_dir) / "overfit")
        cfg.train.resume = None
        cfg.report.output = str(Path(cfg.experiment.output_dir) / "report")

    output_dir = Path(cfg.experiment.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier()
    setup_logging(output_dir, rank=rank)

    seed_all(seed, bool(cfg.experiment.get("deterministic", False)))

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        props = torch.cuda.get_device_properties(device)
        logger.info(
            "GPU: %s, %.1f GiB, compute %d.%d%s",
            props.name,
            props.total_memory / 1024**3,
            props.major,
            props.minor,
            f"  (rank {rank}/{world_size})" if world_size > 1 else "",
        )
        # On a small card, allocator fragmentation is the difference between a
        # batch size working and OOM-ing several hundred steps in.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    else:
        device = torch.device("cpu")
        logger.warning(
            "CUDA is not available -- training on CPU will be impractically slow."
        )

    # ---------------------------------------------------------------- data
    train_tf, eval_tf = build_transforms(cfg.data)
    dataset = build_dataset(
        cfg.data, transform=train_tf, rank=rank, world_size=world_size, seed=seed
    )
    is_iterable = isinstance(dataset, IterableDataset)
    stats = dataset.summary()
    logger.info(
        "Dataset: %(num_samples)s images, %(num_identities)s identities "
        "(%(min_per_identity)s-%(max_per_identity)s per identity, "
        "mean %(mean_per_identity).1f)%(extra)s",
        {
            **stats,
            "extra": (
                f"  streamed from {stats.get('num_shards')} shards, "
                f"{stats.get('epoch_mode')} epochs of {stats.get('samples_per_epoch'):,}"
                if is_iterable
                else ""
            ),
        },
    )

    # ------------------------------------------------------ class map
    # When resuming, the checkpoint's index assignment is the authority: head
    # row i was trained for identity ckpt_map[i]. Start from that map and
    # append whatever identities this dataset adds (more shards, a new folder,
    # a checkpoint produced by scripts/extend_classmap.py), so training resumes
    # against the right rows and grows the head only for genuinely new people.
    class_map = dataset.class_map
    resume_path = args.resume or cfg.train.get("resume")
    if resume_path == "auto":
        resume_path = find_latest_checkpoint(output_dir)
    if resume_path:
        ckpt_map = peek_class_map(resume_path)
        if ckpt_map is not None:
            merged = ClassMap.from_dict(ckpt_map.to_dict())
            result = merged.extend(dataset.identities, source=str(cfg.experiment.name))
            if result.added:
                logger.warning(
                    "%s -- the head will be extended with %d new rows (random init; "
                    "use scripts/extend_classmap.py --init mean_embedding for a "
                    "better start)", result, result.num_added,
                )
            dataset.reindex(merged)
            class_map = merged

    loader = build_loader(cfg, args, dataset, device, rank, world_size, seed)
    if len(loader) == 0:
        hint = (
            "raise data.adapter.samples_per_epoch or lower the batch size"
            if is_iterable
            else "lower the batch size"
        )
        raise RuntimeError(
            f"DataLoader is empty: {len(loader.dataset)} samples per rank with "
            f"batch_size={cfg.data.batch_size} and drop_last=True -- {hint}."
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
    # Sampled softmax for very large class counts (auto above 300k classes).
    head = maybe_wrap_partial_fc(
        head, cfg.model.head.get("partial_fc"), class_map.num_classes
    )

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

    # ------------------------------------------------------- DDP / compile
    backbone = wrap_ddp(backbone, local_rank)
    head = wrap_ddp(head, local_rank)

    if cfg.train.get("torch_compile", False):
        if platform.system() == "Windows":
            logger.warning(
                "torch.compile is disabled on Windows (needs an MSVC toolchain "
                "and gains little here). Enable it on Linux/AWS."
            )
        else:
            backbone = torch.compile(backbone, mode="max-autotune")
            logger.info("torch.compile enabled")

    total_batch = int(cfg.data.batch_size) * world_size * int(cfg.train.get("grad_accum_steps", 1))
    log_config_banner(
        logger,
        cfg,
        extra={
            "device": f"{device} x {world_size}" if world_size > 1 else str(device),
            "total batch": f"{total_batch} (lr {cfg.optim.lr} applies to this)",
            "dataset": "streaming" if is_iterable else "map-style",
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
        dataset_stats=stats,
        is_main=is_main,
    )
    history_obj = trainer.train()

    # --------------------------------------------------------------- report
    if is_main and cfg.get("report", {}).get("enabled", True) and not args.overfit:
        try:
            from frs.report.render import write_report

            paths = write_report(
                output_dir=cfg.report.output,
                cfg=cfg,
                history=history_obj,
                dataset_stats=dataset.report_stats() if hasattr(dataset, "report_stats") else stats,
                best_metrics=trainer.best_metrics,
                formats=cfg.report.get("formats", ["html", "md"]),
            )
            for path in paths:
                logger.info("Report written: %s", path)
        except Exception as exc:
            logger.exception("Report generation failed: %s", exc)

    return 0


def main() -> int:
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    try:
        return run(args, rank, world_size, local_rank)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
