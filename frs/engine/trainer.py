"""The training loop.

Reading order for the interesting parts:

* :meth:`Trainer._train_step` -- the forward/backward, including the one AMP
  subtlety that matters (the head runs in fp32 even under autocast).
* :meth:`Trainer.train` -- epoch loop, evaluation, checkpointing.
* :meth:`Trainer._log_step` -- what gets monitored and why.

Gradient accumulation caveat
----------------------------
Accumulation gives a larger *effective* batch for the optimiser, but BatchNorm
still computes its statistics on the micro-batch. At micro-batch 64 that is
fine; below ~32 the BN noise starts to cost accuracy and the fix is a bigger
micro-batch (or SyncBatchNorm under DDP), not more accumulation.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.class_map import ClassMap
from ..eval.verification import format_results
from ..models.partial_fc import unwrap_partial_fc
from ..utils.distributed import barrier, unwrap
from ..utils.tensorboard import TensorBoardLogger
from .checkpoint import (
    prune_checkpoints,
    save_backbone_only,
    save_checkpoint,
)
from .meters import (
    AverageMeter,
    MetricHistory,
    ThroughputMeter,
    gpu_memory_stats,
    topk_accuracy,
)

logger = logging.getLogger(__name__)


class Trainer:
    """Owns the training loop and everything it touches."""

    def __init__(
        self,
        cfg: Any,
        backbone: torch.nn.Module,
        head: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        train_loader: DataLoader,
        class_map: ClassMap,
        device: torch.device,
        eval_fn: Any = None,
        start_epoch: int = 0,
        global_step: int = 0,
        history: list[dict] | None = None,
        best_metrics: dict[str, float] | None = None,
        dataset_stats: dict[str, Any] | None = None,
        is_main: bool = True,
        start_step_in_epoch: int = 0,
    ) -> None:
        self.cfg = cfg
        self.backbone = backbone
        self.head = head
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.class_map = class_map
        self.device = device
        self.eval_fn = eval_fn
        # Stored in every checkpoint so `evaluate --report` can describe the
        # dataset without re-scanning (or re-streaming) it.
        self.dataset_stats = dict(dataset_stats or {})
        # Under DDP only rank 0 logs to TensorBoard, evaluates and writes
        # checkpoints; the other ranks just train and wait at the barriers.
        self.is_main = bool(is_main)

        self.epoch = start_epoch
        self.global_step = global_step
        self.history = MetricHistory.from_list(history or [])
        self.best_metrics = dict(best_metrics or {})

        train_cfg = cfg.train
        self.epochs = int(train_cfg.epochs)
        self.grad_accum = int(train_cfg.get("grad_accum_steps", 1))
        self.log_every = int(train_cfg.get("log_every_n_steps", 20))
        self.save_every = int(train_cfg.get("save_every_n_epochs", 1))
        self.save_every_steps = int(train_cfg.get("save_every_n_steps", 0))
        self.eval_every = int(train_cfg.get("eval_every_n_epochs", 1))
        self.keep_last_n = int(train_cfg.get("keep_last_n_checkpoints", 3))
        self.clip_grad = float(cfg.optim.get("clip_grad_norm", 0) or 0)
        self.label_smoothing = float(cfg.get("loss", {}).get("label_smoothing", 0.0))

        self.amp_enabled = bool(train_cfg.get("amp", False)) and device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16
            if str(train_cfg.get("amp_dtype", "float16")).lower() in ("bf16", "bfloat16")
            else torch.float16
        )
        # bf16 has the range of fp32, so it needs no loss scaling.
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_enabled and self.amp_dtype is torch.float16
        )

        self.output_dir = Path(cfg.experiment.output_dir)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        monitor = cfg.get("monitor", {})
        self.tb = TensorBoardLogger(
            self.output_dir / "tensorboard",
            enabled=bool(monitor.get("tensorboard", True)) and self.is_main,
        )
        self.log_grad_norm = bool(monitor.get("log_grad_norm", True))
        self.log_feature_norm = bool(monitor.get("log_feature_norm", True))
        self.log_gpu_memory = bool(monitor.get("log_gpu_memory", True))
        self.hist_every = int(monitor.get("histogram_every_n_steps", 500))

        self.primary_metric = cfg.get("eval", {}).get("primary")
        self.steps_per_epoch = len(train_loader)

        # Resuming from a mid-epoch checkpoint continues *inside* that epoch:
        # only the batches it had not reached are run. A checkpoint taken on the
        # epoch's last batch is simply an epoch boundary.
        self.resume_step_in_epoch = max(0, int(start_step_in_epoch))
        if self.resume_step_in_epoch >= self.steps_per_epoch > 0:
            self.epoch += 1
            self.resume_step_in_epoch = 0

        # Classifier row-norm monitor. Margin heads normalise their rows, so a
        # row's effective step size grows as 1/||w||^2: rows collapsing toward
        # zero is an early, cheap-to-see sign of a run about to degrade.
        self._row_norm_reference: float | None = None
        self._row_norm_warned = False

    # ---------------------------------------------------------------- one step

    def _forward(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backbone under autocast, margin head in fp32.

        Returns ``(loss, logits, norms, labels_for_logits)``. The last item is
        the label tensor that indexes ``logits`` -- identical to ``labels``
        unless Partial-FC remapped them onto its sampled class subset.

        The head must not run in fp16: it computes ``acos`` near the domain edge
        and a softmax over a (B x num_classes) matrix, both of which lose
        catastrophic precision at half precision. The backbone -- where all the
        memory actually is -- still gets the fp16 speedup.
        """
        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled
        ):
            embeddings, norms = self.backbone(images)

        with torch.autocast(device_type=self.device.type, enabled=False):
            out = self.head(embeddings.float(), norms.float(), labels)
            # Partial-FC returns (logits over the sampled subset, remapped
            # labels); a bare head returns logits over every class.
            if isinstance(out, tuple):
                logits, loss_labels = out
            else:
                logits, loss_labels = out, labels
            loss = F.cross_entropy(
                logits, loss_labels, label_smoothing=self.label_smoothing
            )

        return loss, logits, norms, loss_labels

    def _decay_penalty(self) -> torch.Tensor | None:
        """Partial-FC's weight decay on the rows it sampled this step, if any."""
        pop = getattr(getattr(self.head, "module", self.head), "pop_decay_penalty", None)
        return pop() if callable(pop) else None

    def _train_step(
        self, images: torch.Tensor, labels: torch.Tensor, accum_index: int
    ) -> dict[str, float]:
        loss, logits, norms, labels = self._forward(images, labels)
        penalty = self._decay_penalty()

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss ({loss.item()}) at step {self.global_step}. "
                f"Usual causes: LR too high for the warmup, or the margin head's "
                f"cosine clamp removed. Try scheduler.warmup_epochs up, optim.lr "
                f"down, or model.head.type=cosface to isolate the angular branch."
            )

        # The logged loss stays pure cross-entropy so runs remain comparable;
        # the weight-decay term only shapes the gradient.
        objective = loss if penalty is None else loss + penalty

        # Scale so the effective gradient equals the mean over the full
        # accumulated batch rather than the sum of micro-batch means.
        self.scaler.scale(objective / self.grad_accum).backward()

        stats = {"loss": loss.item()}
        is_update_step = (accum_index + 1) % self.grad_accum == 0

        if is_update_step:
            grad_norm = None
            if self.clip_grad > 0 or self.log_grad_norm:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(self.backbone.parameters()) + list(self.head.parameters()),
                    max_norm=self.clip_grad if self.clip_grad > 0 else math.inf,
                )
                stats["grad_norm"] = float(grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

        stats["acc1"] = topk_accuracy(logits.detach(), labels, ks=(1,))[0]
        if self.log_feature_norm:
            stats["feature_norm"] = float(norms.detach().mean())
        return stats

    # --------------------------------------------------------------- one epoch

    def train_epoch(self, skip_steps: int = 0) -> dict[str, float]:
        """Run one epoch, or its remaining ``steps_per_epoch - skip_steps`` batches.

        ``skip_steps`` is non-zero only for the first epoch after resuming from a
        mid-epoch checkpoint. The skipped batches are *not* replayed: the data
        order is re-drawn with a different salt, so the remaining steps see
        fresh samples rather than repeating the ones already trained on.
        """
        self.backbone.train()
        self.head.train()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        norm_meter = AverageMeter()
        grad_meter = AverageMeter()
        throughput = ThroughputMeter()

        skip_steps = max(0, min(int(skip_steps), self.steps_per_epoch))
        max_steps = self.steps_per_epoch - skip_steps

        # Both a DistributedSampler and a streaming dataset need to know the
        # epoch to derive a fresh, reproducible order.
        sampler = getattr(self.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            # Offset the epoch seed on a resumed partial epoch (see docstring).
            sampler.set_epoch(self.epoch * 100_003 + skip_steps if skip_steps else self.epoch)
        dataset = getattr(self.train_loader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(self.epoch, salt=skip_steps)

        self.optimizer.zero_grad(set_to_none=True)
        epoch_start = time.perf_counter()
        throughput.start_data()

        for loader_step, (images, labels) in enumerate(self.train_loader):
            if loader_step >= max_steps:
                break
            # Position within the epoch, counting the batches done before resume.
            step = skip_steps + loader_step
            throughput.end_data()

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if self.cfg.model.backbone.get("channels_last", False):
                images = images.to(memory_format=torch.channels_last)

            stats = self._train_step(images, labels, step)

            loss_meter.update(stats["loss"], images.size(0))
            acc_meter.update(stats["acc1"], images.size(0))
            if "feature_norm" in stats:
                norm_meter.update(stats["feature_norm"], images.size(0))
            if "grad_norm" in stats:
                grad_meter.update(stats["grad_norm"])

            throughput.step(images.size(0))
            self.global_step += 1

            if self.global_step % self.log_every == 0:
                self._log_step(loss_meter, acc_meter, norm_meter, grad_meter, throughput, step)

            if (
                self.save_every_steps > 0
                and self.global_step % self.save_every_steps == 0
                and self.is_main
            ):
                # Mid-epoch checkpoint: what makes spot instances safe. It records
                # the position inside the epoch so a resume continues from here
                # instead of skipping to the next epoch.
                self._save("last.pt", epoch=self.epoch, step_in_epoch=step + 1)

            throughput.start_data()

        return {
            "train_loss": loss_meter.avg,
            "train_acc1": acc_meter.avg,
            "feature_norm": norm_meter.avg,
            "grad_norm": grad_meter.avg,
            "lr": self.optimizer.param_groups[0]["lr"],
            "epoch_time_sec": time.perf_counter() - epoch_start,
            "images_per_sec": throughput.images_per_sec,
            "data_time_frac": throughput.data_time_frac,
        }

    def _log_step(
        self,
        loss_meter: AverageMeter,
        acc_meter: AverageMeter,
        norm_meter: AverageMeter,
        grad_meter: AverageMeter,
        throughput: ThroughputMeter,
        step: int,
    ) -> None:
        lr = self.optimizer.param_groups[0]["lr"]
        row_norms = self._classifier_row_norms()
        logger.info(
            "ep %2d [%4d/%4d] loss %.4f  acc %5.2f%%  lr %.5f  "
            "fnorm %.1f  wnorm %.4f  %.0f img/s",
            self.epoch,
            step + 1,
            self.steps_per_epoch,
            loss_meter.smooth,
            acc_meter.smooth,
            lr,
            norm_meter.smooth,
            row_norms.get("median", float("nan")),
            throughput.images_per_sec,
        )
        if row_norms:
            self.tb.scalars(
                {f"train/head_row_norm_{k}": v for k, v in row_norms.items()},
                self.global_step,
            )
            self._check_row_norms(row_norms["median"])

        self.tb.scalars(
            {
                "train/loss": loss_meter.smooth,
                "train/acc_top1": acc_meter.smooth,
                "train/lr": lr,
                "perf/imgs_per_sec": throughput.images_per_sec,
                "perf/data_time_frac": throughput.data_time_frac,
                "perf/sec_per_step": throughput.sec_per_step,
            },
            self.global_step,
        )
        if self.log_feature_norm:
            self.tb.scalar("train/feature_norm", norm_meter.smooth, self.global_step)
        if self.log_grad_norm and grad_meter.count:
            self.tb.scalar("train/grad_norm", grad_meter.smooth, self.global_step)
        if self.log_gpu_memory:
            self.tb.scalars(gpu_memory_stats(self.device), self.global_step, prefix="perf/")

    @torch.no_grad()
    def _classifier_row_norms(self) -> dict[str, float]:
        """Median / 5th percentile / mean L2 norm of the classifier rows."""
        head = unwrap_partial_fc(unwrap(self.head))
        weight = getattr(head, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            return {}
        norms = weight.detach().float().norm(dim=1)
        return {
            "median": float(norms.median()),
            "p05": float(torch.quantile(norms, 0.05)) if norms.numel() <= 16_000_000 else float("nan"),
            "mean": float(norms.mean()),
        }

    def _check_row_norms(self, median: float) -> None:
        if self._row_norm_reference is None:
            self._row_norm_reference = median
            return
        if not self._row_norm_warned and median < 0.1 * self._row_norm_reference:
            self._row_norm_warned = True
            logger.warning(
                "Classifier rows have collapsed: median norm %.4f is under 10%% of its "
                "starting %.4f. Normalised rows this small take very large angular "
                "steps and training usually degrades from here. Check weight decay "
                "on the head and the learning rate.",
                median, self._row_norm_reference,
            )

    # ---------------------------------------------------------------- the loop

    def train(self) -> MetricHistory:
        logger.info(
            "Training %d epochs, %d steps/epoch, %d total steps",
            self.epochs,
            self.steps_per_epoch,
            self.steps_per_epoch * self.epochs,
        )
        if self.epoch >= self.epochs:
            logger.warning(
                "Checkpoint is already at epoch %d >= train.epochs (%d): nothing to "
                "train. Raise train.epochs to continue this run.",
                self.epoch, self.epochs,
            )
            self.tb.close()
            return self.history
        if self.resume_step_in_epoch:
            logger.info(
                "Resuming inside epoch %d at batch %d of %d",
                self.epoch, self.resume_step_in_epoch, self.steps_per_epoch,
            )

        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch
            skip, self.resume_step_in_epoch = self.resume_step_in_epoch, 0
            metrics = self.train_epoch(skip_steps=skip)

            logger.info(
                "epoch %d done: loss %.4f  acc %.2f%%  %.1f min  %.0f img/s"
                "  data-stall %.1f%%",
                epoch,
                metrics["train_loss"],
                metrics["train_acc1"],
                metrics["epoch_time_sec"] / 60,
                metrics["images_per_sec"],
                metrics["data_time_frac"] * 100,
            )
            if metrics["data_time_frac"] > 0.15:
                logger.warning(
                    "GPU starved %.0f%% of the time waiting for data -- raise "
                    "data.num_workers or use faster storage",
                    metrics["data_time_frac"] * 100,
                )

            eval_results: dict[str, Any] = {}
            if self.eval_fn and (epoch + 1) % self.eval_every == 0:
                if self.is_main:
                    eval_results = self.eval_fn(unwrap(self.backbone))
                barrier()  # other ranks wait for rank 0's evaluation
                if eval_results:
                    logger.info("\n%s", format_results(eval_results))
                    for name, res in eval_results.items():
                        metrics[name] = res["accuracy"]
                        self.tb.scalars(
                            {
                                f"eval/{name}/accuracy": res["accuracy"],
                                f"eval/{name}/auc": res.get("auc", 0),
                                f"eval/{name}/threshold": res.get("threshold", 0),
                            },
                            self.global_step,
                        )

            self.history.append(epoch, **metrics)
            if self.is_main:
                self._maybe_save_best(metrics)
                if (epoch + 1) % self.save_every == 0:
                    self._save("last.pt")
                    prune_checkpoints(self.ckpt_dir, self.keep_last_n)
            barrier()

        if self.is_main:
            self._save("last.pt")
            save_backbone_only(
                self.ckpt_dir / "backbone_only.pt",
                self.backbone,
                model_name=str(self.cfg.get_path("experiment.name", "unnamed")),
            )
        self.tb.close()
        logger.info("Training complete. Best: %s", self.best_metrics or "(no eval)")
        return self.history

    def _maybe_save_best(self, metrics: dict[str, Any]) -> None:
        if not self.primary_metric or self.primary_metric not in metrics:
            return
        value = float(metrics[self.primary_metric])
        best = self.best_metrics.get(self.primary_metric)
        if best is None or value > best:
            self.best_metrics[self.primary_metric] = value
            self.best_metrics["epoch"] = self.epoch
            self._save("best.pt")
            logger.info(
                "New best %s: %.4f (epoch %d)", self.primary_metric, value, self.epoch
            )

    def _save(
        self, filename: str, epoch: int | None = None, step_in_epoch: int = 0
    ) -> None:
        """Write a checkpoint.

        By default it marks an epoch boundary: resume starts at the *next*
        epoch. Mid-epoch saves pass the current ``epoch`` plus ``step_in_epoch``,
        the number of batches of that epoch already trained.
        """
        save_checkpoint(
            self.ckpt_dir / filename,
            backbone=self.backbone,
            head=self.head,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            class_map=self.class_map,
            epoch=self.epoch + 1 if epoch is None else epoch,
            step_in_epoch=step_in_epoch,
            global_step=self.global_step,
            config=self.cfg.to_dict(),
            metrics=self.best_metrics,
            history=self.history.to_list(),
            extra={"dataset_stats": self.dataset_stats} if self.dataset_stats else None,
        )
