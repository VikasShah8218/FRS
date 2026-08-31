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
from ..eval.verification import evaluate_target, format_results
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
            self.output_dir / "tensorboard", enabled=bool(monitor.get("tensorboard", True))
        )
        self.log_grad_norm = bool(monitor.get("log_grad_norm", True))
        self.log_feature_norm = bool(monitor.get("log_feature_norm", True))
        self.log_gpu_memory = bool(monitor.get("log_gpu_memory", True))
        self.hist_every = int(monitor.get("histogram_every_n_steps", 500))

        self.primary_metric = cfg.get("eval", {}).get("primary")
        self.steps_per_epoch = len(train_loader)

    # ---------------------------------------------------------------- one step

    def _forward(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backbone under autocast, margin head in fp32.

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
            logits = self.head(embeddings.float(), norms.float(), labels)
            loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

        return loss, logits, norms

    def _train_step(
        self, images: torch.Tensor, labels: torch.Tensor, accum_index: int
    ) -> dict[str, float]:
        loss, logits, norms = self._forward(images, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss ({loss.item()}) at step {self.global_step}. "
                f"Usual causes: LR too high for the warmup, or the margin head's "
                f"cosine clamp removed. Try scheduler.warmup_epochs up, optim.lr "
                f"down, or model.head.type=cosface to isolate the angular branch."
            )

        # Scale so the effective gradient equals the mean over the full
        # accumulated batch rather than the sum of micro-batch means.
        self.scaler.scale(loss / self.grad_accum).backward()

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

    def train_epoch(self) -> dict[str, float]:
        self.backbone.train()
        self.head.train()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        norm_meter = AverageMeter()
        grad_meter = AverageMeter()
        throughput = ThroughputMeter()

        sampler = getattr(self.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.epoch)

        self.optimizer.zero_grad(set_to_none=True)
        epoch_start = time.perf_counter()
        throughput.start_data()

        for step, (images, labels) in enumerate(self.train_loader):
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
            ):
                # Mid-epoch checkpoint: what makes spot instances safe.
                self._save("last.pt")

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
        logger.info(
            "ep %2d [%4d/%4d] loss %.4f  acc %5.2f%%  lr %.5f  "
            "fnorm %.1f  %.0f img/s",
            self.epoch,
            step + 1,
            self.steps_per_epoch,
            loss_meter.smooth,
            acc_meter.smooth,
            lr,
            norm_meter.smooth,
            throughput.images_per_sec,
        )

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

    # ---------------------------------------------------------------- the loop

    def train(self) -> MetricHistory:
        logger.info(
            "Training %d epochs, %d steps/epoch, %d total steps",
            self.epochs,
            self.steps_per_epoch,
            self.steps_per_epoch * self.epochs,
        )

        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch
            metrics = self.train_epoch()

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
                eval_results = self.eval_fn(self.backbone)
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
            self._maybe_save_best(metrics)

            if (epoch + 1) % self.save_every == 0:
                self._save("last.pt")
                prune_checkpoints(self.ckpt_dir, self.keep_last_n)

        self._save("last.pt")
        save_backbone_only(self.ckpt_dir / "backbone_only.pt", self.backbone, self.class_map)
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

    def _save(self, filename: str) -> None:
        save_checkpoint(
            self.ckpt_dir / filename,
            backbone=self.backbone,
            head=self.head,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            class_map=self.class_map,
            epoch=self.epoch + 1,  # resume starts at the *next* epoch
            global_step=self.global_step,
            config=self.cfg.to_dict(),
            metrics=self.best_metrics,
            history=self.history.to_list(),
        )
