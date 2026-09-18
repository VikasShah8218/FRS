"""Trainer resume semantics and the classifier row-norm monitor.

The failure this guards against: a checkpoint written partway through an epoch
used to record the *next* epoch, so resuming silently skipped the rest of the
interrupted epoch -- on Glint360K up to 88,000 steps (six hours of training).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from frs.config import Config  # noqa: E402
from frs.data.adapters.base import Sample  # noqa: E402
from frs.data.class_map import ClassMap  # noqa: E402
from frs.engine.checkpoint import load_checkpoint  # noqa: E402
from frs.engine.optim import build_optimizer, build_scheduler  # noqa: E402
from frs.engine.trainer import Trainer  # noqa: E402
from frs.models.heads import ArcFaceHead  # noqa: E402

DIM, CLASSES, SAMPLES, BATCH = 16, 5, 40, 4  # -> 10 steps per epoch
STEPS_PER_EPOCH = SAMPLES // BATCH


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(DIM, DIM)
        self.arch, self.input_size, self.embedding_size = "tiny", (112, 112), DIM

    def forward(self, x):
        out = self.fc(x)
        norm = torch.norm(out, 2, 1, True)
        return out / norm, norm


class Interrupted(Exception):
    pass


def make_cfg(tmp_path, epochs=2, save_every_n_steps=3):
    return Config._wrap({
        "experiment": {"name": "resume_test", "output_dir": str(tmp_path / "run"), "seed": 0},
        "model": {"backbone": {"channels_last": False}},
        "optim": {"type": "sgd", "lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4, "clip_grad_norm": 5.0},
        "scheduler": {"type": "polylr", "warmup_epochs": 0},
        "train": {
            "epochs": epochs, "amp": False, "grad_accum_steps": 1, "log_every_n_steps": 1,
            "save_every_n_epochs": 1, "save_every_n_steps": save_every_n_steps,
            "eval_every_n_epochs": 1, "keep_last_n_checkpoints": 3,
        },
        "monitor": {"tensorboard": False, "log_gpu_memory": False},
        "eval": {"primary": None},
    })


def make_trainer(cfg, **resume):
    torch.manual_seed(0)
    backbone, head = TinyBackbone(), ArcFaceHead(DIM, CLASSES)
    data = TensorDataset(torch.randn(SAMPLES, DIM), torch.randint(0, CLASSES, (SAMPLES,)))
    loader = DataLoader(data, batch_size=BATCH, shuffle=True, drop_last=True)
    optimizer = build_optimizer(cfg.optim, backbone, head)
    scheduler = build_scheduler(cfg.scheduler, optimizer, steps_per_epoch=len(loader), epochs=cfg.train.epochs)
    class_map = ClassMap.from_samples([Sample(path=f"{i}", identity=f"id{i}") for i in range(CLASSES)])
    trainer = Trainer(
        cfg=cfg, backbone=backbone, head=head, optimizer=optimizer, scheduler=scheduler,
        train_loader=loader, class_map=class_map, device=torch.device("cpu"), **resume,
    )
    return trainer, backbone, head, optimizer, scheduler, class_map


def count_steps(trainer):
    calls = {"n": 0}
    original = trainer._train_step

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    trainer._train_step = counted
    return calls


def test_mid_epoch_checkpoint_resumes_inside_the_epoch(tmp_path):
    cfg = make_cfg(tmp_path)
    trainer, *_ = make_trainer(cfg)

    # Interrupt on the 8th batch of epoch 0; the last save was after batch 6.
    original = trainer._train_step

    def interrupt_at_8(*args, **kwargs):
        if trainer.global_step == 7:
            raise Interrupted()
        return original(*args, **kwargs)

    trainer._train_step = interrupt_at_8
    with pytest.raises(Interrupted):
        trainer.train()

    ckpt_path = tmp_path / "run" / "checkpoints" / "last.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 0, "a mid-epoch save must record the CURRENT epoch"
    assert ckpt["step_in_epoch"] == 6
    assert ckpt["global_step"] == 6

    # Resume exactly as scripts/train.py does.
    resumed, backbone, head, optimizer, scheduler, class_map = make_trainer(cfg)
    state = load_checkpoint(
        ckpt_path, backbone=backbone, head=head, optimizer=optimizer,
        scheduler=scheduler, class_map=class_map,
    )
    assert (state.epoch, state.step_in_epoch, state.global_step) == (0, 6, 6)

    resumed, *_ = make_trainer(
        cfg, start_epoch=state.epoch, global_step=state.global_step,
        start_step_in_epoch=state.step_in_epoch,
    )
    calls = count_steps(resumed)
    resumed.train()

    # 4 remaining batches of epoch 0, then all 10 of epoch 1.
    assert calls["n"] == (STEPS_PER_EPOCH - 6) + STEPS_PER_EPOCH
    assert resumed.global_step == 2 * STEPS_PER_EPOCH
    assert len(resumed.history) == 2


def test_checkpoint_on_last_batch_is_an_epoch_boundary(tmp_path):
    cfg = make_cfg(tmp_path)
    trainer, *_ = make_trainer(
        cfg, start_epoch=0, global_step=STEPS_PER_EPOCH, start_step_in_epoch=STEPS_PER_EPOCH,
    )
    assert trainer.epoch == 1 and trainer.resume_step_in_epoch == 0
    calls = count_steps(trainer)
    trainer.train()
    assert calls["n"] == STEPS_PER_EPOCH


def test_epoch_end_checkpoint_points_at_next_epoch(tmp_path):
    cfg = make_cfg(tmp_path, epochs=1, save_every_n_steps=0)
    trainer, *_ = make_trainer(cfg)
    trainer.train()
    ckpt = torch.load(tmp_path / "run" / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 1 and ckpt["step_in_epoch"] == 0


def test_row_norm_monitor_logs_and_warns_on_collapse(tmp_path, caplog):
    cfg = make_cfg(tmp_path, epochs=1, save_every_n_steps=0)
    trainer, _, head, *_ = make_trainer(cfg)

    norms = trainer._classifier_row_norms()
    assert set(norms) == {"median", "p05", "mean"} and norms["median"] > 0

    with caplog.at_level(logging.WARNING):
        trainer._check_row_norms(norms["median"])          # sets the reference
        trainer._check_row_norms(norms["median"] * 0.5)    # shrinking, not yet alarming
        assert "collapsed" not in caplog.text
        trainer._check_row_norms(norms["median"] * 0.05)   # 20x collapse
    assert "Classifier rows have collapsed" in caplog.text

    with torch.no_grad():
        head.weight.mul_(0.01)
    assert trainer._classifier_row_norms()["median"] == pytest.approx(norms["median"] * 0.01, rel=1e-4)
