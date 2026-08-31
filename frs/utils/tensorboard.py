"""A thin SummaryWriter wrapper that degrades to a no-op.

Training must never fail because TensorBoard is missing or a log write errored.
Every method here swallows exceptions by design.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TensorBoardLogger:
    """Wraps ``torch.utils.tensorboard.SummaryWriter`` if available."""

    def __init__(self, log_dir: str | Path | None, enabled: bool = True) -> None:
        self.enabled = False
        self.writer: Any = None
        if not enabled or log_dir is None:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter

            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
            self.enabled = True
            logger.info("TensorBoard logging to %s", log_dir)
            logger.info("  view with: tensorboard --logdir %s", Path(log_dir).parent)
        except Exception as exc:
            logger.warning(
                "TensorBoard unavailable (%s) -- continuing without it. "
                "Install with: pip install tensorboard",
                exc,
            )

    def scalar(self, tag: str, value: float, step: int) -> None:
        if not self.enabled:
            return
        try:
            self.writer.add_scalar(tag, float(value), step)
        except Exception:
            pass

    def scalars(self, values: dict[str, float], step: int, prefix: str = "") -> None:
        for key, value in values.items():
            if value is None:
                continue
            self.scalar(f"{prefix}{key}" if prefix else key, value, step)

    def histogram(self, tag: str, values: Any, step: int) -> None:
        if not self.enabled:
            return
        try:
            self.writer.add_histogram(tag, values, step)
        except Exception:
            pass

    def image(self, tag: str, image: Any, step: int, dataformats: str = "HWC") -> None:
        if not self.enabled:
            return
        try:
            self.writer.add_image(tag, image, step, dataformats=dataformats)
        except Exception:
            pass

    def text(self, tag: str, value: str, step: int = 0) -> None:
        if not self.enabled:
            return
        try:
            self.writer.add_text(tag, value, step)
        except Exception:
            pass

    def flush(self) -> None:
        if self.enabled:
            try:
                self.writer.flush()
            except Exception:
                pass

    def close(self) -> None:
        if self.enabled:
            try:
                self.writer.close()
            except Exception:
                pass
            self.enabled = False

    def __enter__(self) -> "TensorBoardLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
