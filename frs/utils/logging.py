"""Console + file logging, rank-aware for distributed runs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


class _RankFilter(logging.Filter):
    """Silence non-zero ranks so distributed logs stay readable."""

    def __init__(self, rank: int) -> None:
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        return self.rank == 0 or record.levelno >= logging.WARNING


def setup_logging(
    output_dir: str | Path | None = None,
    level: int = logging.INFO,
    rank: int = 0,
    filename: str = "train.log",
) -> logging.Logger:
    """Configure the root logger. Safe to call more than once."""
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console.addFilter(_RankFilter(rank))
    root.addHandler(console)

    if output_dir is not None and rank == 0:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path / filename, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    # These libraries are chatty at INFO and drown out training output.
    for noisy in ("PIL", "matplotlib", "matplotlib.font_manager"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root


def log_config_banner(logger: logging.Logger, cfg, extra: dict | None = None) -> None:
    """Print the run's key settings once at startup.

    Worth the vertical space: most "why did this run behave differently"
    questions are answered by these ten lines in the log file.
    """
    lines = [
        "=" * 74,
        f"  experiment : {cfg.get_path('experiment.name')}",
        f"  output     : {cfg.get_path('experiment.output_dir')}",
        f"  backbone   : {cfg.get_path('model.backbone.arch')}"
        f"  |  head: {cfg.get_path('model.head.type')}",
        f"  batch      : {cfg.get_path('data.batch_size')}"
        f" x {cfg.get_path('train.grad_accum_steps', 1)} accum"
        f" = {cfg.get_path('data.batch_size') * cfg.get_path('train.grad_accum_steps', 1)} effective",
        f"  optim      : {cfg.get_path('optim.type')} lr={cfg.get_path('optim.lr')}"
        f" wd={cfg.get_path('optim.weight_decay', 0)}",
        f"  schedule   : {cfg.get_path('scheduler.type')}"
        f"  warmup={cfg.get_path('scheduler.warmup_epochs', 0)}ep"
        f"  epochs={cfg.get_path('train.epochs')}",
        f"  amp        : {cfg.get_path('train.amp', False)}"
        f" ({cfg.get_path('train.amp_dtype', 'float16')})",
    ]
    for key, value in (extra or {}).items():
        lines.append(f"  {key:<10} : {value}")
    lines.append("=" * 74)
    for line in lines:
        logger.info(line)
