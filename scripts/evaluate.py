"""Evaluate a trained checkpoint and optionally regenerate its report.

    python -m scripts.evaluate --checkpoint runs/<exp>/checkpoints/best.pt
    python -m scripts.evaluate --checkpoint ... --config configs/... --report
    python -m scripts.evaluate --checkpoint ... --inspect
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from frs.config import load_config  # noqa: E402
from frs.data.transforms import build_transforms  # noqa: E402
from frs.engine.checkpoint import inspect_checkpoint  # noqa: E402
from frs.eval.pairs import load_eval_images  # noqa: E402
from frs.eval.verification import evaluate_target, format_results  # noqa: E402
from frs.models.backbones.iresnet import build_backbone  # noqa: E402
from frs.utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", help="config providing eval.targets (default: the one in the checkpoint)")
    p.add_argument("--inspect", action="store_true", help="print checkpoint contents and exit")
    p.add_argument("--report", action="store_true", help="regenerate the HTML/MD report")
    p.add_argument("--output", help="where to write results JSON")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()

    if args.inspect:
        print(json.dumps(inspect_checkpoint(args.checkpoint), indent=2, default=str))
        return 0

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    cfg = load_config(args.config) if args.config else None
    if cfg is None:
        stored = ckpt.get("config")
        if not stored:
            print(
                "This checkpoint has no embedded config; pass --config explicitly.",
                file=sys.stderr,
            )
            return 1
        from frs.config import Config

        cfg = Config._wrap(stored)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    arch = ckpt["backbone"]["arch"]
    input_size = tuple(ckpt["backbone"].get("input_size", (112, 112)))
    backbone = build_backbone(arch, input_size).to(device)
    backbone.load_state_dict(ckpt["backbone"]["state_dict"])
    backbone.eval()
    logger.info(
        "Loaded %s from epoch %s (step %s)",
        arch, ckpt.get("epoch"), ckpt.get("global_step"),
    )

    _, eval_tf = build_transforms(cfg.data)
    targets = cfg.get("eval", {}).get("targets", []) or []
    if not targets:
        print("No eval.targets configured.", file=sys.stderr)
        return 1

    results = {}
    for target in targets:
        try:
            results[target["name"]] = evaluate_target(
                backbone,
                dict(target),
                load_images=partial(load_eval_images, transform=eval_tf),
                device=device,
                batch_size=int(cfg.eval.get("batch_size", 128)),
                flip_test=bool(cfg.eval.get("flip_test", True)),
            )
        except Exception as exc:
            logger.exception("Target %s failed: %s", target["name"], exc)

    print("\n" + format_results(results) + "\n")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Wrote %s", args.output)

    if args.report:
        from frs.engine.meters import MetricHistory
        from frs.report.render import write_report

        # The trainer stores the dataset statistics in the checkpoint; fall back
        # to re-scanning (cached, so nearly free) for older checkpoints. A failure
        # here must not lose the evaluation results we just computed.
        dataset_stats: dict = dict((ckpt.get("extra") or {}).get("dataset_stats") or {})
        if not dataset_stats:
            try:
                from frs.data.dataset import build_dataset

                dataset = build_dataset(cfg.data, transform=eval_tf, strict_labels=False)
                dataset_stats = dataset.report_stats()
            except Exception as exc:
                logger.warning("Could not summarise the dataset for the report: %s", exc)

        paths = write_report(
            output_dir=cfg.get_path("report.output", "report"),
            cfg=cfg,
            history=MetricHistory.from_list(ckpt.get("history", [])),
            dataset_stats=dataset_stats,
            best_metrics=ckpt.get("metrics", {}),
            eval_results=results,
        )
        for path in paths:
            logger.info("Report written: %s", path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
