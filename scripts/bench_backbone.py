"""Measure raw training throughput of a backbone on this GPU.

    python -m scripts.bench_backbone                       # ir_50, batch 64
    python -m scripts.bench_backbone --arch ir_100 --batch 128
    python -m scripts.bench_backbone --classes 360232      # + Partial-FC 0.1

No data is involved: random tensors through backbone + margin head, forward and
backward under AMP, exactly as the trainer does it. Use it to pick settings
before a long run rather than trusting folklore:

* ``channels_last`` (NHWC) is often quoted as +10-15% on Ampere. On an RTX 3050
  with torch 2.13 / cu126 it measured **6.7x slower** (27 vs 181 img/s), which
  is why every config now ships with it off. Re-measure on each new GPU/driver.
* ``--compile`` tries ``torch.compile`` (Linux only in practice).

Compare the printed img/s with what the trainer logs; if the trainer is much
slower and ``perf/data_time_frac`` is near zero, the loss is in the model
settings, not the data pipeline.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from frs.models.backbones.iresnet import build_backbone  # noqa: E402
from frs.models.heads import build_head  # noqa: E402
from frs.models.partial_fc import maybe_wrap_partial_fc  # noqa: E402


def bench(
    arch: str,
    batch: int,
    classes: int,
    channels_last: bool,
    amp_dtype: torch.dtype,
    head_type: str,
    partial_fc: float,
    compile_model: bool,
    steps: int,
    device: torch.device,
) -> tuple[float, float, float]:
    backbone = build_backbone(arch, (112, 112)).to(device)
    if channels_last:
        backbone = backbone.to(memory_format=torch.channels_last)
    if compile_model:
        backbone = torch.compile(backbone, mode="max-autotune")
    head = build_head({"type": head_type}, embedding_size=512, num_classes=classes).to(device)
    head = maybe_wrap_partial_fc(
        head, {"enabled": partial_fc > 0, "sample_rate": partial_fc or 0.1}, classes
    )
    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype is torch.float16)

    x = torch.randn(batch, 3, 112, 112, device=device)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)
    y = torch.randint(0, classes, (batch,), device=device)

    def step() -> None:
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            emb, norms = backbone(x)
        with torch.autocast(device.type, enabled=False):
            out = head(emb.float(), norms.float(), y)
            logits, labels = out if isinstance(out, tuple) else (out, y)
            loss = torch.nn.functional.cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    for _ in range(5):  # warm-up: cudnn autotune, allocator, compile
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    return steps * batch / dt, dt / steps * 1000, mem


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default="ir_50")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--classes", type=int, default=2000)
    p.add_argument("--head", default="adaface", choices=["arcface", "cosface", "adaface"])
    p.add_argument("--partial-fc", type=float, default=0.0, help="sample rate; 0 = off")
    p.add_argument("--amp-dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--compile", action="store_true")
    p.add_argument(
        "--channels-last", default="both", choices=["both", "on", "off"],
        help="which memory formats to measure",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if args.compile and platform.system() == "Windows":
        print("torch.compile is not supported on Windows here; ignoring --compile")
        args.compile = False

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    variants = {"both": (True, False), "on": (True,), "off": (False,)}[args.channels_last]
    print(
        f"{args.arch}, batch {args.batch}, {args.classes} classes, {args.head}"
        f"{f', partial_fc {args.partial_fc}' if args.partial_fc else ''}, {args.amp_dtype}\n"
    )
    for channels_last in variants:
        ips, ms, mem = bench(
            args.arch, args.batch, args.classes, channels_last, amp_dtype, args.head,
            args.partial_fc, args.compile, args.steps, device,
        )
        print(
            f"  channels_last={str(channels_last):<5}  {ips:7.0f} img/s   "
            f"{ms:6.0f} ms/step   peak mem {mem:.2f} GiB"
        )
        torch.cuda.empty_cache() if device.type == "cuda" else None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
