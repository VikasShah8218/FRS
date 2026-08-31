"""Export a trained backbone for deployment (TorchScript / ONNX).

    python -m scripts.export --checkpoint runs/<exp>/checkpoints/best.pt --format onnx
    python -m scripts.export --checkpoint ... --format torchscript

Only the backbone is exported. The margin head exists solely to shape the
embedding space during training; at inference you compare embeddings by cosine
similarity and never evaluate the classifier. Dropping it also avoids shipping a
matrix that encodes your training identity list.

The exported model takes a normalised NCHW float32 batch and returns
L2-normalised 512-d embeddings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# torch.onnx's progress output contains emoji, which crashes on a Windows
# console using the cp1252 code page. Force UTF-8 before torch is imported.
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from frs.models.backbones.iresnet import build_backbone  # noqa: E402


class EmbeddingModel(torch.nn.Module):
    """Wraps the backbone to return a single tensor.

    The training backbone returns ``(embedding, norm)``; deployment code almost
    always wants just the embedding, and ONNX consumers in particular are
    simpler with one output.
    """

    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding, _ = self.backbone(x)
        return embedding


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", help="output path (default: alongside the checkpoint)")
    p.add_argument("--format", default="onnx", choices=["onnx", "torchscript", "both"])
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    p.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="fix the batch dimension (default: 0 = dynamic)",
    )
    p.add_argument("--verify", action="store_true", help="compare exported vs. eager output")
    return p.parse_args()


def load_backbone(checkpoint_path: str) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Accept a full training checkpoint or a backbone_only.pt artifact.
    if "backbone" in ckpt:
        spec, state = ckpt["backbone"], ckpt["backbone"]["state_dict"]
    else:
        spec, state = ckpt, ckpt["state_dict"]

    arch = spec.get("arch", "ir_50")
    input_size = tuple(spec.get("input_size", (112, 112)))

    backbone = build_backbone(arch, input_size)
    backbone.load_state_dict(state)
    # eval() is essential: the final BatchNorm1d must use running statistics,
    # not batch statistics, or single-image inference produces garbage.
    backbone.eval()

    meta = {
        "arch": arch,
        "input_size": list(input_size),
        "embedding_size": spec.get("embedding_size", 512),
        "epoch": ckpt.get("epoch"),
        "num_training_identities": (
            ckpt.get("head", {}).get("num_classes")
            or ckpt.get("num_training_identities")
        ),
        "metrics": ckpt.get("metrics", {}),
        "preprocessing": {
            "color": "RGB",
            "layout": "NCHW",
            "dtype": "float32",
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "note": "resize/crop to input_size, scale to [0,1], then (x-mean)/std",
        },
    }
    return backbone, meta


def main() -> int:
    args = parse_args()

    backbone, meta = load_backbone(args.checkpoint)
    model = EmbeddingModel(backbone).eval()

    size = meta["input_size"][0]
    batch = args.batch_size or 1
    example = torch.randn(batch, 3, size, size)

    base = Path(args.output) if args.output else Path(args.checkpoint).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        reference = model(example)
    print(f"Model: {meta['arch']}, input {batch}x3x{size}x{size} -> {tuple(reference.shape)}")

    written = []

    if args.format in ("torchscript", "both"):
        path = base.with_suffix(".torchscript.pt")
        traced = torch.jit.trace(model, example)
        traced = torch.jit.freeze(traced)
        traced.save(str(path))
        written.append(path)
        print(f"Wrote {path}")

        if args.verify:
            with torch.no_grad():
                torch.testing.assert_close(
                    torch.jit.load(str(path))(example), reference, rtol=1e-4, atol=1e-5
                )
            print("  verified against eager output")

    if args.format in ("onnx", "both"):
        path = base.with_suffix(".onnx")
        dynamic_axes = (
            None if args.batch_size else {"input": {0: "batch"}, "embedding": {0: "batch"}}
        )
        torch.onnx.export(
            model,
            example,
            str(path),
            input_names=["input"],
            output_names=["embedding"],
            opset_version=args.opset,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
        written.append(path)
        print(f"Wrote {path}")

        if args.verify:
            try:
                import numpy as np
                import onnxruntime as ort

                session = ort.InferenceSession(
                    str(path), providers=["CPUExecutionProvider"]
                )
                out = session.run(None, {"input": example.numpy()})[0]
                np.testing.assert_allclose(
                    out, reference.numpy(), rtol=1e-3, atol=1e-4
                )
                print("  verified against eager output")
            except ImportError:
                print("  (onnxruntime not installed; skipping verification)")

    meta_path = base.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {meta_path}")

    print(
        "\nDeployment notes:\n"
        f"  - Input: RGB, NCHW float32, {size}x{size}, scaled to [0,1] then (x-0.5)/0.5\n"
        "  - Output: L2-normalised 512-d embedding; compare with cosine similarity\n"
        "  - The match threshold is in the checkpoint's eval metrics"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
