"""Extend a trained checkpoint to cover new identities.

This is the "don't start from zero" path. Given an existing checkpoint and a new
dataset, it:

1. Loads the checkpoint's ClassMap -- the record of which class index means which
   person.
2. Scans the new dataset and appends any identities not already known, leaving
   every existing index untouched.
3. Grows the classifier weight matrix, preserving all learned prototypes and
   initialising only the new rows.
4. Grows the optimizer state to match (the step that otherwise crashes on the
   first update after extension).
5. Writes a new checkpoint you can resume training from.

Usage
-----
    # See what would change, without writing anything:
    python -m scripts.extend_classmap --checkpoint runs/exp/checkpoints/best.pt \\
        --config configs/new_data.yaml --dry-run

    # Extend, initialising new rows from real embeddings (recommended):
    python -m scripts.extend_classmap --checkpoint runs/exp/checkpoints/best.pt \\
        --config configs/new_data.yaml --output runs/exp2/checkpoints/start.pt \\
        --init mean_embedding

Then train with ``train.resume`` pointing at the new checkpoint.

Recommended fine-tuning recipe after extending
----------------------------------------------
1. One epoch with ``model.backbone.freeze_backbone: true`` at ``optim.lr: 0.01``
   so the new head rows settle without disturbing the backbone.
2. Unfreeze, drop to 0.1x the original LR, cosine to zero over 4-8 epochs.

Watch accuracy on the *old* identities as well as the new ones -- some
forgetting (typically 1-2%) is normal, and a large drop means the LR is too high.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from frs.config import load_config  # noqa: E402
from frs.data.class_map import ClassMap  # noqa: E402
from frs.data.dataset import build_dataset  # noqa: E402
from frs.data.transforms import build_transforms  # noqa: E402
from frs.engine.checkpoint import extend_head_and_optimizer  # noqa: E402
from frs.models.backbones.iresnet import build_backbone  # noqa: E402
from frs.utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger("extend")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True, help="existing trained checkpoint")
    p.add_argument("--config", required=True, help="config describing the NEW dataset")
    p.add_argument("--output", help="where to write the extended checkpoint")
    p.add_argument(
        "--init",
        default="normal",
        choices=["normal", "mean_embedding"],
        help="how to initialise new class prototypes (mean_embedding is better)",
    )
    p.add_argument(
        "--source",
        default=None,
        help="provenance tag recorded for the new identities (default: config name)",
    )
    p.add_argument(
        "--proto-samples",
        type=int,
        default=16,
        help="images per new identity used to compute mean_embedding prototypes",
    )
    p.add_argument(
        "--proto-max-images",
        type=int,
        default=200_000,
        help="streaming datasets: stop scanning for prototypes after this many images",
    )
    p.add_argument("--dry-run", action="store_true", help="report changes, write nothing")
    return p.parse_args()


@torch.no_grad()
def compute_mean_prototypes(
    backbone: torch.nn.Module,
    dataset,
    class_map: ClassMap,
    new_indices: list[int],
    samples_per_identity: int,
    device: torch.device,
    max_images: int = 200_000,
) -> torch.Tensor:
    """Average normalised embeddings per new identity.

    Starting a new class at the centroid of its own images puts it near its true
    angular position from step one, instead of at a random point on the sphere.
    That removes the loss spike random init causes and roughly halves the time to
    converge.
    """
    backbone.eval()
    dim = int(getattr(backbone, "embedding_size", 512))
    prototypes = torch.zeros(len(new_indices), dim)

    def embed_mean(images: torch.Tensor) -> torch.Tensor:
        embeddings, _ = backbone(images.to(device))
        return torch.nn.functional.normalize(embeddings.float().mean(0), dim=0).cpu()

    def to_tensor(img) -> torch.Tensor:
        arr = dataset.transform(img) if dataset.transform is not None else (
            img.astype("float32").transpose(2, 0, 1) / 255.0
        )
        return torch.from_numpy(arr.copy())

    if hasattr(dataset, "collect_by_class"):
        # Streaming dataset: one bounded pass collects images per new class.
        collected = dataset.collect_by_class(
            set(new_indices), per_class=samples_per_identity, max_images=max_images
        )
        for row, index in enumerate(new_indices):
            imgs = collected.get(index, [])
            if not imgs:
                logger.warning(
                    "identity %s: no samples seen within %d images; random init for it",
                    class_map.identity_of(index), max_images,
                )
                torch.nn.init.normal_(prototypes[row], std=0.01)
                continue
            prototypes[row] = embed_mean(torch.stack([to_tensor(i) for i in imgs]))
        return prototypes

    by_class: dict[int, list[int]] = {index: [] for index in new_indices}
    for position, target in enumerate(dataset.targets):
        bucket = by_class.get(int(target))
        if bucket is not None and len(bucket) < samples_per_identity:
            bucket.append(position)

    for row, index in enumerate(new_indices):
        positions = by_class.get(index, [])
        if not positions:
            logger.warning(
                "identity %s has no samples; falling back to random init for it",
                class_map.identity_of(index),
            )
            torch.nn.init.normal_(prototypes[row], std=0.01)
            continue
        prototypes[row] = embed_mean(torch.stack([dataset[p][0] for p in positions]))

    return prototypes


def main() -> int:
    args = parse_args()
    setup_logging()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not ckpt.get("class_map"):
        print(
            "This checkpoint has no class map, so its head rows cannot be "
            "matched to identities. Extension is not possible; train fresh.",
            file=sys.stderr,
        )
        return 1

    old_map = ClassMap.from_dict(ckpt["class_map"])
    logger.info("Checkpoint: %d classes, epoch %s", old_map.num_classes, ckpt.get("epoch"))

    cfg = load_config(args.config)
    train_tf, _ = build_transforms(cfg.data)
    # strict_labels=False: unknown identities are exactly what we are here to add.
    dataset = build_dataset(cfg.data, transform=train_tf, strict_labels=False)
    new_identities = list(dataset.identities)
    logger.info("New dataset: %d identities, %d images", len(new_identities), len(dataset))

    extended = ClassMap.from_dict(old_map.to_dict())
    result = extended.extend(
        new_identities, source=args.source or cfg.get_path("experiment.name", "new")
    )

    print(f"\n{result}")
    if result.added:
        preview = ", ".join(result.added[:5])
        print(f"  new identities: {preview}{' ...' if len(result.added) > 5 else ''}")
    print(f"  reused (already known): {len(result.reused)}")

    if not result.changed:
        print("\nNothing to extend -- every identity is already in the class map.")
        return 0

    if args.dry_run:
        print("\n--dry-run: no files written.")
        return 0

    if not args.output:
        print("--output is required (unless --dry-run)", file=sys.stderr)
        return 1

    head_state = dict(ckpt["head"]["state_dict"])
    optim_state = ckpt.get("optimizer", {}).get("state_dict")

    prototypes = None
    if args.init == "mean_embedding":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = build_backbone(
            ckpt["backbone"]["arch"], tuple(ckpt["backbone"].get("input_size", (112, 112)))
        ).to(device)
        backbone.load_state_dict(ckpt["backbone"]["state_dict"])

        # Re-index the dataset against the extended map so targets are correct.
        dataset.reindex(extended)
        new_indices = [extended.index_of(i) for i in result.added]
        logger.info("Computing mean-embedding prototypes for %d new identities", len(new_indices))
        prototypes = compute_mean_prototypes(
            backbone, dataset, extended, new_indices, args.proto_samples, device,
            max_images=args.proto_max_images,
        )

    head_state, optim_state = extend_head_and_optimizer(
        head_state=head_state,
        optimizer_state=optim_state,
        old_num_classes=old_map.num_classes,
        new_num_classes=extended.num_classes,
        init=args.init,
        new_prototypes=prototypes,
    )

    ckpt["head"]["state_dict"] = head_state
    ckpt["head"]["num_classes"] = extended.num_classes
    if optim_state is not None and "optimizer" in ckpt:
        ckpt["optimizer"]["state_dict"] = optim_state
    ckpt["class_map"] = extended.to_dict()
    # Fine-tuning restarts the epoch counter; the history is kept for provenance.
    ckpt["epoch"] = 0
    ckpt["global_step"] = 0
    ckpt.pop("rng", None)
    ckpt.pop("scheduler", None)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(output)

    print(
        f"\nWrote {output}\n"
        f"  classes: {old_map.num_classes} -> {extended.num_classes}\n"
        f"  new rows initialised with: {args.init}\n\n"
        f"Next:\n"
        f"  1. Set train.resume: {output}\n"
        f"  2. Freeze the backbone for 1 epoch at optim.lr=0.01\n"
        f"  3. Unfreeze at 0.1x the original LR, cosine to zero over 4-8 epochs\n"
        f"  4. Track accuracy on BOTH old and new identities"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
