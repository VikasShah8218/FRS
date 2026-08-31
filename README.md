# ESSI-FRS

A from-scratch face recognition training framework in PyTorch.

- **Backbones** — IResNet-18/34/50/100/152/200, with SE variants
- **Losses** — ArcFace, CosFace and AdaFace margin heads, switchable by config
- **Alignment** — offline RetinaFace/SCRFD detection + similarity warp to the
  ArcFace canonical template
- **Data** — pluggable adapters; a new dataset format is one file and one import
- **Checkpoints** — fully resumable, and **extensible to new identities without
  retraining from scratch**
- **Monitoring** — TensorBoard plus an automated HTML/Markdown accuracy report

Runs on a 4 GB laptop GPU and scales unchanged to multi-GPU AWS.

---

## Quick start

```bash
workon ml

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 1. Hold out 300 identities for honest validation (run once)
python -m scripts.build_meglass_pairs

# 2. Prove the plumbing works (should drive loss to ~0)
python -m scripts.train --config configs/meglass_ir50_adaface.yaml --overfit 100

# 3. Train (~9-10 h on an RTX 3050 Laptop; add --set train.epochs=6 for a ~2.5 h run)
python -m scripts.train --config configs/meglass_ir50_adaface.yaml

# 4. Watch it
tensorboard --logdir runs/
```

The report lands at `runs/meglass_ir50_adaface/report/report.html`.

**→ Read [docs/GUIDE.md](docs/GUIDE.md) for the full walkthrough.**

---

## What to expect from the bundled dataset

MeGlass has **47,917 images across 1,710 identities**. Production face
recognition trains on 100-250x more (MS1MV3: 5.8 M / 93 K). This pipeline will
converge and report high MeGlass validation accuracy (~97-99%), but that
reflects a small single-domain dataset — not production generalisation.

That is the intended use at this stage: the deliverable is a **correct, reusable
pipeline**. Real accuracy arrives when you feed it a large dataset on AWS — and
because checkpoints carry the identity↔class mapping, that run can build on this
one instead of starting over.

---

## Documentation

| Document | Contents |
|---|---|
| **[docs/GUIDE.md](docs/GUIDE.md)** | Full walkthrough: theory, setup, training, monitoring, interpreting results, troubleshooting |
| [docs/ADAPTERS.md](docs/ADAPTERS.md) | Adding a new dataset format |
| [docs/AWS.md](docs/AWS.md) | Instance selection, data staging, multi-GPU, cost estimates |

---

## Layout

```
configs/          YAML configs (base + experiments)
frs/
  registry.py     name -> class registry (the plug-in mechanism)
  config.py       YAML inheritance, ${} interpolation, startup validation
  models/         IResNet backbones + ArcFace/CosFace/AdaFace heads
  data/           adapters, class map, dataset, transforms, samplers
  align/          SCRFD detector + Umeyama similarity warp
  engine/         trainer, checkpointing, optimisers, meters
  eval/           LFW-protocol verification
  report/         HTML/Markdown report generation
  utils/          logging, seeding, tensorboard
scripts/          CLI entry points
tests/            pytest suite
```

---

## Common commands

```bash
# Train with config overrides
python -m scripts.train --config configs/meglass_ir50_adaface.yaml \
    --set train.epochs=30 optim.lr=0.02

# Compare loss functions on identical data
python -m scripts.train --config configs/meglass_ir50_arcface.yaml

# Evaluate / inspect a checkpoint
python -m scripts.evaluate --checkpoint runs/<exp>/checkpoints/best.pt
python -m scripts.evaluate --checkpoint runs/<exp>/checkpoints/best.pt --inspect

# Check the resize policy visually before a long run
python -m scripts.align_dataset --probe --src MeGlass_120x120

# Align a new, unaligned dataset
python -m scripts.download_models
python -m scripts.align_dataset --src raw_photos --dst aligned_112

# Add new identities to an already-trained model
python -m scripts.extend_classmap --checkpoint runs/<exp>/checkpoints/best.pt \
    --config configs/new_data.yaml --dry-run

pytest tests/ -q
```

---

## Design notes

A few decisions worth knowing, each explained in the source:

**Identities are strings, not integers.** Adapters emit string identity keys; the
`ClassMap` assigns integer class indices and guarantees they never change. This
is what makes a trained classifier head extensible to new people later.

**The margin head runs in fp32 even under AMP.** `acos` near ±1 loses
catastrophic precision at half precision. The backbone — where the memory
actually is — still gets the fp16 speedup.

**Weight decay skips BatchNorm, bias and PReLU** (filtered by `ndim <= 1`, not by
name — PReLU's parameter is called `weight`). Worth ~0.5-1% accuracy.

**Validation identities are excluded from training**, not merely from the pair
list. Anything less measures memorisation.

**Alignment is offline.** Running a detector inside the DataLoader would redo
identical work every epoch while stealing GPU from training.

---

## Requirements

- NVIDIA GPU with CUDA (developed on an RTX 3050 Laptop, 4 GB)
- Python 3.10-3.13
- PyTorch from the CUDA index (not plain PyPI, which is CPU-only)

## Credits

The IResNet backbone in `frs/models/backbones/iresnet.py` is vendored verbatim
from [AdaFace](https://github.com/mk-minchul/AdaFace) (Kim, Jain & Liu, CVPR
2022). Margin-loss formulations follow ArcFace (Deng et al., CVPR 2019), CosFace
(Wang et al., CVPR 2018) and AdaFace.
