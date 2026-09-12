# ESSI-FRS — Complete Training Guide

A from-scratch face recognition system: IResNet backbones, ArcFace/AdaFace
margin losses, RetinaFace alignment, PyTorch. Built to run locally on a 4 GB GPU
and scale unchanged to AWS.

---

## Table of contents

1. [What this is, and what to expect](#1-what-this-is-and-what-to-expect)
2. [How face recognition training actually works](#2-how-face-recognition-training-actually-works)
3. [Setup](#3-setup)
4. [The dataset](#4-the-dataset)
5. [Step-by-step: your first training run](#5-step-by-step-your-first-training-run)
6. [Monitoring: what to watch and what it means](#6-monitoring-what-to-watch-and-what-it-means)
7. [Understanding your results](#7-understanding-your-results)
8. [Checkpoints and incremental training](#8-checkpoints-and-incremental-training)
9. [Using a different dataset format](#9-using-a-different-dataset-format)
10. [Scaling to AWS](#10-scaling-to-aws)
11. [Choosing hyperparameters](#11-choosing-hyperparameters)
12. [Troubleshooting](#12-troubleshooting)
13. [Project layout](#13-project-layout)

---

## 1. What this is, and what to expect

This codebase trains a face recognition model from random initialisation. It
produces a **backbone** that maps a face image to a 512-dimensional embedding
such that two photos of the same person land close together and photos of
different people land far apart. At deployment you compare embeddings by cosine
similarity; the classifier head used during training is discarded.

### Set your expectations correctly

The bundled dataset (MeGlass) has **47,917 images across 1,710 identities**.
Production face recognition models train on:

| Dataset | Images | Identities |
|---|---:|---:|
| **MeGlass (yours)** | **47,917** | **1,710** |
| MS1MV3 | 5.8 M | 93 K |
| WebFace4M | 4.2 M | 360 K |
| WebFace12M | 12 M | 600 K |

That is a 100-250x gap. The pipeline will train correctly and converge, and you
will see high MeGlass validation accuracy (~97-99%) — but that number reflects a
small, single-domain dataset, not production generalisation. A model trained
here will *not* match a WebFace-trained model on real-world faces.

**This is the intended use.** The deliverable at this stage is a correct,
reusable pipeline. Real accuracy comes when you feed it a large dataset on AWS —
and because the class map and checkpoints are designed for extension, that later
run can build on this one rather than starting over.

---

## 2. How face recognition training actually works

Worth understanding before you turn the knobs.

### The problem with ordinary classification

The naive approach is to train a classifier over your 1,710 identities. It
works, but it optimises the wrong thing: it only needs to make each identity
*separable*, not *compact*. Embeddings end up spread out within each identity,
and at test time — where you compare two embeddings of people the model has
never seen — that spread destroys you.

### Margin softmax: the fix

1. **Normalise both the embedding and each class prototype.** Now the logit is
   exactly `cos(θ)`, the angle between a sample and its class centre. Everything
   lives on a hypersphere and only *direction* carries identity.

2. **Make the correct class artificially harder** by subtracting a margin before
   the softmax. To get low loss the model must push each sample not just onto
   the correct side of the boundary, but a comfortable distance past it. The
   result is tight within-identity clusters and wide between-identity gaps —
   exactly the geometry cosine similarity needs.

3. **Scale by `s` (=64).** Cosines live in [-1, 1], too narrow a range for
   cross-entropy to produce useful gradients.

### The three margin variants, and which to use

| Loss | Margin | When |
|---|---|---|
| **CosFace** | `cos(θ) − m` | Simple, unconditionally stable. Use as a debugging baseline. |
| **ArcFace** | `cos(θ + m)` | The established standard. Geodesically uniform margin. |
| **AdaFace** | margin scaled by feature norm | **Default here.** Best on quality-varying data. |

**AdaFace's insight:** the L2 norm of an un-normalised embedding correlates with
image quality — blurry, occluded or badly-lit faces produce low norms. Applying
a large margin to such a sample forces the model to memorise noise. AdaFace
scales the margin by the normalised feature norm, emphasising clean images and
de-emphasising degraded ones.

This is why the backbone's `forward` returns `(embedding, norm)` — the norm is an
input to the loss, not a diagnostic.

Switch losses with one config line:
```yaml
model:
  head:
    type: arcface   # or cosface, adaface
    m: 0.5
```

---

## 3. Setup

### Requirements
- NVIDIA GPU with CUDA (verified on an RTX 3050 Laptop, 4 GB)
- Python 3.10-3.13

### Install

```bash
workon ml          # or: .\path\to\env\Scripts\activate

# PyTorch must come from the CUDA index, not PyPI:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
`torch.cuda.is_available()` must print `True`. If it prints `False` you have a
CPU-only build — reinstall from the `cu126` index URL.

### Sanity check the code

```bash
pytest tests/ -q
```

---

## 4. The dataset

### What is in `MeGlass_120x120/`

47,917 flat 120x120 JPEGs. Identity is encoded in the filename:

```
7276470@N03_identity_3@8733080112_2.jpg
└──────────────────┬─┘ └────┬────┘ └┬┘
              identity   photo id  face index
```

**The identity key is everything before the *last* `@`.** The Flickr user id
itself contains an `@` (`7276470@N03`), so splitting on the first one is wrong —
a mistake that silently produces the wrong number of classes.

Verified: **47,917 images, 1,710 identities, 4-578 images each** (mean 28) — a
144:1 imbalance.

### The 120 → 112 question

IResNet expects 112x112, and the ArcFace landmark template is defined at that
size. Three options, configured by `data.resize_policy`:

| Policy | Effect |
|---|---|
| **`center_crop_112`** (default) | Crops a 4 px border. **Preserves face scale and eye positions exactly.** |
| `resize_112` | Bilinear 120→112. Shrinks the face 6.7%, moving eyes off the template. |
| `realign` | Re-detect and warp. For genuinely unaligned data. |

Verify the choice yourself in ten seconds:
```bash
python -m scripts.align_dataset --probe --src MeGlass_120x120
```
This renders a montage with the canonical landmark positions overlaid. The red
markers should sit on the eyes, green on the nose, blue on the mouth corners.
**On MeGlass they do** — the dataset comes from the same alignment lineage —
which is why `center_crop_112` is the default and no detection step is needed.

---

## 5. Step-by-step: your first training run

### Step 1 — Build the validation split (do this first)

```bash
python -m scripts.build_meglass_pairs
```

Output:
```
47,917 images, 1,710 identities (4-578 per identity, mean 28.0)
Split (seed=42):
  train : 1,410 identities, 38,864 images
  val   : 300 identities, 9,053 images (held out)
Pairs: 6,000 total (3,000 positive, 3,000 negative)
```

**Why this must come first.** It holds out 300 identities *entirely*. If
validation pairs came from people the model trained on, a high score would prove
memorisation, not verification ability. The training config's
`exclude_identities_file` removes exactly these identities from training, so the
reported number is honest.

You lose ~15% of training data. That trade is correct — an unmeasurable model is
worse than a slightly smaller one.

### Step 2 — Smoke-test the plumbing

Before committing two hours, prove backpropagation works end to end:

```bash
python -m scripts.train --config configs/meglass_ir50_adaface.yaml --overfit 100
```

The loss should fall toward zero within a few dozen epochs. It *should* overfit
— that is the point. If loss stays flat, something is broken upstream and no
amount of real training will fix it.

### Step 3 — Train

```bash
python -m scripts.train --config configs/meglass_ir50_adaface.yaml
```

Expect roughly (**measured** on an RTX 3050 *Laptop*, 4 GB):
- 607 steps/epoch (38,864 images ÷ 64)
- **~27 images/second**, so **~24 minutes/epoch**
- **~9-10 hours for the full 24 epochs**

The laptop RTX 3050 is roughly 8x slower than a desktop 3050 here: fewer SMs, a
lower power limit, and 4 GB forcing a smaller batch. Run it overnight, or reduce
scope while iterating:

```bash
# ~2.5 hours, still a usable model
python -m scripts.train --config configs/meglass_ir50_adaface.yaml --set train.epochs=6

# ~2 hours, better accuracy per hour than IR-50 at fewer epochs
python -m scripts.train --config configs/meglass_ir50_adaface.yaml     --set model.backbone.arch=ir_18 train.epochs=24
```

Training is **compute-bound, not data-bound** here — `perf/data_time_frac` sits
near 0.000, and the GPU runs at 98-100% with full clocks. More `num_workers`
will not help; only a faster GPU or a smaller model will.

### Step 4 — Watch it

In a second terminal:
```bash
tensorboard --logdir runs/
```
Open <http://localhost:6006>.

### Step 5 — Read the report

At the end:
```
runs/meglass_ir50_adaface/report/report.html   <- open in a browser
runs/meglass_ir50_adaface/report/report.md
```

Regenerate it any time:
```bash
python -m scripts.evaluate --checkpoint runs/meglass_ir50_adaface/checkpoints/best.pt --report
```

---

## 6. Monitoring: what to watch and what it means

| Metric | Healthy | What it tells you |
|---|---|---|
| `train/loss` | Falls steadily; may plateau then drop after warmup | The basic signal. Sudden spikes precede divergence. |
| `train/acc_top1` | Climbs to 90%+ | Accuracy on *margin-penalised* logits, so it reads lower than plain classification. Progress only, not quality. |
| `train/feature_norm` | **Rises steadily**, settles ~20-40 | Confidence proxy. **Collapsing toward zero means the model is degenerating.** Directly feeds AdaFace's margin. |
| `train/grad_norm` | Stable, no spikes | Earliest divergence warning — spikes appear before the loss moves. |
| `train/lr` | Warmup ramp, then smooth decay to 0 | Confirms the schedule is doing what you configured. |
| `perf/imgs_per_sec` | Steady | Throughput. |
| `perf/data_time_frac` | **< 0.10** | Fraction of time waiting on data. Above 0.15 the GPU is starving — raise `num_workers` or use faster storage. The trainer warns automatically. |
| `perf/gpu_mem_reserved_gb` | Below your card's limit | `reserved` is the real limit, not `allocated` — the gap is fragmentation. |
| `eval/*/accuracy` | Rises, then plateaus | **The number that actually matters.** |

### The most important pattern

**Training accuracy rising while validation accuracy falls = overfitting.** On a
48K-image dataset this typically appears after epoch ~25. It is why 24 epochs is
the recommendation rather than 50.

---

## 7. Understanding your results

The report gives a table like:

| Benchmark | Accuracy | Std | AUC | Threshold | TAR@FAR=1e-3 |
|---|---:|---:|---:|---:|---:|
| meglass_val | 98.20% | 0.45 | 0.9971 | 0.2841 | 94.30% |

**Accuracy** — mean over 10 folds, threshold fitted on 9 and applied to the
held-out one. Fitting one threshold on all the data would leak test labels.

**Std** — spread across folds. Large std (>1%) means too few pairs to trust the
mean.

**Threshold** — the cosine similarity above which two faces are called a match.
**You need this number to deploy.** It is stored in the checkpoint.

**TAR@FAR** — true accept rate at a fixed false accept rate. For access control
this matters far more than accuracy: "if I tolerate one impostor in a thousand,
what fraction of genuine users get in?" A system at 99% accuracy but 60%
TAR@FAR=1e-3 is unusable in practice.

### Contextualising the number

A 98% MeGlass validation accuracy does **not** mean a 98% production system. It
means the model learned this dataset's domain well. Real benchmarks:

| Model | Trained on | LFW |
|---|---|---|
| Yours (MeGlass) | 48 K images | ~97-98% expected |
| ArcFace R100 | MS1MV3 (5.8 M) | 99.83% |
| AdaFace IR101 | WebFace12M | 99.82% |

To measure comparably, download the standard eval packs and add them to
`eval.targets`:
```yaml
eval:
  targets:
    - {name: lfw, type: bin, path: /data/eval/lfw.bin}
    - {name: cfp_fp, type: bin, path: /data/eval/cfp_fp.bin}
    - {name: agedb_30, type: bin, path: /data/eval/agedb_30.bin}
```

---

## 8. Checkpoints and incremental training

### What is saved

Every checkpoint (`runs/<exp>/checkpoints/last.pt`) contains everything needed to
continue as if training never stopped:

- backbone weights + architecture spec
- **margin head weights and buffers** (including AdaFace's EMA norm statistics)
- **optimizer state** (momentum / Adam moments)
- **LR scheduler state**
- **AMP GradScaler state**
- **epoch and global step**
- **RNG state** (python, numpy, torch, CUDA)
- **the ClassMap** — identity string ↔ class index
- metric history and the fully-resolved config

Three files are written:

| File | Purpose |
|---|---|
| `last.pt` | Most recent. Resume from here. |
| `best.pt` | Best validation accuracy so far. |
| `backbone_only.pt` | Deployment: weights only, no head, no optimizer (~166 MB). |

### Resuming an interrupted run

```bash
python -m scripts.train --config configs/meglass_ir50_adaface.yaml
```
`train.resume: auto` finds `last.pt` automatically. Loss continues smoothly
across the seam.

### Why the ClassMap matters

Head row *i* is the learned prototype for class *i*. Without a durable record of
which identity *i* means, a later run cannot reuse those rows — it would be
starting from zero no matter how many weights it loaded. Storing the map is what
turns "load some weights" into genuine incremental training.

The map is **append-only**: an identity's index, once assigned, never changes.
The loader verifies this and refuses to load a checkpoint whose map disagrees
with your dataset, rather than silently training against scrambled prototypes.

### Adding new identities later

Say you train on MeGlass now and get 500 new people next month.

```bash
# 1. See what would change:
python -m scripts.extend_classmap \
    --checkpoint runs/meglass_ir50_adaface/checkpoints/best.pt \
    --config configs/new_data.yaml --dry-run

# 2. Extend, initialising new rows from real embeddings:
python -m scripts.extend_classmap \
    --checkpoint runs/meglass_ir50_adaface/checkpoints/best.pt \
    --config configs/new_data.yaml \
    --output runs/exp2/checkpoints/start.pt \
    --init mean_embedding

# 3. Fine-tune from there (set train.resume to that path).
```

What happens internally:
1. New identities are appended; existing indices are untouched.
2. The classifier weight grows from `(1710, 512)` to `(2210, 512)`, with rows
   0-1709 **bit-identical**.
3. New rows are initialised at the centroid of each new identity's own
   embeddings (`--init mean_embedding`) rather than randomly — this starts them
   near their true angular position and roughly halves fine-tuning time.
4. **The optimizer's momentum buffer grows to match.** This is the step everyone
   forgets; without it the first update after extension raises a shape-mismatch
   `RuntimeError`.

**Recommended fine-tuning recipe:**
1. One epoch with `model.backbone.freeze_backbone: true` at `optim.lr: 0.01`, to
   let the new head rows settle without disturbing the backbone.
2. Unfreeze, drop to 0.1x the original LR, cosine to zero over 4-8 epochs.
3. Track accuracy on **both** old and new identities. Some forgetting (1-2%) is
   normal; a large drop means the LR is too high.

---

## 9. Using a different dataset format

Adding a format takes about 40 lines and touches no training code. See
[ADAPTERS.md](ADAPTERS.md) for the full walkthrough.

Five adapters ship already:

| `type` | Layout | Kind |
|---|---|---|
| `flat_regex` | Flat folder, identity in the filename (MeGlass) | map-style |
| `folder_per_identity` | `root/alice/*.jpg`, `root/bob/*.jpg` | map-style |
| `csv_manifest` | A CSV of `path,identity` | map-style |
| `mxnet_rec` | `.rec`/`.idx` packs (MS1MV3, WebFace4M) | map-style |
| `webdataset` | `.tar`/`.tar.gz` shards of `<key>.jpg` + `<key>.cls` (Glint360K) | **streaming** |

Switching is a config change:
```yaml
data:
  adapter:
    type: folder_per_identity
    root: /data/my_faces
    min_images_per_identity: 3
```

The contract that makes this work: adapters return **string** identities, never
integer indices. The ClassMap assigns indices. That single decision is what lets
a new dataset extend an existing head.

### Map-style vs streaming

Map-style adapters enumerate every sample once (`scan()`) and read any sample
on demand. Streaming adapters never enumerate: a one-off **census** counts
images per identity, and each epoch every DataLoader worker streams its share
of the shards. Same training loop, same checkpoints, same reports. The
differences you will notice:

| | map-style | streaming |
|---|---|---|
| Samplers (`balanced_identity`, `sqrt_frequency`) | yes | `random` only |
| `persistent_workers` | yes | forced off (a few seconds per epoch) |
| `--overfit N` | first N samples | first N samples, decoded into RAM |
| Resume | epoch-granular | epoch-granular |
| Multi-GPU | `DistributedSampler` | `epoch_mode: resampled` |

### Training on Glint360K

Glint360K (17.1M images, 360k identities) ships on HuggingFace as 1,385
WebDataset shards of ~94 MB. The images are already RetinaFace-aligned at
112x112 -- **no detection, no cropping**; they go straight into the network.

```bash
# 1. Download shards -- standalone script, run it in its own terminal.
#    Two shards is enough to prove the plumbing; "all" is ~130 GB.
python scripts/download_glint360k.py --out D:/data/glint360k --shards 0-1

# 2. Census + hold-out split (the streaming equivalent of build_meglass_pairs)
python -m scripts.scan_webdataset --config configs/essi_fr_v1_local.yaml \
    --holdout 100 --holdout-min-images 2

# 3. Plumbing check, then train
python -m scripts.train --config configs/essi_fr_v1_local.yaml --overfit 64
python -m scripts.train --config configs/essi_fr_v1_local.yaml
```

To train on more data, change **only** `data.adapter.shards` (a brace pattern,
a list, or a directory) and re-run the scan script: the census is keyed by the
shard list and rebuilds itself. Because the shards are globally shuffled, a
small subset has ~1 image per identity -- keep `min_images_per_identity: 2`
and expect a plumbing run, not an accuracy run, until you have hundreds of
shards.

---

## 10. Scaling to AWS

See [AWS.md](AWS.md) for the full playbook. In brief:

1. **Validate on one small instance first** — `g5.2xlarge` (1× A10G 24 GB,
   ~$1.21/hr) for an hour, to prove the pipeline runs end to end on real data.
2. **Then scale** — `g5.12xlarge` (4× A10G, ~$5.67/hr) is the value pick.
3. **Use spot instances** — ~70% cheaper. The checkpointing system is what makes
   this safe: with `save_every_n_steps: 2000` and `resume: auto`, an interruption
   costs a few thousand steps at most.
4. **Pack your data** — 4M small JPEGs on EBS will bottleneck you at ~40% GPU
   utilisation. Use `.rec` packs or WebDataset shards on instance-store NVMe.
   Watch `perf/data_time_frac`.
5. **Enable `torch_compile: true`** on Linux — 20-30% on A10G/A100. It stays off
   on Windows automatically.
6. **Multi-GPU is one command** —
   `torchrun --nproc_per_node=4 -m scripts.train --config ...`. Streaming
   datasets need `epoch_mode: resampled` so every rank runs the same number of
   steps; `optim.lr` refers to the *total* batch across GPUs.

---

## 11. Choosing hyperparameters

### Local run (MeGlass, IR-50, 4 GB) — already in the config

| Setting | Value | Why |
|---|---|---|
| Backbone | `ir_50` | IR-100 at batch 32 gives worse BN statistics *and* is 1.7x slower. Wrong trade for 48 K images. |
| Batch | 64 (+2 accum = 128 effective) | Fits 4 GB under AMP with fragmentation headroom. Margins get noisy below ~128 effective. |
| Optimizer | SGD, momentum 0.9 | Reliably beats AdamW on margin-softmax face recognition. |
| LR | 0.05 | Linear rule gives 0.025 at batch 128; raised because 1,710 classes is far easier than 90 K. |
| Weight decay | 5e-4, excluding BN/bias | Worth ~0.5-1%. Filtered by `ndim <= 1` — PReLU's parameter is called `weight` and would be missed by name matching. |
| Warmup | 1 epoch | **Not optional.** Prevents the early divergence people blame on "ArcFace instability". |
| Schedule | PolyLR power 2 | InsightFace's default; decays to exactly zero and needs no milestone tuning. |
| **Epochs** | **24** | Below ~16 the margin has not converged (feature norms still climbing). Above ~30, a 48 K set overfits. |

### Memory budget on 4 GB

Activations dominate. The IResNet stem is a **stride-1** 3x3 conv, so stage 1
runs at 56x56 and memory is higher than a stock ResNet of the same depth.

| Config | Max batch | Recommended |
|---|---:|---:|
| IR-50, fp32 | ~41 | 32 |
| **IR-50, AMP fp16** | ~76 | **64** |
| IR-50, AMP + grad checkpointing | ~250 | 192 |
| IR-100, AMP fp16 | ~40 | 32 |

Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the trainer does this
automatically) — on a 4 GB card, allocator fragmentation is the difference
between batch 64 working and OOM at step 400.

### Scaling the LR

**Linear rule: LR scales with total batch size.** The ArcFace reference is 0.1 at
batch 256. Doubling the batch doubles the LR; warmup becomes more important as
batch grows (2 epochs at batch ≥512).

### How many epochs, really

| Dataset | Epochs | Reasoning |
|---|---:|---|
| MeGlass (48 K) | **24** | Only ~608 steps/epoch. Fewer undertrains; more overfits. |
| MS1MV3 (5.8 M) | **20** | LFW saturates by epoch 8; hard benchmarks improve to ~18-20 then flatten. |
| WebFace4M | **20-24** | As above. |
| Budget-constrained | **16** | Recovers ~99% of final accuracy at 80% of the cost. |

---

## 12. Troubleshooting

### `CUDA out of memory`
Lower `data.batch_size` (try 48, then 32) and raise `train.grad_accum_steps` to
keep the effective batch at 128. Or enable
`model.backbone.gradient_checkpointing: true` — ~4x the batch size at ~30% slower
steps.

### Loss becomes `nan`
The trainer raises a descriptive error rather than training silently on garbage.
In order:
1. Increase `scheduler.warmup_epochs` to 2-3.
2. Lower `optim.lr`.
3. Set `model.head.type: cosface` — if that trains cleanly, the problem is in the
   angular branch, not the data.
4. Set `model.head.easy_margin: true` (ArcFace only) as a temporary crutch.

The head has three independent NaN guards (cosine clamp, the `θ+m > π` branch,
and fp32 computation under AMP). Reaching NaN anyway almost always means the LR
is too high for the warmup.

### Training is very slow / GPU under-utilised
Check `perf/data_time_frac` in TensorBoard. Above 0.15 means data starvation:
raise `data.num_workers`, ensure `persistent_workers: true`, or move data to a
faster disk.

### `num_workers > 0` hangs or spawns endless processes (Windows)
All top-level code must be under `if __name__ == "__main__":`. Windows uses
`spawn`, which re-imports the module in every worker. `scripts/train.py` is
already correct — this bites when you write your own entry point.

### Validation accuracy is suspiciously high (99.9%)
Check that the held-out identities are actually excluded:
```bash
pytest tests/test_adapters.py::test_meglass_validation_split_is_excluded -q
```

### Resume produces a different loss than before the interruption
Model state is restored bit-identically, but DataLoader workers are reseeded per
epoch, so augmentation order differs. This is expected. A *large* jump means the
optimizer state failed to load — check the log for a warning.

### I added shards (or a folder) and resumed -- what happens to the head?
The trainer starts from the checkpoint's class map and *appends* any new
identities, so existing head rows keep their meaning and new rows are added
(random init) automatically -- the log says how many. For a better start for
the new people, extend first with
`python -m scripts.extend_classmap --checkpoint ... --config <new data> --output ... --init mean_embedding`
and resume from that checkpoint. A `class map mismatch at index N` error means
a checkpoint whose map was *rebuilt* rather than extended; use the script.

### Training is slow and `perf/data_time_frac` is ~0
The data pipeline is not the bottleneck; the model settings are. Run
`python -m scripts.bench_backbone` -- on an RTX 3050 with torch 2.13,
`model.backbone.channels_last: true` measured **6.7x slower** than off (27 vs
181 img/s), which is why the configs ship with it off. Re-measure on each
GPU/driver before enabling it.

### `N of M shards are missing`
`data.adapter.shards` names files the download script has not fetched yet.
Narrow the brace range to what is present, or point `shards` at the directory
so whatever has landed is used.

### `persistent_workers is forced OFF for streaming datasets`
Informational. Each epoch's shard order is derived from the epoch number, and
workers only see it when they are (re)started. The restart costs a few
seconds per epoch.

### Streaming epoch has fewer steps than `len(loader)` (natural mode)
With several workers each drops its own partial batch. Harmless; use
`epoch_mode: resampled` if you need the count to be exact (DDP requires it).

---

## 13. Project layout

```
ESSI-FRS/
├── configs/                    YAML configs (base + experiments)
├── frs/
│   ├── registry.py             name -> class registry (the plug-in mechanism)
│   ├── config.py               YAML inheritance, ${} interpolation, validation
│   ├── models/
│   │   ├── backbones/iresnet.py  IResNet-18..200 (vendored from AdaFace)
│   │   └── heads.py            ArcFace / CosFace / AdaFace margin heads
│   ├── data/
│   │   ├── adapters/           one module per dataset format
│   │   │   ├── streaming.py    the streaming contract + census cache
│   │   │   └── webdataset.py   .tar/.tar.gz shards (Glint360K)
│   │   ├── class_map.py        identity <-> index, append-only
│   │   ├── dataset.py          map-style torch Dataset + scan caching
│   │   ├── iterable_dataset.py streaming torch IterableDataset
│   │   ├── transforms.py       resize policies and augmentation
│   │   └── sampler.py          long-tail samplers
│   ├── align/                  SCRFD detector + Umeyama warp
│   ├── engine/
│   │   ├── trainer.py          the training loop
│   │   ├── checkpoint.py       full-state save/load + head extension
│   │   ├── optim.py            optimiser, param groups, LR schedules
│   │   └── meters.py           metric accumulators
│   ├── eval/                   LFW-protocol verification (pair files + .bin packs)
│   ├── report/                 HTML/Markdown report generation
│   └── utils/                  logging, seeding, tensorboard
├── scripts/                    CLI entry points
├── tests/                      pytest suite
└── docs/                       this guide, ADAPTERS.md, AWS.md
```

### Command reference

```bash
# Build the validation split (run once, before training)
python -m scripts.build_meglass_pairs

# Sanity check: overfit 100 images
python -m scripts.train --config configs/meglass_ir50_adaface.yaml --overfit 100

# Train
python -m scripts.train --config configs/meglass_ir50_adaface.yaml

# Train with overrides
python -m scripts.train --config configs/... --set train.epochs=30 optim.lr=0.02

# Evaluate a checkpoint
python -m scripts.evaluate --checkpoint runs/<exp>/checkpoints/best.pt

# Inspect a checkpoint's contents
python -m scripts.evaluate --checkpoint runs/<exp>/checkpoints/best.pt --inspect

# Check the resize policy visually
python -m scripts.align_dataset --probe --src MeGlass_120x120

# Align a new, unaligned dataset
python -m scripts.download_models
python -m scripts.align_dataset --src raw_photos --dst aligned_112

# Add new identities to a trained model
python -m scripts.extend_classmap --checkpoint ... --config ... --dry-run

# Glint360K (streaming): download -> census + hold-out -> train
python scripts/download_glint360k.py --out D:/data/glint360k --shards 0-1
python -m scripts.scan_webdataset --config configs/essi_fr_v1_local.yaml --holdout 100 --holdout-min-images 2
python -m scripts.train --config configs/essi_fr_v1_local.yaml

# Multi-GPU (Linux)
torchrun --nproc_per_node=4 -m scripts.train --config configs/essi_fr_v1_aws.yaml

# Raw GPU throughput of a backbone (pick channels_last / batch before a long run)
python -m scripts.bench_backbone --arch ir_50 --batch 64

# Tests
pytest tests/ -q
```
