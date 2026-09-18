# ESSI-FR v1 — Training Pipeline Update

**Date:** 12 September 2026
**Branch:** `AWS` (34 commits, 36 files, +3,077 / −227 lines)
**Status:** Training in progress on AWS

---

## Summary

The training pipeline has been extended to train on **Glint360K** — 17,091,657
images across 360,232 identities — replacing MeGlass (47,917 images, 1,710
identities) as the production dataset. That is a **357x increase in images** and
a **211x increase in identities**.

The MeGlass path is untouched and still works. Glint360K was added as a second,
parallel data path rather than a replacement.

---

## 1. Why a new data path was needed

The existing pipeline scanned a dataset once into a list of samples, then read
any sample on demand by index. That design is correct for a folder of JPEGs. It
does not work for Glint360K, which ships as 1,385 gzip-compressed tar shards:

- **gzip archives are not seekable.** There is no way to jump to image number
  8,000,000 without decompressing everything before it.
- **A sample list would not fit.** 17 million Python objects costs several GB
  *per data-loading worker*.

So a **streaming** data path was built alongside the existing one. Both produce
identical `(image, class index)` output, so the trainer, checkpoints, evaluation
and reporting work unchanged with either.

---

## 2. What was added

### Streaming dataset support

| File | Purpose |
|---|---|
| `frs/data/adapters/streaming.py` | The streaming adapter contract and the census cache |
| `frs/data/adapters/webdataset.py` | Reads `.tar`/`.tar.gz` shards (`<key>.jpg` + `<key>.cls`) |
| `frs/data/iterable_dataset.py` | The PyTorch `IterableDataset` the trainer consumes |

Instead of listing every sample, a one-time **census** reads only the tiny label
entry inside each shard and counts images per identity. From that single pass the
system derives the identity list, the epoch length, the dataset statistics for
the report, and the minimum-images and held-out-identity filters. The result is
cached as JSON, so it is paid once.

During training each worker streams its own share of the shards. Only a 1.4 MB
integer lookup table crosses into each worker, not a sample list.

**Measured:** the census over all 1,385 shards took 17 minutes and returned
17,091,657 images / 360,232 identities — matching the published dataset figures
exactly.

### Standard benchmark evaluation

`frs/eval/bin_pack.py` reads InsightFace `.bin` verification packs, so the model
can be scored on **LFW, CFP-FP and AgeDB-30** — the benchmarks published results
are measured against. Previously only custom pair files were supported.

### Multi-GPU training

`torchrun` multi-GPU now works end to end. The distributed helpers existed but
were never called; the training script now initialises the process group, wraps
the model, splits data per rank, and confines logging, checkpointing and
evaluation to rank 0.

### New tools

| Script | Purpose |
|---|---|
| `scripts/download_glint360k.py` | Resumable shard download; imports nothing from the project so it runs in a separate process |
| `scripts/scan_webdataset.py` | Runs the census and builds the held-out validation split |
| `scripts/bench_backbone.py` | Measures raw GPU throughput before committing to a long run |

---

## 3. Bugs found and fixed

Three defects would each have stopped a full-scale run. All were found by
testing rather than by reading.

**Partial-FC was broken and unreachable.** Partial-FC samples a subset of
classes per step, which is mandatory above ~300,000 identities or the classifier
alone consumes gigabytes. It was never called from the training script, and when
called it crashed immediately — it assigned a plain tensor to a PyTorch
parameter. Both are fixed; it now auto-enables above 300,000 classes and samples
10% of classes per step.

**Checkpoints broke under wrapping.** Saving a model wrapped in Partial-FC,
multi-GPU or compilation produced renamed tensors that could not be loaded back.
Checkpoints now always store the bare inner model, so a run can switch any of
those on or off between restarts.

**Adding data invalidated a checkpoint.** Resuming after adding shards rebuilt
the identity list from scratch, so every learned classifier row silently pointed
at a different person. Resume now starts from the checkpoint's own identity
mapping and appends new identities to it.

---

## 4. Performance work

Two settings were measured rather than assumed, and both findings were
counter-intuitive.

**`channels_last` was 6.7x slower, not faster.** This memory layout is widely
recommended for NVIDIA Ampere GPUs and was enabled everywhere. Measured on the
development GPU it gave 27 images/sec against 181 with it off. It is now off by
default, with `scripts/bench_backbone.py` provided to re-measure on any new GPU.

**`torch.compile` crashed training after a few steps.** The `max-autotune` mode
enables CUDA graphs, which reuse one output buffer per compiled region. This
pipeline deliberately runs the loss head *outside* the compiled region in full
precision for numerical stability, so the next iteration overwrote the data
before it was used. Fixed by using `max-autotune-no-cudagraphs`, which keeps the
speed-up and drops the incompatible part.

**TF32 enabled.** The loss head runs in full precision by design, and at 360,000
classes its matrix multiply is a real share of each step. Enabling TF32 cut it
from 0.78 ms to 0.46 ms. Precision was verified on the actual computation, not a
synthetic one: 7e-05 error, finer than the half precision already used elsewhere.

---

## 5. Branding

The pipeline and its outputs are now identified as **ESSI-FR v1**. Run
directories, reports and exported models carry that name.

The deployable artifact was tightened. Exporting now produces the weights plus
an inference contract — input size, colour order, normalisation, embedding
dimension and the model name — and nothing else. Verified by inspection: the
internal checkpoint exposes the training identity list and every hyperparameter;
the exported model exposes neither.

| Artifact | Contains | Shareable |
|---|---|---|
| `backbone_only.pt`, `.onnx`, `.meta.json` | Weights + inference contract | **Yes** |
| `best.pt`, `last.pt` | The above, plus identity list and all hyperparameters | **No** |

Component names (the backbone source and the loss functions) remain in the
source code as required attribution. They do not appear in anything shipped.

---

## 6. Testing

The test suite grew from 57 to **86 tests**, all passing. New coverage:

- Streaming pipeline: census accuracy, filters, identity mapping, epoch
  determinism, multi-worker behaviour, and the scan tool end to end
- Partial-FC: the crash that was fixed, gradient correctness, checkpoint
  round-trip
- Benchmark pack loading
- Resume-after-adding-data

Tests build small synthetic shards on the fly, so none of them need the real
130 GB dataset.

---

## 7. Current training run

| Setting | Value |
|---|---|
| Dataset | 17,030,462 images / 359,232 identities |
| Held out | 1,000 identities, excluded from training entirely |
| Model | IResNet-50 + AdaFace, 43.6M parameters |
| Classifier | Partial-FC, 10% class sampling |
| Optimiser | SGD, momentum 0.9, LR 0.075, poly decay, 2-epoch warm-up |
| Batch | 192 |
| Hardware | AWS g5.xlarge, 1x NVIDIA A10G |
| Throughput | 778 images/sec, GPU at 99% |
| Schedule | 20 epochs, ~6.1 h each, ~5 days total |

**Progress at time of writing:** epoch 0, 35% complete. Loss has fallen from
40.1 to ~17.5 and accuracy has risen from 0% to ~1.5% against 359,232 classes,
where random chance is 0.0003%. Both indicate healthy convergence.

IResNet-50 was chosen over IResNet-100 after measurement: 778 images/sec against
467, cutting training from 8.5 days to 5. The trade is roughly 1-2% accuracy on
the hardest benchmarks.

---

## 8. Open items

1. **Standard benchmark packs are not on the training machine.** Until LFW,
   CFP-FP and AgeDB-30 are available, the only score is on held-out Glint360K
   identities, which is in-domain and will read optimistically high. These
   should be in place before the run finishes.

2. **Dataset licence.** Glint360K's terms should be confirmed as compatible with
   the intended commercial use. This is independent of any technical work.

3. **Uncommitted changes.** The TF32 setting and the compilation fix are applied
   and running but not yet committed to the branch.

---

## 9. Expected outcome

Based on published results for this architecture and dataset, a reasonable
expectation is **~99.7% on LFW** and **93-95% on CFP-FP**. That is a solid,
defensible model — not state of the art, which would require the larger backbone
and longer training.

The pipeline itself is the more durable asset: it scales from a 4 GB laptop GPU
to multi-GPU cloud training without code changes, survives interruption, and can
be extended with new identities later without retraining from scratch.
