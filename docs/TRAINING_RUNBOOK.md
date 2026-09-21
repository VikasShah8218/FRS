# ESSI-FR Training Runbook

How to set up an AWS instance, download Glint360K, and train ESSI-FR on 1 or 4 GPUs,
with IResNet-50 or IResNet-100. Everything here is done with commands and configs;
no code changes are needed.

---

## Quick start (4 GPUs, IResNet-100)

```bash
# 0. once per instance: code + env (section 3)
cd ~/FRS && source ~/venv/bin/activate

# 1. download the dataset, ~130 GB (section 4)
python scripts/download_glint360k.py --out /mnt/data/glint360k --shards all --workers 8 --verify

# 2. census + validation hold-out (section 5)
python -m scripts.scan_webdataset --config configs/essi_fr_v1_aws.yaml --workers 16

# 3. train on all 4 GPUs inside tmux (section 7)
tmux new -s train
torchrun --standalone --nproc_per_node=4 -m scripts.train \
    --config configs/essi_fr_v1_aws.yaml \
    --set experiment.name=essi_fr_v2 eval.primary=glint_val
# Ctrl-B then D to detach. `tmux attach -t train` to come back.
```

---

## 1. Training architecture

```
Glint360K shards  (1,385 x .tar.gz, 17.09 M images, 360,232 identities, 112x112)
      |   streamed from disk, shuffled in a 5,000-image buffer, decoded by DataLoader workers
      v
Augmentation      flip 0.5 · colour jitter 0.2 · grayscale 0.05 · random erasing 0.2
      |           normalised to [-1, 1]
      v
Backbone          IResNet-50 or IResNet-100  ->  512-d face embedding   <- the product
      |
      v
Margin head       AdaFace (s=64, m=0.4, h=0.333) + Partial-FC (10% of classes per step)
      |
      v
Loss / optimiser  cross-entropy -> SGD momentum 0.9 -> poly LR with warm-up
                  mixed precision fp16, TF32, torch.compile
```

| Component | What it is | Why |
|---|---|---|
| **Streaming WebDataset** | Reads tar shards sequentially instead of millions of loose JPEGs | Keeps GPUs fed; data stall was 0.1% on the v1 run |
| **IResNet** | ResNet built for 112x112 faces, outputs a 512-d embedding | Industry-standard face backbone |
| **AdaFace head** | Margin softmax whose margin adapts to image quality (via the embedding norm) | Robust to blurry and low-quality faces |
| **Partial-FC** | Each step uses only 10% of the ~359k class centres | Makes 360k classes affordable in memory and time |
| **SGD + poly LR** | LR rises linearly during warm-up, then decays smoothly to 0 | Standard, stable recipe for margin losses |
| **AMP fp16 + TF32** | Mixed precision; the head stays fp32 | About 2x faster, same accuracy |
| **DDP** | One process per GPU, gradients averaged each step | Near-linear speed-up on 4 GPUs |

Only the backbone ships: `backbone_only.pt` or the ONNX export. The head and optimizer
state stay internal.

---

## 2. Files you will use

| File | Purpose |
|---|---|
| `configs/essi_fr_v1_aws.yaml` | **4-GPU config.** IResNet-100, batch 128 per GPU, LR 0.05 |
| `configs/essi_fr_v1_aws_1gpu.yaml` | **1-GPU config** (the v1 run). Inherits the file above and overrides: IResNet-50, batch 192, LR 0.02 |
| `configs/base.yaml` | Defaults for every key (augmentation, embedding size, ...) |
| `scripts/download_glint360k.py` | Downloads the dataset shards from HuggingFace |
| `scripts/scan_webdataset.py` | Label census + builds the validation hold-out |
| `scripts/train.py` | Training |
| `scripts/evaluate.py` / `scripts/export.py` | Re-evaluate a checkpoint / export to ONNX |
| `data/splits/glint360k_val_identities.txt` | 1,000 identities excluded from training |
| `data/pairs/glint360k_val_pairs.txt` | 6,000 validation pairs (3,000 same + 3,000 different) |
| `scripts/aws/*` | Auto-resume service and server-stability settings (section 3.3) |

---

## 3. One-time setup on a new instance

### 3.1 Storage

The configs expect data under `/mnt/data`:

```bash
sudo mkdir -p /mnt/data && sudo chown ubuntu:ubuntu /mnt/data
df -h /mnt/data          # need ~150 GB free for data, plus space for checkpoints
```

> Use the **EBS volume**, which survives a stop/start. Instance-store NVMe is faster,
> but it is **wiped every time the instance stops**, so you would have to download the
> dataset again.

### 3.2 Code and Python environment

```bash
sudo apt-get update && sudo apt-get install -y python3-venv tmux
cd ~ && git clone <your-git-remote> FRS && cd FRS && git checkout AWS-2

python3 -m venv ~/venv && source ~/venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python -c "import torch; print(torch.__version__, torch.cuda.device_count())"   # expect 4
python -m pytest tests -q                                                       # all pass
```

Always work inside the venv (`source ~/venv/bin/activate`). **Never run the scripts with
`sudo`**: sudo's Python cannot see the venv packages (`No module named huggingface_hub`).

### 3.3 Server-stability settings (recommended)

These fix the two problems that killed training on 13 Sep 2026. Ubuntu auto-updates
restarted the training service, and logging out of SSH deleted the shared memory the
DataLoader uses.

```bash
sudo mkdir -p /etc/systemd/logind.conf.d /etc/needrestart/conf.d
sudo cp ~/FRS/scripts/aws/logind-essi-train.conf     /etc/systemd/logind.conf.d/essi-train.conf
sudo cp ~/FRS/scripts/aws/needrestart-essi-train.conf /etc/needrestart/conf.d/essi-train.conf
sudo loginctl enable-linger ubuntu
sudo systemctl restart systemd-logind
```

The **auto-resume service** (`essi-train.service`) restarts training after a reboot, but
it launches **single-GPU** training only. On the 4-GPU instance, **do not install it**, or
turn it off:

```bash
sudo systemctl disable essi-train      # or: touch ~/FRS/NO_AUTORESUME
```

---

## 4. Download the dataset

```bash
source ~/venv/bin/activate && cd ~/FRS
python scripts/download_glint360k.py --out /mnt/data/glint360k --shards all --workers 8 --verify
```

| Option | Default | Meaning |
|---|---|---|
| `--out` | **required** | Folder to save shards in (the configs expect `/mnt/data/glint360k`) |
| `--shards` | `0-1` | Which shards to fetch: `all`, a range `0-49`, or a list `0-49,100,200-210` |
| `--workers` | `2` | Parallel downloads (use 8 on AWS) |
| `--verify` | off | After downloading, gzip-reads every shard to catch corrupt files |
| `--force` | off | Re-download shards that already exist |
| `--dry-run` | off | Shows what would be downloaded, and the shard pattern for the config, without downloading |
| `--revision` | `main` | HuggingFace dataset revision |

Good to know:
- **Resumable.** If it stops, run the same command again; finished shards are skipped.
- It checks free disk space before starting.
- Optional: `export HF_TOKEN=hf_xxx` first, for fewer rate limits.
- It is independent of training. Run it in its own tmux window (`tmux new -s download`).
- Full dataset: 1,385 shards, ~94 MB each, **~130 GB**. The configs use
  `shards: "/mnt/data/glint360k/glint360k-{0000..1384}.tar.gz"`.

---

## 5. Census and validation hold-out

Training needs two things before it starts:

1. **A census**, the per-identity image counts. It is cached in `.cache/scan/census_<hash>.json`.
2. **The validation hold-out**: 1,000 identities removed from training, with their
   images in `/mnt/data/glint360k_val`. The pair file points to these paths, so the
   images must exist.

The identity and pair files are already in the repo. **Only the images and the census
cache are missing on a new instance.** Choose A or B.

### A. Copy from the old instance (recommended, and exactly the same benchmark as v1)

Go through your PC, since the instances are in different regions:

```bash
# on the OLD instance
tar czf ~/val_and_census.tgz -C /mnt/data glint360k_val -C ~/FRS .cache/scan
# on your PC
scp -i <old-key.pem> ubuntu@16.16.41.227:~/val_and_census.tgz .
scp -i <new-key.pem> val_and_census.tgz ubuntu@<NEW_IP>:~
# on the NEW instance
mkdir -p ~/FRS/.cache && tar xzf ~/val_and_census.tgz -C /mnt/data glint360k_val
tar xzf ~/val_and_census.tgz -C ~/FRS .cache/scan
```

The census cache is only reused if the shard path is identical
(`/mnt/data/glint360k/glint360k-{0000..1384}.tar.gz`), and it is.

### B. Rebuild (if the old instance is gone)

Rebuild with the same settings the v1 run used. `--force` is needed because the
split files already exist:

```bash
python -m scripts.scan_webdataset --config configs/essi_fr_v1_aws.yaml --workers 16 \
    --holdout 1000 --holdout-min-images 8 \
    --holdout-out /mnt/data/glint360k_val \
    --identities-out data/splits/glint360k_val_identities.txt \
    --pairs-out data/pairs/glint360k_val_pairs.txt --force
```

If `git diff data/` then shows changes, the benchmark differs slightly from v1. Commit
the new files so every future run uses the same ones.

**Census only** (the images already exist): `python -m scripts.scan_webdataset --config configs/essi_fr_v1_aws.yaml --workers 16`

> **Before 4-GPU training, the census must already exist.** Otherwise all 4 GPU processes
> each scan the full 130 GB at the same time. That is slow and can hit the multi-GPU
> start-up timeout.

| Scan option | Default | Meaning |
|---|---|---|
| `--config` | required | Config whose `data.adapter.shards` is scanned |
| `--workers` | up to 8 | Parallel shard readers (16 on a 48-vCPU box) |
| `--force` | off | Redo the census / overwrite split files |
| `--holdout N` | 0 | Number of identities to hold out (0 = census only) |
| `--holdout-min-images` | 4 | Only identities with at least this many images qualify (v1 used 8) |
| `--holdout-max-images` | 20 | Max images extracted per held-out identity |
| `--holdout-out` | `data/glint360k_val` | Where held-out JPEGs go (v1: `/mnt/data/glint360k_val`) |
| `--num-positive` / `--num-negative` | 3000 / 3000 | Number of same / different pairs |
| `--seed` | 42 | Makes the selection reproducible |

---

## 6. Start training (1 GPU)

```bash
tmux new -s train
cd ~/FRS && source ~/venv/bin/activate
python -m scripts.train --config configs/essi_fr_v1_aws_1gpu.yaml
```

| Flag | Meaning |
|---|---|
| `--config PATH` | The YAML config (required) |
| `--set key=value ...` | Override any config key without editing the file, e.g. `--set train.epochs=16 optim.lr=0.02` |
| `--resume PATH` | Resume from a specific checkpoint (normally not needed; see below) |
| `--overfit N` | Sanity check: train on only N images; loss should fall to ~0. Writes to a separate `overfit/` folder |
| `--no-eval` | Skip validation at the end of each epoch |

**Resuming is automatic.** With `train.resume: auto`, re-running the **exact same
command** continues from `runs/<experiment.name>/checkpoints/last.pt`, even mid-epoch.
A checkpoint is saved every 2,000 steps and at every epoch end.

> Always give a **new model a new `experiment.name`** (e.g. `--set experiment.name=essi_fr_v2`).
> Otherwise `resume: auto` finds the old v1 checkpoint and tries to continue it.

---

## 7. Train on all 4 GPUs

```bash
tmux new -s train
cd ~/FRS && source ~/venv/bin/activate
torchrun --standalone --nproc_per_node=4 -m scripts.train \
    --config configs/essi_fr_v1_aws.yaml \
    --set experiment.name=essi_fr_v2 eval.primary=glint_val
```

`torchrun` starts one process per GPU. GPU 0 does the logging, checkpoints and
evaluation, and the other GPUs wait for it.

**Why the two `--set` values:**
- `experiment.name=essi_fr_v2`: the new run gets its own folder (`runs/essi_fr_v2/`) and
  does not resume v1.
- `eval.primary=glint_val`: the config scores `best.pt` on LFW, but the LFW/CFP/AgeDB
  `.bin` files are not on the server. Those targets are skipped with a warning, and
  without this override `best.pt` would never be written. If you copy
  `lfw.bin`, `cfp_fp.bin` and `agedb_30.bin` into `/mnt/data/eval/`, they are
  evaluated too.

### Rules when changing GPUs or batch size

| Setting | Rule |
|---|---|
| **Learning rate** | `lr = 0.1 x total_batch / 1024`, where `total_batch = batch_size x GPUs`. Too high an LR ruined the first v1 attempt |
| **Warm-up** | `warmup_epochs: 2` when the total batch is 512 or more |
| **`num_workers`** | Per GPU: about `vCPUs / GPUs - 2`. With 48 vCPUs and 4 GPUs, 10 (already set) |
| **`epoch_mode`** | Must stay `resampled` for multi-GPU. `natural` refuses to start |

| GPUs x batch/GPU | Total batch | `optim.lr` | `warmup_epochs` | Steps/epoch |
|---|---:|---:|---:|---:|
| 1 x 192 (v1 run) | 192 | 0.02 | 1 | 88,698 |
| 4 x 128 (**config default**) | 512 | 0.05 | 2 | ~33,260 |
| 4 x 256 (48 GB GPUs only) | 1,024 | 0.1 | 2 | ~16,630 |

Example for the last row:
`--set data.batch_size=256 optim.lr=0.1 experiment.name=essi_fr_v2 eval.primary=glint_val`

### Check that all 4 GPUs are working

- `nvidia-smi` shows **4 python processes**, all at high utilisation.
- The log start shows `GPU: ... (rank 0/4)`.
- The `img/s` in the log is for **one GPU**. Total speed is about 4x that.
- The warning about `OMP_NUM_THREADS` at start-up is normal.

### After a stop or reboot

The auto-resume service does not handle torchrun. Re-attach with `tmux attach -t train`,
or open a new tmux session, and run **the same torchrun command** again. It resumes
from the last checkpoint by itself.

---

## 8. Switch IResNet-50 to IResNet-100

**Only the config changes.** The key is `model.backbone.arch`.

| Want | How |
|---|---|
| IResNet-100 on 4 GPUs | Nothing to do: `essi_fr_v1_aws.yaml` is already `ir_100` |
| IResNet-100 on 1 GPU | `--set model.backbone.arch=ir_100 data.batch_size=128 optim.lr=0.0125 experiment.name=essi_fr_v2` |
| IResNet-50 on 4 GPUs | `--set model.backbone.arch=ir_50` (plus your other `--set` values) |

Available: `ir_18`, `ir_34`, `ir_50`, `ir_100`, `ir_152`, `ir_200`, and the
`ir_se_*` versions with squeeze-excitation.

What changes with IResNet-100:

| | IResNet-50 | IResNet-100 |
|---|---|---|
| Speed on one A10G | ~789 img/s | ~467 img/s (about 40% slower) |
| GPU memory | — | ~13.4 GB at batch 128 |
| Accuracy | v1 result: 99.95% on glint_val | Usually better on hard cases (pose, age, low quality) |

Rules:
- Use a **new `experiment.name`**. An IResNet-50 checkpoint cannot be loaded into
  IResNet-100, because the layers differ.
- It trains from scratch. `model.backbone.pretrained` only works with a checkpoint of
  the **same** architecture.

---

## 9. Monitoring

```bash
tmux attach -t train                                    # live console (Ctrl-B, D to leave)
tail -f ~/FRS/runs/essi_fr_v2/train.log                 # full log
grep -E "epoch .* done|glint_val " ~/FRS/runs/essi_fr_v2/train.log   # one line per epoch + eval
watch -n 5 nvidia-smi                                   # GPU use and memory
```

For TensorBoard, run `tensorboard --logdir ~/FRS/runs --port 6006` on the server. Then
on your PC run `ssh -i <key.pem> -L 6006:localhost:6006 ubuntu@<IP>` and open
http://localhost:6006.

A step line looks like:
`ep 19 [  88/88698] loss 0.9740  acc 80.45%  lr 0.00006  fnorm 22.6  wnorm 0.2563  789 img/s`

| Field | Meaning | Healthy |
|---|---|---|
| `ep [step/total]` | Epoch (from 0) and step inside it | — |
| `loss` | Training loss | Falls steadily; v1 ended at 0.91 |
| `acc` | Training accuracy on the sampled classes | Rises; v1 ended at 81.3% |
| `lr` | Current learning rate | Rises in warm-up, then decays to ~0 |
| `fnorm` | Average embedding length | Stable (~20–23) |
| `wnorm` | Typical class-centre length | Stable (v1: ~0.25). **Falling steadily towards 0 = problem.** The trainer warns |
| `img/s` | Speed of this GPU | Steady |

At each epoch end:
- `epoch N done: ... data-stall X%`. Above ~15% means the CPU or disk can't keep up;
  raise `num_workers`.
- An eval table: `benchmark | accuracy | std | thresh | AUC | TAR@1e-3`.
  - **accuracy** is the verification accuracy on the 6,000 pairs.
  - **TAR@1e-3** is the share of genuine pairs accepted when only 0.1% of impostors are accepted.

**Time estimate:** hours per epoch ≈ 17,030,000 / (img/s x number of GPUs) / 3600.

---

## 10. Outputs

```
runs/<experiment.name>/
├── train.log                      full training log
├── checkpoints/
│   ├── last.pt                    latest state: resume from here (internal)
│   ├── best.pt                    best eval.primary score so far (internal)
│   └── backbone_only.pt           final embedding model, written when training ends
├── tensorboard/                   curves
└── report/                        HTML / Markdown report
```

```bash
# re-evaluate a checkpoint and rebuild the report
python -m scripts.evaluate --checkpoint runs/essi_fr_v2/checkpoints/best.pt --report

# export for deployment (ONNX, clean metadata, named essi_fr_v2)
python -m scripts.export --checkpoint runs/essi_fr_v2/checkpoints/best.pt \
    --format onnx --verify --model-name essi_fr_v2
```

`last.pt` and `best.pt` contain the full head and optimizer state and are **internal only**.
Ship `backbone_only.pt` or the ONNX file.

**When training is done, stop the instance. Don't terminate it:** the EBS volume and
checkpoints are kept, and an idle 4-GPU instance is expensive.

---

## 11. Parameter reference

Values used by the two AWS configs. Override any of them with `--set key=value`.

### Data

| Key | 1-GPU (v1) | 4-GPU | Meaning |
|---|---|---|---|
| `data.adapter.shards` | `/mnt/data/glint360k/glint360k-{0000..1384}.tar.gz` | same | Shards to train on |
| `data.adapter.min_images_per_identity` | 2 | 2 | Drop identities with fewer images |
| `data.adapter.exclude_identities_file` | `data/splits/glint360k_val_identities.txt` | same | Hold-out identities, never trained on |
| `data.adapter.epoch_mode` | `resampled` | `resampled` | Required for multi-GPU |
| `data.adapter.samples_per_epoch` | null | null | null = whole dataset (~17.03 M) per epoch |
| `data.adapter.shuffle_buffer` | 5000 | 5000 | Images kept in memory for shuffling, per worker |
| `data.batch_size` | 192 | 128 | Images per GPU per step |
| `data.num_workers` | 3 | 10 | Loader processes per GPU |
| `data.prefetch_factor` | 4 | 6 | Batches queued per worker |
| `data.census_workers` | 4 | 16 | Parallel readers if training has to run the census |
| `data.augment.*` | flip 0.5, jitter 0.2, gray 0.05, erasing 0.2 | same | Augmentation (in `base.yaml`) |

### Model

| Key | 1-GPU (v1) | 4-GPU | Meaning |
|---|---|---|---|
| `model.backbone.arch` | `ir_50` | `ir_100` | Backbone depth |
| `model.backbone.embedding_size` | 512 | 512 | Embedding length |
| `model.head.type` | `adaface` | `adaface` | Margin loss head |
| `model.head.scale` / `m` / `h` / `t_alpha` | 64 / 0.4 / 0.333 / 0.01 | same | AdaFace settings (leave as is) |
| `model.head.partial_fc.enabled` | `auto` | `auto` | On automatically above 300k classes |
| `model.head.partial_fc.sample_rate` | 0.1 | 0.1 | Fraction of classes used per step |

### Optimiser and schedule

| Key | 1-GPU (v1) | 4-GPU | Meaning |
|---|---|---|---|
| `optim.type` | `sgd` | `sgd` | Optimiser |
| `optim.lr` | 0.02 | 0.05 | Peak LR, **scaled to total batch** (section 7) |
| `optim.momentum` | 0.9 | 0.9 | SGD momentum |
| `optim.weight_decay` | 5e-4 | 5e-4 | Regularisation (not applied to BN, bias, or unsampled class rows) |
| `optim.clip_grad_norm` | 5.0 | 5.0 | Gradient clipping for stability |
| `scheduler.type` | `polylr` | `polylr` | Poly decay to 0 |
| `scheduler.warmup_epochs` | 1 | 2 | Linear warm-up length |
| `scheduler.warmup_start_factor` | 0.1 | 0.1 | Warm-up starts at 10% of the peak LR |
| `scheduler.power` | 2.0 | 2.0 | Decay curve shape |

### Training loop

| Key | 1-GPU (v1) | 4-GPU | Meaning |
|---|---|---|---|
| `experiment.name` | `essi_fr_v1` | set a new one | Run folder name, and the model name on export |
| `train.epochs` | 20 | 20 | 16 gives ~99% of the accuracy for 80% of the cost |
| `train.amp` / `amp_dtype` | true / float16 | same | Mixed precision |
| `train.grad_accum_steps` | 1 | 1 | Keep at 1 |
| `train.torch_compile` | true | true | ~20–30% faster after a few minutes of warm-up compile |
| `train.save_every_n_steps` | 2000 | 2000 | Mid-epoch checkpoint interval |
| `train.keep_last_n_checkpoints` | 3 | 3 | Old checkpoints pruned |
| `train.resume` | `auto` | `auto` | Continue from `last.pt` if it exists |
| `train.log_every_n_steps` | 50 | 50 | Log line interval |

### Evaluation

| Key | 1-GPU (v1) | 4-GPU | Meaning |
|---|---|---|---|
| `eval.targets` | glint_val | glint_val + lfw, cfp_fp, agedb_30 (`.bin`) | Benchmarks run each epoch; missing `.bin` files are skipped |
| `eval.primary` | `glint_val` | `lfw`, so **pass `--set eval.primary=glint_val`** | Metric that decides `best.pt` |
| `eval.flip_test` | true | true | Averages each image with its mirror |
| `eval.batch_size` | 256 | 256 | Eval batch |

---

## 12. v1 reference result

| | ESSI-FR v1 |
|---|---|
| Setup | 1 x A10G (g5.xlarge), IResNet-50, batch 192, LR 0.02 |
| Duration | 20 epochs, ~6 h 04 min per epoch, ~5 days, 789 img/s |
| Final training | loss 0.909, accuracy 81.30% |
| glint_val | **99.95%** (3 of 6,000 pairs wrong), TAR@1e-3 **99.97%** |

A new run should beat or match these numbers on glint_val.

---

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named huggingface_hub` / `torch` | Activate the venv; don't use `sudo` |
| `the following arguments are required: --out` | Add `--out /mnt/data/glint360k` |
| Missing shard error at training start | Download didn't finish. Re-run the download command |
| Multi-GPU start hangs, or times out at start-up | Run the census first (section 5), then start torchrun |
| `natural mode` error under torchrun | Keep `data.adapter.epoch_mode: resampled` |
| CUDA out of memory | Lower `data.batch_size` **and** `optim.lr` by the same factor |
| `data-stall` above 15% | Raise `data.num_workers` (within the vCPU rule) |
| `best.pt` never appears | Add `--set eval.primary=glint_val` |
| New run "finishes" instantly or loads the wrong model | It resumed an old run: use a new `experiment.name` |
| Glint validation skipped: file not found | Hold-out images missing at `/mnt/data/glint360k_val` (section 5) |
| Training died after SSH logout | Install the logind settings (section 3.3) |
| `wnorm` keeps falling towards 0, accuracy drops | LR too high: check it against the section 7 table |
