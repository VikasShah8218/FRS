# Scaling to AWS

Moving from the 4 GB local smoke test to a real training run on a large dataset.
The codebase does not change — only the config and the data.

---

## 1. Choosing an instance

| Instance | GPUs | VRAM | ~$/hr (on-demand) | Use for |
|---|---|---:|---:|---|
| `g5.2xlarge` | 1× A10G | 24 GB | ~$1.21 | **Pipeline validation** |
| `g5.12xlarge` | 4× A10G | 96 GB | ~$5.67 | **The value pick for real training** |
| `p4d.24xlarge` | 8× A100 | 320 GB | ~$32.77 | 5-6x faster; worth it on long runs, not while iterating |

Prices vary by region and change over time — check current rates.

### Do this in order

1. **Validate on one `g5.2xlarge` for an hour.** Prove the data loads, the model
   trains, checkpoints save and resume works — on the *real* dataset. An hour at
   $1.21 is far cheaper than discovering a path bug three hours into a
   `g5.12xlarge` run.
2. **Then scale up.**

### Use spot instances

Spot is roughly 70% cheaper, and this codebase is built to survive interruption:

```yaml
train:
  save_every_n_epochs: 1
  save_every_n_steps: 2000    # mid-epoch checkpoints
  resume: auto
```

An interruption costs at most a few thousand steps. Keep checkpoints on EBS or
sync them to S3 so a replacement instance can pick up where the last left off.

---

## 2. Environment setup

Use the **Deep Learning AMI** — CUDA and drivers are preinstalled.

```bash
sudo apt-get update && sudo apt-get install -y python3-venv
python3 -m venv ~/frs && source ~/frs/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
nvidia-smi
```

---

## 3. Getting data onto the instance

**This is where most AWS training runs actually lose their time.**

4 million small JPEGs on an EBS gp3 volume will bottleneck you at ~40% GPU
utilisation — you pay for GPUs and spend the money on IOPS.

### Do this

1. **Use a packed format.** Either `.rec`/`.idx` packs (MS1MV3, WebFace4M —
   the `mxnet_rec` adapter, no mxnet dependency) or WebDataset tar shards
   (Glint360K on HuggingFace — the streaming `webdataset` adapter).
2. **Put it on instance-store NVMe**, not EBS:
   ```bash
   sudo mkfs -t xfs /dev/nvme1n1
   sudo mkdir -p /mnt/data && sudo mount /dev/nvme1n1 /mnt/data
   sudo chown $USER /mnt/data
   ```
   Instance store is ephemeral — it disappears when the instance stops. Keep the
   source of truth in S3 (or re-download from HuggingFace) and checkpoints on
   EBS or S3.
3. **Watch `perf/data_time_frac` in TensorBoard.** Above 0.15 means the GPU is
   starving. The trainer warns automatically.

### Glint360K: the full recipe

```bash
# all 1,385 shards, ~130 GB, resumable -- run it under tmux, in its own shell
python scripts/download_glint360k.py --out /mnt/data/glint360k --shards all --workers 8

# one pass over the labels (~15-30 min from NVMe) + hold out 1,000 identities
python -m scripts.scan_webdataset --config configs/glint360k_ir100_adaface_aws.yaml \
    --workers 16 --holdout 1000 --holdout-min-images 8
```

The download script imports nothing from `frs`; it can run while you are still
editing configs, and training uses whatever `data.adapter.shards` names. The
scan writes `data/splits/glint360k_val_identities.txt` (excluded from training)
and `data/pairs/glint360k_val_pairs.txt` (the honest in-domain benchmark).
Commit both: they define the protocol.

### Standard benchmarks (`.bin` packs)

`lfw.bin`, `cfp_fp.bin`, `agedb_30.bin` are the InsightFace verification packs
that every published number is measured on. They are distributed alongside the
InsightFace training sets (see the `_datasets_` page of the insightface GitHub
repository; the MS1MV3 / Glint360K archives contain them). Copy them to
`/mnt/data/eval/` and the `type: bin` targets in the AWS config light up.

---

## 4. Config for a full-scale run

`configs/glint360k_ir100_adaface_aws.yaml` (abridged; the file is commented):

```yaml
_base_: base.yaml

experiment:
  name: glint360k_ir100_adaface
  seed: 3407

data:
  adapter:
    type: webdataset
    shards: "/mnt/data/glint360k/glint360k-{0000..1384}.tar.gz"
    min_images_per_identity: 2
    exclude_identities_file: data/splits/glint360k_val_identities.txt
    epoch_mode: resampled          # required for DDP (equal steps per rank)
    samples_per_epoch: null        # null -> one nominal pass (~17M)
    shuffle_buffer: 5000
  input_size: [112, 112]
  resize_policy: center_crop_112   # shards are already 112x112 (no-op)
  batch_size: 128                  # per GPU (A10G 24 GB); 256 on A100 40 GB
  num_workers: 10
  prefetch_factor: 6

model:
  backbone: {arch: ir_100, channels_last: true}
  head:
    type: adaface
    partial_fc: {enabled: auto, sample_rate: 0.1}   # on: 360k > 300k classes

optim: {type: sgd, lr: 0.2, momentum: 0.9, weight_decay: 5.0e-4}   # 0.1 x (512/256)
scheduler: {type: polylr, warmup_epochs: 2, power: 2.0}

train:
  epochs: 20
  amp: true
  save_every_n_steps: 2000         # spot-instance safety
  resume: auto
  torch_compile: true              # Linux only

eval:
  targets:
    - {name: glint_val, type: pairs, pair_file: data/pairs/glint360k_val_pairs.txt}
    - {name: lfw,       type: bin,   path: /mnt/data/eval/lfw.bin}
    - {name: cfp_fp,    type: bin,   path: /mnt/data/eval/cfp_fp.bin}
    - {name: agedb_30,  type: bin,   path: /mnt/data/eval/agedb_30.bin}
  primary: lfw
```

`configs/aws_ir100_adaface.yaml` is the equivalent for `.rec` packs (MS1MV3).

---

## 5. Launching

### Single GPU
```bash
python -m scripts.train --config configs/glint360k_ir100_adaface_aws.yaml
```

### Multi-GPU

```bash
torchrun --nproc_per_node=4 -m scripts.train --config configs/glint360k_ir100_adaface_aws.yaml
```

`scripts/train.py` initialises the process group from the `torchrun`
environment (`frs/utils/distributed.py`), wraps backbone and head in
`DistributedDataParallel`, splits the data per rank (a `DistributedSampler`
for map-style datasets; the shard stream itself for streaming ones) and lets
rank 0 do all logging, checkpointing and evaluation. Things to know:

- **LR scales with the *total* batch.** 4 GPUs × 128 = 512, so LR 0.2. The
  start-up banner prints the total batch.
- **Streaming needs `epoch_mode: resampled`.** Every rank and worker then yields
  exactly the same number of batches, which DDP needs to not deadlock.
  `natural` mode refuses to start under DDP.
- **Partial-FC under DDP** replicates the full classifier on every rank and
  all-reduces its (sparse-in-practice) gradient every step: ~737 MB at 360k
  classes. It works and is simple; a model-parallel head that shards the
  classifier across GPUs (the InsightFace design) is the next optimisation if
  scaling efficiency on 4-8 GPUs matters.
- **Leave `SyncBatchNorm` off.** At 128 per GPU the per-device statistics are
  already good, and SyncBN costs throughput.

### Keeping it alive across SSH drops
```bash
tmux new -s train
python -m scripts.train --config configs/aws_ir100_adaface.yaml
# Ctrl-B then D to detach; `tmux attach -t train` to return
```

---

## 6. Time and cost estimates

MS1MV3 (5.8 M images, 93 K identities), IR-100:

| Setup | Per epoch | 20 epochs | On-demand | Spot (~70% off) |
|---|---:|---:|---:|---:|
| 1× A10G (`g5.2xlarge`) | ~10 h | ~200 h | ~$242 | ~$73 |
| 4× A10G (`g5.12xlarge`) | ~2.7 h | ~55 h | ~$310 | ~$95 |
| 8× A100 (`p4d.24xlarge`) | ~0.5 h | ~10 h | ~$330 | ~$100 |

Costs land in the same range; what differs is **wall-clock**. Pick by how fast
you need the answer.

---

## 7. How many epochs

| Dataset | Epochs | Why |
|---|---:|---|
| MS1MV3 (5.8 M) | **20** | LFW saturates by epoch 8; IJB-C and CFP-FP keep improving to ~18-20 then flatten. |
| WebFace4M (4.2 M) | **20-24** | As above. |
| WebFace12M (12 M) | **20** | With Partial-FC enabled. |
| Budget-constrained | **16** | ~99% of final accuracy at 80% of cost. |

Past ~24 epochs you overfit the training identity distribution with no benchmark
gain.

---

## 8. Partial-FC (very large class counts)

The classifier weight is `num_classes × 512 × 4 bytes`, and SGD momentum doubles
it:

| Classes | FC weight | + momentum | Verdict |
|---|---:|---:|---|
| 1,710 (MeGlass) | 3.5 MB | 7 MB | Full FC, trivially |
| 93 K (MS1MV3) | 190 MB | 380 MB | Full FC fine on ≥16 GB |
| 360 K (WebFace4M) | 737 MB | 1.5 GB | Full FC OK on A100-40G, tight on 24 GB |
| 2 M (WebFace42M) | 4.1 GB | 8.2 GB | **Partial-FC required** |

`partial_fc.enabled: auto` turns it on above 300 K classes. It samples a subset
of negative class rows each step, so memory and compute scale with the sample
rate rather than the class count, at negligible accuracy cost.

---

## 9. Getting results back

```bash
# Checkpoints and report
aws s3 sync runs/ms1mv3_ir100_adaface/ s3://your-bucket/runs/ms1mv3_ir100_adaface/

# The deployment artifact -- backbone only, no head (~250 MB for IR-100)
aws s3 cp runs/ms1mv3_ir100_adaface/checkpoints/backbone_only.pt \
          s3://your-bucket/models/
```

Keep `best.pt` too — it is what you extend when new identities arrive later.

---

## 10. Pre-flight checklist

Before starting an expensive run:

- [ ] Validated the whole pipeline for an hour on a small instance
- [ ] `python -m scripts.train --config <cfg> --overfit 100` drives loss to ~0
- [ ] Data is packed and on instance-store NVMe, not EBS
- [ ] Streaming: census done (`scripts/scan_webdataset.py`), hold-out written,
      `epoch_mode: resampled` for multi-GPU
- [ ] `perf/data_time_frac` < 0.10 in the validation run
- [ ] Validation identities are excluded from training (or you are using standard
      `.bin` eval packs)
- [ ] `save_every_n_steps` set if on spot
- [ ] Checkpoint directory is on persistent storage, not instance store
- [ ] LR scaled to the *total* batch size across all GPUs
- [ ] `warmup_epochs: 2` at batch ≥ 512
- [ ] `torch_compile: true` (Linux only)
- [ ] Running under `tmux` or `nohup`
- [ ] A billing alarm is set
