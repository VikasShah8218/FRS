# ESSI-FR v1: FRVT 1:1 setup and local validation

This guide takes a fresh clone of NIST's official `frvt` repo, adds the ESSI-FR v1 implementation from this folder, and runs NIST's 1:1 validation. The result is a submission package.

**Assumes the AWS layout:** user `ubuntu` (uid 1000) and everything under `/home/ubuntu`. For a different machine or user, see [Different machine or user](#different-machine-or-user).

The run is **CPU only**, because NIST 1:1 is evaluated on CPU. It uses **Docker**, because NIST's script refuses any OS except the one it names (currently Ubuntu 24.04.3 LTS).

---

## What goes where

| Source | Destination |
|---|---|
| `frvt_env_code/11/src/yourImpl/` (this folder) | `~/frvt/11/src/yourImpl/` |
| `frvt_env_code/11/doc/` | `~/frvt/11/doc/` |
| `frvt_env_code/frvt_env/` | `~/frvt_env/` |
| `essi_fr_v1_last.onnx` (recognition model) | `~/frvt/11/config/` |
| `face_detection_yunet_2023mar.onnx` (face detector) | `~/frvt/11/config/` |
| ONNX Runtime 1.20.1 `libonnxruntime.so*` | `~/frvt/11/lib/` |

When you're done, the layout is:

```
~/frvt/                              <- NIST repo (git clone)
  11/
    src/yourImpl/                    <- our code (5 files)
    doc/version.txt                  <- our version info
    doc/THIRD_PARTY_NOTICES.txt      <- MIT licences (YuNet, ONNX Runtime)
    config/essi_fr_v1_last.onnx
    config/face_detection_yunet_2023mar.onnx
    lib/libonnxruntime.so*           <- copied from ONNX Runtime
    lib/libfrvt_11_essi_000.so       <- built in step 6
~/frvt_env/                          <- Dockerfile + run scripts
~/onnxruntime-linux-x64-1.20.1/      <- ONNX Runtime (headers + libs)
~/frvt_logs/                         <- console logs of each run
```

---

## 1. Get the NIST repo

```bash
cd ~ && git clone https://github.com/usnistgov/frvt.git       # first time
cd ~/frvt && git pull                                         # later updates

grep reqOS= ~/frvt/common/scripts/utils.sh                    # OS NIST requires
```

Our files are untracked in this repo, so `git pull` never touches them. If `reqOS` has changed, update the Docker base image in step 5.

## 2. Copy this folder to the server

From the Windows PC (PowerShell):

```powershell
scp -i "D:\Drive-1\AWS_Keys\ESSI-FRS\Shah-Access.pem" -r "D:\Drive-1\Facial Rec\FRS\frvt_env_code" ubuntu@16.16.41.227:~/
```

If `frvt_env_code` is committed to the FRS repo instead, run `git pull` in `~/FRS` and use `~/FRS/frvt_env_code` below.

## 3. Add our code to the NIST repo

```bash
SRC=~/frvt_env_code
cp -r $SRC/11/src/yourImpl ~/frvt/11/src/
mkdir -p ~/frvt/11/doc && cp $SRC/11/doc/* ~/frvt/11/doc/
mkdir -p ~/frvt_env    && cp $SRC/frvt_env/* ~/frvt_env/

# Remove Windows line endings (they break shell scripts), and make the scripts executable
sed -i 's/\r$//' ~/frvt/11/src/yourImpl/*.* ~/frvt/11/doc/* ~/frvt_env/*
chmod +x ~/frvt_env/*.sh
```

## 4. Models and ONNX Runtime

```bash
mkdir -p ~/frvt/11/config ~/frvt/11/lib

# Recognition model (from the training run)
cp ~/FRS/runs/essi_fr_v1/export/essi_fr_v1_last.onnx ~/frvt/11/config/

# Face detector: YuNet, OpenCV Zoo, MIT licence
curl -L -o ~/frvt/11/config/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

sha256sum ~/frvt/11/config/*.onnx
# 6872693382e27a98933163021e46076cc7c2f22d4ccbf7b97b9ff6d5c92e1646  essi_fr_v1_last.onnx
# 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4  face_detection_yunet_2023mar.onnx

# ONNX Runtime 1.20.1 (CPU)
cd ~ && curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz | tar xz
cp -a ~/onnxruntime-linux-x64-1.20.1/lib/libonnxruntime.so* ~/frvt/11/lib/
```

`config/` must contain **only** these two `.onnx` files, because NIST packages the whole folder.

## 5. Build the Docker image (once)

```bash
docker build -t essi-frvt11:24.04.3 ~/frvt_env
docker run --rm essi-frvt11:24.04.3 lsb_release -ds      # must print: Ubuntu 24.04.3 LTS
```

The output must match `reqOS` from step 1. If NIST ever requires a newer release, change the `FROM ubuntu:noble-…` line in `~/frvt_env/Dockerfile` to an image of that release.

## 6. Build our library (inside the container)

```bash
cd ~/frvt/11 && rm -rf src/yourImpl/build lib/libfrvt_11_essi_000.so
docker run --rm --user 1000:1000 -e HOME=/tmp \
  -v /home/ubuntu/frvt:/home/ubuntu/frvt \
  -v /home/ubuntu/onnxruntime-linux-x64-1.20.1:/home/ubuntu/onnxruntime-linux-x64-1.20.1:ro \
  essi-frvt11:24.04.3 bash -c \
  'mkdir -p /home/ubuntu/frvt/11/src/yourImpl/build && cd /home/ubuntu/frvt/11/src/yourImpl/build && cmake .. && make'

ls -l ~/frvt/11/lib        # libfrvt_11_essi_000.so + libonnxruntime.so*
```

Build inside the container rather than with the host compiler. NIST's machine has Ubuntu 24.04's GCC 13, and a library built with a newer compiler can fail to load there.

## 7. Run the validation

```bash
bash ~/frvt_env/start_frvt_tmux.sh     # starts tmux session "frvt"
tmux attach -t frvt                    # watch it; Ctrl-B then D to leave it running
```

It takes about **4.5 min** on a g5.xlarge. The last line shows `=== finished … exit code 0 …`. Console and step-time logs are saved in `~/frvt_logs/`.

---

## Check the result

```bash
cd ~/frvt/11
for f in enroll verif verif_multiperson match match_multiperson; do
  echo "$f: $(sed 1d validation/$f.log | wc -l) rows, $(sed 1d validation/$f.log | awk '$4!=0' | wc -l) non-zero"
done
cat validation/os.txt; ls -l libfrvt_11_essi_*.tar.gz
```

| Log | Rows | Non-zero return codes (expected) |
|---|---:|---|
| `enroll.log` | 665 | 2: `white.ppm`, `black.ppm` (blank test images; correct) |
| `verif.log` | 653 | 0 |
| `verif_multiperson.log` | 28 | 2: the same blank images |
| `match.log` | 653 | 0 |
| `match_multiperson.log` | 28 | 0 |

- `os.txt` should read `Ubuntu 24.04.3 LTS`.
- The package is `~/frvt/11/libfrvt_11_essi_000.v3.1.tar.gz`, about 171 MB. **Do not rename it.** Encrypt it as described in the instructions printed at the end of the run.

## Run again

```bash
tmux kill-session -t frvt              # close the previous session
bash ~/frvt_env/start_frvt_tmux.sh
```

NIST's script clears `validation/`, `bin/` and `build/` itself and overwrites the package. Move the old `.tar.gz` somewhere else first if you want to keep it. **If you changed any code, redo step 6 first.**

## Change the submission sequence number

Every package sent to NIST needs a new three-digit number (`000` → `001` → …):

```bash
cd ~/frvt/11
sed -i 's/frvt_11_essi_000/frvt_11_essi_001/g' src/yourImpl/CMakeLists.txt doc/version.txt doc/THIRD_PARTY_NOTICES.txt
rm -f lib/libfrvt_11_essi_000.so       # lib/ may hold only ONE libfrvt_11_* library
```

Then redo step 6, run again, and make the same change in `frvt_env_code` on the PC.

---

## Different machine or user

These are the only machine-specific values:

| File | Line | Value |
|---|---|---|
| `11/src/yourImpl/CMakeLists.txt` | 7 | `set(ORT_DIR /home/ubuntu/onnxruntime-linux-x64-1.20.1)` |
| `frvt_env/run_frvt11_essi.sh` | 6, 7 | `/home/ubuntu/frvt/11`, `/home/ubuntu/frvt_logs` |
| `frvt_env/start_frvt_tmux.sh` | 10, 16, 19–21 | `/home/ubuntu/frvt_env`, `/home/ubuntu/frvt_logs`, `--user 1000:1000` (your `id -u`:`id -g`), `/home/ubuntu/frvt` |
| Step 6 command | — | the same `/home/ubuntu/…` paths and `--user` |

**On a machine that already runs Ubuntu 24.04.3** (like the VPS), Docker isn't needed. Build natively and run NIST's script directly:

```bash
cd ~/frvt/11/src/yourImpl && mkdir -p build && cd build && cmake .. && make
cd ~/frvt/11 && ./run_validate_11.sh
```

## Troubleshooting

| Message | Fix |
|---|---|
| `You are not running the correct version of the operating system` | You ran the script on the host instead of through `start_frvt_tmux.sh`, or NIST changed `reqOS` (see step 5). |
| `Could not find core implementation library` | Step 6 wasn't run, or it failed. |
| `more than one libraries in …/lib` | Remove the extra `libfrvt_11_*` file from `lib/`. |
| `ConfigError` / `onnxruntime:` error in `enroll.log` | Model file names in `config/` must be exactly `essi_fr_v1_last.onnx` and `face_detection_yunet_2023mar.onnx`. |
| `tmux session 'frvt' already exists` | `tmux kill-session -t frvt` |
| `Permission denied` when editing `config/` | NIST's script leaves it read-only: `chmod 775 ~/frvt/11/config` |
| `permission denied … docker.sock` | `sudo usermod -aG docker $USER`, then log in again. |
| `GLIBCXX_… not found` | The library was built with the host compiler. Redo step 6. |
| `$'\r': command not found` | Windows line endings. Rerun the `sed` line from step 3. |
