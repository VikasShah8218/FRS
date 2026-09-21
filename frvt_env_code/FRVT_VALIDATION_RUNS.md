# ESSI-FR v1: FRVT 1:1 local validation runs

These are three runs of NIST's official `run_validate_11.sh` with the ESSI-FR v1 recognition model. **Bold** marks a Run 3 value that differs from Run 2.

| | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| **What changed** | first clean run | same code and models, different machine | **face detector swapped** |
| Machine | VPS `shah-ho-vps` | AWS g5.xlarge | AWS g5.xlarge |
| Face detector | InsightFace SCRFD | InsightFace SCRFD | **YuNet** |
| Result | ✅ PASS | ✅ PASS | ✅ PASS |

## 1. Setup

| | Run 1: VPS · SCRFD | Run 2: AWS · SCRFD | Run 3: AWS · YuNet |
|---|---|---|---|
| Date (IST) | 18 Sep 2026, 23:23 – 23:59 | 19 Sep 2026, 11:55 – 12:05 | 19 Sep 2026, 12:34 – 12:38 |
| CPU | QEMU virtual CPU, 10 vCPU | AMD EPYC 7R32, 4 vCPU | AMD EPYC 7R32, 4 vCPU |
| RAM | 15 GB | 15 GB | 15 GB |
| OS the script ran on | Ubuntu 24.04.3 LTS (native) | Ubuntu 24.04.3 LTS (Docker `essi-frvt11:24.04.3`; host is 26.04.1) | same as Run 2 |
| Compiler / CMake | GCC 13.3.0 / 3.28.3 | GCC 13.3.0 / 3.28.3 | GCC 13.3.0 / 3.28.3 |
| Compute | CPU, 1 thread | CPU, 1 thread | CPU, 1 thread |
| Detector file | `det_10g.onnx` (16.9 MB) | `det_10g.onnx` (16.9 MB) | **`face_detection_yunet_2023mar.onnx` (232 KB)** |
| Detector source / licence | InsightFace SCRFD-10G, non-commercial research | same | **OpenCV Zoo YuNet, MIT** |
| Detector SHA256 | `5838f7fe…` | `5838f7fe…` | **`8f2383e4…`** |
| Detector input | RGB, (px − 127.5) / 128 | same | **BGR, raw 0–255** |
| Detector scales | 1 pass at 640 | 1 pass at 640 | **3 passes (640 / 320 / 160), merged** |
| Score threshold / NMS | 0.5 / 0.4 | 0.5 / 0.4 | **0.6 / 0.3** |
| Recognition model | `essi_fr_v1_last.onnx` (`68726933…`) | same | same |
| Alignment | 5-point, 112×112 | same | same |
| Runtime | ONNX Runtime 1.20.1 | same | same |
| Library | `libfrvt_11_essi_000.so` (80,880 B) | same (80,880 B) | **85,456 B** |
| Source changes | — | none (only the ONNX Runtime path in `CMakeLists.txt`) | **`face_engine.cpp/.h` detection rewritten; comment in `essiimplfrvt11.cpp`** |
| Package `config/` | `det_10g.onnx`, `essi_fr_v1_last.onnx` | same | **`face_detection_yunet_2023mar.onnx`, `essi_fr_v1_last.onnx`** |
| Package `doc/` | `version.txt` | `version.txt` | **`version.txt`, `THIRD_PARTY_NOTICES.txt`** |
| Package size | 186.3 MB | 186.3 MB | **170.8 MB** |

## 2. NIST checks

| Check | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| OS version (24.04.3) | ✅ | ✅ | ✅ |
| No hard-coded config path | ✅ | ✅ | ✅ |
| Single-threaded (no warning) | ✅ | ✅ | ✅ |
| Works with `fork()` (4 processes) | ✅ | ✅ | ✅ |
| Multi-image and multi-person templates | ✅ | ✅ | ✅ |
| ≥ 50% unique match scores | ✅ 614/653 | ✅ 614/653 | ✅ **651/653** |
| No negative scores | ✅ | ✅ | ✅ |
| Exit code / package created | 0 / ✅ | 0 / ✅ | 0 / ✅ |

## 3. Face detection

| | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| Enroll: no face found (of 665) | 29 | 29 | **2** |
| · NIST test images¹ | 5 | 5 | **2** (white, black) |
| · ordinary close-up photos | 24 | 24 | **0** |
| Verif: no face found (of 653) | 15 | 15 | **0** |
| Multi-person: templates written | 34 | 34 | **28** |
| Multi-person: images with no face | 5 | 5 | **2** (white, black) |
| Group photos `multi1` / `multi2` / `multi3`: faces found² | 2 / 5 / 2 | 2 / 5 / 2 | **1 / 3 / 2** |
| Eye positions | — | identical to Run 1 | **median 3.5% of face width from Run 2** |

## 4. Matching

Scores run 0–100, where score = (cosine + 1) × 50.

| | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| Pairs usable (of 653) | 614 | 614 | **653** |
| · same person / different person | 367 / 247 | 367 / 247 | **389 / 264** |
| · with a failed template (score 50) | 39 | 39 | **0** |
| Same person: lowest / mean score | 60.39 / 88.10 | 60.39 / 88.10 | **62.86 / 87.66** |
| Different person: highest / mean score | 65.58 / 51.18 | 65.58 / 51.18 | **65.08 / 51.22** |
| Best accuracy (at score) | 99.84% (66.0) | 99.84% (66.0) | **99.85% (66.4)** |
| Same-person pairs below the highest different-person score | 1 | 1 | 1 |
| Largest score difference vs Run 1 | — | 0.0002 | — |
| Template size | 2,048 B (512 floats) | 2,048 B | 2,048 B |

## 5. Time (min:sec)

| Step | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Hard-coded config check (10 images) | 0:22 | 0:05 | 0:03 |
| Enroll, 1 process (665 images) | 22:04 | 4:59 | **2:06** |
| Enroll, multiple images per subject (8) | 1:00 | 0:14 | 0:05 |
| Enroll, 4 processes (665) | 5:37 | 2:15 | 0:55 |
| Verif, 4 processes (653) | 5:35 | 2:13 | 0:55 |
| Match, 4 processes (653) | 0:01 | 0:01 | 0:00 |
| Verif, multi-person | 0:48 | 0:11 | 0:06 |
| Match, multi-person | 0:01 | 0:01 | 0:00 |
| Create package | 0:11 | 0:09 | 0:09 |
| **Total** | **35:45** | **10:13** | **4:23** |
| Time per image, 1 process | 1.99 s | 0.45 s | **0.19 s** |

## 6. Where to find each run

| | Logs | Package |
|---|---|---|
| Run 1 | VPS `FRS_NIST/frvt_logs/frvt11_essi_fr_v1_20260918_232320.*` | VPS `FRS_NIST/frvt/11/libfrvt_11_essi_000.v3.1.tar.gz` |
| Run 2 | AWS `~/frvt_logs/frvt11_essi_fr_v1_20260919_062523.*` and `~/frvt_logs/baseline_scrfd_20260919/` | AWS `~/frvt_previous_run_backup_20260919_070224/11/` |
| Run 3 | AWS `~/frvt_logs/frvt11_essi_fr_v1_20260919_070424.*` | AWS `~/frvt/11/libfrvt_11_essi_000.v3.1.tar.gz` |

---
¹ NIST's deliberate test images: `white.ppm`, `black.ppm`, `8bit.pgm` (greyscale), `rotated180.ppm`, `small.ppm` (7×9 px). Only white and black contain no face.
² Distinct faces. NIST's input lists each group photo twice, so the log has two rows per face. YuNet skips people who are partly hidden in the background; in `multi1` that person scored 0.52, just under the 0.6 threshold.
