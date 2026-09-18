# ESSI-FR v1 — Training Progress Log

**Run:** `configs/essi_fr_v1_aws_1gpu.yaml`, started 13 September 2026, 06:44 UTC
**Hardware:** AWS g5.xlarge, 1x NVIDIA A10G 24 GB
**Model:** IResNet-50 + AdaFace + Partial-FC at 10% sampling, 43.6M parameters
**Data:** Glint360K, 17,030,462 images, 359,232 identities, 1,000 identities held out
**Schedule:** SGD, momentum 0.9, batch 192, peak LR 0.02, 1 warm-up epoch, polynomial decay over 20 epochs
**Speed:** 789 images/sec, 6 hours 4 minutes per epoch, 88,698 steps per epoch

All times are server time (UTC). Add 5 hours 30 minutes for IST.

---

## Status

**Epoch 16 in progress**, 21% through as of 17 Sep 09:18 UTC. Epochs 0 to 15 are
done, the run is **81% complete**, and it is on track to finish **18 September at
about 08:20 UTC, 13:50 IST**. Epochs 7 through 15, nine in a row, each finished
at exactly the predicted minute.

Training keeps improving: loss has fallen every single epoch, now 1.22, and
training accuracy has risen every epoch, now 75.59%. **Epoch 14 holds the run
records: 99.950% accuracy, only 3 wrong pairs out of 6,000, TAR@1e-3 of 99.97%,
and the tightest fold spread at 0.076**. `best.pt` holds that model. Epoch 15
scored 6 wrong pairs, which is within the noise of this test.

---

## Epoch summary

`glint_val` is verification on 6,000 face pairs from 1,000 people the model has
never trained on. "Wrong pairs" is out of 6,000.

| Epoch | Avg loss       | Avg train acc    | `glint_val`     | Wrong pairs | TAR@1e-3         | std             | AUC    | LR at end | `wnorm` at end | Best? |
| ----- | -------------- | ---------------- | ----------------- | ----------- | ---------------- | --------------- | ------ | --------- | ---------------- | ----- |
| 0     | 22.19          | 4.18%            | 98.150%           | 111         | 92.40%           | 0.411           | 0.9945 | 0.02000   | 0.2612           | yes   |
| 1     | 7.53           | 20.51%           | 99.567%           | 26          | 99.17%           | 0.213           | 0.9998 | 0.01795   | 0.3034           | yes   |
| 2     | 4.98           | 32.38%           | 99.767%           | 14          | 99.50%           | 0.170           | 0.9999 | 0.01601   | 0.3080           | yes   |
| 3     | 4.17           | 38.91%           | 99.667%           | 20          | 99.43%           | 0.236           | 0.9999 | 0.01418   | 0.3054           | no    |
| 4     | 3.66           | 43.72%           | 99.700%           | 18          | 99.67%           | 0.145           | 0.9998 | 0.01247   | 0.3009           | no    |
| 5     | 3.31           | 47.39%           | 99.817%           | 11          | 99.83%           | 0.138           | 0.9997 | 0.01086   | 0.2959           | yes   |
| 6     | 3.01           | 50.72%           | **99.883%** | **7** | 99.70%           | **0.107** | 0.9998 | 0.00936   | 0.2908           | yes   |
| 7     | 2.76           | 53.59%           | **99.883%** | **7** | 99.83%           | 0.150           | 0.9999 | 0.00798   | 0.2861           | tie   |
| 8     | 2.51           | 56.68%           | 99.867%           | 8           | 99.90%           | 0.125           | 0.9999 | 0.00670   | 0.2818           | no    |
| 9     | 2.29           | 59.48%           | 99.867%           | 8           | 99.90%           | 0.180           | 0.9998 | 0.00554   | 0.2777           | no    |
| 10    | 2.09           | 62.25%           | 99.900%           | 6           | 99.93%           | 0.111           | 0.9999 | 0.00449   | 0.2738           | yes   |
| 11    | 1.89           | 65.10%           | 99.933%           | 4           | **99.97%** | 0.082           | 0.9999 | 0.00355   | 0.2702           | yes   |
| 12    | 1.71           | 67.68%           | 99.833%           | 10          | 99.90%           | 0.167           | 0.9998 | 0.00271   | 0.2666           | no    |
| 13    | 1.54           | 70.35%           | 99.917%           | 5           | 99.93%           | 0.134           | 0.9999 | 0.00199   | 0.2632           | no    |
| 14    | 1.38           | 72.96%           | **99.950%** | **3** | **99.97%** | **0.076** | 0.9999 | 0.00138   | 0.2604           | yes   |
| 15    | **1.22** | **75.59%** | 99.900%           | 6           | 99.93%           | 0.133           | 0.9999 | 0.00089   | 0.2583           | no    |
| 16    | in progress    | 76 to 79%        |                   |             |                  |                 |        | 0.00080   | 0.2579           |       |

---

## Epoch by epoch

### Epoch 0 — finished 13 Sep 13:03 UTC

`glint_val` **98.150%**, 111 wrong pairs, TAR@1e-3 92.40%

The first honest read on the rebuilt pipeline, and it immediately beat the
failed run's 97.22%. Loss fell from 40 to about 10.5 by the end of the epoch,
and training accuracy reached 13 to 15%. `fnorm` dipped to about 8 early on,
meaning feature lengths were very uneven between faces, then recovered to its
natural ceiling of 22.5. `wnorm` grew from 0.226 to 0.261, the opposite of the
collapse that destroyed the previous run.

### Epoch 1 — finished 13 Sep 19:07 UTC

`glint_val` **99.567%**, 26 wrong pairs, TAR@1e-3 99.17%

The decisive epoch. The failed run collapsed here, falling to 91.8%. This run
jumped instead: wrong pairs dropped from 111 to 26 and TAR@1e-3 went from 92.4%
to 99.2%, confirming both fixes worked. Warm-up ended and the learning rate hit
its 0.02 peak, after which it only decays. Loss more than halved, 7.53 against
22.19.

### Epoch 2 — finished 14 Sep 01:11 UTC

`glint_val` **99.767%**, 14 wrong pairs, TAR@1e-3 99.50%

Errors halved again. `wnorm` peaked here at 0.3080 and began easing afterwards,
which is the expected response to the decaying learning rate. Training accuracy
passed 32%.

### Epoch 3 — finished 14 Sep 07:15 UTC

`glint_val` 99.667%, 20 wrong pairs, TAR@1e-3 99.43%

The first epoch with no new best. Accuracy slipped by 6 pairs out of 6,000,
which is inside the noise of this test, with a fold spread of about 4 pairs.
Training itself never wavered: loss 4.17 against 4.98, accuracy 38.91% against
32.38%. Reading this as a regression would have been wrong.

### Epoch 4 — finished 14 Sep 13:19 UTC

`glint_val` 99.700%, 18 wrong pairs, TAR@1e-3 99.67%

Again no new best on accuracy, yet TAR@1e-3 reached its highest value so far
and the fold spread fell to 0.145. The two metrics disagreeing is the clearest
sign that differences of a few pairs at this level are noise, not signal.

### Epoch 5 — finished 14 Sep 19:23 UTC

`glint_val` **99.817%**, 11 wrong pairs, TAR@1e-3 **99.83%**

New best. The plateau broke. Loss dropped below the 3.3 mark that had looked
stuck, and training accuracy passed 47%. TAR@1e-3 reached its highest value of
the run, meaning only about 5 genuine pairs in 3,000 were rejected at a strict
threshold.

### Epoch 6 — finished 15 Sep 01:27 UTC

`glint_val` **99.883%**, 7 wrong pairs, TAR@1e-3 99.70%

New best again. Errors are down to 7 pairs out of 6,000, and consistency reached
its best value yet at 0.107. Loss broke below 3. The small TAR@1e-3 dip from
99.83% is about 4 pairs out of 3,000, well inside noise.

### Epoch 7 — finished 15 Sep 07:31 UTC

`glint_val` **99.883%**, 7 wrong pairs, TAR@1e-3 **99.83%**

Tied epoch 6 exactly at 7 wrong pairs, so no new `best.pt` was written: a new
best needs a strictly higher score. TAR@1e-3 returned to the run high of 99.83%
and AUC reached 0.9999. The fold spread widened slightly to 0.150 from 0.107,
which is noise at this error count. Loss fell another 8% to 2.76, in line with
the 9 to 10% drops of the two epochs before, and late-epoch lines touched loss
2.26 with 59.6% accuracy. The epoch finished at the exact minute predicted two
days earlier, confirming the timing is rock steady.

### Epoch 8 — finished 15 Sep 13:35 UTC

`glint_val` 99.867%, 8 wrong pairs, TAR@1e-3 **99.90%**

No new best on accuracy: one wrong pair more than the epoch 6 record, which is
noise. But TAR@1e-3 hit the highest value of the run. At the strict threshold,
only about 3 of 3,000 genuine pairs were rejected. The fold spread tightened to
0.125. Loss fell another 9% to 2.51, training accuracy rose to 56.68%, and
late-epoch lines touched loss 2.12 with 61.7% accuracy.

This epoch is the clearest example of why `best.pt` should not choose the
release model: it still holds epoch 6, although epoch 8 is better on the stricter
metric. It finished at the predicted minute again.

### Epoch 9 — finished 15 Sep 19:39 UTC

`glint_val` 99.867%, 8 wrong pairs, TAR@1e-3 99.90%

A repeat of epoch 8 on the validation set, same 8 wrong pairs and same TAR@1e-3,
while training kept improving underneath: loss down another 9% to 2.29 and
accuracy up to 59.48%. The fold spread widened to 0.180, which at 8 errors is
the difference of a single pair moving between folds.

### Epoch 10 — finished 16 Sep 01:43 UTC

`glint_val` **99.900%**, **6 wrong pairs**, TAR@1e-3 **99.93%**

The best epoch of the run on every measure. Errors fell to 6 out of 6,000, a new
low. TAR@1e-3 reached 99.93%, meaning only about 2 of 3,000 genuine pairs were
rejected at the strict threshold. AUC 0.9999 and the fold spread back to 0.111.

`best.pt` was rewritten and now holds the epoch 10 model, replacing epoch 6.
Training loss fell another 9% to 2.09 and accuracy reached 62.25%, with
late-epoch lines touching loss 1.94 and 64% accuracy. The learning rate is now
0.0045, under a quarter of its peak, which is the region where the loss usually
falls fastest.

### Epoch 11 — finished 16 Sep 07:47 UTC

`glint_val` **99.933%**, **4 wrong pairs**, TAR@1e-3 **99.97%**

Better than epoch 10 on every measure, and a new `best.pt`. Errors are down to 4
out of 6,000. TAR@1e-3 of 99.97% means roughly 1 genuine pair in 3,000 is
rejected at the strict threshold. The fold spread reached its lowest value of
the run, 0.082, so the result is also the most consistent so far.

Training loss fell nearly 10% to 1.89 and accuracy reached 65.10%. The fifth
epoch in a row to finish at the exact predicted minute.

### Epoch 12 — finished 16 Sep 13:51 UTC

`glint_val` 99.833%, 10 wrong pairs, TAR@1e-3 99.90%

Training improved as usual: loss down 9.3% to 1.71 and accuracy up to 67.68%,
with late-epoch lines around 70%. Validation went the other way, from 4 wrong
pairs to 10.

Is that a problem? Almost certainly not. At counts this small the random spread
on 7 errors is plus or minus about 2.6 pairs, so 4 and 10 are roughly two
standard deviations apart, and the same pattern appeared at epoch 3, which
recovered immediately. TAR@1e-3 only moved from 99.97% to 99.90%, about 2 pairs
in 3,000, far inside the 0.3% warning threshold. The one thing to watch is
whether epoch 13 recovers. Two consecutive weak epochs would be worth a second
look.

### Epoch 13 — finished 16 Sep 19:55 UTC

`glint_val` 99.917%, 5 wrong pairs, TAR@1e-3 99.93%

The recovery the epoch 12 note was waiting for. Wrong pairs fell from 10 back to
5, and TAR@1e-3 rose to 99.93%, confirming the dip was noise. Training loss fell
10% to 1.54 and accuracy passed 70%, reaching 70.35%. The last line of the epoch
touched loss 1.24 with 74.9% accuracy. No new `best.pt`, since 5 errors does not
beat epoch 11's 4.

### Epoch 14 — finished 17 Sep 01:59 UTC

`glint_val` **99.950%**, **3 wrong pairs**, TAR@1e-3 **99.97%**

The best epoch of the run. Only 3 wrong pairs out of 6,000, a new low. TAR@1e-3
tied the run high at 99.97%, about 1 rejected genuine pair in 3,000. The fold
spread fell to 0.076, the most consistent result so far, which means the 3 errors
were scattered one per fold rather than clustered.

`best.pt` was rewritten and now holds epoch 14. Training loss fell another 10.5%
to 1.38, the largest relative drop since epoch 5, and accuracy reached 72.96%.
This is the accelerating late-training phase the schedule was built for: the
learning rate is now 0.0014, under 7% of its peak.

### Epoch 15 — finished 17 Sep 08:03 UTC

`glint_val` 99.900%, 6 wrong pairs, TAR@1e-3 99.93%

Training had its strongest relative gain of the run: loss down 11.4% to 1.22 and
accuracy up to 75.59%, with late-epoch lines steady around 76%. Validation
slipped from 3 wrong pairs to 6 and TAR@1e-3 by a single pair, from 99.97% to
99.93%. No new `best.pt`; epoch 14 remains the record.

This follows the pattern of the whole second half: the last five epochs scored
4, 10, 5, 3 and 6 wrong pairs while training loss fell every single time. The
validation count is bouncing around its floor, not trending. The ninth
consecutive epoch to finish at the exact predicted minute.

### Epoch 16 — in progress

At 21% on 17 Sep 09:18 UTC, the loss runs between 1.03 and 1.16 with accuracy
between 76.3 and 79.3%, already ahead of epoch 15. On schedule to finish
17 Sep 14:07 UTC, 19:37 IST.

---

## Stability checks

These are the numbers that caught the earlier failure. All are healthy.

| Check                            | Reading                                                                                                 | Verdict                                              |
| -------------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `wnorm`, classifier row size   | 0.3080 peak, easing to 0.2579 now, with the per-epoch drop slowing from 0.0045 to about 0.002           | Steady and decelerating. The failed run fell 30x.    |
| Effective step, LR ÷`wnorm`² | 0.293, 0.195, 0.169, 0.152, 0.138, 0.124, 0.111, 0.098, 0.084, 0.072, 0.060, 0.049, 0.038, 0.029, 0.020, 0.013 | Falling every epoch, so updates keep getting gentler |
| `fnorm`, feature strength      | 22.5 to 22.6 since epoch 1                                                                              | At its natural ceiling of √512                      |
| Learning rate                    | Matches the formula to 5 decimals at every check                                                        | Schedule exact                                       |
| Throughput                       | 789 img/s, data stall 0.1%                                                                              | GPU fully fed, no bottleneck                         |

---

## Incidents

| When         | What                                                                                                                                | Effect                                                                  |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| 12 Sep       | Previous run degraded, 97.2% then 91.8%. Two bugs: learning rate 4x too high, and weight decay collapsing unsampled classifier rows | Run abandoned, both fixed, restarted from scratch                       |
| 13 Sep 06:43 | Ubuntu auto-updates restarted the training service three times                                                                      | 3 minutes lost. Fixed with an updater exclusion and`KillMode=process` |
| 13 Sep 07:19 | Last SSH session closed, Linux deleted the shared memory PyTorch uses                                                               | Crash at step 7,700. Fixed with`RemoveIPC=no` and user lingering      |
| 13 Sep 07:23 | Resumed from the step-6,000 checkpoint, inside epoch 0                                                                              | 7 minutes lost, loss and learning rate continuous                       |

No incidents since 13 Sep 07:23. The run has been clean for over 97 hours.

---

## Remaining schedule

| Epoch               | Completes, IST                               |
| ------------------- | -------------------------------------------- |
| 7                   | Done, Tue 15 Sep 13:01, exactly as predicted |
| 8                   | Done, Tue 15 Sep 19:05, exactly as predicted |
| 9                   | Done, Wed 16 Sep 01:09, exactly as predicted |
| 10                  | Done, Wed 16 Sep 07:13, exactly as predicted |
| 11                  | Done, Wed 16 Sep 13:17, exactly as predicted |
| 12                  | Done, Wed 16 Sep 19:21, exactly as predicted |
| 13                  | Done, Thu 17 Sep 01:25, exactly as predicted |
| 14                  | Done, Thu 17 Sep 07:29, exactly as predicted |
| 15                  | Done, Thu 17 Sep 13:33, exactly as predicted |
| 16                  | Thu 17 Sep, 19:37                            |
| 17                  | Fri 18 Sep, 01:41                            |
| 18                  | Fri 18 Sep, 07:45                            |
| **19, final** | **Fri 18 Sep, 13:49**                  |

---

## What to expect, and what to watch

**The learning rate is 4% of its peak** and the loss fell 11.4% in epoch 15, the
fastest relative drop of the run. With four epochs left, the estimate for epoch
19 holds at **loss around 0.8 to 1.0 and training accuracy around 80 to 86%**.
Gains will taper in the last epoch or two as the learning rate approaches zero.

**`glint_val` has all but run out of room.** The last five epochs scored 4, 10,
5, 3 and 6 wrong pairs out of 6,000. That sequence is exactly what noise looks
like at this level: a handful of hard pairs decides the number. This test can no
longer show how much the model is still improving.

**Warning signs**, none present so far:

- Epoch-average loss rising two epochs in a row
- `wnorm` falling faster each epoch, or the effective step rising
- TAR@1e-3 dropping by more than about 0.3%

---

## Open items

1. **Get the standard benchmark packs onto the server:** LFW, CFP-FP and
   AgeDB-30. They test pose and age variation that `glint_val` barely covers,
   and they are the only way to measure the real gains over the final epochs.
2. **Don't release `best.pt` by default.** It is selected on `glint_val`
   accuracy alone, which is nearly saturated. It currently holds epoch 14, which
   ties the best TAR@1e-3, so the two agree for now. They disagreed at epochs 6
   to 8, and at 3 wrong pairs a single pair can decide which epoch is kept. `backbone_only.pt`, written from the final epoch, is expected to be the
   better model. Choose the release model on the stricter tests.
3. **For NIST FRVT 1:1**, report FNMR at FMR of 1e-5 or 1e-6. The 3,000
   impostor pairs here cannot go below about 1e-3, but the 17,715 held-out
   images can form roughly 150 million impostor pairs, which is enough.
