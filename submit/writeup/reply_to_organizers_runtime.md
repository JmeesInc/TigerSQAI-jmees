# 主催者への返信ドラフト（実行時間 9h 超過の件）

宛先: tso-tiger-sqai-challenge@groups.tu-dresden.de (Anneli)
件名: Re: TIGER SQ-AI container runtime / model count — corrected submission

---

Dear Anneli,

Thank you very much for running our container and for the detailed feedback. It
caught a real defect on our side, and we have a corrected image.

**1. Is the write-up final?**

No, it was a draft. Appendix A in the version you have is wrong, and a corrected
write-up will come with the new image before the deadline.

**2. Which model count is accurate — 17 or 75?**

**The container log is accurate.** The image in your queue (`...:v6`) contains
**17 segmentation models and 15 Task-3 encoders**, and that is what ran.

The "75" in the write-up was the size of the **pool of trained candidates** we
selected from, not the ensemble deployed in the container. It was written into
the inference section by mistake. Our corrected image (`...:v7`) contains **16
segmentation models and 15 Task-3 fold encoders**, chosen by 5-fold
cross-validation, and the write-up now states the deployed ensemble and the
candidate pool separately.

**3. Can the runtime be brought under the limit? — Yes.**

The dominant cause is almost certainly that **the container ran on CPU**. Our
entry point silently falls back to CPU when `torch.cuda.is_available()` is
false, and we measured the difference directly:

| execution | one member, one frame |
|---|---|
| GPU (Quadro RTX 8000, fp16) | 0.10 - 0.14 s |
| CPU (16 threads) | **4.01 s** |

32 models x 139 frames x 4.0 s is 4.9 h of forward passes alone; with model
loading, post-processing and a slower host CPU this reaches the ~9 h you
measured. On a GPU the same v6 configuration needs roughly 25-30 minutes. We
could never verify this ourselves because our own host is missing
`nvidia-container-runtime`, so every container test we ran was on CPU. That was
our mistake.

Three further defects made the old code fragile, each the opposite of the advice
in your instructions, and all are fixed:

| Problem | v6 | v7 |
|---|---|---|
| Frame decoding | model-outer / frame-inner loop, so every PNG was decoded **32 times** | frame-outer, chunked: decoded **once** |
| Memory | softmax of **all** frames buffered on the host: 576x1024 x (31+16) ch float32 = 110 MB/frame, **~15 GB for 139 frames** | accumulated on the GPU for a 16-frame chunk only; resident memory is independent of test-set size |
| Output | written only after the whole loop, so a timeout produced **nothing** | PNGs and `task3.csv` written and flushed after every chunk |

We also build each network architecture once (the members share only 7 distinct
architectures) and swap members by copying fp16 weights into the existing
parameter storage, and we added an explicit **wall-clock budget controller**:
the container measures its own throughput on the first three frames and then,
before each chunk, sizes the ensemble to `remaining_time / remaining_frames`.
Members are ordered by cross-validation rank, so a tight budget drops the
weakest ones first, and if no GPU is visible the ensemble shrinks to a single
model and the run still completes. The container therefore cannot exceed the
budget.

Measured end to end on one Quadro RTX 8000 with 139 frames at the resolution mix
of the training data (101 x 1080p, 29 x 4K, 9 x 720p), 16 segmentation members
and 10 Task-3 encoders: **about 33 minutes**, of which 12 minutes is the one-off
cold read of the weights.

**Our questions back to you**

To confirm we fixed the right thing, could you check the log you kept?

1. **Did the container see a GPU?** The very first line the v6 image prints is
   literally `[HH:MM:SS] device=cuda` or `[HH:MM:SS] device=cpu`, immediately
   followed by `[HH:MM:SS] /app/model: 17 models across 17 members` (the line you
   quoted, which confirms the image was v6). If the first line says `device=cpu`,
   that confirms the diagnosis above. We re-ran the exact submitted image here to
   check that this line is present and is the first thing in the log.
2. **Was the image started once, or more than once?** We would like to know
   whether the ~9 h is a single pass over the 139 frames or includes retries or
   queueing.
3. **How much host RAM was available to the container?** With the old code the
   process needed about 15 GB for 139 frames; below that it would have been
   swapping. The new code stays well under that regardless of test-set size.

One clarification on the budget: we read §4 as **1 min x n_frames x n_tasks**,
so 417 min for 139 frames with a three-task image. Please correct us if the
intended budget is the flat 420 min.

Thank you again for catching this before the deadline.

Best regards,
Shunsuke Kikuchi (Team Jmees)
