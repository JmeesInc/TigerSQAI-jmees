# TIGER SQ-AI 2026: Submission Instructions

**Challenge:** AI-based surgical quality assessment for thoracic lymphadenectomy in minimally invasive esophagectomy
**Venue:** Endoscopic Vision Challenge (EndoVis), MICCAI 2026
**Platform:** Synapse.org
**Reference code:** https://gitlab.com/nct_tso_public/challenges/miccai2026/tigersqai_challenge

> Every submission is a Docker image that runs offline on our evaluation hardware. We mount one input folder holding the complete test set and one empty output folder. Your container starts once, processes all frames one after another, and writes its results into the output folder. This page describes exactly what goes in and what has to come out. Please validate your container against the sanity check data before you upload.

---

## 1. Tasks

| Task | Name | Output |
|---|---|---|
| 1 | Semantic segmentation, merged labels | One RGB PNG per frame, 15 merged classes |
| 2 | Semantic segmentation, full labels | One RGB PNG per frame, 30 fine-grained classes |
| 3 | Lymph node station visibility | One CSV file for the whole test set |

You may enter one, two, or all three tasks. Tasks 1 and 2 are ranked by Dice similarity coefficient and normalized Hausdorff distance. Task 3 is ranked by macro averaged F1 score and AUROC. Details are given in the challenge design document.

### One container per task, or one container for several tasks

**Both are allowed. Pick whichever suits your method.**

* **Separate images.** One image per task, each writing only its own output folder. Choose this if your models are independent, if you enter only some of the tasks, or if you want to iterate on one task without rebuilding the others.
* **One combined image.** A single image that covers two or all three tasks in one run. Choose this if your tasks share a backbone or an encoder, since you then pay for loading and for the forward pass once instead of two or three times.

The interface is identical in both cases. There is no advantage or disadvantage in the ranking either way, and combined submissions are not ranked against separate ones. Each task is ranked on its own.

**How we know which tasks an image covers:** by the paths it creates under `/output`. An image writes `task1/`, `task2/`, and `task3.csv` for the tasks it handles, and creates nothing for the tasks it skips. An absent path means the image does not enter that task. An empty or incomplete folder counts as a failed submission for that task, not as a skipped one.

**One image per task, at most.** For every task you enter, exactly one image must be responsible. Do not submit a combined image covering Task 2 and also a separate Task 2 image, since we would not know which one to score. Declare in the submission form which image covers which task. If a task is claimed twice, we will ask you once, and score neither if there is no answer before the deadline.

---

## 2. What your container receives

One **case** is one thoracic esophagectomy and consists of 14 frames, one per lymph node station. The test set holds 10 cases (140 frames): 5 laparoscopic and 5 robotic, from 5 different centers, one of which is not represented in the training data.

The complete test set is mounted **read only** as a flat folder at `/input`:

```
/input/
├── center_x_case_y_<station>.png
├── center_x_case_y_<station>.png
└── ...
```

* Frames are RGB PNG in the native resolution of the recording. Resolutions differ between centers and between laparoscopic and robotic cases. **Do not assume a fixed input size.**
* The folder is flat. There are no subfolders and no metadata file.
* For Task 3: Do not use the station information to get a prediction.
* No patient level context information is provided.
* Iterate over the folder yourself. Do not hard code filenames, do not hard code the number of frames, and do not rely on any particular order beyond the one you produce.
* Nothing outside `/input` may be read at runtime. There is no internet access.

**Filenames are your only key.** Every output must carry the input filename through unchanged. We group frames into cases and match them to the reference by filename, so a renamed, suffixed, or lowercased file cannot be scored.

---

## 3. What your container must produce

`/output` is mounted empty and writable. Create the task folders yourself, only for the tasks you enter. Write nothing anywhere else.

```
/output/
├── task1/
│   ├── center_x_case_y_<station>.png
│   └── ...
├── task2/
│   ├── center_x_case_y_<station>.png
│   └── ...
└── task3.csv
```

A Task 3 only image produces just `/output/task3.csv`. A combined image produces both folders and the CSV in one run.

### 3.1 Tasks 1 and 2 (segmentation)

For every input frame, write **one RGB PNG** into `/output/task1/` or `/output/task2/`, with:

* the **same filename** as the input frame,
* the **same width and height** as the input frame,
* each pixel coloured with the **exact RGB value** of the predicted class (see tables below).

Do **not** use JPEG — lossy compression corrupts pixel colours and therefore class boundaries. Pixels with an unrecognised colour are treated as unknown and excluded from scoring. Assign every pixel to one of the listed classes; do not leave pixels with colours outside the tables.

Anything that is not a valid mask (wrong resolution, wrong colour, renamed file) makes the frame unscorable and is penalized.

**Task 1 — 15 merged classes:**

| Class ID | Merged label | RGB | Fine-grained structures included |
|----------|-------------|-----|----------------------------------|
| 0  | Background | `(0, 0, 0)` | Background |
| 1  | Respiratory Tract | `(0, 255, 220)` | Trachea, R/L Main Bronchus |
| 2  | Gastroesophageal | `(253, 189, 0)` | Esophagus, Fatty Tissue Esophagus, Gastric Conduit, Omentum |
| 3  | Pleura | `(15, 126, 87)` | R/L Inferior Pulmonary Ligament, Pleura |
| 4  | Heart | `(153, 76, 13)` | Pericardium, Inferior Pulmonary Vein |
| 5  | Vessels | `(255, 142, 142)` | R/L Subclavian Artery, Right Bronchial Artery |
| 6  | Nerves | `(255, 239, 179)` | Right Vagal Nerve, R/L Recurrent Laryngeal Nerve |
| 7  | Aorta | `(255, 0, 7)` | Aorta |
| 8  | Azygos Vein | `(0, 6, 255)` | Azygos Vein |
| 9  | Superior Caval Vein | `(50, 183, 250)` | Superior Caval Vein |
| 10 | Lung | `(17, 139, 57)` | Lung |
| 11 | Lymphatic Tissue | `(255, 0, 204)` | Lymph Node |
| 12 | Non-anatomical Other | `(184, 61, 245)` | Instrument, Other |
| 13 | Fatty Tissue | `(250, 250, 55)` | Fatty Tissue |
| 14 | Pulmonary Artery | `(255, 161, 136)` | Pulmonary Artery |
| 15 | Anatomical Other | `(173, 0, 0)` | Pool of Blood, Resection Area, Thoracic Duct |

**Task 2 — 30 fine-grained classes:**

| Class ID | Structure | RGB |
|----------|-----------|-----|
| 0  | Background | `(0, 0, 0)` |
| 1  | Instrument | `(184, 61, 245)` |
| 2  | Other | `(134, 0, 251)` |
| 3  | Trachea | `(0, 255, 220)` |
| 4  | Right Main Bronchus | `(131, 224, 112)` |
| 5  | Left Main Bronchus | `(191, 224, 112)` |
| 6  | Esophagus | `(253, 189, 0)` |
| 7  | Fatty Tissue Esophagus | `(255, 224, 32)` |
| 8  | Right Inferior Pulmonary Ligament | `(207, 164, 117)` |
| 9  | Left Inferior Pulmonary Ligament | `(234, 204, 159)` |
| 10 | Pleura | `(15, 126, 87)` |
| 11 | Pericardium | `(153, 76, 13)` |
| 12 | Inferior Pulmonary Vein | `(51, 221, 255)` |
| 13 | Right Subclavian Artery | `(255, 142, 142)` |
| 14 | Right Vagal Nerve | `(255, 239, 179)` |
| 15 | Aorta | `(255, 0, 7)` |
| 16 | Azygos Vein | `(0, 6, 255)` |
| 17 | Superior Caval Vein | `(50, 183, 250)` |
| 18 | Lung | `(17, 139, 57)` |
| 19 | Lymph Node | `(255, 0, 204)` |
| 20 | Fatty Tissue | `(250, 250, 55)` |
| 21 | Left Subclavian Artery | `(255, 136, 178)` |
| 22 | Right Bronchial Artery | `(255, 73, 162)` |
| 23 | Pulmonary Artery | `(255, 161, 136)` |
| 24 | Pool of Blood | `(173, 0, 0)` |
| 25 | Resection Area | `(128, 128, 128)` |
| 26 | Gastric Conduit | `(245, 147, 49)` |
| 27 | Right Recurrent Laryngeal Nerve | `(218, 161, 69)` |
| 28 | Left Recurrent Laryngeal Nerve | `(255, 217, 129)` |
| 29 | Omentum | `(219, 219, 56)` |
| 30 | Thoracic Duct | `(201, 138, 118)` |

### 3.2 Task 3 (station visibility)

Write **exactly one CSV file** for the entire test set:

```
/output/task3.csv
```

Format:

* UTF-8, comma separated, one header row, one row per input frame, 140 data rows in total.
* Column `case_id`: the input filename stem **without the `.png` extension** (e.g. `center_1_case_3_6R`). We derive the case from it.
* One column per lymph node station, holding a **continuous score in [0, 1]**.
* Station columns must appear in this exact order: `6L,6R,7L,7R,8,9,10L,10R,11L,11R,12L,12R,13L,13R`
* Row order does not matter. The header names must match exactly.
* Multiple stations can be visible in one frame, so the scores of a row are **not** required to sum to 1.

```csv
case_id,6L,6R,7L,7R,8,9,10L,10R,11L,11R,12L,12R,13L,13R
center_1_case_3_6R,0.11,0.98,0.03,0.76,0.02,0.41,0.00,0.00,0.01,0.00,0.00,0.02,0.00,0.00
center_1_case_3_6L,0.91,0.03,0.78,0.06,0.01,0.05,0.03,0.09,0.02,0.01,0.04,0.02,0.03,0.01
```

Continuous scores are mandatory. AUROC is threshold independent and cannot be computed from binary 0/1 entries. For the F1 score we binarize at 0.5 unless announced otherwise. A frame missing from the CSV is scored as a complete miss.

---

## 4. Container requirements

| Requirement | Value |
|---|---|
| Architecture | linux/amd64 |
| GPU | Default NVIDIA GPU, CUDA. RTXA5000 (in case you need more than this, please contact us!)|
| Network at runtime | none. The container is run with `--network none` |
| Filesystem | `/input` read only, `/output` writable, nothing else |
| Invocation | started once for the whole test set, no arguments |
| Runtime limit | 1 minute per frame **and per task covered by the image**. See below |
| Image size | Keep it as small as possible |
| Exit | must terminate on its own with exit code 0 |
| User interaction | none. Only fully automatic methods are allowed |

**Runtime budget.** The limit is 1 minute per frame per task, so the budget scales with what your image does:

| Image covers | Budget |
|---|---|
| 1 task | 140 minutes |
| 2 tasks | 280 minutes |
| 3 tasks | 420 minutes |

A combined image is therefore never penalized for doing more work in one run, and a shared backbone remains an advantage rather than a risk. If the limit is exceeded, the submission is invalid for **every** task the image covers, so budget with a margin. Our hardware may be slower than yours.

Practical consequences:

* **Bake everything in.** Model weights, pretrained checkpoints, and Python packages live inside the image. Nothing is downloaded at runtime. A container that calls `torch.hub.load`, `from_pretrained`, or `pip install` at startup will fail.
* Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `TORCH_HOME=/opt/checkpoints`, or the equivalent for your framework, so that no library silently tries to reach the network.
* **Load your models once**, before the loop over the frames. You get the whole test set in a single run, so loading and warm up cost you almost nothing once and far too much 140 times.
* In a combined image, loop over frames on the outside and over tasks on the inside. Reading and decoding each frame once instead of three times is free performance.
* Write results **incrementally**. Save each label map right after inference and flush the CSV as you go. A crash on frame 130 should not throw away the first 129 results.
* Do not buffer all frames in memory. Read, predict, write, release.
* A combined image holds two or three models in VRAM at once. Check that they fit, or move each to the GPU in turn.
* Log progress to stdout. Logs are kept for debugging, but nothing in the logs is scored.

---

## 5. Minimal examples

### 5.1 `Dockerfile`

Identical for both submission modes.

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.10-py3

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TORCH_HOME=/opt/checkpoints

WORKDIR /opt/algorithm

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# weights must be part of the image
COPY checkpoints/ /opt/checkpoints/
COPY src/ /opt/algorithm/src/
COPY predict.py .

ENTRYPOINT ["python", "predict.py"]
```

### 5.2 Single task image, segmentation (Task 1 or 2)

```python
from pathlib import Path
import numpy as np
from PIL import Image

IN_DIR  = Path("/input")
OUT_DIR = Path("/output/task1")     # /output/task2 for Task 2

# RGB colours indexed by class ID — Task 1 (15 merged classes, see Section 3.1)
PALETTE = np.array([
    [  0,   0,   0], [  0, 255, 220], [253, 189,   0], [ 15, 126,  87],
    [153,  76,  13], [255, 142, 142], [255, 239, 179], [255,   0,   7],
    [  0,   6, 255], [ 50, 183, 250], [ 17, 139,  57], [255,   0, 204],
    [184,  61, 245], [250, 250,  55], [255, 161, 136], [173,   0,   0],
], dtype=np.uint8)    # shape (16, 3); index = class ID


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model("/opt/checkpoints/model.pt")     # once, not per frame

    frames = sorted(IN_DIR.glob("*.png"))
    print(f"found {len(frames)} frames", flush=True)

    for i, frame_path in enumerate(frames, 1):
        image = Image.open(frame_path).convert("RGB")
        width, height = image.size

        label_map = np.asarray(model.predict(image), dtype=np.uint8)    # HxW, values 0-15

        assert label_map.shape == (height, width), "label map must match input resolution"

        # convert class IDs → RGB colours and save as RGB PNG
        rgb_mask = PALETTE[label_map]
        Image.fromarray(rgb_mask, mode="RGB").save(OUT_DIR / frame_path.name)

        print(f"[{i}/{len(frames)}] {frame_path.name}", flush=True)


if __name__ == "__main__":
    main()
```

### 5.3 Single task image, classification (Task 3)

```python
import csv
from pathlib import Path
from PIL import Image

IN_DIR  = Path("/input")
OUT_CSV = Path("/output/task3.csv")
STATIONS = ["6L","6R","7L","7R","8","9","10L","10R","11L","11R","12L","12R","13L","13R"]


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    model = load_model("/opt/checkpoints/model.pt")
    frames = sorted(IN_DIR.glob("*.png"))

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id"] + STATIONS)

        for i, frame_path in enumerate(frames, 1):
            image = Image.open(frame_path).convert("RGB")
            scores = model.predict_proba(image)            # 14 floats in [0, 1]

            writer.writerow([frame_path.stem] + [f"{float(s):.6f}" for s in scores])
            f.flush()
            print(f"[{i}/{len(frames)}] {frame_path.name}", flush=True)


if __name__ == "__main__":
    main()
```

### 5.4 Combined image covering all three tasks

One pass over the frames, three outputs. Set `TASKS` to the subset you enter, for example `{2, 3}`.

```python
import csv
from pathlib import Path
import numpy as np
from PIL import Image

IN_DIR  = Path("/input")
OUT_DIR = Path("/output")
TASKS   = {1, 2, 3}                                       # the tasks this image enters

STATIONS = ["6L","6R","7L","7R","8","9","10L","10R","11L","11R","12L","12R","13L","13R"]

# Palettes indexed by class ID → RGB (see Section 3.1 colour tables)
PALETTE_T1 = np.array([
    [  0,   0,   0], [  0, 255, 220], [253, 189,   0], [ 15, 126,  87],
    [153,  76,  13], [255, 142, 142], [255, 239, 179], [255,   0,   7],
    [  0,   6, 255], [ 50, 183, 250], [ 17, 139,  57], [255,   0, 204],
    [184,  61, 245], [250, 250,  55], [255, 161, 136], [173,   0,   0],
], dtype=np.uint8)    # 15 merged classes
PALETTE_T2 = np.array([
    [  0,   0,   0], [184,  61, 245], [134,   0, 251], [  0, 255, 220],
    [131, 224, 112], [191, 224, 112], [253, 189,   0], [255, 224,  32],
    [207, 164, 117], [234, 204, 159], [ 15, 126,  87], [153,  76,  13],
    [ 51, 221, 255], [255, 142, 142], [255, 239, 179], [255,   0,   7],
    [  0,   6, 255], [ 50, 183, 250], [ 17, 139,  57], [255,   0, 204],
    [250, 250,  55], [255, 136, 178], [255,  73, 162], [255, 161, 136],
    [173,   0,   0], [128, 128, 128], [245, 147,  49], [218, 161,  69],
    [255, 217, 129], [219, 219,  56], [201, 138, 118],
], dtype=np.uint8)    # 30 fine-grained classes


def save_rgb_mask(label_array, palette, out_path, size):
    label_map = np.asarray(label_array, dtype=np.uint8)
    assert label_map.shape == (size[1], size[0]), "label map must match input resolution"
    Image.fromarray(palette[label_map], mode="RGB").save(out_path)


def main():
    # create only the output paths for the tasks we enter
    for t in sorted(TASKS):
        if t in (1, 2):
            (OUT_DIR / f"task{t}").mkdir(parents=True, exist_ok=True)

    models = {t: load_model(f"/opt/checkpoints/task{t}.pt") for t in sorted(TASKS)}

    frames = sorted(IN_DIR.glob("*.png"))
    print(f"found {len(frames)} frames, entering tasks {sorted(TASKS)}", flush=True)

    csv_file = writer = None
    if 3 in TASKS:
        csv_file = (OUT_DIR / "task3.csv").open("w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["case_id"] + STATIONS)

    for i, frame_path in enumerate(frames, 1):
        image = Image.open(frame_path).convert("RGB")      # decode once, reuse
        size = image.size

        if 1 in TASKS:
            save_rgb_mask(models[1].predict(image),
                          PALETTE_T1, OUT_DIR / "task1" / frame_path.name, size)

        if 2 in TASKS:
            save_rgb_mask(models[2].predict(image),
                          PALETTE_T2, OUT_DIR / "task2" / frame_path.name, size)

        if 3 in TASKS:
            scores = models[3].predict_proba(image)
            writer.writerow([frame_path.stem] + [f"{float(s):.6f}" for s in scores])
            csv_file.flush()

        print(f"[{i}/{len(frames)}] {frame_path.name}", flush=True)

    if csv_file is not None:
        csv_file.close()


if __name__ == "__main__":
    main()
```

---

## 6. Test your container locally

Run exactly the command we run. If it does not work this way on your machine, it will not work on ours.

```bash
docker build -t tigersqai_myteam .

mkdir -p results

docker run --rm \
  --gpus all \
  --network none \
  --shm-size 8g \
  -v /path/to/sanity_set:/input:ro \
  -v $(pwd)/results:/output \
  tigersqai_myteam
```

Check afterwards that `results/` contains a `task1/` and/or `task2/` folder and/or `task3.csv` for every task you enter, and nothing else. Then score your output with the official script, which detects the task paths on its own:

```bash
git clone https://gitlab.com/nct_tso_public/challenges/miccai2026/tigersqai_challenge
cd tigersqai_challenge
python metrics/01_evaluate_challenge.py \
    --pred /path/to/results \
    --gt   /path/to/sanity_labels \
    --out  report.md
```

If you submit separate images, run each of them into its **own** results folder and score them one at a time, exactly as we will.

A small sanity check set drawn from the training data is released with the second training data package, so that you can verify your I/O contract end to end. It is **not** the test set and its scores carry no ranking information.

The most common reasons a submission fails:

1. output resolution differs from input resolution,
2. single-channel or palette PNG instead of an RGB colour-coded mask,
3. pixel colour not in the official colour table (treated as unknown and excluded from scoring),
4. filenames renamed, suffixed, or lowercased,
5. results written to `/output` directly instead of `/output/task1`, `/output/task2`, or the `case_id` column missing/wrong in `task3.csv`,
6. an empty task folder created for a task the image does not actually enter, which is scored as a failure and not as a skip,
7. weights downloaded at runtime, which fails without network,
8. hard coded input size that breaks on a center with a different resolution,
9. models reloaded per frame, which blows the runtime budget.

---

## 7. Submission

1. Register your team on Synapse
2. Save and compress each image you submit:

```bash
docker save <image> | gzip -c > <filename>.tar.gz
```

3. Naming convention:

| Submission mode | Filename |
|---|---|
| One image per task | `tigersqai_<teamname>_task<1\|2\|3>.tar.gz` |
| One image, several tasks | `tigersqai_<teamname>_task<list>.tar.gz`, for example `tigersqai_<teamname>_task123.tar.gz` or `tigersqai_<teamname>_task23.tar.gz` |

The tasks in the filename must match the folders the image writes.

4. Upload to Synapse
5. In the submission form, state for each task which image covers it, and submit a short method description 

For our previous challenges, we generated some detailed instructions with tips and tricks which you can follow.
Please have a look [here](https://caruscloud.uniklinikum-dresden.de/index.php/s/Tq33KRs54ogWcPM)!

Rules:

* **Only the final submission per team and task is evaluated.** Earlier uploads are ignored, not averaged.
* Every task you enter must be covered by exactly one image. Mixing a combined image and a separate image for the same task is not allowed.
* You may mix modes across tasks, for example one combined image for Tasks 1 and 2 and a separate image for Task 3.
* Submission deadline: **15 September 2026**. Evaluation starts on 1 September 2026.
* Allowed training data: the provided challenge data plus publicly available data, including open pretrained networks. Private or non public data, additional non public annotations, and models pretrained on such data are **not** permitted. Declare every external dataset and every pretrained checkpoint in your method description.
* Members of the organizing institutes may participate but are not eligible for awards. Their submissions are marked as such.
* No results may be published before the joint challenge paper is published.
* Making your code open source is encouraged but not required. Submitted Docker images are not shared by the organizers.


---

## 8. Checklist before you upload

For every image you submit:

* [ ] Builds for linux/amd64 and runs with `--network none`
* [ ] Starts once, discovers all frames in the flat `/input` folder itself, and exits with code 0
* [ ] Creates `task1/`, `task2/`, and/or `task3.csv` under `/output` for **every** task it enters, and for **no** other task
* [ ] Finishes within 140 minutes per task covered
* [ ] Segmentation: one RGB PNG per input frame, input resolution, all pixel colours in the official colour table, filename unchanged
* [ ] Classification: exactly one `/output/task3.csv`, one row per frame with `case_id` = filename stem (no `.png`), continuous scores for all 14 stations in the correct column order
* [ ] All weights inside the image, no runtime downloads
* [ ] Official evaluation script runs on the output without errors
* [ ] Filename lists exactly the tasks the image writes
* [ ] Across all your images, every task is claimed exactly once
* [ ] Method description written, external data and pretrained models declared
* [ ] Write the organizers an email that you have submitted something

---

## 9. Contact

tso-tiger-sqai-challenge@groups.tu-dresden.de 


For technical questions about the container interface, please ask in the challenge wiki so that other teams benefit from the answer.

