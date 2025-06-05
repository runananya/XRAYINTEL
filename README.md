# XRAY‑INTEL

**Interpretable two‑stage fracture detection pipeline** built on a lightweight Prototype Vision Transformer (ProtoViT) gate in front of a YOLOv8 detector.  Designed for the [FracAtlas](https://fracatlas.org) X‑ray dataset, but the code is dataset‑agnostic with a pluggable `XRayCOCODataset` loader.

---

## Table of contents

* [Why XRAY‑INTEL?](#why-xray-intel)
* [Key features](#key-features)
* [Project structure](#project-structure)
* [Quick start](#quick-start)
* [Training](#training)
* [Inference](#inference)
* [Prototype visualisation](#prototype-visualisation)
* [Benchmarks](#benchmarks)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)

---

## Why XRAY‑INTEL?

Radiograph fracture screening often demands **both high recall** (don’t miss a fracture) **and clinician trust**.  A single heavy detector can be accurate but slow; a single lightweight classifier is fast but opaque.  XRAY‑INTEL combines the best of both:

1. **ProtoViT filter** — interpretable, milliseconds‑fast toggle that routes only suspicious cases onward.
2. **YOLOv8 detector** — localises fractures with high precision.

This cascade saves compute on obviously healthy images **while surfacing prototype explanations** for every decision.

---

## Key features

| Module                            | Highlights                                                                                                           |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **`xr_intel/datasets.py`**        | COCO‑style loader that stitches polygon masks into binary masks and returns canonical Tensor shapes.                 |
| **`xr_intel/models/protovit.py`** | Vision Transformer with learnable class prototypes & similarity heatmaps.                                            |
| **`xr_intel/models/yolo.py`**     | Thin wrapper around [Ultralytics‑YOLOv8](https://github.com/ultralytics/ultralytics) for seamless training & export. |
| **`train.py`**                    | End‑to‑end trainer with `argparse` flags, deterministic seeding, and optional Weights & Biases logging.              |
| **`predict.py`**                  | Performs ProtoViT → YOLO cascade, emits JSON or COCO predictions, and saves annotated images.                        |
| **`tests/`**                      | Smoke tests that load 5 images and complete a forward pass.                                                          |

---

## Project structure

```text
XRAY_INTEL/
├── xr_intel/
│   ├── __init__.py
│   ├── datasets.py
│   └── models/
│       ├── protovit.py
│       └── yolo.py
├── configs/
│   ├── protovit_default.yaml
│   └── yolo_default.yaml
├── train.py
├── predict.py
├── tests/
│   ├── test_dataset.py
│   └── test_pipeline.py
├── environment.yml
├── docs/
│   └── architecture.svg
└── README.md  ← you are here
```

---

## Quick start

```bash
# 1. Clone repo & install
conda env create -f environment.yml
conda activate xray_intel

# 2. Download FracAtlas (≈1.2 GB) into ./data
bash scripts/download_fracatlas.sh

# 3. Train both stages with defaults
python train.py \  
    --config configs/protovit_default.yaml \  
    --output runs/protovit

python train.py \  
    --config configs/yolo_default.yaml \  
    --output runs/yolo
```

> **Tip** – All file/dir arguments can be overridden from the CLI; see `--help`.

---

## Training

### ProtoViT

```bash
python train.py \
  --config configs/protovit_default.yaml \
  --epochs 50 \
  --batch-size 32 \
  --lr 3e-4
```

### YOLOv8

```bash
python train.py \
  --config configs/yolo_default.yaml \
  --epochs 100 \
  --img 640 640 \
  --device 0,1
```

Both trainers log to **Weights & Biases** if `WANDB_API_KEY` is in your env.

---

## Inference

Run the full cascade on a folder of DICOM/JPEG/PNG images:

```bash
python predict.py \
  --input data/test_images \
  --protovit-checkpoint runs/protovit/best.pt \
  --yolo-checkpoint runs/yolo/best.pt \
  --output results.json \
  --vis-dir vis/
```

The script outputs:

* `results.json` – COCO‑style detections.
* `vis/` – images with bounding boxes & prototype heatmaps.

---

## Prototype visualisation

ProtoViT exposes a `--save-prototypes` flag that dumps every prototype patch and its activation map.  Example CLI:

```bash
python predict.py \
  ... \
  --save-prototypes prototypes/
```

---

## Benchmarks

| Metric                 | Value          | Notes                                   |
| ---------------------- | -------------- | --------------------------------------- |
| **Classifier Recall**  | 98.9 %         | ProtoViT threshold @ 0.15               |
| **Classifier Latency** | 3.1 ms / image | RTX A6000, FP16                         |
| **Detector mAP\@0.5**  | 84.7 %         | YOLOv8‑s, 640×640                       |
| **End‑to‑end FPS**     | 62 FPS         | 75 % images short‑circuited by ProtoViT |

See `docs/benchmarks.md` for the full ablation table.

---

## Roadmap

*

---

## Contributing

PRs are welcome!  Please fork, create a feature branch, run `pytest` and `ruff`, then open a pull request.  Check the [contributing guide](CONTRIBUTING.md) for coding conventions.

---

## License

[MIT](LICENSE) © 2025 Ananya Tomar and contributors
