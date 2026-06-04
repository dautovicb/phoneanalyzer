# Smartphone Analyzer

> AI-powered analysis of second-hand smartphone listings from [OLX.ba](https://olx.ba).

**Smartphone Analyzer** takes a smartphone listing (its photos, title, and description) and automatically extracts the features a buyer actually cares about — model, storage, battery health, packaging, and condition red flags such as cracked screens or iCloud locks. It combines **Computer Vision** and **NLP** into a single pipeline that turns a messy listing into one clean, structured record.

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Output Schema](#output-schema)
- [Models](#models)
  - [Object Detection (RF-DETR)](#object-detection-rf-detr)
  - [Description Parsing (BERTić NER)](#description-parsing-bertić-ner)
  - [Crack Detection (CNN)](#crack-detection-cnn)
  - [Spec OCR (RapidOCR)](#spec-ocr-rapidocr)
- [Training](#training)
- [License](#license)

---

## Overview

When buying a used phone, the key signals are scattered: some live in the **photos** (a cracked back, a battery-health screenshot, the original box), and some live in the **text** (the title says "iPhone 13 Pro 256GB", the description mentions "iCloud locked"). This project fuses both:

- **Computer Vision** — detects the phone, its packaging, and on-screen settings; flags physical damage.
- **OCR** — reads battery health and storage from screenshots of the iOS settings screens.
- **NLP** — extracts model, storage, condition, and red flags from the free-text Bosnian/Croatian/Serbian listing description.
- **Fusion** — a merger reconciles all three sources into a single record with explicit priority rules.

The result is a structured, database-ready record per listing.

> **Note:** OCR-based spec extraction is currently tuned for **iPhones** (iOS Battery and Storage settings screens).

---

## How It Works

```
                       OLX.ba listing URL
                               │
               ┌───────────────┴────────────────┐
               ▼                                 ▼
            Images                       Title + Description
               │                                 │
               ▼                                 ▼
        ┌─────────────┐                   ┌───────────────┐
        │   RF-DETR   │                   │    BERTić     │
        │  detection  │                   │      NER      │
        └──────┬──────┘                   └──────┬────────┘
               │                                 │
       ┌───────┴────────┐                  model, storage,
       ▼                ▼                  battery, condition,
   UI parts      phone front / back        red flags
       │                │                        │
       ▼                ▼                        │
 ┌───────────┐    ┌─────────────┐                │
 │  RapidOCR │    │   ConvNeXt  │                │
 │battery/mem│    │     CNN     │                │
 └─────┬─────┘    └──────┬──────┘                │
       └────────┬────────┘                       │
                ▼                                │
           CV summary ──────────┐                │
                                ▼                ▼
                         ┌──────────────────────────┐
                         │          Merger          │
                         │  (priority-based fusion) │
                         └────────────┬─────────────┘
                                      ▼
                              Merged listing record
```

The single entry point, `analyze_listing(url)`, runs this whole flow and returns the merged record plus per-source diagnostics.

---

## Project Structure

```
phoneanalyzer/
├── core/                       # Orchestration + fusion (framework-agnostic)
│   ├── pipeline.py             # analyze_listing(), AnalysisResult, PipelineError
│   ├── merger.py               # Reconciles OLX + NLP + CV into one record
│   └── olx_client.py           # OLX.ba API client
├── demo/
│   └── app.py                  # Streamlit demo UI
├── models/
│   ├── detection_model/        # RF-DETR object detector (ONNX) + OCR utils
│   ├── description_model/      # BERTić NER for listing text
│   └── crack_detection/        # Keras CNN crack classifier
├── requirements.txt            # Inference dependencies
└── requirements-train.txt      # Training/export/eval dependencies
```

---

## Installation

### Prerequisites

- **Python 3.10+**
- **[Git LFS](https://git-lfs.com/)** — model weights (`.onnx`, `.keras`, `.safetensors`) are stored via Git LFS. Install it **before** cloning, then pull the weights:

```bash
git lfs install
git clone <repo-url>
cd phoneanalyzer
git lfs pull
```

### Inference (running the analyzer)

```bash
pip install -r requirements.txt
```

This installs everything needed to run the pipeline and the demo app: ONNX Runtime, Transformers/PyTorch (NER), TensorFlow (crack CNN), RapidOCR, and Streamlit.

> By default the detector runs on CPU (`onnxruntime`). For GPU inference, install `onnxruntime-gpu` instead.

### Training / export / evaluation

```bash
pip install -r requirements-train.txt
```

This is a superset of the inference requirements, adding RF-DETR, Weights & Biases, scikit-learn, and plotting libraries.

---

## Usage

### Python API

The pipeline is exposed through `core.pipeline`. Run from the module root (or add it to `sys.path`):

```python
from core.pipeline import analyze_listing, PipelineError

try:
    result = analyze_listing("https://olx.ba/artikal/12345678")
except PipelineError as err:
    print(f"Could not analyze listing: {err}")
else:
    print(result.merged)          # final database-ready record
    print(result.bertic_summary)  # what the NLP model extracted
    print(result.cv_summary)      # what the vision pipeline extracted
```

`analyze_listing` returns an `AnalysisResult` dataclass:

| Field            | Description                                                        |
| ---------------- | ------------------------------------------------------------------ |
| `merged`         | The final, fused listing record (see [Output Schema](#output-schema)) |
| `cv_summary`     | Specs and flags extracted from the photos                          |
| `bertic_summary` | Specs and flags extracted from the description text                |
| `bertic_raw`     | Raw NER entities (for debugging/transparency)                      |
| `analysis`       | Low-level detection output: crops, detections, OCR text            |

### Demo app

A Streamlit UI is provided to try the pipeline interactively — paste a listing URL and inspect every stage (merged record, detections, crops, NER entities):

```bash
streamlit run demo/app.py
```

---

## Output Schema

`result.merged` is a flat dictionary ready to be written to a database:

| Field                       | Type        | Source priority            | Description                            |
| --------------------------- | ----------- | -------------------------- | -------------------------------------- |
| `id`                        | int         | OLX                        | Listing ID                             |
| `title`, `description`      | str         | OLX                        | Listing text                           |
| `price`                     | number      | OLX                        | Asking price (KM)                      |
| `url`, `images`, `date`     | str         | OLX                        | Listing metadata                       |
| `model`                     | str         | NLP                        | Phone model (e.g. `iphone 15 pro`)     |
| `storage_gb`                | int \| None | NLP → CV(OCR) → OLX attr   | Internal storage                       |
| `battery_health_pct`        | int \| None | NLP → CV(OCR)              | Battery health %                       |
| `condition_claim`           | str \| None | NLP                        | Seller's condition wording             |
| `red_flag_cracked_front`    | 0/1         | NLP **or** CV              | Cracked front detected                 |
| `red_flag_cracked_back`     | 0/1         | NLP **or** CV              | Cracked back detected                  |
| `red_flag_icloud_locked`    | 0/1         | NLP                        | iCloud lock mentioned                  |
| `red_flag_sim_locked`       | 0/1         | NLP                        | SIM lock mentioned                     |
| `red_flag_not_functioning`  | 0/1         | NLP                        | "not working" / defect mentioned       |
| `has_original_box`          | 0/1         | NLP **or** CV              | Original box present                   |
| `damage_confidence`         | float       | CV                         | Confidence in damage detection         |
| `analyzed`                  | 1           | —                          | Pipeline-state flag                    |

**Fusion rules** (implemented in [`core/merger.py`](core/merger.py)): specs prefer the NLP reading (parsed from text), fall back to OCR from photos, then to OLX attributes; red flags are raised if **either** the text or the vision pipeline detects them.

---

## Models

### Object Detection (RF-DETR)

The starting point of the vision pipeline is an **RF-DETR** object detector (exported to ONNX) that locates the phone, its packaging, and relevant on-screen settings.

**Supported classes (v2):**
- **Hardware:** `phone_front`, `phone_back`, `box`, `case` (phone back with case)
- **Software/UI screens:** `ui_battery`, `ui_memory`, `ui_memory_about`

**Metrics:**
- mAP @ 0.5: **0.9474**
- mAP @ 0.5–0.95: **0.8386**
- *Trained in <20 epochs*

**Dataset:** The [dataset](models/detection_model/dataset) includes 1143 images (369 original source images with a 70/30 split, 258 training images), annotated in COCO format.

Pre-processing:
- Auto-orientation of pixel data (with EXIF-orientation stripping)

Augmentation (4 versions per source image):
- 50% probability of horizontal flip
- 50% probability of vertical flip
- Random rotation between −11° and +11°
- Random brightness adjustment between −15% and +15%
- Random Gaussian blur between 0 and 1.3 pixels

<!-- TODO: replace with your own chart -->
<img width="395" height="207" alt="RF-DETR training metrics" src="https://github.com/user-attachments/assets/589e2359-d339-4254-bc93-18aae5fea5c2" />

### Description Parsing (BERTić NER)

Listing titles and descriptions are written in Bosnian/Croatian/Serbian (often mixed with English). A fine-tuned **[BERTić](https://huggingface.co/classla/bcms-bertic)** token-classification (NER) model extracts structured fields from this free text.

**Entity tags:**

| Tag     | Meaning            | Example                  |
| ------- | ------------------ | ------------------------ |
| `BRAND` | Manufacturer       | *Apple, Samsung*         |
| `MOD`   | Model              | *iPhone 13 Pro*          |
| `MEM`   | Storage            | *256GB*                  |
| `BATT`  | Battery health     | *89%*                    |
| `COND`  | Condition claim    | *kao nov / like new*     |
| `FAIL`  | Issues / defects   | *ne radi, puknut ekran*  |
| `ICL`   | iCloud status      | *icloud locked*          |
| `SIM`   | SIM lock status    | *sim locked*             |
| `BOX`   | Packaging          | *kutija / box*           |

Entities below a confidence threshold are discarded; the highest-scoring entity per field wins. See [`models/description_model/inference.py`](models/description_model/inference.py).

<!-- TODO: add NER training/evaluation chart -->
<!-- ![BERTić NER — training & evaluation metrics](docs/images/ner_training.png) -->
> _Training/evaluation charts for the NER model go here._

### Crack Detection (CNN)

A binary **Keras CNN** classifies `phone_front` and `phone_back` crops as cracked / not cracked (224×224 input, sigmoid output). Either a positive crack prediction here **or** a textual mention in the description raises the corresponding red flag.

<!-- TODO: add crack-detection training chart -->
<!-- ![Crack detector — training metrics](docs/images/crack_training.png) -->
> _Training/evaluation charts for the crack detector go here._

### Spec OCR (RapidOCR)

For UI screenshots (`ui_battery`, `ui_memory`, `ui_memory_about`), **RapidOCR** reads the raw text and dedicated parsers extract **battery health %** and **internal storage (GB)** — currently tuned for iOS settings screens. See [`models/detection_model/ocr_utils.py`](models/detection_model/ocr_utils.py).

---

## Training

Training, export, and evaluation scripts live alongside each model under `models/`:

| Task                          | Script                                           |
| ----------------------------- | ------------------------------------------------ |
| Train the RF-DETR detector    | `models/detection_model/train.py`                |
| Export the detector to ONNX   | `models/detection_model/export.py`               |
| Evaluate the detector         | `models/detection_model/test.py`                 |
| Train the crack classifier    | `models/crack_detection/training/cracks.py`      |

Install the training dependencies first:

```bash
pip install -r requirements-train.txt
```

> Detection training logs to [Weights & Biases](https://wandb.ai/); set up your W&B account (or disable logging) before running.

---

## License

Released under the [MIT License](LICENSE).
