# Smartphone Analyzer

> AI-powered analysis of second-hand smartphone listings from [OLX.ba](https://olx.ba).

**Smartphone Analyzer** takes a smartphone listing (its photos, title, and description) and extracts the features a buyer actually cares about - model, storage, battery health, packaging and red flags such as a cracked screen, iCloud lock, non-functioning parts and more. It combines **Computer Vision** and **NLP** into a single pipeline that turns a messy listing into one clean, structured record.

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)

---

## Table of Contents

- [Overview](#overview)
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

- **Object Detection** - detects the phone and the orientation, packaging, ui elements (battery health, storage) 
- **OCR** - reads battery health and storage from detected images.
- **NLP** - extracts model, storage, condition, and red flags from the natural language, Bosnian/Croatian/Serbian listing description.
- **Fusion** - a merger reconciles all three sources into a single record with explicit priority rules.

The result is a structured, database-ready record per listing.

> **Note:** OCR-based spec extraction is currently tuned for **iPhones** (iOS Battery and Storage settings screens).

---

## Installation

### Prerequisites

- **Python 3.10+**
- **[Git LFS](https://git-lfs.com/)** - model weights (`.onnx`, `.keras`, `.safetensors`) and datasets are stored via Git LFS. Install it **before** cloning, then pull:

```bash
git lfs install
git clone https://github.com/dautovicb/phoneanalyzer.git
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

### CLI

Analyze one or more listings from the command line:

```bash
python cli.py "https://olx.ba/artikal/12345678"               # readable summary
python cli.py "https://olx.ba/artikal/12345678" --json        # merged record as JSON
python cli.py <url1> <url2> <url3>                            # models load once, then each URL is analyzed
```

Status messages go to stderr, results to stdout, so `--json` output pipes cleanly. Exits non-zero if any listing failed.

### Python API

The pipeline is exposed through `core.pipeline`. Run from the module root (or add it to `sys.path`):

```python
from core.pipeline import analyze_listing, preload_models, PipelineError

preload_models()  # optional: warm all three models up front (long-running processes)

try:
    result = analyze_listing("https://olx.ba/artikal/12345678")
except PipelineError as err:
    print(f"Could not analyze listing: {err}")
else:
    print(result.summary())       # human-readable report
    print(result.merged)          # final database-ready record
    print(result.bertic_summary)  # what the NLP model extracted
    print(result.cv_summary)      # what the vision pipeline extracted
```

Models are lazy singletons - they load on first use and stay cached for the life of the process, so repeated `analyze_listing` calls only pay the load cost once. `preload_models()` just moves that cost to startup.

`analyze_listing` returns an `AnalysisResult` dataclass:

| Field            | Description                                                        |
| ---------------- | ------------------------------------------------------------------ |
| `merged`         | The final, fused listing record (see [Output Schema](#output-schema)) |
| `cv_summary`     | Specs and flags extracted from the photos                          |
| `bertic_summary` | Specs and flags extracted from the description text                |
| `bertic_raw`     | Raw NER entities (for debugging/transparency)                      |
| `analysis`       | Low-level detection output: crops, detections, OCR text            |

### Demo app

A Streamlit UI is provided to try the pipeline interactively - paste a listing URL and inspect every stage (merged record, detections, crops, NER entities):

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

**Supported classes:**
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

<img width="395" height="207" alt="RF-DETR training metrics" src="https://github.com/user-attachments/assets/589e2359-d339-4254-bc93-18aae5fea5c2" />

### Description Parsing (BERTić NER)

Listing titles and descriptions are written in Bosnian/Croatian/Serbian. A fine-tuned **[BERTić](https://huggingface.co/classla/bcms-bertic)** token-classification (NER) model extracts structured fields from this free text.

**Entity tags:**

| Tag     | Meaning            | Example                  |
| ------- | ------------------ | ------------------------ |
| `BRAND` | Manufacturer       | *Apple, Samsung*         |
| `MOD`   | Model              | *iPhone 13 Pro*          |
| `MEM`   | Storage            | *256GB*                  |
| `BATT`  | Battery health     | *89%*                    |
| `COND`  | Condition claim    | *kao nov*     |
| `FAIL`  | Issues / defects   | *ne radi, puknut ekran*  |
| `ICL`   | iCloud status      | *icloud locked*          |
| `SIM`   | SIM lock status    | *sim locked*             |
| `BOX`   | Packaging          | *kutija*           |

Entities below a confidence threshold are discarded; the highest-scoring entity per field wins. See [`models/description_model/inference.py`](models/description_model/inference.py).


### Crack Detection (ConvNeXt)

A binary **ConvNeXt** classifies `phone_front` and `phone_back` crops as cracked / not cracked (224×224 input, sigmoid output). Either a positive crack prediction here **or** a textual mention in the description raises the corresponding red flag.

![Accuracy crack detection](image.png)
### Spec OCR (RapidOCR)

For UI screenshots (`ui_battery`, `ui_memory`, `ui_memory_about`), **RapidOCR** reads the raw text and dedicated parsers extract **battery health %** and **internal storage (GB)** - currently tuned for iOS settings screens. See [`models/detection_model/ocr_utils.py`](models/detection_model/ocr_utils.py).

---

## Training

Training, export, and evaluation scripts live alongside each model under `models/`:

| Task                          | Script                                           |
| ----------------------------- | ------------------------------------------------ |
| Train the RF-DETR detector    | `models/detection_model/training/train.py`       |
| Export the detector to ONNX   | `models/detection_model/training/export.py`      |
| Evaluate the detector         | `models/detection_model/training/test.py`        |
| Train the crack classifier    | `models/crack_detection/training/cracks.py`      |

Install the training dependencies first:

```bash
pip install -r requirements-train.txt
```

> Detection training logs to [Weights & Biases](https://wandb.ai/); set up your W&B account (or disable logging) before running.

---

## License

Released under the [MIT License](LICENSE).
