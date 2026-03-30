from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
from rapidocr_onnxruntime import RapidOCR

from inference import detect_and_crop, load_model, MODEL_PATH

# Common image suffixes for listing photos.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# class_name -> (confidence, source_image_path, crop_image)
BestResult = Dict[str, Tuple[float, Path, Image.Image]]
UI_CLASSES = {"ui_battery", "ui_memory", "ui_memory_about"}


def iter_images(input_dir: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from (p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    else:
        yield from (p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def collect_best_detections(input_dir: Path, model_path: str, threshold: float, recursive: bool) -> BestResult:
    session = load_model(model_path)
    best_by_class: BestResult = {}

    for image_path in iter_images(input_dir, recursive):
        try:
            image = Image.open(image_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as err:
            print(f"Skipping unreadable image: {image_path} ({err})")
            continue

        crops = detect_and_crop(image, session, threshold=threshold)
        if not crops:
            continue

        for cls_name, conf, crop in crops:
            current = best_by_class.get(cls_name)
            if current is None or conf > current[0]:
                best_by_class[cls_name] = (conf, image_path, crop)

    return best_by_class


def run_ocr(ocr_engine: RapidOCR, image: Image.Image) -> List[str]:
    ocr_results, _ = ocr_engine(np.array(image))
    print(f"OCR results: {ocr_results}")
    if not ocr_results:
        return []
    texts = []
    for row in ocr_results:
        if len(row) < 2:
            continue
        text_part = row[1]
        if isinstance(text_part, tuple) and text_part:
            text = str(text_part[0])
        else:
            text = str(text_part)
        if text.strip():
            texts.append(text.strip())
    return texts


def extract_battery_health(texts: List[str]) -> Optional[str]:
    text_blob = " ".join(texts)
    direct_match = re.search(r"(\d{2,3})\s*%", text_blob)
    if direct_match:
        value = int(direct_match.group(1))
        if 50 <= value <= 100:
            return f"{value}%"

    for value in re.findall(r"\d{2,3}", text_blob):
        n = int(value)
        if 50 <= n <= 100:
            return f"{n}%"
    return None


def extract_internal_memory(texts: List[str]) -> Optional[str]:
    text_blob = " ".join(texts)
    normalized = text_blob.replace(",", ".")

    tb_match = re.search(r"(\d+(?:\.\d{1,2})?)\s*T\s*B", normalized, re.IGNORECASE)
    if tb_match:
        size = tb_match.group(1)
        if size.endswith(".0"):
            size = size[:-2]
        return f"{size} TB"

    gb_match = re.search(r"(\d+)\s*G\s*B", normalized, re.IGNORECASE)
    if gb_match:
        return f"{int(gb_match.group(1))} GB"

    for candidate in re.findall(r"\d{2,4}", normalized):
        value = int(candidate)
        if value in {64, 128, 256, 512}:
            return f"{value} GB"
    return None


def save_results(best_by_class: BestResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for cls_name, (conf, source_path, crop) in sorted(best_by_class.items()):
        out_path = output_dir / f"{cls_name}.jpg"
        crop.save(out_path)
        summary.append(
            {
                "class": cls_name,
                "confidence": round(conf, 6),
                "source_image": str(source_path),
                "output_crop": str(out_path),
            }
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_ocr_extractions(best_by_class: BestResult, output_dir: Path) -> None:
    ocr_engine = RapidOCR()
    ui_ocr: Dict[str, List[str]] = {}

    for cls_name, (_, _, crop) in best_by_class.items():
        if cls_name not in UI_CLASSES:
            continue
        texts = run_ocr(ocr_engine, crop)
        ui_ocr[cls_name] = texts

    extraction = {
        "battery_health_percent": extract_battery_health(ui_ocr.get("ui_battery", [])),
        "internal_memory": extract_internal_memory(ui_ocr.get("ui_memory", []) + ui_ocr.get("ui_memory_about", [])),
        "ocr_text": ui_ocr,
    }
    extraction_path = output_dir / "extracted_specs.json"
    extraction_path.write_text(json.dumps(extraction, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run phone detection on all images in a folder and keep only the highest-confidence crop per class."
    )
    parser.add_argument("input_dir", type=str, help="Folder containing listing images")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Path to ONNX model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Minimum confidence threshold")
    parser.add_argument("--output-dir", type=str, default="batch_output", help="Directory to write best crops")
    parser.add_argument("--recursive", action="store_true", help="Scan input folder recursively")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_dir}")

    best_by_class = collect_best_detections(
        input_dir=input_dir,
        model_path=args.model,
        threshold=args.threshold,
        recursive=args.recursive,
    )

    if not best_by_class:
        print("No matching detections found in the input folder.")
        return

    output_dir = Path(args.output_dir)
    save_results(best_by_class, output_dir)
    save_ocr_extractions(best_by_class, output_dir)

    print("Saved best detections:")
    for cls_name, (conf, source_path, _) in sorted(best_by_class.items()):
        print(f"- {cls_name}: {conf:.4f} from {source_path}")
    print(f"Summary written to: {output_dir / 'summary.json'}")
    print(f"Specs extraction written to: {output_dir / 'extracted_specs.json'}")


if __name__ == "__main__":
    main()
