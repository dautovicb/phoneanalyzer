from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

from PIL import Image, UnidentifiedImageError

from inference import detect_and_crop, load_model, MODEL_PATH
from ocr_utils import extract_specs_from_best

# Common image suffixes for listing photos.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# class_name -> (confidence, source_image_path, crop_image)
BestResult = Dict[str, Tuple[float, Path, Image.Image]]


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


def build_analysis(best_by_class: BestResult) -> Dict:
    specs = extract_specs_from_best(best_by_class)
    return {
        "internal_memory": specs["internal_memory"],
        "battery_percentage": specs["battery_health_percent"],
        "hasBox": "box" in best_by_class,
        "detections": {
            cls_name: {
                "confidence": round(info[0], 4),
                "source_image": str(info[1]),
            }
            for cls_name, info in sorted(best_by_class.items())
        },
        "ocr_text": specs["ocr_text"],
        "best_crops": best_by_class,
    }


def analyze_folder(input_dir: Path, model_path: str, threshold: float, recursive: bool = False) -> Dict:
    best_by_class = collect_best_detections(
        input_dir=input_dir,
        model_path=model_path,
        threshold=threshold,
        recursive=recursive,
    )
    return build_analysis(best_by_class)


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
    extraction = extract_specs_from_best(best_by_class)
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
