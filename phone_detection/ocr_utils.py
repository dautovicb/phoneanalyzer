from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

UI_CLASSES = {"ui_battery", "ui_memory", "ui_memory_about"}
BestResult = Dict[str, Tuple[float, Path, Image.Image]]


def run_ocr(ocr_engine: RapidOCR, image: Image.Image) -> List[str]:
    ocr_results, _ = ocr_engine(np.array(image))
    if not ocr_results:
        return []

    texts: List[str] = []
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
    lowered = text_blob.lower()

    direct_match = re.search(r"(\d{2,3})\s*%", text_blob)
    if direct_match:
        value = int(direct_match.group(1))
        if 50 <= value <= 100:
            return f"{value}%"

    for value in re.findall(r"\d{2,3}", text_blob):
        n = int(value)
        if 50 <= n <= 100:
            return f"{n}%"

    battery_context = any(k in lowered for k in ("maximum capacity", "battery", "capacity"))
    if battery_context:
        for token in re.findall(r"\d{3,4}", text_blob):
            first_two = int(token[:2])
            if 50 <= first_two <= 100:
                return f"{first_two}%"

    decimal_match = re.search(r"(\d{2,3})[\.,](\d)", text_blob)
    if decimal_match:
        whole = int(decimal_match.group(1))
        if 50 <= whole <= 100:
            return f"{whole}%"

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


def extract_specs_from_best(best_by_class: BestResult, ocr_engine: Optional[RapidOCR] = None) -> Dict:
    engine = ocr_engine or RapidOCR()
    ui_ocr: Dict[str, List[str]] = {}

    for cls_name in UI_CLASSES:
        best = best_by_class.get(cls_name)
        if not best:
            continue
        ui_ocr[cls_name] = run_ocr(engine, best[2])

    battery = extract_battery_health(ui_ocr.get("ui_battery", []))
    memory = extract_internal_memory(ui_ocr.get("ui_memory", []) + ui_ocr.get("ui_memory_about", []))

    return {
        "battery_health_percent": battery,
        "internal_memory": memory,
        "ocr_text": ui_ocr,
    }
