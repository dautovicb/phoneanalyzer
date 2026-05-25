from __future__ import annotations

import importlib.util
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import requests
import streamlit as st

MODEL_DIR = Path(__file__).resolve().parent / "detection_model"
if str(MODEL_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(MODEL_DIR))

from batch_inference import IMAGE_SUFFIXES, analyze_folder  # type: ignore
from inference import MODEL_PATH  # type: ignore
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPTS_DIR))

from merger import merge as merge_records  # type: ignore
from olx_client import extract_listing_id, fetch_listing_detail, fetch_listing_images

BERTIC_INFERENCE_PATH = Path(__file__).resolve().parent / "description_model" / "inference.py"
_bertic_spec = importlib.util.spec_from_file_location("bertic_inference", str(BERTIC_INFERENCE_PATH))
if _bertic_spec is None or _bertic_spec.loader is None:
    raise RuntimeError("Failed to load bertic inference module.")
_bertic_module = importlib.util.module_from_spec(_bertic_spec)
_bertic_spec.loader.exec_module(_bertic_module)
extract_features = getattr(_bertic_module, "extract_features")

_CRACKED_RE = re.compile(r"(puknut|pukla|crack|cracked|razbijen|razbijeno)")
_NOT_WORKING_RE = re.compile(r"(ne\s*rad(i|e)|not\s*work|does\s*not\s*work|defekt)")
_LOCKED_RE = re.compile(r"(lock|locked|zakljucan|zakljucana)")
_NO_RE = re.compile(r"\b(no|bez|nije)\b")
_BOX_RE = re.compile(r"(box|kutija|full\s*box|puna\s*kutija)")
_CHARGER_RE = re.compile(r"(charger|punjac|punjacc|punjac\b)")


def apply_custom_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .results-card {
            padding: 0.8rem 1rem;
            border: 1px solid rgba(120, 120, 120, 0.25);
            border-radius: 12px;
            background: rgba(248, 249, 251, 0.45);
            margin-bottom: 0.9rem;
        }
        .small-note {
            font-size: 0.85rem;
            opacity: 0.8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def download_images(image_urls: List[str], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for idx, url in enumerate(image_urls):
        ext = Path(url.split("?")[0]).suffix.lower()
        if ext not in IMAGE_SUFFIXES:
            ext = ".jpg"
        target = output_dir / f"{idx}{ext}"
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            target.write_bytes(r.content)
            saved += 1
        except requests.RequestException:
            continue
    return saved


def save_uploaded_files(files, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for idx, f in enumerate(files):
        suffix = Path(f.name).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            suffix = ".jpg"
        target = output_dir / f"upload_{idx}{suffix}"
        target.write_bytes(f.getbuffer())
        saved += 1
    return saved


def _parse_storage_gb(memory_str: Optional[str]) -> Optional[int]:
    if not memory_str:
        return None
    value = memory_str.upper().replace(" ", "")
    if "TB" in value:
        try:
            return int(float(value.replace("TB", "")) * 1024)
        except (ValueError, TypeError):
            return None
    if "GB" in value:
        try:
            return int(float(value.replace("GB", "")))
        except (ValueError, TypeError):
            return None
    return None


def _parse_battery_pct(battery_str: Optional[str]) -> Optional[int]:
    if not battery_str:
        return None
    try:
        return int(battery_str.replace("%", "").strip())
    except (ValueError, TypeError):
        return None


def build_raw_listing(
    detail: Optional[Dict],
    listing_id: int,
    url: str,
    title_fallback: str,
    image_urls: List[str],
    description_override: Optional[str] = None,
) -> Dict:
    title = title_fallback
    description = description_override or ""
    price = None
    date = None

    if detail:
        title = str(detail.get("title") or title_fallback)
        additional = detail.get("additional") or {}
        description = description_override or additional.get("description") or detail.get("short_description") or ""
        price = detail.get("price")
        date = detail.get("date")

    return {
        "id": listing_id,
        "title": title,
        "price": price,
        "description": description,
        "date": date,
        "model": "",
        "images": ",".join(image_urls),
        "url": url,
    }


def summarize_cv(analysis: Dict) -> Dict:
    return {
        "storage_gb": _parse_storage_gb(analysis.get("internal_memory")),
        "battery_health_pct": _parse_battery_pct(analysis.get("battery_percentage")),
        "red_flag_cracked_front": 0,
        "red_flag_cracked_back": 0,
        "has_charger": 0,
        "has_original_box": 1 if analysis.get("hasBox") else 0,
        "condition_rating": float(analysis.get("condition_rating") or 0.0),
        "damage_confidence": float(analysis.get("damage_confidence") or 0.0),
    }


def _bool_from_phrase(value: str | None, positive_re: re.Pattern, negative_re: re.Pattern | None = None) -> int:
    if not value:
        return 0
    normalized = value.lower()
    if negative_re and negative_re.search(normalized):
        return 0
    return int(bool(positive_re.search(normalized)))


def summarize_bertic(text: str) -> Dict:
    features = extract_features(text or "")

    issues = features.get("issues") or ""
    icloud = features.get("icloud") or ""
    sim = features.get("sim") or ""
    box = features.get("box") or ""

    return {
        "storage_gb": features.get("storage_gb"),
        "battery_health_pct": features.get("battery_pct"),
        "condition_claim": features.get("condition"),
        "red_flag_cracked_front": _bool_from_phrase(issues, _CRACKED_RE),
        "red_flag_cracked_back": 0,
        "red_flag_icloud_locked": _bool_from_phrase(icloud, _LOCKED_RE),
        "red_flag_sim_locked": _bool_from_phrase(sim, _LOCKED_RE),
        "red_flag_not_functioning": _bool_from_phrase(issues, _NOT_WORKING_RE),
        "has_original_box": _bool_from_phrase(box, _BOX_RE, _NO_RE),
        "has_charger": _bool_from_phrase(text or "", _CHARGER_RE, _NO_RE),
    }


def render_result(merged: Dict, cv_summary: Dict, bertic_summary: Dict, analysis: Dict) -> None:
    st.subheader("Merged Data")
    c1, c2, c3 = st.columns(3)
    storage_value = merged.get("storage_gb")
    storage_label = f"{storage_value} GB" if storage_value else "N/A"
    battery_value = merged.get("battery_health_pct")
    battery_label = f"{battery_value}%" if battery_value else "N/A"
    c1.metric("Internal memory", storage_label)
    c2.metric("Battery health", battery_label)
    c3.metric("Has box", "Yes" if merged.get("has_original_box") else "No")

    with st.expander("Merged record", expanded=False):
        st.json(merged)

    with st.expander("Bertic summary", expanded=False):
        st.json(bertic_summary)

    with st.expander("CV summary", expanded=False):
        st.json(cv_summary)

    with st.expander("Detection summary", expanded=False):
        st.json(analysis.get("detections", {}))

    with st.expander("OCR text", expanded=False):
        st.json(analysis.get("ocr_text", {}))

    st.subheader("Best crops by class")
    best_crops = analysis.get("best_crops", {})
    if not best_crops:
        st.info("No crops available for display.")
        return

    items = sorted(best_crops.items())
    cols_per_row = 4
    for row_start in range(0, len(items), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, (cls_name, (conf, source, crop)) in zip(cols, items[row_start: row_start + cols_per_row]):
            with col:
                st.image(crop, width=220)
                st.caption(f"{cls_name} ({conf:.2f})")
                st.markdown(
                    f'<div class="small-note">{Path(str(source)).name}</div>',
                    unsafe_allow_html=True,
                )


def main() -> None:
    st.set_page_config(page_title="Smartphone Analyzer", layout="wide")
    apply_custom_styles()
    st.title("Smartphone Analyzer")
    st.write("Analyze OLX.ba listing images or your own uploads to extract memory, battery health, and box presence.")

    model_path = str((MODEL_DIR / MODEL_PATH).resolve())
    threshold = 0.45

    mode = st.radio("Input mode", ["OLX listing URL", "Upload images"], horizontal=True)

    if mode == "OLX listing URL":
        olx_url = st.text_input("OLX listing URL", placeholder="https://olx.ba/artikal/12345678")
        if st.button("Analyze listing", type="primary"):
            if not olx_url.strip():
                st.error("Please enter an OLX listing URL.")
                return

            listing_id = extract_listing_id(olx_url)
            if listing_id is None:
                st.error("Could not parse listing ID from URL.")
                return

            with st.spinner("Fetching listing and running analysis..."):
                with tempfile.TemporaryDirectory() as tmp:
                    img_dir = Path(tmp) / str(listing_id)
                    try:
                        detail: Optional[Dict] = None
                        detail = fetch_listing_detail(listing_id)
                        title = str(detail.get("title") or f"Listing {listing_id}")
                        image_urls = detail.get("images") or []
                        image_urls = [u for u in image_urls if isinstance(u, str) and u.strip()]
                        if not image_urls:
                            title, image_urls = fetch_listing_images(listing_id)
                        saved = download_images(image_urls, img_dir)
                    except Exception as err:
                        st.error(f"Failed to fetch listing: {err}")
                        return

                    if saved == 0:
                        st.error("No listing images could be downloaded.")
                        return

                    analysis = analyze_folder(img_dir, model_path=model_path, threshold=threshold)

                description = ""
                if detail:
                    additional = detail.get("additional") or {}
                    description = additional.get("description") or detail.get("short_description") or ""

                bertic_text = f"{title}\n{description}".strip()
                bertic_summary = summarize_bertic(bertic_text)
                cv_summary = summarize_cv(analysis)
                raw_listing = build_raw_listing(
                    detail=detail,
                    listing_id=listing_id,
                    url=olx_url,
                    title_fallback=title,
                    image_urls=image_urls,
                )
                merged = merge_records(raw_listing, bertic_summary, cv_summary)

            st.success(f"Analyzed listing {listing_id}: {title}")
            render_result(merged, cv_summary, bertic_summary, analysis)

    else:
        uploads = st.file_uploader(
            "Upload iPhone listing images",
            type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
        )
        upload_title = st.text_input("Listing title (optional)")
        upload_description = st.text_area("Listing description (optional)", height=120)
        if st.button("Analyze uploads", type="primary"):
            if not uploads:
                st.error("Please upload at least one image.")
                return

            with st.spinner("Running analysis on uploaded images..."):
                with tempfile.TemporaryDirectory() as tmp:
                    img_dir = Path(tmp) / "uploads"
                    save_uploaded_files(uploads, img_dir)
                    analysis = analyze_folder(img_dir, model_path=model_path, threshold=threshold)

                bertic_text = f"{upload_title}\n{upload_description}".strip()
                bertic_summary = summarize_bertic(bertic_text)
                cv_summary = summarize_cv(analysis)
                raw_listing = build_raw_listing(
                    detail=None,
                    listing_id=0,
                    url="",
                    title_fallback=upload_title or "Uploaded images",
                    image_urls=[],
                    description_override=upload_description,
                )
                merged = merge_records(raw_listing, bertic_summary, cv_summary)

            st.success(f"Analyzed {len(uploads)} uploaded images.")
            render_result(merged, cv_summary, bertic_summary, analysis)


if __name__ == "__main__":
    main()
