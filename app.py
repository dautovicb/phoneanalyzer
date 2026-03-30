from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List

import requests
import streamlit as st

MODEL_DIR = Path(__file__).resolve().parent / "detection_model"
if str(MODEL_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(MODEL_DIR))

from batch_inference import IMAGE_SUFFIXES, analyze_folder  # type: ignore
from inference import MODEL_PATH  # type: ignore
from olx_client import extract_listing_id, fetch_listing_images


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


def render_result(result: Dict) -> None:
    st.subheader("Extracted Data")
    c1, c2, c3 = st.columns(3)
    c1.metric("Internal memory", result.get("internal_memory") or "N/A")
    c2.metric("Battery health", result.get("battery_percentage") or "N/A")
    c3.metric("Has box", "Yes" if result.get("hasBox") else "No")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Detection summary", expanded=False):
        st.json(result.get("detections", {}))

    with st.expander("OCR text", expanded=False):
        st.json(result.get("ocr_text", {}))

    st.subheader("Best crops by class")
    best_crops = result.get("best_crops", {})
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
                        title, image_urls = fetch_listing_images(listing_id)
                        saved = download_images(image_urls, img_dir)
                    except Exception as err:
                        st.error(f"Failed to fetch listing: {err}")
                        return

                    if saved == 0:
                        st.error("No listing images could be downloaded.")
                        return

                    result = analyze_folder(img_dir, model_path=model_path, threshold=threshold)

            st.success(f"Analyzed listing {listing_id}: {title}")
            render_result(result)

    else:
        uploads = st.file_uploader(
            "Upload iPhone listing images",
            type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
        )
        if st.button("Analyze uploads", type="primary"):
            if not uploads:
                st.error("Please upload at least one image.")
                return

            with st.spinner("Running analysis on uploaded images..."):
                with tempfile.TemporaryDirectory() as tmp:
                    img_dir = Path(tmp) / "uploads"
                    save_uploaded_files(uploads, img_dir)
                    result = analyze_folder(img_dir, model_path=model_path, threshold=threshold)

            st.success(f"Analyzed {len(uploads)} uploaded images.")
            render_result(result)


if __name__ == "__main__":
    main()
