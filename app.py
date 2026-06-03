from __future__ import annotations

import importlib.util
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests
import streamlit as st # type: ignore
from PIL import Image # type: ignore

MODEL_DIR = Path(__file__).resolve().parent / "detection_model"
if str(MODEL_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(MODEL_DIR))

from batch_inference import IMAGE_SUFFIXES, analyze_folder  # type: ignore
from inference import MODEL_PATH, load_model as load_detection_model  # type: ignore
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

CRACK_MODEL_PATH = Path(__file__).resolve().parent / "crack_detection" / "model" / "crack_detector.keras"

_CRACKED_RE = re.compile(r"(puknut|pukla|crack|cracked|razbijen|razbijeno)")
_NOT_WORKING_RE = re.compile(r"(ne\s*rad(i|e)|not\s*work|does\s*not\s*work|defekt)")
_LOCKED_RE = re.compile(r"(lock|locked|zakljucan|zakljucana)")
_NO_RE = re.compile(r"\b(no|bez|nije)\b")
_BOX_RE = re.compile(r"(box|kutija|full\s*box|puna\s*kutija)")


def apply_custom_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        .listing-banner {
            padding: 1rem 1.25rem;
            border-radius: 14px;
            background: linear-gradient(135deg, rgba(37,99,235,0.06) 0%, rgba(99,102,241,0.09) 100%);
            border: 1px solid rgba(99,102,241,0.2);
            margin-bottom: 1.4rem;
        }
        .listing-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.25rem; }
        .listing-price { font-size: 1.5rem; font-weight: 700; color: #2563eb; }
        .results-card {
            padding: 0.75rem 1rem;
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 12px;
            background: rgba(248,249,251,0.5);
            margin-bottom: 0.8rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .small-note { font-size: 0.83rem; opacity: 0.7; }
        .flag-pill {
            display: inline-block;
            padding: 0.3rem 0.85rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.01em;
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



def _parse_storage_gb(memory_str: Optional[str]) -> Optional[int]:
    if not memory_str:
        return None
    value = memory_str.upper().replace(" ", "")
    if "TB" in value:
        unit, multiplier = "TB", 1024
    elif "GB" in value:
        unit, multiplier = "GB", 1
    else:
        return None
    try:
        return int(float(value.replace(unit, "")) * multiplier)
    except (ValueError, TypeError):
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
        "images": ",".join(image_urls),
        "url": url,
    }


def _check_crack(best_crops: Dict, key: str) -> tuple[bool, float, Optional[str]]:
    if key not in best_crops:
        return False, 0.0, None
    _, source, crop = best_crops[key]
    cracked, confidence = predict_crack(crop)
    return cracked, confidence, str(source)


def summarize_cv(analysis: Dict) -> Dict:
    best_crops = analysis.get("best_crops", {})
    crack_sources = {}
    crack_confidence = 0.0

    front_cracked, front_conf, front_src = _check_crack(best_crops, "phone_front")
    back_cracked, back_conf, back_src = _check_crack(best_crops, "phone_back")

    for key, src, conf in [("phone_front", front_src, front_conf), ("phone_back", back_src, back_conf)]:
        if src:
            crack_sources[key] = src
            crack_confidence = max(crack_confidence, conf)

    return {
        "storage_gb": _parse_storage_gb(analysis.get("internal_memory")),
        "battery_health_pct": _parse_battery_pct(analysis.get("battery_percentage")),
        "red_flag_cracked_front": int(front_cracked),
        "red_flag_cracked_back": int(back_cracked),
        "has_original_box": 1 if analysis.get("hasBox") else 0,
        "condition_rating": float(analysis.get("condition_rating") or 0.0),
        "damage_confidence": max(float(analysis.get("damage_confidence") or 0.0), crack_confidence),
        "crack_sources": crack_sources,
    }


def _bool_from_phrase(value: str | None, positive_re: re.Pattern, negative_re: re.Pattern | None = None) -> int:
    if not value:
        return 0
    normalized = value.lower()
    if negative_re and negative_re.search(normalized):
        return 0
    return int(bool(positive_re.search(normalized)))


@st.cache_resource(show_spinner=False)
def load_detection_session(model_path: str):
    return load_detection_model(model_path)


@st.cache_resource(show_spinner=False)
def load_crack_model():
    return _load_keras_model(CRACK_MODEL_PATH)


def _load_keras_model(model_path: Path):
    try:
        import tensorflow as tf

        return tf.keras.models.load_model(str(model_path), compile=False)
    except TypeError as exc:
        try:
            import keras

            if hasattr(keras, "config") and hasattr(keras.config, "enable_legacy_deserialization"):
                keras.config.enable_legacy_deserialization()

            if hasattr(keras, "saving") and hasattr(keras.saving, "load_model"):
                return keras.saving.load_model(str(model_path), compile=False)

            return keras.models.load_model(str(model_path), compile=False)
        except Exception:
            raise exc


def predict_crack(crop: Image.Image) -> tuple[bool, float]:
    import tensorflow as tf

    model = load_crack_model()
    img = crop.convert("RGB").resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    score = float(predictions.item())

    if score < 0.5:
        return True, 1 - score
    return False, score


def summarize_bertic(text: str) -> tuple[Dict, list]:
    features, raw_entities = extract_features(text or "")

    issues = features.get("issues") or ""
    icloud = features.get("icloud") or ""
    sim = features.get("sim") or ""
    box = features.get("box") or ""

    summary = {
        "model": features.get("model"),
        "storage_gb": features.get("storage_gb"),
        "battery_health_pct": features.get("battery_pct"),
        "condition_claim": features.get("condition"),
        "red_flag_cracked_front": _bool_from_phrase(issues, _CRACKED_RE),
        "red_flag_cracked_back": 0,
        "red_flag_icloud_locked": _bool_from_phrase(icloud, _LOCKED_RE),
        "red_flag_sim_locked": _bool_from_phrase(sim, _LOCKED_RE),
        "red_flag_not_functioning": _bool_from_phrase(issues, _NOT_WORKING_RE),
        "has_original_box": _bool_from_phrase(box, _BOX_RE, _NO_RE),
    }
    return summary, raw_entities


def render_result(merged: Dict, cv_summary: Dict, bertic_summary: Dict, bertic_raw: list, analysis: Dict) -> None:
    title = merged.get("title") or ""
    price = merged.get("price")
    price_str = f"{price}" if price else "Price not listed"
    st.markdown(
        f'<div class="listing-banner">'
        f'<div class="listing-title">{title}</div>'
        f'<div class="listing-price">{price_str}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    model_label = ((merged.get("model") or "").title() or "N/A").upper()
    storage_value = merged.get("storage_gb")
    storage_label = f"{storage_value} GB" if storage_value else "N/A"
    battery_value = merged.get("battery_health_pct")
    battery_label = f"{battery_value}%" if battery_value else "N/A"
    box_label = "Yes" if merged.get("has_original_box") else "No"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", model_label)
    c2.metric("Storage", storage_label)
    c3.metric("Battery health", battery_label)
    c4.metric("Original box", box_label)

    st.divider()

    flags = [
        ("Cracked front", merged.get("red_flag_cracked_front")),
        ("Cracked back", merged.get("red_flag_cracked_back")),
        ("iCloud locked", merged.get("red_flag_icloud_locked")),
        ("SIM locked", merged.get("red_flag_sim_locked")),
        ("Not functioning", merged.get("red_flag_not_functioning")),
    ]
    pills_html = ""
    for label, flagged in flags:
        if flagged:
            pills_html += f'<span class="flag-pill" style="background:#fef2f2;color:#c0392b;border:1px solid #f5c6c6">{label}</span> '
        else:
            pills_html += f'<span class="flag-pill" style="background:#f0fdf4;color:#16a34a;border:1px solid #bbf7d0">{label}</span> '
    st.markdown(
        f'<div style="margin-bottom:1.2rem">'
        f'<div class="small-note" style="margin-bottom:0.45rem">RED FLAGS</div>'
        f'{pills_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    with st.expander("Merged record", expanded=False):
        st.json(merged)
    with st.expander("Bertic summary", expanded=False):
        st.json(bertic_summary)
    with st.expander("Bertic raw NER entities", expanded=False):
        st.json(bertic_raw)
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
    st.write("Paste an OLX.ba listing URL to extract storage, battery health, and condition signals.")

    model_path = str((MODEL_DIR / MODEL_PATH).resolve())
    threshold = 0.45
    detection_session = load_detection_session(model_path)

    olx_url = st.text_input("OLX listing URL", placeholder="https://olx.ba/artikal/12345678")
    if st.button("Analyze", type="primary"):
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

                analysis = analyze_folder(img_dir, threshold=threshold, session=detection_session)

            additional = (detail.get("additional") or {}) if detail else {}
            description = additional.get("description") or (detail.get("short_description") if detail else None) or ""

            bertic_text = f"{title}\n{description}".strip()
            bertic_summary, bertic_raw = summarize_bertic(bertic_text)
            cv_summary = summarize_cv(analysis)
            raw_listing = build_raw_listing(
                detail=detail,
                listing_id=listing_id,
                url=olx_url,
                title_fallback=title,
                image_urls=image_urls,
                description_override=description,
            )
            merged = merge_records(raw_listing, bertic_summary, cv_summary)

        render_result(merged, cv_summary, bertic_summary, bertic_raw, analysis)


if __name__ == "__main__":
    main()
