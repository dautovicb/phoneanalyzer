from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

# Put the module root on sys.path so `core` and the model packages import cleanly
# when launched via `streamlit run demo/app.py`. Removed once the package is
# pip-installed (`pip install -e .`).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # type: ignore

from core import pipeline
from core.pipeline import AnalysisResult


def apply_custom_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        .price-tag {
            font-size: 2.4rem;
            font-weight: 800;
            color: #2563eb;
            line-height: 1.1;
            margin: 0.2rem 0 1.4rem;
        }
        .price-tag .price-unit { font-size: 1.3rem; font-weight: 600; opacity: 0.75; margin-left: 0.35rem; }
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


def render_result(merged: Dict, cv_summary: Dict, bertic_summary: Dict, bertic_raw: list, analysis: Dict) -> None:
    price = merged.get("price")
    if price:
        st.markdown(
            f'<div class="price-tag">{price}<span class="price-unit">KM</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="price-tag">Price not listed</div>', unsafe_allow_html=True)

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
    detected = [label for label, flagged in flags if flagged]
    if detected:
        pills_html = "".join(
            f'<span class="flag-pill" style="background:#fef2f2;color:#c0392b;border:1px solid #f5c6c6">{label}</span> '
            for label in detected
        )
    else:
        pills_html = '<span class="flag-pill" style="background:#f0fdf4;color:#16a34a;border:1px solid #bbf7d0">None detected</span>'
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

    olx_url = st.text_input("OLX listing URL", placeholder="https://olx.ba/artikal/12345678")
    if not st.button("Analyze", type="primary"):
        return

    if not olx_url.strip():
        st.error("Please enter an OLX listing URL.")
        return

    try:
        with st.spinner("Fetching listing and running analysis..."):
            result: AnalysisResult = pipeline.analyze_listing(olx_url)
    except pipeline.PipelineError as err:
        st.error(str(err))
        return

    render_result(
        result.merged,
        result.cv_summary,
        result.bertic_summary,
        result.bertic_raw,
        result.analysis,
    )


if __name__ == "__main__":
    main()
