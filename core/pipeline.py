"""Pipeline — framework-agnostic orchestration for analyzing an OLX listing.

Wires the data sources together: OLX fetch -> object detection (+OCR) -> BERTic NLP
-> merge into a single listing record. Owns the model lifecycle via the lazy
singletons exposed by the model packages. No Streamlit/UI dependency.
"""

from __future__ import annotations

import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests

from models.crack_detection.inference import load_crack_model, predict_crack
from models.description_model.inference import extract_features, get_ner
from models.detection_model.batch_inference import IMAGE_SUFFIXES, analyze_folder
from models.detection_model.inference import get_session

from .merger import merge
from .olx_client import extract_listing_id, fetch_listing_detail, fetch_listing_images

DETECTION_THRESHOLD = 0.45

# ── Red-flag categorisation ────────────────────────────────────────────────
# BERTic already decided which phrases are defects (the FAIL spans); these
# patterns only ROUTE each span to a red-flag column. They run against the FAIL
# spans after diacritic folding, so they can stay ASCII and still match both
# "koči"/"koci", "zvučnik"/"zvucnik", etc. Because the input is already the set
# of model-flagged defects, the patterns can be permissive without over-firing
# on ordinary description text.
_CRACK_RE = re.compile(
    r"(puknu|pukla|puklo|pukotin|napuk|naprsl|razbij|slomljen|crack|smrskan|ostec|ostet)"
)
_CRACK_BACK_HINT_RE = re.compile(r"(zadnj|pozadi|nazad|ledj|\bback\b|kucist)")
_CRACK_FRONT_HINT_RE = re.compile(r"(ekran|displej|display|lcd|oled|naprijed|prednj|touch)")

# Everything that means "this phone is not fully working". Per the product
# decision a replaced/aftermarket SCREEN ("mijenjan/zamijenjen displej") folds
# in here; a replaced BATTERY does not (a new battery is neutral/positive), so
# the replaced-part clause is anchored to a screen/glass noun.
_NOT_WORKING_RE = re.compile(
    r"ne\s*rad"                                              # ne radi / ne rade
    r"|ne\s*pali|ne\s*ukljuc|nece\s*(da\s*)?(se\s*)?(upali|ukljuc)|ne\s*mogu\s*da\s*(ga\s*)?upal"
    r"|gasi\s*se|ugasi\s*se|se\s*gasi|zna\s*(se|da\s*se)\s*ugasi|pali\s*i\s*gasi|sam\s*gasi"
    r"|restart|resetuje\s*se\s*sam|resetira|vrti\s*logo|logo\s*blinka|blinka|boot\s*loop"
    r"|koci|zamrzava|smrzava|bagu"
    r"|ne\s*reagira|ne\s*reaguje|ne\s*registr|ne\s*prepoznaje"
    r"|za\s*dij?elove"
    r"|pokvaren|neispravan|u\s*kvaru|\bkvar\b|defekt"
    r"|mrtav?\s*piksel|mrtvih?\s*piksel|mrtva\s*piksel|dead\s*pixel|mrtvi\s*piksel"
    r"|burn|zapecen|gorenje\s*ekran|zut[ae]\s*(mrlj|fleka)"
    r"|linij|mrlj|flek|prasin"
    r"|napuh|brzo\s*se\s*prazni|kratko\s*traje|slaba\s*bater|drzi\s*(jako\s*)?kratko"
    r"|ne\s*puni|slabo\s*se\s*puni|ne\s*hvata\s*signal|nema\s*signal"
    r"|(mijenj|mjenj|zamijenj|zamjenj|promijenj|promjenj|zamjensk|nije\s*original|aftermarket)"
        r"[\s\w]{0,12}?(displej|ekran|display|lcd|oled|panel|staklo|touch)"
    r"|(displej|ekran|lcd|oled|touch|panel)\s*(je\s*)?(mijenj|mjenj|zamijenj|zamjenj|promijenj)"
    r"|not\s*work|does\s*not\s*work|problem\s*sa|ima\s*problem|ima\s*gresku|javlja\s*gresku"
    r"|\bmdm\b|bypass|odvojio|grije|pregrijava|fokusira|zamagljen|mutn|slika\s*mutna"
)
_LOCKED_RE = re.compile(r"(lock|locked|zakljucan|zakljucana)")
_NO_RE = re.compile(r"\b(no|bez|nije)\b")
_BOX_RE = re.compile(r"(box|kutija|full\s*box|puna\s*kutija)")


def _fold(text: str) -> str:
    """Lower-case and strip diacritics so keyword matching is accent-insensitive."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return norm.lower()


def _issue_red_flags(issues) -> Dict[str, int]:
    """Route the BERTic FAIL spans to red-flag columns.

    `issues` is the list of defect phrases the model extracted (possibly empty,
    or — for backwards safety — a single string). A listing can carry several
    independent defects, so we fold them into one blob and OR the families.
    """
    if isinstance(issues, str):
        issues = [issues] if issues else []
    blob = _fold(" ; ".join(issues))

    crack = bool(_CRACK_RE.search(blob))
    back = bool(_CRACK_BACK_HINT_RE.search(blob))
    front_hint = bool(_CRACK_FRONT_HINT_RE.search(blob))
    glass = "staklo" in blob  # "staklo" without a screen word means the back glass
    return {
        # Back if there's a back hint, or bare "staklo" damage with no screen
        # word. Front if a screen word is named, or a hint-less/glass-less crack
        # ("razbijen") — the front screen is the default and more penalising.
        "cracked_back": int(crack and (back or (glass and not front_hint))),
        "cracked_front": int(crack and (front_hint or (not back and not glass))),
        "not_functioning": int(bool(_NOT_WORKING_RE.search(blob))),
    }


class PipelineError(Exception):
    """Raised for expected, user-facing failures (bad URL, no images, fetch error)."""


class ListingNotFoundError(PipelineError):
    """Raised when the OLX listing no longer exists (HTTP 404/410) — do not retry."""


def preload_models() -> None:
    """Load all three models up front so the first analyze_listing() call is not slow.

    Optional — each model is a lazy singleton and loads on first use anyway.
    Call this once at startup in long-running processes (UI, server, worker).
    """
    get_session()
    load_crack_model()
    get_ner()


_RED_FLAG_LABELS = [
    ("red_flag_cracked_front", "Cracked front"),
    ("red_flag_cracked_back", "Cracked back"),
    ("red_flag_icloud_locked", "iCloud locked"),
    ("red_flag_sim_locked", "SIM locked"),
    ("red_flag_not_functioning", "Not functioning"),
]


@dataclass
class AnalysisResult:
    merged: Dict
    cv_summary: Dict
    bertic_summary: Dict
    bertic_raw: list
    analysis: Dict

    def summary(self) -> str:
        """Human-readable report of the merged record."""
        m = self.merged
        price = f"{m['price']} KM" if m.get("price") else "not listed"
        storage = f"{m['storage_gb']} GB" if m.get("storage_gb") else "unknown"
        battery = f"{m['battery_health_pct']}%" if m.get("battery_health_pct") else "unknown"
        flags = [label for key, label in _RED_FLAG_LABELS if m.get(key)]

        lines = [
            m.get("title") or f"Listing {m.get('id')}",
            m.get("url") or "",
            "",
            f"  Price:           {price}",
            f"  Model:           {(m.get('model') or 'unknown').title()}",
            f"  Storage:         {storage}",
            f"  Battery health:  {battery}",
            f"  Condition claim: {m.get('condition_claim') or 'none'}",
            f"  Original box:    {'yes' if m.get('has_original_box') else 'no'}",
            f"  Red flags:       {', '.join(flags) if flags else 'none detected'}",
        ]
        if m.get("damage_confidence"):
            lines.append(f"  Damage conf.:    {m['damage_confidence']:.2f}")
        return "\n".join(line for line in lines if line is not None)

    def __str__(self) -> str:
        return self.summary()


# --- Image download -------------------------------------------------------

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


# --- Field parsers --------------------------------------------------------

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


def _bool_from_phrase(value: str | None, positive_re: re.Pattern, negative_re: re.Pattern | None = None) -> int:
    if not value:
        return 0
    normalized = value.lower()
    if negative_re and negative_re.search(normalized):
        return 0
    return int(bool(positive_re.search(normalized)))


# --- Raw OLX record -------------------------------------------------------

def attributes_map(detail: Optional[Dict]) -> Dict[str, str]:
    """Flatten OLX's structured attributes into {attr_code: value}.

    OLX exposes specs (storage, RAM, colour, …) as a top-level `attributes` list
    of {attr_code, name, value, …}. We key by attr_code (lower-cased) so the merger
    can fall back to them when text/image extraction misses a field.
    """
    out: Dict[str, str] = {}
    for a in (detail or {}).get("attributes") or []:
        if not isinstance(a, dict):
            continue
        code = a.get("attr_code") or a.get("name")
        value = a.get("value")
        if code and value is not None:
            out[str(code).strip().lower()] = str(value)
    return out


def build_raw_listing(
    detail: Optional[Dict],
    listing_id: int,
    url: str,
    title_fallback: str,
    image_urls: List[str],
    description_override: Optional[str] = None,
    attributes: Optional[Dict[str, str]] = None,
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
        "attributes": attributes or {},
    }


# --- CV summary -----------------------------------------------------------

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
        "damage_confidence": max(float(analysis.get("damage_confidence") or 0.0), crack_confidence),
        "crack_sources": crack_sources,
    }


# --- NLP (BERTic) summary -------------------------------------------------

def summarize_bertic(text: str) -> tuple[Dict, list]:
    features, raw_entities = extract_features(text or "")

    issues = features.get("issues") or []
    icloud = features.get("icloud") or ""
    sim = features.get("sim") or ""
    box = features.get("box") or ""

    flags = _issue_red_flags(issues)

    summary = {
        "model": features.get("model"),
        "storage_gb": features.get("storage_gb"),
        "battery_health_pct": features.get("battery_pct"),
        "condition_claim": features.get("condition"),
        "red_flag_cracked_front": flags["cracked_front"],
        "red_flag_cracked_back": flags["cracked_back"],
        "red_flag_icloud_locked": _bool_from_phrase(icloud, _LOCKED_RE),
        "red_flag_sim_locked": _bool_from_phrase(sim, _LOCKED_RE),
        "red_flag_not_functioning": flags["not_functioning"],
        "has_original_box": _bool_from_phrase(box, _BOX_RE, _NO_RE),
    }
    return summary, raw_entities


# --- Public entry point ---------------------------------------------------

def analyze_listing(url: str, detail: Optional[Dict] = None) -> AnalysisResult:
    """Run the full pipeline for an OLX listing URL.

    If `detail` (a raw OLX listing-detail dict) is supplied, it's used as-is and
    no detail fetch is made — this lets a caller that already fetched the listing
    (e.g. the backend ingestion job) avoid a redundant round-trip. When omitted,
    the detail is fetched from the URL so standalone/CLI use still works from a
    URL alone. Images are always downloaded here regardless.

    Raises PipelineError for expected, user-facing failures.
    """
    listing_id = extract_listing_id(url)
    if listing_id is None:
        raise PipelineError("Could not parse listing ID from URL.")

    with tempfile.TemporaryDirectory() as tmp:
        img_dir = Path(tmp) / str(listing_id)
        try:
            if detail is None:
                detail = fetch_listing_detail(listing_id)
            title = str(detail.get("title") or f"Listing {listing_id}")
            image_urls = detail.get("images") or []
            image_urls = [u for u in image_urls if isinstance(u, str) and u.strip()]
            if not image_urls:
                title, image_urls = fetch_listing_images(listing_id)
            saved = download_images(image_urls, img_dir)
        except requests.HTTPError as err:
            status = err.response.status_code if err.response is not None else None
            if status in (404, 410):
                raise ListingNotFoundError(
                    f"Listing {listing_id} no longer exists on OLX (HTTP {status})."
                ) from err
            raise PipelineError(f"Failed to fetch listing: {err}") from err
        except Exception as err:
            raise PipelineError(f"Failed to fetch listing: {err}") from err

        # Missing/undownloadable images are non-fatal: we can still recover the
        # model, storage and battery from the description text and OLX attributes.
        if saved == 0:
            analysis = {}
        else:
            analysis = analyze_folder(img_dir, threshold=DETECTION_THRESHOLD, session=get_session())

    additional = (detail.get("additional") or {}) if detail else {}
    description = additional.get("description") or (detail.get("short_description") if detail else None) or ""

    bertic_text = f"{title}\n{description}".strip()
    bertic_summary, bertic_raw = summarize_bertic(bertic_text)
    cv_summary = summarize_cv(analysis)
    raw_listing = build_raw_listing(
        detail=detail,
        listing_id=listing_id,
        url=url,
        title_fallback=title,
        image_urls=image_urls,
        description_override=description,
        attributes=attributes_map(detail),
    )
    merged = merge(raw_listing, bertic_summary, cv_summary)

    return AnalysisResult(
        merged=merged,
        cv_summary=cv_summary,
        bertic_summary=bertic_summary,
        bertic_raw=bertic_raw,
        analysis=analysis,
    )
