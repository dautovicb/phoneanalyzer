import re

# Plausible iPhone storage capacities (GB) — guards against grabbing a stray
# number (e.g. a RAM value or a price) as the storage size.
_KNOWN_STORAGE_GB = {16, 32, 64, 128, 256, 512, 1024}


def _int_or_none(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def _parse_storage_from_attribute(raw_value: str) -> int | None:
    if not raw_value:
        return None
    value = raw_value.upper().replace(" ", "")
    mult = 1024 if "TB" in value else 1
    m = re.search(r"(\d+)", value)
    if not m:
        return None
    gb = int(m.group(1)) * mult
    return gb if gb in _KNOWN_STORAGE_GB else None


def _storage_from_attributes(attrs: dict | None) -> int | None:
    """OLX exposes storage as a structured attribute (attr_code 'interna-memorija',
    value like '256 GB'). Use it when text/image extraction came up empty."""
    if not attrs:
        return None
    for code, val in attrs.items():
        if "memorij" in code:  # interna-memorija
            gb = _parse_storage_from_attribute(str(val))
            if gb:
                return gb
    return None


def _battery_from_attributes(attrs: dict | None) -> int | None:
    """Battery health isn't a standard OLX attribute today, but read it defensively
    in case a listing carries one (attr_code containing 'bateri')."""
    if not attrs:
        return None
    for code, val in attrs.items():
        if "bateri" in code or "battery" in code:
            m = re.search(r"(\d{1,3})", str(val))
            if m:
                pct = int(m.group(1))
                if 1 <= pct <= 100:
                    return pct
    return None


def _storage_from_text(*texts: str | None) -> int | None:
    """Last-resort storage parse from the title/description (e.g. 'iPhone 13 128GB')."""
    blob = " ".join(t for t in texts if t)
    for num, unit in re.findall(r"(\d{2,4})\s*(tb|gb)", blob, flags=re.IGNORECASE):
        gb = int(num) * (1024 if unit.lower() == "tb" else 1)
        if gb in _KNOWN_STORAGE_GB:
            return gb
    return None


def _battery_from_text(*texts: str | None) -> int | None:
    """Parse battery health from the title/description. Most sellers write it as
    '87% baterije' / 'baterija 100%' — require a battery keyword near the number so
    we don't grab an unrelated percentage (discount, screen size, etc.)."""
    blob = " ".join(t for t in texts if t).lower()

    # "<pct>% ... bateri" or "bateri ... <pct>%" — keyword within a short window.
    for m in re.finditer(r"(\d{2,3})\s*%", blob):
        window = blob[max(0, m.start() - 22): m.end() + 22]
        if "bateri" in window or "batter" in window or "zdravlje" in window:
            pct = int(m.group(1))
            if 50 <= pct <= 100:
                return pct

    # "baterija 87" (no % sign)
    m = re.search(r"(?:bateri\w*|batter\w*|zdravlje)\D{0,12}(\d{2,3})", blob)
    if m:
        pct = int(m.group(1))
        if 50 <= pct <= 100:
            return pct
    return None


def _either_bool(nlp: dict, cv: dict, key: str) -> int:
    return int(bool(nlp.get(key)) or bool(cv.get(key)))


def merge(raw: dict, nlp: dict, cv: dict) -> dict:
    """Merge the three data sources into a single flat listing record.

    Args:
        raw: Output of build_raw_listing() — OLX API fields.
        nlp: Output of summarize_bertic() — BERTic NER text-extracted fields.
        cv:  Output of summarize_cv() — computer-vision fields.

    Returns:
        Dict whose keys match exactly the columns in db.upsert_listing().
    """

    attrs = raw.get("attributes") or {}
    title = raw.get("title", "")
    description = raw.get("description", "")

    # NLP wins (parsed from description), then CV (OCR), then the OLX structured
    # attribute, then a plain title/description parse — so we only fall back to
    # weaker sources when the stronger ones extracted nothing.
    storage_gb = (
        _int_or_none(nlp.get("storage_gb"))
        or _int_or_none(cv.get("storage_gb"))
        or _storage_from_attributes(attrs)
        or _storage_from_text(title, description)
    )

    # NLP wins (text mention), then CV (photo of battery settings screen), then an
    # OLX attribute if present, then a title/description parse ('87% baterije').
    battery_health_pct = (
        _int_or_none(nlp.get("battery_health_pct"))
        or _int_or_none(cv.get("battery_health_pct"))
        or _battery_from_attributes(attrs)
        or _battery_from_text(title, description)
    )

    return {
        # Core OLX fields — passed through unchanged
        "id":          raw["id"],
        "title":       raw.get("title", ""),
        "price":       raw.get("price"),
        "description": raw.get("description", ""),
        "date":        raw.get("date"),
        "images":      raw.get("images", ""),
        "url":         raw.get("url", ""),

        # Resolved fields
        "model":              nlp.get("model") or "",
        "storage_gb":         storage_gb,
        "battery_health_pct": battery_health_pct,
        "condition_claim":    nlp.get("condition_claim"),

        # Red flags — cracked can be caught by NLP (description) OR CV (photos)
        "red_flag_cracked_front":    _either_bool(nlp, cv, "red_flag_cracked_front"),
        "red_flag_cracked_back":     _either_bool(nlp, cv, "red_flag_cracked_back"),

        # Red flags — NLP only (not visually detectable)
        "red_flag_icloud_locked":    int(bool(nlp.get("red_flag_icloud_locked"))),
        "red_flag_sim_locked":       int(bool(nlp.get("red_flag_sim_locked"))),
        "red_flag_not_functioning":  int(bool(nlp.get("red_flag_not_functioning"))),

        # Packaging — NLP OR CV
        "has_original_box": _either_bool(nlp, cv, "has_original_box"),

        # CV only
        "damage_confidence": float(cv.get("damage_confidence") or 0.0),

        # Pipeline state
        "analyzed": 1,
    }
