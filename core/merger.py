import re

def _int_or_none(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def _parse_storage_from_attribute(raw_value: str) -> int | None:
    if not raw_value:
        return None
    m = re.search(r"(\d+)", raw_value)
    return int(m.group(1)) if m else None


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

    # NLP wins (parsed from description), then CV (OCR), then OLX attribute
    storage_gb = (
        _int_or_none(nlp.get("storage_gb"))
        or _int_or_none(cv.get("storage_gb"))
        or _parse_storage_from_attribute(raw.get("interna_memorija", ""))
    )

    # NLP wins (text mention), then CV (photo of battery settings screen)
    battery_health_pct = (
        _int_or_none(nlp.get("battery_health_pct"))
        or _int_or_none(cv.get("battery_health_pct"))
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

        # Accessories — NLP OR CV
        "has_charger":      _either_bool(nlp, cv, "has_charger"),
        "has_original_box": _either_bool(nlp, cv, "has_original_box"),

        # CV only
        "condition_rating":  float(cv.get("condition_rating") or 0.0),
        "damage_confidence": float(cv.get("damage_confidence") or 0.0),

        # Pipeline state
        "analyzed": 1,
    }
