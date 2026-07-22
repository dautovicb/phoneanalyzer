from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "model"

# Spec fields (brand/model/memory/…) are high-confidence, so keep a firm gate.
# FAIL spans run lower-confidence — the model is often only ~0.3 sure a phrase is
# a defect — and missing a red flag is costlier than missing a spec, so gate FAIL
# lower. Validated against real listings: dropping from 0.40 to 0.30 does not
# raise the false-positive rate (junk sub-word fragments are filtered separately).
_MIN_SCORE = 0.4
_FAIL_MIN_SCORE = 0.3

_ner = None


def get_ner():
    """Lazily load and cache the BERTic NER pipeline."""
    global _ner
    if _ner is None:
        from transformers import pipeline

        _ner = pipeline(
            "ner",
            model=str(MODEL_DIR),
            aggregation_strategy="simple",
        )
    return _ner

def merge_entities(results):
    if not results:
        return []

    merged = []
    current = dict(results[0])

    for entity in results[1:]:
        same_group = entity["entity_group"] == current["entity_group"]
        word = entity["word"]
        is_subword = word.startswith("##")

        # Character gap to the previous entity (fast tokenizer supplies offsets;
        # None if unavailable, in which case we fall back to the old behaviour).
        start, end = entity.get("start"), current.get("end")
        gap = (start - end) if (start is not None and end is not None) else None

        # Merge subword continuations ("25" + "##6GB" -> "256GB") and pieces that
        # are physically contiguous (gap 0). Do NOT merge same-group entities
        # separated by whitespace/newline — those are distinct mentions. Without
        # this, a model named in both the title and the description merged into
        # "iphone 12 iphone 12"; keeping them separate lets the caller pick the
        # single best-scoring one.
        if same_group and (is_subword or gap is None or gap <= 0):
            if is_subword:
                current["word"] += word[2:]
            elif gap == 0:
                current["word"] += word
            else:
                current["word"] += " " + word
            current["score"] = min(current["score"], entity["score"])
            if start is not None:
                current["end"] = entity.get("end", end)
        else:
            merged.append(current)
            current = dict(entity)

    merged.append(current)
    return merged

def extract_features(text: str) -> tuple[dict, list]:
    results = get_ner()(text)
    results = merge_entities(results)
    raw_entities = results
    
    features = {
        "brand": (None, 0),
        "model": (None, 0),
        "storage_gb": (None, 0),
        "battery_pct": (None, 0),
        "condition": (None, 0),
        "icloud": (None, 0),
        "sim": (None, 0),
        "box": (None, 0),
    }
    # FAIL is multi-valued: one listing routinely reports several independent
    # defects ("restartuje se ... prednja kamera ne radi ... zadnje staklo ima
    # liniju"). Keeping only the top-scoring span (as the single-value fields do)
    # would drop every defect but one, so collect them all — the pipeline ORs the
    # red-flag families across the whole list.
    issues: list[str] = []

    def update(key, value, score):
        if score > features[key][1]:
            features[key] = (value, score)

    for entity in results:
        group = entity["entity_group"]
        word = entity["word"].strip()
        score = entity["score"]

        if group == "FAIL":
            # Own (lower) threshold, and drop junk sub-word fragments that
            # survive with no meaning — e.g. "##i", "ot" clipped off a longer
            # word — which would otherwise pollute the extracted defect list.
            w = word.lower()
            if (
                score >= _FAIL_MIN_SCORE
                and len(w) >= 3
                and not w.startswith("##")
                and w not in issues
            ):
                issues.append(w)
            continue

        if score < _MIN_SCORE:
            continue

        if group == "BRAND":
            update("brand", word.lower(), score)
        elif group == "MOD":
            update("model", word.lower(), score)
        elif group == "MEM":
            digits = ''.join(filter(str.isdigit, word))
            if digits:
                update("storage_gb", int(digits), score)
        elif group == "BATT":
            digits = ''.join(filter(str.isdigit, word))
            if digits:
                update("battery_pct", int(digits), score)
        elif group == "COND":
            update("condition", word.lower(), score)
        elif group == "ICL":
            update("icloud", word.lower(), score)
        elif group == "SIM":
            update("sim", word.lower(), score)
        elif group == "BOX":
            update("box", word.lower(), score)

    out = {k: v[0] for k, v in features.items()}
    out["issues"] = issues
    return out, raw_entities


if __name__ == "__main__":
    import sys

    for text in sys.argv[1:]:
        print(f"{'-'*60}")
        print(f"{text} -> {extract_features(text)}")
