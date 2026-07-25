"""Run the FULL pipeline over whole OLX listings and output a combined verdict.

For each listing this fetches the ad, downloads every photo, and combines two
independent damage signals the way the product should:

  * VISION  — object-detect phone_front/back crops across all photos, score cracks
              with a swappable model (default v3) and MAX-POOL (a crack in any
              photo = cracked). Mirrors core/pipeline.py's CV path.
  * TEXT    — run the BERTic description model over title+description and route
              its FAIL spans to red flags (not_functioning, replaced screen,
              iCloud/SIM lock, text-mentioned cracks). The description routinely
              states damage the photos hide, so it's often the stronger signal.

A listing is FLAGGED if either signal fires; the reason codes show which.

Input:  a text file of listing URLs (default test_listings.txt), one per line,
        full-line '#' comments / blanks ignored. olx.ba/artikal/<id>/ URLs, raw
        IDs, and /api/listings/<id> URLs all work. A *trailing* '# LABEL — note'
        comment is ground truth (CLEAN/FRONT/BACK/BOTH/PIXEL/NOTWORK) and is
        scored, not discarded — without it the run has no meaning.
Output: console table + results.csv (every field + ground truth + extracted
        issues + desc) + a metrics block: confusion matrices and P/R/F1 for the
        overall flag decision and per red flag, plus the misses by id.
        With --save-crops, the decisive front/back crop per listing is written
        under listing_test_out/<verdict>/ for review.

Run from ml/phoneanalyzer/:
    python models/crack_detection/test_listings.py --save-crops
    options: --urls FILE  --crack-model v3  --threshold 0.50  --no-text
             --limit N  --delay 0.3  --out DIR
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
ML_ROOT = HERE.parent.parent  # ml/phoneanalyzer
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from core.olx_client import extract_listing_id, fetch_listing_detail
from core.pipeline import (
    DETECTION_THRESHOLD, download_images,
    _issue_red_flags, _bool_from_phrase, _LOCKED_RE, _BOX_RE, _NO_RE,
)
from models.detection_model.batch_inference import analyze_folder
from models.detection_model.inference import get_session
from models.crack_detection.inference import _load_keras_model
from models.description_model.inference import extract_features

MODEL_DIR = HERE / "model"
REGISTRY = {
    "v1": MODEL_DIR / "crack_detector.keras",
    "v2": MODEL_DIR / "crack_detector_v2.keras",
    "v3": MODEL_DIR / "crack_detector_v3.keras",
    "v4": MODEL_DIR / "crack_detector_v4.keras",
    # v4 was saved by Keras 3.13 (it writes `quantization_config` into every
    # Dense config) and will NOT load on the pinned 3.12 runtime. v4_local is the
    # same net re-exported by 3.12 without optimizer state — the one to benchmark
    # here. Drop this alias once the runtime moves to >= 3.13.
    "v4_local": MODEL_DIR / "crack_detector_v4_local.keras",
}


def model_input_size(model, default: int = 224) -> int:
    """Square input side the model expects (v2=224, v3=384), read off the model."""
    shp = getattr(model, "input_shape", None)
    if isinstance(shp, list) and shp:
        shp = shp[0]
    if shp and len(shp) == 4 and shp[1] and shp[2]:
        return int(shp[1])
    return default


class CrackScorer:
    """Loads a crack model and scores a PIL crop -> P(cracked). See module notes on
    polarity: models trained by cracks.py (cracked=0<not_cracked=1) and the shipped
    v1 path both make P(cracked) = 1 - output, so flip defaults to True."""

    def __init__(self, model_path: Path, flip: bool = True):
        self.model = _load_keras_model(model_path)
        self.size = model_input_size(self.model)
        self.flip = flip

    def p_cracked(self, crop: Image.Image) -> float:
        arr = np.asarray(crop.convert("RGB").resize((self.size, self.size)), dtype="float32")
        raw = float(self.model.predict(arr[None], verbose=0).reshape(-1)[0])
        p = (1.0 - raw) if self.flip else raw
        return float(np.clip(p, 0.0, 1.0))

    def max_pool(self, crops):
        """Return (best_p, best_crop, best_source) over a class's crops."""
        best_p, best_crop, best_src = 0.0, None, None
        for _conf, source, crop in (crops or []):
            p = self.p_cracked(crop)
            if p >= best_p:
                best_p, best_crop, best_src = p, crop, source
        return best_p, best_crop, best_src


def analyze_text(title: str, description: str) -> dict:
    """BERTic over title+description -> the same red flags core/pipeline.py derives."""
    text = f"{title}\n{description}".strip()
    features, _ = extract_features(text or "")
    issues = features.get("issues") or []
    flags = _issue_red_flags(issues)
    return {
        "not_functioning": int(flags["not_functioning"]),
        "cracked_front": int(flags["cracked_front"]),
        "cracked_back": int(flags["cracked_back"]),
        "icloud": _bool_from_phrase(features.get("icloud") or "", _LOCKED_RE),
        "sim": _bool_from_phrase(features.get("sim") or "", _LOCKED_RE),
        "condition": features.get("condition") or "",
        "issues": issues,
    }


# ── ground truth ──────────────────────────────────────────────────────────
# Every URL in test_listings.txt carries the seller-stated truth inline
# ("# BACK — razbijeno zadnje staklo"). Parsing it is what turns this script
# from a list of guesses into a scoreboard, so the labels are part of the input
# format, not a comment to strip.
GT_CATEGORIES = ("CLEAN", "FRONT", "BACK", "BOTH", "PIXEL", "NOTWORK")
_GT_TOKEN_RE = re.compile(r"\b(CLEAN|FRONT|BACK|BOTH|PIXEL|NOTWORK)\b")


def parse_labels(comment: str) -> dict | None:
    """Ground-truth flags from a trailing '# FRONT + NOTWORK — ...' comment.

    Categories are read from the head of the comment only (before the em dash);
    the tail is free Bosnian prose describing the defect and would produce
    spurious matches. The one exception is the explicit '(+ iCloud)' note,
    which is a label the seller stated, so it is scanned for over the whole
    comment. Returns None when the line carries no label — those listings are
    still analysed, just excluded from the metrics.
    """
    head = comment.split("—")[0].split("–")[0]
    cats = set(_GT_TOKEN_RE.findall(head))
    icloud = "icloud" in comment.lower()
    if not cats and not icloud:
        return None
    return {
        "categories": cats,
        # BOTH is shorthand for FRONT+BACK. PIXEL (dead/stuck pixel) is routed to
        # not_functioning because that is where core/pipeline.py's _NOT_WORKING_RE
        # sends "piksel" spans — scoring it as its own class would measure a
        # distinction the pipeline does not make.
        "cracked_front": int(bool(cats & {"FRONT", "BOTH"})),
        "cracked_back": int(bool(cats & {"BACK", "BOTH"})),
        "not_functioning": int(bool(cats & {"NOTWORK", "PIXEL"})),
        "icloud": int(icloud),
        # A listing is a positive if it carries any defect at all; CLEAN alone
        # is the only negative. This is the decision the product actually makes.
        "problem": int(bool(cats - {"CLEAN"}) or icloud),
    }


def read_listings(path: Path):
    """Yield {url, labels, comment} per line. Full-line '#' comments skipped."""
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        url, _, comment = line.partition("#")
        entries.append({
            "url": url.strip(),
            "comment": comment.strip(),
            "labels": parse_labels(comment) if comment.strip() else None,
        })
    return entries


# ── scoring ───────────────────────────────────────────────────────────────

def confusion(pairs):
    """(tp, fp, fn, tn) over [(gt_bool, pred_bool), ...]."""
    tp = sum(1 for g, p in pairs if g and p)
    fp = sum(1 for g, p in pairs if not g and p)
    fn = sum(1 for g, p in pairs if g and not p)
    tn = sum(1 for g, p in pairs if not g and not p)
    return tp, fp, fn, tn


def prf(tp: int, fp: int, fn: int, tn: int):
    """precision, recall, f1, accuracy — 0.0 where the denominator is empty."""
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else 0.0
    return prec, rec, f1, acc


def print_matrix(name: str, pairs) -> None:
    """A labelled 2x2 confusion matrix plus its P/R/F1."""
    tp, fp, fn, tn = confusion(pairs)
    prec, rec, f1, acc = prf(tp, fp, fn, tn)
    print(f"  {name}   (n={len(pairs)})")
    print(f"      {'':14}{'pred +':>8}{'pred -':>8}")
    print(f"      {'actual +':14}{tp:>8}{fn:>8}    recall     {rec:.3f}")
    print(f"      {'actual -':14}{fp:>8}{tn:>8}    precision  {prec:.3f}")
    print(f"      {'':30}    F1         {f1:.3f}")
    print(f"      {'':30}    accuracy   {acc:.3f}")


def print_signal_table(rows) -> None:
    """One row per signal for the binary flag decision — shows who carries it."""
    print(f"  {'signal':10}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}"
          f"{'prec':>8}{'rec':>8}{'F1':>8}{'acc':>8}")
    for name, pairs in rows:
        tp, fp, fn, tn = confusion(pairs)
        prec, rec, f1, acc = prf(tp, fp, fn, tn)
        print(f"  {name:10}{tp:>5}{fp:>5}{fn:>5}{tn:>5}"
              f"{prec:>8.3f}{rec:>8.3f}{f1:>8.3f}{acc:>8.3f}")


def report(scored: list, use_text: bool) -> None:
    """Metrics block: category breakdown, confusion matrices, and the misses."""
    if not scored:
        print("\n=== no labelled listings scored — nothing to score against ===")
        return

    print(f"\n=== ground truth ({len(scored)} labelled listings analysed) ===")
    for cat in GT_CATEGORIES:
        members = [s for s in scored if cat in s["labels"]["categories"]]
        if not members:
            continue
        hit = sum(1 for s in members if s["pred_problem"])
        # For CLEAN, "flagged" is the error; for everything else it is the hit.
        note = f"false alarms {hit}/{len(members)}" if cat == "CLEAN" else \
               f"caught {hit}/{len(members)} = {hit / len(members):.3f}"
        print(f"  {cat:9} n={len(members):<4} {note}")

    print("\n=== flagged vs clean (the product decision) ===")
    signals = [("vision", [(s["gt_problem"], s["pred_vision"]) for s in scored])]
    if use_text:
        signals.append(("text", [(s["gt_problem"], s["pred_text"]) for s in scored]))
    signals.append(("combined", [(s["gt_problem"], s["pred_problem"]) for s in scored]))
    print_signal_table(signals)
    print()
    print_matrix("combined", [(s["gt_problem"], s["pred_problem"]) for s in scored])

    print("\n=== per red flag ===")
    # Front/back cracks can fire from either signal; not_functioning is text-only
    # by construction, so scoring it against vision would be meaningless.
    print_matrix("cracked_front (vision|text)",
                 [(s["gt"]["cracked_front"], s["pred_front"]) for s in scored])
    print()
    print_matrix("cracked_back  (vision|text)",
                 [(s["gt"]["cracked_back"], s["pred_back"]) for s in scored])
    if use_text:
        print()
        print_matrix("cracked_front (vision only)",
                     [(s["gt"]["cracked_front"], s["vis_front"]) for s in scored])
        print()
        print_matrix("cracked_back  (vision only)",
                     [(s["gt"]["cracked_back"], s["vis_back"]) for s in scored])
        print()
        print_matrix("not_functioning (text only)",
                     [(s["gt"]["not_functioning"], s["txt_notfunc"]) for s in scored])

    misses = [s for s in scored if s["gt_problem"] and not s["pred_problem"]]
    alarms = [s for s in scored if not s["gt_problem"] and s["pred_problem"]]
    if misses:
        print(f"\n=== missed ({len(misses)}) — real damage, not flagged ===")
        for s in misses:
            print(f"  {s['id']:>9}  {s['comment'][:64]}")
            print(f"            p_front={s['p_front']:.2f} p_back={s['p_back']:.2f} "
                  f"crops={s['n_front']}/{s['n_back']}  issues={s['issues'][:70] or '-'}")
    if alarms:
        print(f"\n=== false alarms ({len(alarms)}) — CLEAN, flagged anyway ===")
        for s in alarms:
            print(f"  {s['id']:>9}  {s['comment'][:64]}")
            print(f"            reasons={s['reasons']}  p_front={s['p_front']:.2f} "
                  f"p_back={s['p_back']:.2f}  issues={s['issues'][:70] or '-'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Full-pipeline (vision + text) test over OLX listings.")
    ap.add_argument("--urls", type=str, default=str(HERE / "test_listings.txt"))
    ap.add_argument("--crack-model", type=str, default="v3",
                    help="Registry name (v1/v2/v3) or a path to a .keras file.")
    ap.add_argument("--threshold", type=float, default=0.50,
                    help="P(cracked) decision threshold. 0.50 mirrors what ships — "
                         "inference.predict_crack flags cracked on score > 0.5.")
    ap.add_argument("--no-flip", action="store_true",
                    help="Use raw output as P(cracked) instead of 1-output.")
    ap.add_argument("--no-text", action="store_true", help="Skip the BERTic text pass.")
    ap.add_argument("--det-threshold", type=float, default=DETECTION_THRESHOLD)
    ap.add_argument("--save-crops", action="store_true",
                    help="Save the decisive front/back crop per listing for review.")
    ap.add_argument("--limit", type=int, default=0, help="Only the first N listings.")
    ap.add_argument("--delay", type=float, default=0.3, help="Seconds between listings.")
    ap.add_argument("--out", type=str, default=str(HERE / "listing_test_out"))
    args = ap.parse_args()

    urls_path = Path(args.urls)
    if not urls_path.exists():
        raise SystemExit(f"URL file not found: {urls_path}")
    entries = read_listings(urls_path)
    if args.limit:
        entries = entries[: args.limit]
    if not entries:
        raise SystemExit(f"No URLs in {urls_path} (blank / all comments).")
    n_labelled = sum(1 for e in entries if e["labels"])

    model_path = REGISTRY.get(args.crack_model, Path(args.crack_model))
    if not model_path.exists():
        raise SystemExit(f"Crack model not found: {model_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_text = not args.no_text

    print(f"{len(entries)} listing(s) ({n_labelled} labelled) | crack {args.crack_model} "
          f"@ thr {args.threshold} | text(BERTic): {'on' if use_text else 'off'} "
          f"| det thr {args.det_threshold}")
    det_session = get_session()
    scorer = CrackScorer(model_path, flip=not args.no_flip)
    print(f"  crack input {scorer.size}x{scorer.size}, P(cracked) = "
          f"{'1 - output' if scorer.flip else 'output'}\n")

    hdr = (f"{'id':>9}  {'title':24}  {'crops':>6}  {'P_f':>5} {'P_b':>5}  "
           f"{'vision':8}  {'text':14}  {'VERDICT':8}  {'TRUTH':14}  {'':3}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    scored = []
    counts = {"problem": 0, "clean": 0, "error": 0}
    reason_tally = {"vis-crack": 0, "txt-crack": 0, "not-func": 0, "icloud": 0, "sim": 0}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for entry in entries:
            url, gt, comment = entry["url"], entry["labels"], entry["comment"]
            lid = extract_listing_id(url)
            if lid is None:
                print(f"{'?':>9}  bad url: {url[:50]}")
                counts["error"] += 1
                rows.append({"url": url, "verdict": "bad_url", "gt_labels": comment})
                continue
            try:
                detail = fetch_listing_detail(lid)
                title = str(detail.get("title") or f"Listing {lid}")
                additional = detail.get("additional") or {}
                description = additional.get("description") or detail.get("short_description") or ""
                image_urls = [u for u in (detail.get("images") or []) if isinstance(u, str) and u.strip()]
                img_dir = tmp_root / str(lid)
                saved = download_images(image_urls, img_dir)
                analysis = analyze_folder(img_dir, threshold=args.det_threshold, session=det_session) if saved else {}
            except Exception as e:
                # Dead listings (404) are common — they drop out of the metrics
                # rather than counting as misses, so a pruned file scores fairly.
                print(f"{lid:>9}  ERROR  {type(e).__name__}: {e}")
                counts["error"] += 1
                rows.append({"url": url, "id": lid, "gt_labels": comment,
                             "verdict": f"error:{type(e).__name__}"})
                continue

            # ── vision ──
            all_crops = analysis.get("all_crops", {}) if analysis else {}
            nF = len(all_crops.get("phone_front") or [])
            nB = len(all_crops.get("phone_back") or [])
            pF, cropF, srcF = scorer.max_pool(all_crops.get("phone_front"))
            pB, cropB, srcB = scorer.max_pool(all_crops.get("phone_back"))
            vfront, vback = pF > args.threshold, pB > args.threshold
            if nF == 0 and nB == 0:
                vision = "no-phone"
            elif vfront or vback:
                vision = "CRACK-" + ("F" if vfront else "") + ("B" if vback else "")
            else:
                vision = "clean"

            # ── text ──
            txt = analyze_text(title, description) if use_text else {}

            # ── combine ──
            reasons = []
            if vfront or vback:
                reasons.append("vis-crack")
            if txt.get("cracked_front") or txt.get("cracked_back"):
                reasons.append("txt-crack")
            if txt.get("not_functioning"):
                reasons.append("not-func")
            if txt.get("icloud"):
                reasons.append("icloud")
            if txt.get("sim"):
                reasons.append("sim")
            for r in reasons:
                reason_tally[r] += 1
            verdict = "PROBLEM" if reasons else "clean"
            counts["problem" if reasons else "clean"] += 1

            text_codes = ",".join(
                c for c, on in [
                    ("NF", txt.get("not_functioning")), ("cF", txt.get("cracked_front")),
                    ("cB", txt.get("cracked_back")), ("iCL", txt.get("icloud")),
                    ("SIM", txt.get("sim")),
                ] if on
            ) or "-"

            # ── score against the inline ground truth ──
            truth = "+".join(sorted(gt["categories"])) if gt else "?"
            if gt and gt["icloud"]:
                truth += "+iCL"
            mark = "   "
            if gt:
                mark = "OK " if bool(gt["problem"]) == bool(reasons) else \
                       ("FN!" if gt["problem"] else "FP!")

            print(f"{lid:>9}  {title[:24]:24}  {f'{nF}/{nB}':>6}  {pF:5.2f} {pB:5.2f}  "
                  f"{vision:8}  {text_codes:14}  {verdict:8}  {truth:14}  {mark}")
            issues = txt.get("issues") or []
            if verdict == "PROBLEM" and (issues or txt.get("condition")):
                extra = f"issues={issues}" if issues else ""
                if txt.get("condition"):
                    extra += f"  cond='{txt['condition']}'"
                print(f"           └ {extra}")

            if gt:
                scored.append({
                    "id": lid, "comment": comment, "labels": gt, "gt": gt,
                    "gt_problem": bool(gt["problem"]),
                    # The two signals kept apart so the report can attribute
                    # every hit and every miss to the one that produced it.
                    "pred_vision": bool(vfront or vback),
                    "pred_text": bool(txt.get("cracked_front") or txt.get("cracked_back")
                                      or txt.get("not_functioning") or txt.get("icloud")
                                      or txt.get("sim")),
                    "pred_problem": bool(reasons),
                    "pred_front": bool(vfront or txt.get("cracked_front")),
                    "pred_back": bool(vback or txt.get("cracked_back")),
                    "vis_front": bool(vfront), "vis_back": bool(vback),
                    "txt_notfunc": bool(txt.get("not_functioning")),
                    "p_front": pF, "p_back": pB, "n_front": nF, "n_back": nB,
                    "issues": "; ".join(issues), "reasons": ",".join(reasons),
                })

            if args.save_crops and vision != "no-phone":
                dest = out_dir / ("cracked" if (vfront or vback) else "clean")
                dest.mkdir(parents=True, exist_ok=True)
                if cropF is not None:
                    cropF.save(dest / f"{lid}_front_p{pF:.2f}.jpg")
                if cropB is not None:
                    cropB.save(dest / f"{lid}_back_p{pB:.2f}.jpg")

            rows.append({
                "url": url, "id": lid, "title": title,
                "gt_labels": truth,
                "gt_cracked_front": gt["cracked_front"] if gt else "",
                "gt_cracked_back": gt["cracked_back"] if gt else "",
                "gt_not_functioning": gt["not_functioning"] if gt else "",
                "gt_problem": gt["problem"] if gt else "",
                "outcome": mark.strip(),
                "n_front": nF, "n_back": nB,
                "p_front": round(pF, 4), "p_back": round(pB, 4), "vision": vision,
                "txt_not_functioning": txt.get("not_functioning", ""),
                "txt_cracked_front": txt.get("cracked_front", ""),
                "txt_cracked_back": txt.get("cracked_back", ""),
                "txt_icloud": txt.get("icloud", ""), "txt_sim": txt.get("sim", ""),
                "condition": txt.get("condition", ""),
                "issues": "; ".join(issues),
                "verdict": verdict, "reasons": ",".join(reasons),
                "trigger": str(srcF if pF >= pB else srcB) if (nF or nB) else "",
                "description": (description or "")[:500],
            })
            if args.delay:
                time.sleep(args.delay)

    csv_path = out_dir / "results.csv"
    cols = ["url", "id", "title", "gt_labels", "gt_cracked_front", "gt_cracked_back",
            "gt_not_functioning", "gt_problem", "outcome",
            "n_front", "n_back", "p_front", "p_back", "vision",
            "txt_not_functioning", "txt_cracked_front", "txt_cracked_back", "txt_icloud",
            "txt_sim", "condition", "issues", "verdict", "reasons", "trigger", "description"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== summary ({len(entries)} listings) ===")
    print(f"  PROBLEM:  {counts['problem']}   clean: {counts['clean']}   errors: {counts['error']}")
    print("  reasons:  " + ", ".join(f"{k}={v}" for k, v in reason_tally.items()))

    report(scored, use_text)

    print(f"\n  csv: {csv_path}")
    if args.save_crops:
        print(f"  crops: {out_dir}\\cracked , {out_dir}\\clean")


if __name__ == "__main__":
    main()
