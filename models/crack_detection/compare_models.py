"""Compare crack classifiers (v1 / v2 / v3 ...) on a labeled test set and tune
the decision threshold.

Usage (from ml/phoneanalyzer/):
    python models/crack_detection/compare_models.py \
        --test-dir models/crack_detection/dataset/test \
        --models v2,v3
    # extra: --per-image   --threshold 0.5 (pin one instead of sweeping)

Test-set layout — sub-folders give the ground truth:
    <test-dir>/cracked/*.jpg        (label 1)
    <test-dir>/not_cracked/*.jpg    (label 0)   ("clean"/"ok"/"good" also work)
A flat folder of images (no sub-folders) still runs, but only the raw scores and
model-vs-model agreement are shown — no metrics or tuning without labels.

All models are 224x224 single-sigmoid classifiers fed RAW [0,255] RGB, exactly as
production feeds them (models/crack_detection/inference.py). The one gotcha is
output polarity: a model trained by training/cracks.py on folders `cracked` /
`not_cracked` learns cracked=0 < not_cracked=1 alphabetically, so its sigmoid is
P(not_cracked) and P(cracked) = 1 - output; the shipped v1 path also uses
1 - output. Rather than trust any convention, this script AUTO-ORIENTS each model
from the ground truth (picks the polarity under which truly-cracked phones score
higher) and prints which way it chose. Recall is weighted over precision in the
recommendation: on a resale listing, missing a crack is the costly error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
ML_ROOT = HERE.parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

MODEL_DIR = HERE / "model"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# short name -> weights file
REGISTRY = {
    "v1": MODEL_DIR / "crack_detector.keras",
    "v2": MODEL_DIR / "crack_detector_v2.keras",
    "v3": MODEL_DIR / "crack_detector_v3.keras",
    "v4": MODEL_DIR / "crack_detector_v4.keras",
}


def load_model(path: Path):
    from models.crack_detection.inference import _load_keras_model
    return _load_keras_model(path)


def gather_images(root: Path):
    """[(path, label)] with label 1=cracked, 0=clean, None=unknown (from sub-folder)."""
    items = []
    if not root.exists():
        return items
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        name = sub.name.lower()
        if "crack" in name and "not" not in name and "no_crack" not in name:
            label = 1
        elif any(k in name for k in ("clean", "ok", "good", "not", "nocrack", "undamaged")):
            label = 0
        else:
            label = None
        for f in sorted(sub.rglob("*")):
            if f.suffix.lower() in IMG_EXT:
                items.append((f, label))
    for f in sorted(root.glob("*")):
        if f.is_file() and f.suffix.lower() in IMG_EXT:
            items.append((f, None))
    return items


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def to_batch(images, size: int) -> np.ndarray:
    """Resize each PIL image to size x size, raw [0,255] float32 — production input."""
    arrs = [np.asarray(im.resize((size, size)), dtype="float32") for im in images]
    return np.stack(arrs, axis=0)


def model_input_size(model, default: int = 224) -> int:
    """Square input side the model expects. v2 is 224; v3 was trained at 384, so
    we can't share one batch tensor — read it off each model instead of assuming."""
    shp = getattr(model, "input_shape", None)
    if isinstance(shp, list) and shp:
        shp = shp[0]
    if shp and len(shp) == 4 and shp[1] and shp[2]:
        return int(shp[1])
    return default


def metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tot = tp + tn + fp + fn
    acc = (tp + tn) / tot if tot else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) else 0.0  # recall-weighted
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "acc": acc, "prec": prec, "rec": rec, "f1": f1, "f2": f2}


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Rank AUC (Mann-Whitney); n is tiny so the O(n^2) form is fine."""
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def orient(raw: np.ndarray, y: np.ndarray):
    """Return (p_cracked, flipped): pick the polarity under which truly-cracked
    phones score higher on average. `flipped` True means P(cracked) = 1 - output."""
    if raw[y == 1].mean() >= raw[y == 0].mean():
        return raw, False
    return 1.0 - raw, True


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare & tune crack detectors.")
    ap.add_argument("--test-dir", type=str, default=str(HERE / "test_images"))
    ap.add_argument("--models", type=str, default="v2,v3",
                    help="Comma-separated names from the registry, or name=path pairs.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Pin a single P(cracked) threshold instead of sweeping.")
    ap.add_argument("--per-image", action="store_true",
                    help="Print every image's scores (auto-on for <=40 images).")
    args = ap.parse_args()

    # Resolve requested models.
    specs = []
    for tok in args.models.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" in tok:
            name, p = tok.split("=", 1)
            specs.append((name.strip(), Path(p).resolve()))
        elif tok in REGISTRY:
            specs.append((tok, REGISTRY[tok]))
        else:
            print(f"[warn] unknown model '{tok}' (registry: {list(REGISTRY)})")
    specs = [(n, p) for n, p in specs if p.exists()]
    if not specs:
        raise SystemExit("No valid models to compare.")

    test_dir = Path(args.test_dir).resolve()
    items = gather_images(test_dir)
    if not items:
        raise SystemExit(f"No images under {test_dir}")
    labeled = any(lbl is not None for _, lbl in items)
    paths = [p for p, _ in items]
    labels = np.array([lbl for _, lbl in items], dtype=object)
    print(f"{len(items)} image(s) under {test_dir.name}  "
          f"({'labeled' if labeled else 'UNLABELED'} mode)")
    if labeled:
        y = np.array([int(l) for l in labels])
        print(f"  ground truth: {int((y==1).sum())} cracked, {int((y==0).sum())} clean")
    print(f"  models: {', '.join(n for n, _ in specs)}\n")

    images = [load_rgb(p) for p in paths]

    # Score + orient each model. Each may want a different input size (v2=224,
    # v3=384), so build the batch per model rather than sharing one tensor.
    p_cracked, raw_out, flip = {}, {}, {}
    for name, path in specs:
        print(f"Loading {name} ({path.name}, {path.stat().st_size/1e6:.0f} MB)...")
        model = load_model(path)
        size = model_input_size(model)
        print(f"    input {size}x{size}")
        batch = to_batch(images, size)
        raw = np.asarray(model.predict(batch, verbose=0)).reshape(-1)
        raw_out[name] = raw
        if labeled:
            p, fl = orient(raw, y)
        else:
            p, fl = 1.0 - raw, True  # production convention as fallback
        p_cracked[name] = np.clip(p, 0.0, 1.0)
        flip[name] = fl
    print()

    names = [n for n, _ in specs]
    for n in names:
        print(f"  {n}: P(cracked) = {'1 - output' if flip[n] else 'output'}"
              + (f"   (auto-oriented; AUC={auc(p_cracked[n], y):.3f})" if labeled else ""))
    print()

    # ── Per-image table (small sets only) ──────────────────────────────────
    if args.per_image or len(paths) <= 40:
        thr_show = args.threshold if args.threshold is not None else 0.5
        hdr = f"{'image':34} {'truth':7}"
        for n in names:
            hdr += f" | {n:>5} P(crk)"
        hdr += " | agree"
        print(hdr)
        print("-" * len(hdr))
        tstr = {1: "cracked", 0: "clean", None: "-"}
        for i, p in enumerate(paths):
            row = f"{p.name[:34]:34} {tstr[labels[i]]:7}"
            preds = []
            for n in names:
                pc = float(p_cracked[n][i])
                preds.append(pc > thr_show)
                mark = "*" if (labeled and preds[-1] != bool(labels[i])) else " "
                row += f" | {pc:6.3f}{mark}"
            row += "  | " + ("yes" if len(set(preds)) == 1 else "NO")
            print(row)
        print("  (* = wrong at P>%.2f;  P(crk) shown after orientation)\n" % thr_show)

    if not labeled:
        if len(names) == 2:
            a, b = names
            d = int(np.sum((p_cracked[a] > 0.5) != (p_cracked[b] > 0.5)))
            print(f"Models agree on {len(paths)-d}/{len(paths)} images "
                  f"(no labels -> no metrics).")
        return

    # ── Threshold sweep + tuning ───────────────────────────────────────────
    if args.threshold is not None:
        grid = [args.threshold]
    else:
        grid = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]

    print("=== Threshold sweep (cracked = positive) ===")
    best = {}
    for n in names:
        print(f"\n  {n}  (AUC={auc(p_cracked[n], y):.3f})")
        print(f"    {'thr':>5} {'acc':>5} {'prec':>5} {'rec':>5} {'F1':>5} {'F2':>5}   TP FP FN TN")
        best_f1 = best_f2 = None
        for t in grid:
            m = metrics(y, (p_cracked[n] > t).astype(int))
            print(f"    {t:5.2f} {m['acc']:5.2f} {m['prec']:5.2f} {m['rec']:5.2f} "
                  f"{m['f1']:5.2f} {m['f2']:5.2f}   {m['tp']:2d} {m['fp']:2d} {m['fn']:2d} {m['tn']:2d}")
            # prefer higher F, tie-break toward higher recall then higher threshold
            if best_f1 is None or (m["f1"], m["rec"], t) > (best_f1[1]["f1"], best_f1[1]["rec"], best_f1[0]):
                best_f1 = (t, m)
            if best_f2 is None or (m["f2"], m["rec"], t) > (best_f2[1]["f2"], best_f2[1]["rec"], best_f2[0]):
                best_f2 = (t, m)
        best[n] = {"f1": best_f1, "f2": best_f2}

    # ── Recommendation ─────────────────────────────────────────────────────
    print("\n=== Recommendation ===")
    for n in names:
        tf1, mf1 = best[n]["f1"]
        tf2, mf2 = best[n]["f2"]
        print(f"  {n}: AUC={auc(p_cracked[n], y):.3f}")
        print(f"      best F1 @ thr {tf1:.2f}: acc={mf1['acc']:.2f} "
              f"prec={mf1['prec']:.2f} rec={mf1['rec']:.2f} f1={mf1['f1']:.2f}")
        print(f"      best F2 @ thr {tf2:.2f}: acc={mf2['acc']:.2f} "
              f"prec={mf2['prec']:.2f} rec={mf2['rec']:.2f} f2={mf2['f2']:.2f}  <- recall-first")
    # Pick the better model by AUC (threshold-independent), then show misses.
    winner = max(names, key=lambda n: auc(p_cracked[n], y))
    tf2, _ = best[winner]["f2"]
    print(f"\n  Winner by AUC: {winner}.  At its recall-first thr {tf2:.2f}:")
    preds = (p_cracked[winner] > tf2).astype(int)
    for i, p in enumerate(paths):
        if preds[i] != y[i]:
            kind = "MISSED crack (FN)" if y[i] == 1 else "false alarm (FP)"
            print(f"      {kind:18} {p.name[:44]}  P(crk)={p_cracked[winner][i]:.3f}")
    if int(np.sum(preds != y)) == 0:
        print("      (no misclassifications at this threshold)")
    print(f"\n  NOTE: {len(paths)} test images is a small sample — treat thresholds "
          f"as directional, confirm on a larger/held-out set before shipping.")


if __name__ == "__main__":
    main()
