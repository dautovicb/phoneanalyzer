"""Extract phone_front / phone_back crops from a folder of listing photos, for
building the crack-detection dataset.

The crack classifier is trained on tight phone crops (front screen / back glass),
so raw listing photos have to be run through the object detector first. This does
that in bulk and preserves any label sub-folders, so the crops land ready to move
into dataset/train/<label>/.

Input layout (labels optional):
    <input>/cracked/*.jpg
    <input>/not cracked/*.jpg     -> label normalised to "not_cracked"
  or a flat <input>/*.jpg         -> no label sub-folder

Only phone_front and phone_back are kept (cracks live there; box / ui_* crops are
irrelevant). The detector keeps just the single best front and back per image, so
each photo yields at most one of each — nothing to de-duplicate.

Output (default: a sibling of dataset/, NOT inside it, so the staging crops don't
get pulled into Git-LFS before you've sorted them):
    models/crack_detection/<input-name>_crops/<label>/<class>__<src>__NNNN__cCONF.jpg
    models/crack_detection/<input-name>_crops/manifest.json

Run from ml/phoneanalyzer/:
    python models/crack_detection/extract_crops.py \
        models/crack_detection/dataset/newcrackimages
    # options: --threshold 0.35  --out <dir>  --classes phone_front,phone_back
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

HERE = Path(__file__).resolve().parent
ML_ROOT = HERE.parent.parent  # ml/phoneanalyzer, so `models....` imports resolve
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from models.detection_model.inference import detect_and_crop, load_model

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# A lower default than production (0.45): for dataset-building we want to recover
# even phones the detector is unsure about — notably heavily shattered ones, which
# are exactly the scarce cracked examples worth keeping. The crop box is unchanged
# by the threshold; it only gates whether we get a crop at all. Eyeball the output.
DEFAULT_THRESHOLD = 0.35


def normalize_label(name: str) -> str:
    """'not cracked' / 'Not-Cracked' -> 'not_cracked' (dataset folder convention)."""
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def gather(input_dir: Path):
    """Yield (image_path, label) — label is the top-level sub-folder under input_dir
    (normalized), or '' for images sitting directly in input_dir."""
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMG_EXT:
            continue
        rel = path.relative_to(input_dir)
        label = normalize_label(rel.parts[0]) if len(rel.parts) > 1 else ""
        yield path, label


def safe_stem(path: Path, limit: int = 40) -> str:
    stem = re.sub(r"[^\w.-]+", "_", path.stem)
    return stem[:limit]


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract phone crops for the crack dataset.")
    ap.add_argument("input_dir", type=str, help="Folder of photos (optionally in label sub-folders)")
    ap.add_argument("--out", type=str, default=None,
                    help="Output dir (default: <crack_detection>/<input-name>_crops)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"Detection confidence threshold (default {DEFAULT_THRESHOLD})")
    ap.add_argument("--classes", type=str, default="phone_front,phone_back",
                    help="Comma-separated detector classes to keep")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Not a directory: {input_dir}")
    keep = {c.strip() for c in args.classes.split(",") if c.strip()}
    out_dir = Path(args.out).resolve() if args.out else (HERE / f"{input_dir.name}_crops")

    items = list(gather(input_dir))
    if not items:
        raise SystemExit(f"No images found under {input_dir}")
    labels = sorted({lbl for _, lbl in items})
    print(f"{len(items)} image(s) under {input_dir.name}  "
          f"| labels: {[l or '(none)' for l in labels]}  | keep: {sorted(keep)}  "
          f"| threshold {args.threshold}")

    session = load_model()
    counters: dict[tuple[str, str], int] = {}
    manifest = []
    saved = {cls: 0 for cls in keep}
    n_no_detection = 0
    n_unreadable = 0

    for i, (path, label) in enumerate(items, 1):
        if i % 100 == 0 or i == len(items):
            print(f"  ...{i}/{len(items)}  (saved: "
                  + ", ".join(f"{c}={saved[c]}" for c in sorted(keep)) + ")")
        try:
            image = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as err:
            print(f"  [skip] unreadable: {path.name} ({err})")
            n_unreadable += 1
            continue

        crops = [(c, cf, cr) for (c, cf, cr) in detect_and_crop(image, session, args.threshold)
                 if c in keep]
        if not crops:
            n_no_detection += 1
            continue

        for cls_name, conf, crop in crops:
            dest_dir = out_dir / label if label else out_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            key = (label, cls_name)
            counters[key] = counters.get(key, 0) + 1
            fname = f"{cls_name}__{safe_stem(path)}__{counters[key]:04d}__c{conf:.2f}.jpg"
            crop.save(dest_dir / fname)
            saved[cls_name] = saved.get(cls_name, 0) + 1
            manifest.append({
                "source": str(path),
                "label": label,
                "class": cls_name,
                "confidence": round(float(conf), 4),
                "crop": str(dest_dir / fname),
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== done ===")
    print(f"  images processed:     {len(items)}")
    print(f"  unreadable (skipped): {n_unreadable}")
    print(f"  no phone detected:    {n_no_detection}  "
          f"(dropped — lower --threshold to recover borderline ones)")
    for cls in sorted(keep):
        print(f"  {cls} crops saved:  {saved.get(cls, 0)}")
    # Per-label breakdown so you can sanity-check the cracked vs not_cracked split.
    by_label: dict[str, dict[str, int]] = {}
    for m in manifest:
        by_label.setdefault(m["label"] or "(none)", {}).setdefault(m["class"], 0)
        by_label[m["label"] or "(none)"][m["class"]] += 1
    print("  by label:")
    for lbl, d in sorted(by_label.items()):
        print(f"    {lbl:14} " + ", ".join(f"{k}={v}" for k, v in sorted(d.items())))
    print(f"\n  crops in: {out_dir}")
    print(f"  manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
