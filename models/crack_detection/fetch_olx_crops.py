"""Fetch fresh iPhone listing photos from OLX, run them through the phone
detector, and save front/back crops for the crack-detection dataset.

Candidate listing ids come from the production `listings` table (the same DB
backend/jobs/fetcher.py populates) rather than a fresh OLX search scrape --
that table holds months of history OLX's own search API doesn't (fetcher.py
caps search at created_gte=-1+month). Only the id is read from Postgres; every
photo still comes straight from OLX's public detail endpoint, since the DB
only stores one thumbnail per listing, not the full gallery.

De-duplication: crops already on disk are named
    <olx_id>_<front|back>_<n>_c<confidence>.jpg          (this script)
    r<score>_<olx_id>_<front|back>_<n>_c<confidence>.jpg  (older convention)
so the id is recoverable by regex regardless of which script produced it.
Every listing whose id already appears anywhere under dataset/ or in a
previous run's staging output is skipped -- nothing is ever fetched twice.

Output goes to a staging folder, NOT dataset/ -- these crops aren't labelled
yet, mirroring extract_crops.py's convention. Sort into dataset/cracked or
dataset/not_cracked by hand, then re-run to top up with the next batch.

Run from ml/phoneanalyzer/ (needs backend/'s DB creds on the path -- see
common/appenv.py's dotenv layering -- and the inference deps in
requirements.txt: onnxruntime, pillow, requests, psycopg2, python-dotenv):
    python models/crack_detection/fetch_olx_crops.py
    # options: --limit 300  --threshold 0.35  --out DIR  --delay 0.3
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, UnidentifiedImageError

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
ML_ROOT = HERE.parent.parent          # ml/phoneanalyzer
REPO_ROOT = ML_ROOT.parent.parent     # dealsniper/  (ml/phoneanalyzer -> ml -> repo root)
BACKEND_ROOT = REPO_ROOT / "backend"
for _p in (ML_ROOT, BACKEND_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from models.detection_model.inference import detect_and_crop, load_model  # noqa: E402
from common import db  # noqa: E402

DATASET_DIR = HERE / "dataset"
DEFAULT_OUT = HERE / "olx_fresh_crops"
# Lower than production (0.45), same reasoning as extract_crops.py: recover
# borderline detections too (esp. heavily damaged phones) -- eyeball the output.
DEFAULT_THRESHOLD = 0.35
KEEP_CLASSES = {"phone_front", "phone_back"}
DETAIL_URL = "https://olx.ba/api/listings/{}"
DEFAULT_HEADERS = {"User-Agent": "ulov.ba-dataset/1.0"}

# Recovers the OLX id out of any crop filename we've ever produced -- the id is
# always the only 6+ digit run in the stem (detection/crack-score fragments
# like "0.44" or "0099" are shorter or contain a dot). A stray 6+ digit run in
# an unrelated filename (e.g. a stock-photo id) would at worst cause one real
# listing to be skipped unnecessarily -- harmless, so no need to be stricter.
_ID_RE = re.compile(r"(?<!\d)(\d{6,})(?!\d)")


def used_olx_ids(*roots: Path) -> set:
    """Every OLX id already spent on a crop, scanned out of existing filenames."""
    ids = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.jpg"):
            m = _ID_RE.search(path.stem)
            if m:
                ids.add(m.group(1))
    return ids


def candidate_ids(exclude: set, limit: int) -> list:
    """Up to `limit` DB-known listing ids not already spent, newest-id-first."""
    all_ids = db.get_all_listing_ids()
    fresh = sorted((str(i) for i in all_ids if str(i) not in exclude),
                   key=int, reverse=True)
    return fresh[:limit]


def fetch_detail(session: requests.Session, olx_id: str) -> dict:
    resp = session.get(DETAIL_URL.format(olx_id), timeout=20)
    resp.raise_for_status()
    return resp.json()


def image_urls_from_detail(detail: dict) -> list:
    images = detail.get("images") or []
    return [u for u in images if isinstance(u, str) and u.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch fresh OLX iPhone photos -> front/back crops.")
    ap.add_argument("--limit", type=int, default=300, help="Max new listings to process.")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"Detection confidence threshold (default {DEFAULT_THRESHOLD}).")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                    help="Staging output dir (default: sibling of dataset/, not inside it).")
    ap.add_argument("--delay", type=float, default=0.3,
                    help="Seconds between listings -- OLX politeness (default 0.3).")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    spent = used_olx_ids(DATASET_DIR, out_dir)
    print(f"[fetch] {len(spent)} olx id(s) already used under dataset/ + {out_dir.name}/")

    ids = candidate_ids(spent, args.limit)
    if not ids:
        print("[fetch] nothing new to fetch -- every DB-known listing is already spent.")
        return
    print(f"[fetch] {len(ids)} new listing(s) queued (limit {args.limit}) | threshold {args.threshold}")

    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    det_session = load_model()

    counters: dict[tuple[str, str], int] = {}
    saved = {"phone_front": 0, "phone_back": 0}
    counts = {"processed": 0, "gone": 0, "failed": 0, "no_detection": 0}

    for i, olx_id in enumerate(ids, 1):
        if i % 25 == 0 or i == len(ids):
            print(f"  ...{i}/{len(ids)}  (saved: front={saved['phone_front']}, back={saved['phone_back']})")

        try:
            detail = fetch_detail(session, olx_id)
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code in (404, 410):
                counts["gone"] += 1
            else:
                counts["failed"] += 1
                print(f"  [{olx_id}] HTTP {code}")
            continue
        except requests.RequestException as e:
            counts["failed"] += 1
            print(f"  [{olx_id}] request failed: {e}")
            continue

        urls = image_urls_from_detail(detail)
        if not urls:
            counts["failed"] += 1
            continue

        any_crop = False
        for url in urls:
            try:
                r = session.get(url, timeout=20)
                r.raise_for_status()
                image = Image.open(BytesIO(r.content)).convert("RGB")
            except (requests.RequestException, UnidentifiedImageError, OSError):
                continue

            for cls_name, conf, crop in detect_and_crop(image, det_session, args.threshold):
                if cls_name not in KEEP_CLASSES:
                    continue
                key = (olx_id, cls_name)
                counters[key] = counters.get(key, 0) + 1
                side = "front" if cls_name == "phone_front" else "back"
                fname = f"{olx_id}_{side}_{counters[key]}_c{conf:.2f}.jpg"
                crop.save(out_dir / fname)
                saved[cls_name] += 1
                any_crop = True

        counts["processed"] += 1
        if not any_crop:
            counts["no_detection"] += 1

        if args.delay:
            time.sleep(args.delay)

    print("\n=== done ===")
    print(f"  processed:     {counts['processed']}")
    print(f"  gone (404):    {counts['gone']}")
    print(f"  failed:        {counts['failed']}")
    print(f"  no detection:  {counts['no_detection']}")
    print(f"  phone_front crops saved: {saved['phone_front']}")
    print(f"  phone_back  crops saved: {saved['phone_back']}")
    print(f"\n  crops in: {out_dir}")
    print("  sort into dataset/cracked or dataset/not_cracked by hand, then re-run to top up.")


if __name__ == "__main__":
    main()
