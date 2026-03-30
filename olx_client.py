from __future__ import annotations

import re
from typing import List, Optional, Tuple

import requests

DETAIL_BASE_URL = "https://olx.ba/api/listings"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,bs;q=0.8",
}


def extract_listing_id(olx_url: str) -> Optional[int]:
    value = olx_url.strip()
    if not value:
        return None

    if value.isdigit() and len(value) >= 6:
        return int(value)

    patterns = [
        r"/artikal/(\d{6,})(?:/|$)",
        r"/api/listings/(\d{6,})(?:/|$)",
        r"[?&]id=(\d{6,})(?:&|$)",
        r"(\d{6,})",
    ]
    for pattern in patterns:
        m = re.search(pattern, value)
        if m:
            return int(m.group(1))
    return None


def fetch_listing_images(listing_id: int) -> Tuple[str, List[str]]:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    api_url = f"{DETAIL_BASE_URL}/{listing_id}"
    try:
        resp = session.get(api_url, timeout=20)
        resp.raise_for_status()
        detail = resp.json()
        title = str(detail.get("title") or f"Listing {listing_id}")
        images = detail.get("images") or []
        image_urls = [u for u in images if isinstance(u, str) and u.strip()]
        if image_urls:
            return title, image_urls
    except requests.RequestException:
        pass

    return title, image_urls
