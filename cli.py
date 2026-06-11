from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Put the module root on sys.path so `core` and the model packages import cleanly
# regardless of the working directory. Removed once the package is pip-installed
# (`pip install -e .`).
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.pipeline import PipelineError, analyze_listing, preload_models


def main(argv: list[str] | None = None) -> int:

    for stream in (sys.stdout, sys.stderr):
        if stream.encoding and stream.encoding.lower() != "utf-8":
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        prog="phoneanalyzer",
        description="Analyze OLX.ba phone listings: extract model, storage, battery health, and red flags.",
    )
    parser.add_argument("urls", nargs="+", metavar="URL", help="OLX.ba listing URL(s)")
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the merged database-ready record(s) as JSON instead of a readable summary",
    )
    args = parser.parse_args(argv)

    print("Loading models...", file=sys.stderr)
    preload_models()

    records = []
    failed = 0
    for url in args.urls:
        print(f"Analyzing {url} ...", file=sys.stderr)
        try:
            result = analyze_listing(url)
        except PipelineError as err:
            print(f"Could not analyze {url}: {err}", file=sys.stderr)
            failed += 1
            continue

        if args.json:
            records.append(result.merged)
        else:
            print(f"\n{result.summary()}\n")

    if args.json and records:
        payload = records[0] if len(args.urls) == 1 else records
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
