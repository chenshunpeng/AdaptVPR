"""Export accepted AdaptVPR images from generation records."""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an accepted hard-positive manifest.")
    parser.add_argument("records", type=Path, help="records.jsonl written by run.py")
    parser.add_argument("--output", type=Path, default=Path("adaptvpr_manifest.jsonl"))
    return parser.parse_args()


def build_manifest(records: Path, output: Path) -> int:
    accepted = 0
    with records.open("r", encoding="utf-8") as source, output.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            for candidate in record.get("training_candidates", []):
                if candidate.get("passed") and candidate.get("eligible_for_training"):
                    destination.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                    accepted += 1
    return accepted


def main() -> None:
    args = parse_args()
    accepted = build_manifest(args.records, args.output)
    print(f"Wrote {accepted} accepted samples to {args.output}")


if __name__ == "__main__":
    main()
