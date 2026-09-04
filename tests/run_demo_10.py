#!/usr/bin/env python3
"""Run the public AdaptVPR pipeline on the 10-source demo."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES_CSV = Path(__file__).with_name("demo_10.csv")
REQUIRED_ROUTES = {"global", "local", "dual"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GSV-Cities demo paths with Qwen3-VL-4B planning or released prompts."
    )
    parser.add_argument("--mode", choices=("qwen4b", "prompt", "all"), default="qwen4b")
    parser.add_argument(
        "--gsvcities-root",
        type=Path,
        required=True,
        help="GSV-Cities root containing Images/CITY/source.jpg.",
    )
    parser.add_argument(
        "--prompts-jsonl",
        type=Path,
        help="Released Prompt JSONL; required for prompt/all mode.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reflection", choices=("on", "off"), default="on")
    parser.add_argument("--max-reflections", type=int, choices=range(0, 4), default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode in {"prompt", "all"} and args.prompts_jsonl is None:
        parser.error("--prompts-jsonl is required for prompt/all mode")
    return args


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_sources() -> list[dict[str, str]]:
    with SOURCES_CSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"city", "source_id", "gsvcities_path"}
    if len(rows) != 10:
        raise RuntimeError(f"{SOURCES_CSV} must contain exactly 10 rows")
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"{SOURCES_CSV} must contain columns {sorted(required)}")
    source_ids = [row["source_id"].strip() for row in rows]
    if len(set(source_ids)) != len(source_ids):
        raise RuntimeError(f"{SOURCES_CSV} contains duplicate source_id values")
    for row in rows:
        relative = Path(row["gsvcities_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"GSV-Cities path must be relative and contained: {relative}")
        if relative.name != row["source_id"]:
            raise RuntimeError(f"source_id/path mismatch: {row['source_id']} != {relative}")
    return rows


def prepare_inputs(
    args: argparse.Namespace, samples: list[dict[str, str]]
) -> tuple[Path, Path | None]:
    staging = args.output / "demo_inputs"
    staging.mkdir(parents=True, exist_ok=True)
    wanted = [sample["source_id"] for sample in samples]
    for sample in samples:
        source = args.gsvcities_root / sample["gsvcities_path"]
        if not source.is_file():
            raise FileNotFoundError(source)
        link = staging / sample["source_id"]
        if link.is_symlink() and link.resolve() == source.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise RuntimeError(f"staged input conflicts with fixed CSV: {link}")
        link.symlink_to(source.resolve())

    subset = None
    if args.prompts_jsonl:
        by_source = {
            str(row.get("source_id")): row for row in load_jsonl(args.prompts_jsonl)
        }
        missing = [name for name in wanted if name not in by_source]
        if missing:
            raise RuntimeError(f"released Prompt JSONL is missing demo sources: {missing}")
        subset = args.output / "demo_10_prompts.jsonl"
        subset.write_text(
            "".join(
                json.dumps(by_source[name], ensure_ascii=False) + "\n" for name in wanted
            ),
            encoding="utf-8",
        )
    return staging, subset


def execute(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())


def validate(run: Path) -> None:
    records = load_jsonl(run / "records.jsonl")
    if len(records) != 10:
        raise RuntimeError(f"{run}: expected 10 records, got {len(records)}")
    bad = [
        record.get("sample_id")
        for record in records
        if not record.get("generated") or record.get("status") == "skipped"
    ]
    if bad:
        raise RuntimeError(f"{run}: Skip/non-generated is forbidden: {bad}")
    routes = {record.get("route") for record in records}
    missing_routes = REQUIRED_ROUTES - routes
    if missing_routes:
        raise RuntimeError(f"{run}: missing routes: {sorted(missing_routes)}")
    depths = sorted(
        {max(0, int(record.get("rounds_used", 1)) - 1) for record in records}
    )
    print(f"{run}: observed reflection depths={depths}", flush=True)


def run_one(args: argparse.Namespace, public_mode: str, input_path: Path) -> Path:
    run_mode = "plan" if public_mode == "qwen4b" else "prompt"
    reflection = args.reflection if args.max_reflections > 0 else "off"
    max_reflections = args.max_reflections if reflection == "on" else 0
    output = args.output / f"{public_mode}_reflection_{reflection}"
    command = [
        sys.executable,
        str(ROOT / "run.py"),
        str(input_path),
        "--mode",
        run_mode,
        "--output",
        str(output),
        "--reflection",
        reflection,
        "--max-reflections",
        str(max_reflections),
        "--seed",
        str(args.seed),
        "--fail-fast",
        "--require-generated",
    ]
    if run_mode == "prompt":
        command += ["--image-root", str(args.gsvcities_root / "Images")]
    if args.resume:
        command.append("--resume")
    execute(command)
    validate(output)
    return output


def main() -> None:
    args = parse_args()
    samples = load_sources()
    args.output.mkdir(parents=True, exist_ok=True)
    staged, prompts = prepare_inputs(args, samples)
    completed = []
    if args.mode in {"qwen4b", "all"}:
        completed.append(run_one(args, "qwen4b", staged))
    if args.mode in {"prompt", "all"}:
        completed.append(run_one(args, "prompt", prompts))
    print("Demo-10 validation passed:", *(str(path) for path in completed), sep="\n- ")


if __name__ == "__main__":
    main()
