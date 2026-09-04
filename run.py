"""Public batch entry point for AdaptVPR generation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate verified AdaptVPR hard positives.")
    parser.add_argument(
        "input",
        type=Path,
        help="Planner mode: image/directory. Prompt mode: released prompt JSONL.",
    )
    parser.add_argument("--mode", choices=("plan", "prompt"), default="plan")
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Prompt mode root containing CITY/source_id images.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--reflection", choices=("on", "off"), default="on")
    parser.add_argument("--max-reflections", type=int, default=3)
    parser.add_argument(
        "--max-generations",
        type=int,
        default=None,
        help="Deprecated compatibility alias; equals initial generation plus reflections.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--require-generated",
        action="store_true",
        help="Exit nonzero unless every requested input produced an image (no Skip).",
    )
    parser.add_argument("--mock", action="store_true", help="Use synthetic backends.")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if not 0 <= args.max_reflections <= 3:
        parser.error("--max-reflections must be between 0 and 3")
    if args.max_generations is not None and not 1 <= args.max_generations <= 4:
        parser.error("--max-generations must be between 1 and 4")
    if args.mode == "prompt" and args.image_root is None:
        parser.error("--image-root is required in prompt mode")
    if args.mode == "prompt" and not args.input.is_file():
        parser.error("prompt mode input must be a JSONL file")
    return args


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image extension: {path}")
        return [path]
    if path.is_dir():
        return sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
        )
    raise FileNotFoundError(path)


def planner_sample_id(image_path: Path, input_path: Path) -> str:
    if input_path.is_dir():
        relative = image_path.relative_to(input_path).with_suffix("")
        return "__".join(relative.parts)
    return image_path.stem


def main() -> None:
    args = parse_args()
    os.environ["ADAPTVPR_DISABLE_MOCK"] = "0" if args.mock else "1"
    os.environ["ADAPTVPR_FORCE_MOCK_LLM"] = "1" if args.mock else "0"

    from generation.agent import SceneAugmentAgent
    from generation.batch import (
        atomic_write_json,
        build_summary,
        error_path,
        is_completed_record,
        load_jsonl,
        materialize_records,
        record_path,
        resolve_prompt_source,
        safe_sample_id,
        write_record,
    )

    reflection_enabled = args.reflection == "on" and args.max_reflections > 0
    if args.max_generations is not None:
        max_generations = args.max_generations
        reflection_enabled = reflection_enabled and max_generations > 1
    else:
        max_generations = 1 + args.max_reflections if reflection_enabled else 1

    if args.mode == "plan":
        images = collect_images(args.input)
        tasks = [
            {
                "sample_id": safe_sample_id(planner_sample_id(path, args.input)),
                "image_path": path,
                "entry": None,
            }
            for path in images
        ]
    else:
        tasks = [
            {"sample_id": entry["sample_id"], "image_path": None, "entry": entry}
            for entry in load_jsonl(args.input)
        ]
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        raise RuntimeError("No input records or supported images were found")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "mode": args.mode,
        "input": str(args.input.resolve()),
        "image_root": str(args.image_root.resolve()) if args.image_root else None,
        "output": str(args.output.resolve()),
        "reflection_enabled": reflection_enabled,
        "max_reflections": max(0, max_generations - 1),
        "seed": args.seed,
        "mock": args.mock,
        "requested": len(tasks),
        "require_generated": args.require_generated,
        "planner_model": os.getenv("ADAPTVPR_PLANNER_MODEL", "qwen3-vl-4b-instruct-remote"),
        "planner_api_base": os.getenv("ADAPTVPR_PLANNER_API_BASE", "http://127.0.0.1:23002/v1"),
        "iclight_api_url": os.getenv("ICLIGHT_API_URL", "http://127.0.0.1:8002/generate"),
        "lightx2v_api_url": os.getenv("LIGHTX2V_API_URL", "http://127.0.0.1:8001/generate"),
    }
    manifest_path = args.output / "experiment.json"
    comparable_keys = (
        "mode",
        "input",
        "image_root",
        "reflection_enabled",
        "max_reflections",
        "seed",
        "mock",
        "require_generated",
    )
    if args.resume and manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = [
            key
            for key in comparable_keys
            if existing_manifest.get(key) != manifest.get(key)
        ]
        if mismatches:
            raise RuntimeError(
                "resume configuration differs from experiment.json: "
                + ", ".join(mismatches)
            )
    elif not args.resume and any((args.output / "records").glob("*.json")):
        raise RuntimeError(
            f"{args.output} already contains records; use --resume or a new --output"
        )
    atomic_write_json(manifest_path, manifest)

    agent = None
    for index, task in enumerate(tasks, 1):
        sample_id = task["sample_id"]
        completed_path = record_path(args.output, sample_id)
        if args.resume and is_completed_record(completed_path):
            print(f"[{index}/{len(tasks)}] resume {sample_id}", flush=True)
            continue
        try:
            if agent is None:
                agent = SceneAugmentAgent(
                    mock=args.mock,
                    max_generations=max_generations,
                    planning_enabled=args.mode == "plan",
                    reflection_enabled=reflection_enabled,
                    base_seed=args.seed,
                )
            image_path = task["image_path"]
            if args.mode == "prompt":
                image_path = resolve_prompt_source(task["entry"], args.image_root)
            record = agent.run_path(
                image_path,
                args.output,
                entry=task["entry"],
                frozen_prompt=args.mode == "prompt",
                sample_id=sample_id,
            )
            record["sample_id"] = sample_id
            record["run_mode"] = args.mode
            write_record(args.output, record)
            print(
                f"[{index}/{len(tasks)}] {sample_id}: route={record.get('route')} "
                f"status={record.get('status')} rounds={record.get('rounds_used', 0)}",
                flush=True,
            )
        except Exception as exc:
            error = {
                "sample_id": sample_id,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            atomic_write_json(error_path(args.output, sample_id), error)
            print(
                f"[{index}/{len(tasks)}] {sample_id}: ERROR {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if args.fail_fast:
                raise

    selected_ids = [task["sample_id"] for task in tasks]
    records = materialize_records(args.output, selected_ids)
    errors = sum(error_path(args.output, sample_id).is_file() for sample_id in selected_ids)
    summary = build_summary(records, errors=errors, requested=len(tasks))
    atomic_write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if errors or (args.require_generated and summary["generated"] != len(tasks)):
        if args.require_generated and summary["generated"] != len(tasks):
            print(
                "required every input to generate: "
                f"generated={summary['generated']} requested={len(tasks)}",
                file=sys.stderr,
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
