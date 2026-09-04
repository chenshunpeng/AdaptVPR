"""Reusable batch I/O and reporting for the public AdaptVPR CLI."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


def safe_sample_id(value: Any) -> str:
    raw = str(value or "").strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    if not safe:
        raise ValueError("sample_id is empty or contains no safe characters")
    return safe


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            sample_id = safe_sample_id(record.get("sample_id"))
            if sample_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate sample_id {sample_id!r}")
            seen.add(sample_id)
            record["sample_id"] = sample_id
            records.append(record)
    return records


def resolve_prompt_source(entry: dict[str, Any], image_root: Path) -> Path:
    source_id = str(entry.get("source_id") or "").strip()
    if not source_id:
        raise ValueError(f"{entry.get('sample_id')}: missing source_id")
    source = Path(source_id)
    if source.is_absolute() and source.is_file():
        return source
    candidates = []
    city = str(entry.get("city") or "").strip()
    if city:
        candidates.append(image_root / city / source_id)
    candidates.append(image_root / source_id)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"{entry.get('sample_id')}: source image not found; tried {rendered}")


def record_path(output_root: Path, sample_id: Any) -> Path:
    return output_root / "records" / f"{safe_sample_id(sample_id)}.json"


def is_completed_record(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return record.get("status") != "router_failed"


def error_path(output_root: Path, sample_id: Any) -> Path:
    return output_root / "errors" / f"{safe_sample_id(sample_id)}.json"


def write_record(output_root: Path, record: dict[str, Any]) -> Path:
    path = record_path(output_root, record.get("sample_id"))
    atomic_write_json(path, record)
    stale_error = error_path(output_root, record.get("sample_id"))
    if stale_error.is_file():
        stale_error.unlink()
    return path


def materialize_records(
    output_root: Path, sample_ids: Iterable[Any] | None = None
) -> list[dict[str, Any]]:
    if sample_ids is None:
        paths = sorted((output_root / "records").glob("*.json"))
    else:
        paths = [
            record_path(output_root, sample_id)
            for sample_id in sorted(safe_sample_id(value) for value in sample_ids)
        ]
        paths = [path for path in paths if path.is_file()]
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    jsonl_path = output_root / "records.jsonl"
    temporary = jsonl_path.with_name(f".{jsonl_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(jsonl_path)
    return records


def build_summary(
    records: Iterable[dict[str, Any]], *, errors: int = 0, requested: int | None = None
) -> dict[str, Any]:
    rows = list(records)
    generated = [row for row in rows if row.get("generated") is True]
    passed = [row for row in generated if row.get("passed") is True]
    skipped = [row for row in rows if row.get("status") == "skipped"]
    router_failed = [row for row in rows if row.get("status") == "router_failed"]
    reflected = [row for row in generated if int(row.get("rounds_used") or 0) > 1]
    improved = [
        row
        for row in reflected
        if row.get("reflection_rounds")
        and not bool(row["reflection_rounds"][0].get("eval", {}).get("passed"))
        and row.get("passed") is True
    ]
    route_stats: dict[str, dict[str, int | float]] = {}
    for route in ("global", "local", "dual"):
        route_rows = [row for row in generated if row.get("route") == route]
        route_passed = sum(row.get("passed") is True for row in route_rows)
        route_stats[route] = {
            "generated": len(route_rows),
            "passed": route_passed,
            "pass_rate": round(route_passed / len(route_rows), 6) if route_rows else 0.0,
        }
    return {
        "requested": requested if requested is not None else len(rows) + errors,
        "records": len(rows),
        "errors": int(errors),
        "generated": len(generated),
        "skipped": len(skipped),
        "router_failed": len(router_failed),
        "passed": len(passed),
        "verifier_pass_rate": round(len(passed) / len(generated), 6) if generated else 0.0,
        "reflected": len(reflected),
        "reflection_improved": len(improved),
        "route_stats": route_stats,
    }
