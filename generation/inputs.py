"""Input normalization for released, precomputed AdaptVPR prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from generation.router import ROUTE_TO_MODEL, Route, normalize_route
from prompts.rules import normalize_weather


def parse_condition(condition: Any) -> tuple[str | None, str | None]:
    """Parse a released ``weather+occlusion`` condition."""

    raw_condition = str(condition or "").strip().lower()
    local_taxonomy = {
        "curbside / parked-vehicle occlusion": "vehicle",
        "road-traffic occlusion": "vehicle",
        "other local occlusions": "person",
        "curbside_parked_vehicle_occlusion": "vehicle",
        "road_traffic_occlusion": "vehicle",
        "other_local_occlusions": "person",
    }
    tokens = [
        token.strip().lower()
        for token in raw_condition.split("+")
        if token.strip()
    ]
    weather = next(
        (normalized for token in tokens if (normalized := normalize_weather(token))),
        None,
    )
    occlusion = next(
        (token for token in tokens if token in {"vehicle", "person"}),
        local_taxonomy.get(raw_condition),
    )
    return weather, occlusion


def normalize_frozen_prompt_entry(
    raw: dict[str, Any], image_path: str | Path
) -> dict[str, Any]:
    """Validate a released prompt without rewriting its route or prompt.

    This path deliberately bypasses planner scheduling and prompt construction.
    It is used when callers need the exact released prompt to reach the image
    generator byte-for-byte.
    """

    image_path = Path(image_path)
    route = normalize_route(raw.get("route"))
    if route == Route.SKIP.value:
        raise ValueError("frozen prompt entries must use global, local, or dual")

    prompt = raw.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("frozen prompt entry has no non-empty string prompt")

    weather, occlusion = parse_condition(raw.get("condition"))
    if raw.get("weather") is not None:
        weather = normalize_weather(raw.get("weather"))
    if raw.get("occlusion") is not None:
        value = str(raw.get("occlusion")).strip().lower()
        occlusion = value if value in {"vehicle", "person"} else None

    valid = (
        (route == Route.GLOBAL.value and weather is not None and occlusion is None)
        or (route == Route.LOCAL.value and weather is None and occlusion is not None)
        or (route == Route.DUAL.value and weather is not None and occlusion is not None)
    )
    if not valid:
        raise ValueError(
            "route/condition mismatch: "
            f"route={route!r} condition={raw.get('condition')!r}"
        )

    sample_id = str(raw.get("sample_id") or image_path.stem).strip()
    if not sample_id:
        raise ValueError("frozen prompt entry has no sample_id")

    decision = dict(raw)
    decision.update(
        {
            "sample_id": sample_id,
            "file_name": image_path.name,
            "source_id": str(raw.get("source_id") or image_path.name),
            "source_path": str(image_path),
            "city": str(raw.get("city") or image_path.parent.name),
            "route": route,
            "weather": weather,
            "occlusion": occlusion,
            "selected_model": ROUTE_TO_MODEL[route],
            "position": str(raw.get("position") or raw.get("target_region") or ""),
            "prompt": prompt,
            "prompt_source": "released_prompt",
            "frozen_prompt": True,
        }
    )
    return decision
