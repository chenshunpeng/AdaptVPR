import json
import re
from enum import Enum
from pathlib import Path
from typing import Any

from prompts.rules import (
    OCCLUSIONS,
    build_structured_prompt,
    choose_occlusion_from_context,
    ensure_global_iclight_constraints,
    has_person_surface,
    has_vehicle_surface,
    normalize_weather,
    predict_bad_image,
)


class Route(str, Enum):
    SKIP = "skip"
    GLOBAL = "global"
    LOCAL = "local"
    DUAL = "dual"


VALID_ROUTES = {route.value for route in Route}
ROUTE_ALIASES = {
    "weather": Route.GLOBAL.value,
    "weather_only": Route.GLOBAL.value,
    "occlusion": Route.LOCAL.value,
    "occlusion_only": Route.LOCAL.value,
    "both": Route.DUAL.value,
    "weather_and_occlusion": Route.DUAL.value,
    "pass": Route.SKIP.value,
}

ROUTE_TO_MODEL = {
    Route.SKIP.value: "None",
    Route.GLOBAL.value: "IC-Light",
    Route.LOCAL.value: "LightX2V-Local",
    Route.DUAL.value: "LightX2V-Dual",
}


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_route(raw_route: Any) -> str:
    if hasattr(raw_route, "value"):
        raw_route = raw_route.value
    route = str(raw_route or Route.SKIP.value).lower().strip()
    route = ROUTE_ALIASES.get(route, route)
    if route not in VALID_ROUTES:
        return Route.SKIP.value
    return route


def _normalize_score(value: Any) -> float:
    try:
        score = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    if score > 1.0:
        score = score / 10.0
    return round(max(0.0, min(1.0, score)), 4)


def _is_structured_prompt(prompt: str) -> bool:
    return prompt.startswith("Task:") and "Preserve:" in prompt and "Forbidden:" in prompt


def normalize_decision(raw: dict[str, Any], image_path: str | Path | None = None) -> dict[str, Any]:
    image_path = Path(image_path) if image_path is not None else None
    route = normalize_route(raw.get("route"))

    weather = normalize_weather(raw.get("weather"))

    occlusion = raw.get("occlusion")
    if isinstance(occlusion, str):
        occlusion = occlusion.lower().strip()
    if occlusion not in OCCLUSIONS:
        occlusion = None

    context_parts = (
        raw.get("position"),
        raw.get("target_region"),
        raw.get("prompt"),
        raw.get("reason"),
        raw.get("skip_reason"),
        raw.get("street_scene_quality"),
        raw.get("occlusion_feasibility"),
    )
    inferred_occlusion = choose_occlusion_from_context(*context_parts, requested=occlusion)
    if route in {Route.LOCAL.value, Route.DUAL.value}:
        occlusion = inferred_occlusion

    weather_score = _normalize_score(raw.get("weather_score"))
    occlusion_score = _normalize_score(raw.get("occlusion_score"))
    weather_feasibility = str(raw.get("weather_feasibility", "") or "").lower().strip()
    occlusion_feasibility = str(raw.get("occlusion_feasibility", "") or "").lower().strip()
    surface_context = (
        raw.get("position"),
        raw.get("target_region"),
        raw.get("prompt"),
        raw.get("reason"),
        raw.get("skip_reason"),
    )
    vehicle_surface_ok = has_vehicle_surface(*surface_context)
    person_surface_ok = has_person_surface(*surface_context)
    legal_occlusion_surface = bool(vehicle_surface_ok or person_surface_ok)

    # Optional legacy behavior. It can only upgrade global after a legal
    # vehicle/person placement surface is explicit. Skip is never upgraded.
    force_occlusion = str(raw.get("force_occlusion", "")).lower() in {"1", "true", "yes"}
    if (
        force_occlusion
        and route == Route.GLOBAL.value
        and occlusion_score >= 0.2
        and legal_occlusion_surface
        and inferred_occlusion
    ):
        route = Route.DUAL.value if weather_score >= 0.6 and weather else Route.LOCAL.value
        occlusion = inferred_occlusion

    if route == Route.SKIP.value:
        weather = None
        occlusion = None
    elif route == Route.GLOBAL.value:
        weather = weather or "rain"
        occlusion = None
    elif route == Route.LOCAL.value:
        weather = None
        if not occlusion or not legal_occlusion_surface:
            route = Route.SKIP.value
            skip_reason = str(raw.get("skip_reason", "")).strip() or "no_legal_vehicle_or_person_surface_for_local_occlusion"
        else:
            skip_reason = str(raw.get("skip_reason", "")).strip()
    elif route == Route.DUAL.value:
        weather = weather or "rain"
        if not occlusion or not legal_occlusion_surface:
            route = Route.SKIP.value
            weather = None
            occlusion = None
            skip_reason = str(raw.get("skip_reason", "")).strip() or "no_legal_vehicle_or_person_surface_for_dual_occlusion"
    else:
        weather = None
        occlusion = None

    raw_prompt = str(raw.get("prompt", "")).strip()
    skip_reason = locals().get("skip_reason", str(raw.get("skip_reason", "")).strip())
    if route == Route.SKIP.value and not skip_reason:
        skip_reason = "invalid_or_unsuitable_scene"

    if _is_structured_prompt(raw_prompt):
        prompt = ensure_global_iclight_constraints(raw_prompt) if route == Route.GLOBAL.value else raw_prompt
    elif route == Route.SKIP.value:
        prompt = build_structured_prompt(
            route=route,
            weather=None,
            occlusion=None,
            base_prompt=skip_reason,
        )
    else:
        prompt = build_structured_prompt(
            route=route,
            weather=weather,
            occlusion=occlusion,
            position=str(raw.get("position", "") or raw.get("target_region", "")).strip(),
            base_prompt=raw_prompt,
        )

    bad_image = predict_bad_image(
        route=route,
        prompt=prompt,
        reason=str(raw.get("reason", "")).strip(),
        weather=weather,
        occlusion=occlusion,
    )

    return {
        "file_name": image_path.name if image_path else str(raw.get("file_name", "unknown")),
        "city": str(raw.get("city") or (image_path.parent.name if image_path else "unknown")),
        "route": route,
        "weather": weather,
        "occlusion": occlusion,
        "weather_score": weather_score,
        "occlusion_score": occlusion_score,
        "selected_model": ROUTE_TO_MODEL[route],
        "position": str(raw.get("position", "") or raw.get("target_region", "")).strip(),
        "prompt": prompt,
        "reason": str(raw.get("reason", "")).strip(),
        "skip_reason": skip_reason,
        "street_scene_quality": str(raw.get("street_scene_quality", "") or "").lower().strip(),
        "occlusion_feasibility": occlusion_feasibility,
        "weather_feasibility": weather_feasibility,
        "road_visibility": str(raw.get("road_visibility", "") or "").lower().strip(),
        "sky_visibility": str(raw.get("sky_visibility", "") or "").lower().strip(),
        "vegetation_level": str(raw.get("vegetation_level", "") or "").lower().strip(),
        "facade_density": str(raw.get("facade_density", "") or "").lower().strip(),
        "close_building": str(raw.get("close_building", "") or "").lower().strip(),
        "distant_landmarks_readable": str(raw.get("distant_landmarks_readable", "") or "").lower().strip(),
        "global_weather_risk": str(raw.get("global_weather_risk", "") or "").lower().strip(),
        "safe_global_weathers": raw.get("safe_global_weathers") if isinstance(raw.get("safe_global_weathers"), list) else [],
        "risk_score": bad_image["risk_score"],
        "risk_flags": bad_image["risk_flags"],
        "skip_recommendation": bad_image["skip_recommendation"],
    }


class ImageRouter:
    def process_vlm_output(self, vlm_json_str: str, image_path: str | Path | None = None) -> dict[str, Any]:
        return normalize_decision(extract_json(vlm_json_str), image_path=image_path)
