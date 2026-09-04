from __future__ import annotations

WEATHERS = {"rain", "snow", "night", "overcast", "fog", "rainy_night"}
WEATHER_ALIASES = {
    "rainy-night": "rainy_night",
    "rainy night": "rainy_night",
}
OCCLUSIONS = {"person", "vehicle"}
PROMPT_POLICY_VERSION = "adaptvpr_public_v2_source_anchored_reflection"


def normalize_weather(value: object) -> str | None:
    """Return the canonical internal weather name; rainy-night is rainy_night."""
    weather = str(value or "").lower().strip()
    weather = WEATHER_ALIASES.get(weather, weather)
    return weather if weather in WEATHERS else None

VEHICLE_SURFACE_TERMS = (
    "traffic lane", "road lane", "visible lane", "curbside lane", "curb lane",
    "parking bay", "parking lane", "roadside parking", "roadside parking area",
    "parking area", "parked car", "parked cars", "roadway", "carriageway",
    "street lane", "right lane", "left lane", "near lane", "foreground lane",
)
PERSON_SURFACE_TERMS = (
    "sidewalk", "paved sidewalk", "curb", "kerb", "crosswalk", "zebra crossing",
    "roadside pavement", "road-edge pavement", "visible pavement", "pedestrian area",
)


GLOBAL_ICLIGHT_FACADE_CONSTRAINT = (
    "Preserve the original building facade colors, wall materials, architectural textures, "
    "doors, windows, signs, storefronts, rooflines, and structural geometry. Apply weather "
    "or time-of-day changes through realistic illumination, sky, road surface, shadows, "
    "and atmosphere only. Do not repaint, restyle, relight, replace, or stylize fixed "
    "architectural surfaces. Keep facade color shifts extremely subtle and physically plausible"
)
LIGHTX2V_LOCAL_VEHICLE_CONSTRAINT = (
    "exactly one clearly visible vehicle, occupying approximately 6-10% image area, "
    "partially blocking the nearest visible lane, clearly visible vehicle body, clearly visible wheels, "
    "noticeable but realistic local occlusion"
)
LIGHTX2V_DUAL_VEHICLE_CONSTRAINT = (
        "one large realistic box truck, delivery van, city bus, shuttle bus, or tall service vehicle, preferably white "
        "or light-colored with visible wheels and realistic side panels. The dominant vehicle should occupy "
        "approximately 24-32% image area, placed in the lower-middle or near-midground on a legal traffic lane, "
        "curbside lane, parking bay, or intersection approach. It must interrupt a continuous lower-scene recognition "
        "band while preserving landmark facades, building silhouettes, road geometry, perspective, main sign identity, "
        "and place-defining structures"
    )
LIGHTX2V_PERSON_CONSTRAINT = (
    "exactly one realistic full-body pedestrian, natural scale, normal clothing, feet on the "
    "visible ground plane, matched lighting, realistic shadow"
)
FORBID_CLAUSE = (
    "no new background vehicles, no dense crowds, no traffic jams, no new traffic signs, "
    "no readable hallucinated text, no logo or license plate changes, no warped buildings, "
    "no changed viewpoint, no solid black rectangle, no black frame, no black mask, "
    "no black shadow person, no dark blob, no border occluder, no faceless silhouette, "
    "no object on wall, sky, or building facade, do not cover main building structure, "
    "do not change road geometry"
)
GLOBAL_ICLIGHT_FORBID_CLAUSE = (
    "do not recolor walls, facades, doors, windows, signs, storefronts, roofs, or fixed "
    "architectural structures into neon, rainbow, multicolored, oversaturated, glowing, "
    "or large-area unnatural color-pollution effects; do not change shop-sign colors, sign shapes, "
    "facade materials, window frames, awnings, storefront identity, or roofline colors"
)
LIGHTX2V_FORBID_CLAUSE = (
    "no black silhouette person, no shadow-like person, no tiny pedestrian, no cutout person, "
    "no sticker-like person, no border pedestrian, no person floating, no person on wall, sky, or "
    "building facade, no black vehicle block, no floating vehicle, no vehicle on sidewalk, wall, sky, "
    "or building facade, no oversized foreground vehicle, no flat pasted cutout, no sticker-like "
    "occluder, no border-attached occluder, no object covering place-defining facades, signs, windows, "
    "storefronts, road layout, or lane markings"
)
QUALITY_CLAUSE = (
    "photorealistic street-view image, natural exposure, matched lighting, realistic shadows, "
    "no pasted-object appearance"
)
NEGATIVE_PROMPT = (
    "warped buildings, changed road layout, changed camera viewpoint, new signs, new text, "
    "readable license plates, extra vehicles, dense crowds, traffic jam, unrealistic objects, "
    "floating person, pasted object, pasted cutout, sticker-like object, broken pavement, distorted facade, cartoon, low quality, "
    "solid black rectangle, black bar, black frame, black mask, pure black occluder, "
    "edge shadow person, black shadow person, shadow-like person, faceless silhouette, dark blob, ghost person, "
    "tiny pedestrian, cutout person, sticker-like person, border pedestrian, person floating, "
    "black vehicle block, floating vehicle, vehicle on sidewalk, vehicle on wall, "
    "border occluder, object on wall, object in sky, object on building facade, "
    "close-up person, close-up car, oversized foreground vehicle, large foreground object, blocked main building, over-occlusion, "
    "neon facade, rainbow facade, multicolored building, oversaturated storefront, changed shop sign, "
    "changed facade color, changed wall material, distorted sign, deformed storefront, "
    "tiny occluder, weak occlusion, barely visible object, invisible cyclist, weak local edit"
)
GLOBAL_SNOW_CONSERVATIVE_PROMPT = """Preserve scene geometry, buildings, signs, vehicles and viewpoint.

Apply light snowy winter weather only:
overcast sky, visible snowflakes,
light snow accumulation on horizontal surfaces,


Natural street-view photo, conservative edit.
Avoid structural changes, deformation, blur or detail loss."""

GLOBAL_NIGHT_CONSERVATIVE_PROMPT = """Preserve scene geometry, buildings, signs, vehicles and viewpoint.

Apply urban night conditions only:
dim street lighting,
reduced ambient brightness,


Natural street-view photo, conservative edit.
Avoid structural changes, blur or deformation."""

GLOBAL_RAIN_CONSERVATIVE_PROMPT = """Preserve scene geometry, buildings, signs, vehicles, lane markings, road layout, and viewpoint.

Apply conservative light rainy weather only:
subtle wet pavement, soft overcast sky,
very light visible rain streaks,
mild reflections near the road surface.

Keep building facades, storefronts, signs, windows, trees, cars, and road boundaries unchanged.
Do not add heavy rain, flooding, haze, darkness, strong blur, large reflections, new vehicles, or structural changes.

Natural documentary street-view photo, conservative edit."""

EXPERIENCE_RULES = {
    "global_rain": {
        "target": "wet reflective pavement, overcast sky, subtle visible falling rain",
        "avoid": "rain only shown as wet ground without atmosphere",
    },
    "global_snow": {
        "target": "light snow, realistic winter atmosphere, thin snow on sidewalks and parked cars",
        "avoid": "heavy snow that hides place-defining facades or road geometry",
    },
    "global_night": {
        "target": "dim artificial street lighting, reduced saturation, controlled low-light exposure",
        "avoid": "over-dark image or changed facade/window/fence/tree structure",
    },
    "global_overcast": {
        "target": "uniform cloudy sky, soft diffuse daylight, muted contrast, no strong shadows",
        "avoid": "turning the scene into rain, night, or heavy fog",
    },
    "global_fog": {
        "target": "realistic dense fog with reduced long-range visibility while keeping nearby road and facade geometry readable",
        "avoid": "fog so heavy that it hides place-defining facades, signs, lane markings, or road layout",
    },
    "local_person": {
        "target": "exactly one normal full-body pedestrian, clearly visible, natural scale, realistic clothing color, not a black silhouette, not a dark blob, not tiny, not at the image border, feet firmly on visible sidewalk, curb, crosswalk, or roadside pavement",
        "avoid": "edge shadow person, black shadow person, shadow-like person, faceless black silhouette, dark blob, tiny pedestrian, border pedestrian, floating person, cutout person, sticker-like person, pasted sharp person, crowd",
    },
    "local_vehicle": {
        "target": "exactly one normal street vehicle such as car, van, bus, or taxi, wheels aligned with road perspective, natural scale, not a black block, placed only on visible traffic lane, curbside lane, parking bay, or roadside parking area",
        "avoid": "pure black vehicle blob, black vehicle block, floating vehicle, vehicle on sidewalk, vehicle on wall, vehicle on sky, vehicle on building facade, oversized foreground vehicle, border occluder, traffic jam, extra parked cars, vehicle that changes road layout",
    },
    "dual_rain_person": {
        "target": "wet reflective pavement plus one natural pedestrian on visible sidewalk, crosswalk, curb, or road-edge pavement with matched rainy lighting",
        "avoid": "hallucinated vehicles, edge shadow person, wrong ground-plane position, pasted person",
    },
    "dual_rain_vehicle": {
        "target": "wet reflective pavement plus one normal vehicle on a visible road lane, curb lane, or parking lane with matched rainy lighting",
        "avoid": "pure black vehicle blob, wrong lane position, traffic jam, changed road layout",
    },
    "dual_snow_person": {
        "target": "light snow plus one natural pedestrian on visible sidewalk, crosswalk, curb, or road-edge pavement with matched winter lighting",
        "avoid": "edge shadow person, snow covering place-defining geometry, pasted person",
    },
    "dual_snow_vehicle": {
        "target": "light snow plus one normal vehicle on a visible lane or parking lane with tires aligned to the road perspective",
        "avoid": "pure black vehicle blob, traffic jam, heavy snow hiding landmarks",
    },
    "dual_overcast_person": {
        "target": "soft overcast daylight plus one natural pedestrian on visible sidewalk, crosswalk, curb, or road-edge pavement",
        "avoid": "edge shadow person, rain artifacts, night lighting, pasted person",
    },
    "dual_overcast_vehicle": {
        "target": "heavy low-cloud overcast daylight plus one dominant vehicle that blocks a meaningful lane or curbside segment under flat cool grey lighting",
        "avoid": "rain artifacts, sunny contrast, decorative tiny car, traffic jam, changed road layout",
    },
    "dual_fog_person": {
        "target": "realistic fog plus one natural pedestrian on visible nearby pavement, with background visibility reduced but local geometry preserved",
        "avoid": "faceless silhouette, edge shadow person, fog hiding the ground plane, pasted person",
    },
    "dual_fog_vehicle": {
        "target": "realistic fog plus one normal vehicle on a visible nearby lane or parking lane, with tires aligned to road perspective",
        "avoid": "pure black vehicle blob, fog hiding road layout, traffic jam, changed road geometry",
    },
    "dual_night_person": {
        "target": "dim night lighting plus one natural full-body pedestrian on visible pavement, with readable body shape and matched exposure",
        "avoid": "black silhouette, edge shadow person, over-bright person, changed building details",
    },
    "dual_night_vehicle": {
        "target": "night lighting plus one normal vehicle on a visible lane or curb lane with subtle realistic headlight spill",
        "avoid": "pure black vehicle blob, multiple new cars, traffic jam, changed lane geometry",
    },
}

WEATHER_SCENE_PHRASES = {
    "rain": "rainy street scene",
    "snow": "snowy street scene",
    "night": "night street scene",
    "overcast": "overcast street scene",
    "fog": "foggy street scene",
    "rainy_night": "rainy night street scene",
}

WEATHER_INSTRUCTION_PHRASES = {
    "rain": (
        "Apply strong rainy weather only:\n"
        "replace the sky with dark overcast rain clouds,\n"
        "add clearly visible rain streaks across the image,\n"
        "make the full drivable road surface wet with continuous reflective sheen,\n"
        "add several realistic puddles and elongated reflections on lanes and curbside asphalt,\n"
        "reduce visibility and local contrast slightly, darken the scene modestly, and keep all geometry sharp without blur."
    ),
    "snow": (
        "Apply snowy winter weather only:\n"
        "replace the sky with overcast winter clouds,\n"
        "add visible snowflakes falling across the image,\n"
        "make the ground, sidewalks and parked cars lightly snow-covered,\n"
        "coat rooftops with a thin layer of white snow,\n"
        "create a cold winter atmosphere with frosty surfaces."
    ),
    "night": (
        "Apply stronger urban night time only:\n"
        "darken the sky to nighttime black,\n"
        "add clear pools of artificial street lighting on the road surface,\n"
        "make building windows glow with warm indoor lights,\n"
        "darken the roadway and sidewalk while keeping lane geometry readable,\n"
        "reduce overall saturation and create a distinctly nocturnal atmosphere with stronger foreground-background contrast."
    ),
    "overcast": (
        "Apply heavy overcast weather only:\n"
        "replace the sky with a low dense blanket of thick grey clouds,\n"
        "flatten the entire scene lighting into strongly diffused cool daylight,\n"
        "remove nearly all direct-sun contrast and suppress crisp cast shadows,\n"
        "desaturate the full street scene moderately, cool the white balance slightly, and make distant/background regions look duller and more compressed,\n"
        "make some asphalt and curbside surfaces look faintly damp and less contrasty without adding rain streaks, puddles, or night lighting,\n"
        "preserve facade identity and geometry while making the whole scene feel distinctly gloomier and heavier than normal daylight."
    ),
    "fog": (
        "Apply foggy weather only:\n"
        "add thick realistic fog across the entire scene,\n"
        "reduce long-range visibility significantly,\n"
        "mute colors and soften contrast,\n"
        "keep nearby road and facade geometry readable."
    ),
    "rainy_night": (
        "Apply rainy night weather only:\n"
        "replace the sky with pitch-dark rain clouds,\n"
        "add clearly visible rain streaks under dim street lighting,\n"
        "make road surfaces wet with puddles reflecting street lamps,\n"
        "darken the overall scene to nocturnal atmosphere,\n"
        "keep building windows glowing with warm indoor lights."
    ),
}


def global_negative_prompt() -> str:
    return (
        "oil painting, painterly, illustration, watercolor, brush strokes, stylized image, "
        "cartoon, anime, cinematic color grading, colorful lighting, neon facade, rainbow facade, "
        "oversaturated storefront, changed facade color, changed wall material, changed shop sign, "
        "sign deformation, storefront color shift, warped buildings, changed road layout, "
        "changed camera viewpoint"
    )


def _rule_key(route: str, weather: str | None, occlusion: str | None) -> str:
    if route == "skip":
        return "skip"
    if route == "global":
        return f"global_{weather}"
    if route == "local":
        return f"local_{occlusion}"
    if route == "dual":
        return f"dual_{weather}_{occlusion}"
    return "skip"


def default_position(route: str, occlusion: str | None) -> str:
    if route == "skip":
        return "none"
    if route == "global":
        return "the entire image"
    if occlusion == "person":
        return "visible sidewalk, curb, crosswalk, roadside pavement, or road-edge ground"
    if occlusion == "vehicle":
        return "visible traffic lane, curbside lane, parking bay, or roadside parking area"
    return "a visible street surface with clear ground-plane support"


def _context_text(*parts: object) -> str:
    return " ".join(str(part or "") for part in parts).lower()


def has_vehicle_surface(*parts: object) -> bool:
    text = _context_text(*parts)
    return any(term in text for term in VEHICLE_SURFACE_TERMS)


def has_person_surface(*parts: object) -> bool:
    text = _context_text(*parts)
    return any(term in text for term in PERSON_SURFACE_TERMS)


def choose_occlusion_from_context(*parts: object, requested: str | None = None) -> str | None:
    """Vehicle-first occlusion choice from planner text; never fabricates a person fallback."""
    requested = requested if requested in OCCLUSIONS else None
    vehicle_ok = has_vehicle_surface(*parts)
    person_ok = has_person_surface(*parts)
    if vehicle_ok:
        return "vehicle"
    if requested == "person" and person_ok:
        return "person"
    if requested == "vehicle" and vehicle_ok:
        return "vehicle"
    if person_ok:
        return "person"
    return None


def ensure_global_iclight_constraints(prompt: str) -> str:
    """Append global IC-Light facade constraints when a structured prompt predates them."""
    if "Natural documentary street-view photo" in prompt or "natural documentary street-view photo" in prompt:
        return prompt
    return (
        f"{prompt} Natural documentary street-view photo, realistic lighting, conservative edit. "
        "Preserve road geometry, building facades, storefronts, signs, doors, windows, "
        "wall materials, original facade colors, and storefront identity."
    ).strip()


def _short_weather_phrase(weather: str | None, rule: dict) -> str:
    scene_phrase = WEATHER_SCENE_PHRASES.get(weather or "", f"{weather} street scene")
    target = str(rule.get("target", "")).strip()
    if not target:
        return scene_phrase
    target = target.split(".")[0]
    return f"{scene_phrase}, {target}"


def _short_position(position: str, occlusion: str | None) -> str:
    if not position:
        return default_position("local", occlusion)
    text = " ".join(str(position).replace("\n", " ").split())
    if len(text) > 140:
        text = text[:140].rsplit(" ", 1)[0]
    return text


def _local_vehicle_crowded_remove_prompt(position: str) -> str:
    pos = _short_position(position, "vehicle")
    return (
        "Conservative street-view edit for a vehicle-heavy original scene. The original image already "
        "contains several vehicles or visually crowded curb/road areas. "
        "If more than three vehicles are visible in the target road, curbside, or parking area, reduce "
        "vehicle clutter instead of adding cars: remove 1-2 least important background or curbside "
        "vehicles only, and naturally fill those regions with matching road, curb, sidewalk, parking-lane "
        "texture, shadows, or background continuation. Keep at most 1-2 clearly visible vehicles in the "
        "edited target area. If a local occluder is still needed, prefer one cyclist or one pedestrian near "
        "the road edge rather than adding another car. "
        f"Place any remaining local edit only on or immediately beside: {pos}. "
        "Preserve road geometry, lane markings, camera viewpoint, building facades, storefronts, signs, "
        "windows, doors, existing place identity, and global lighting/weather. Do not remove landmark "
        "objects, traffic signs, storefront identity, lane markings, or vehicles that define the place. "
        "Avoid traffic jams, dense vehicle clusters, new background cars, oversized foreground cars, black "
        "vehicle blocks, pasted cutouts, floating objects, and over-occlusion. Natural photorealistic "
        "street-view image."
    )


def _local_multi_occlusion_prompt(occlusion: str | None, position: str, original_vehicle_crowded: bool = False) -> str:
    if occlusion == "vehicle" and original_vehicle_crowded:
        return _local_vehicle_crowded_remove_prompt(position)

    pos = _short_position(position, occlusion)
    if occlusion == "person":
        occluders = (
            "Add controlled multi-occlusion with 2-4 realistic full-body pedestrians, "
            "optionally one cyclist or scooter rider if a road-edge or crosswalk is visible, "
            "and optionally one small roadside tree or pole-side shrub only near the curb."
        )
    elif occlusion == "vehicle":
        occluders = (
            "Add controlled multi-occlusion with 2-3 realistic street vehicles such as cars, "
            "vans, taxis, or a small bus, optionally one cyclist or scooter rider near the road edge, "
            "and optionally one small roadside tree near the curb."
        )
    else:
        occluders = (
            "Add controlled multi-occlusion with 1-3 realistic local street occluders, including "
            "vehicles, pedestrians, cyclists or scooter riders, and optionally one small roadside tree."
        )
    return (
        f"{occluders} Place them only on or immediately beside: {pos}. "
        "Total new occluder area should occupy approximately 12-20% of the image. "
        "Occluders should be clearly visible at mid-ground or near-ground scale, with realistic texture, "
        "matched lighting, contact shadows, wheels or feet aligned to the ground plane, and noticeable but "
        "natural local occlusion. Prioritize partially occluding traffic lanes, curbside lanes, roadside areas, "
        "sidewalks, and lower street-level foreground. Mildly occluding the lower edge of buildings, shopfront "
        "corners, parked-car edges, or non-core facade margins is allowed. Do not cover main sign text, landmark "
        "structures, central building facade areas, storefront identity, lane geometry, or place-defining details. "
        "Preserve road geometry, camera viewpoint, building structure, road layout, lane markings, signs, windows, "
        "doors, existing place identity, and global lighting/weather. Do not change weather, time of day, color "
        "style, viewpoint, architecture, or road layout. Natural photorealistic street-view image. Avoid tiny "
        "occluders, weak occlusion, barely visible objects, sticker-like cutouts, floating objects, black silhouettes, "
        "edge-attached objects, dense crowds, traffic jams, or over-occlusion."
    )


def build_structured_prompt(
    route: str,
    weather: str | None,
    occlusion: str | None,
    position: str = "",
    base_prompt: str = "",
    refined_prompt: str = "",
    original_vehicle_crowded: bool = False,
) -> str:
    """Build compact model prompts. Positive prompts only say what to generate."""
    weather = normalize_weather(weather)
    key = _rule_key(route, weather, occlusion)
    rule = EXPERIENCE_RULES.get(key, {})
    position = position or default_position(route, occlusion)

    refined_prompt = str(refined_prompt or "").strip()
    if refined_prompt and route in {"local", "dual"}:
        pos = _short_position(position, occlusion)
        if route == "local":
            route_constraints = (
                f"Mandatory Local-route constraints: apply only a realistic {occlusion or 'street-participant'} "
                f"occlusion on or immediately beside {pos}; keep weather, time of day, and global lighting unchanged; "
                "preserve exact road geometry, camera viewpoint, buildings, signs, lane markings, and place identity; "
                "keep every added object photorealistic and aligned with the visible ground plane."
            )
        else:
            route_constraints = (
                f"Mandatory Dual-route constraints: apply the requested {weather or 'global appearance'} change and "
                f"a realistic {occlusion or 'street-participant'} occlusion on or immediately beside {pos}; preserve "
                "exact road geometry, camera viewpoint, buildings, signs, lane markings, and place identity; keep every "
                "added object photorealistic and aligned with the visible ground plane."
            )
        return f"{refined_prompt}\n\n{route_constraints}"

    if route == "skip":
        reason = base_prompt or "image is not suitable for realistic weather/time or street-participant occlusion."
        return (
            "Task: skip generation. "
            f"Reason: {reason} "
            "Do not call generation service."
        )
    if route == "global":
        if weather == "rain":
            return GLOBAL_RAIN_CONSERVATIVE_PROMPT
        if weather == "snow":
            return GLOBAL_SNOW_CONSERVATIVE_PROMPT
        if weather == "night":
            return GLOBAL_NIGHT_CONSERVATIVE_PROMPT
        weather_text = _short_weather_phrase(weather, rule)
        return (
            f"{weather_text}. Natural documentary street-view photo, realistic lighting, conservative edit. "
            "Preserve road geometry, camera viewpoint, building facades, storefronts, signs, doors, "
            "windows, wall materials, rooflines, original facade colors, and storefront identity."
        )

    if route == "local":
        return _local_multi_occlusion_prompt(occlusion, position, original_vehicle_crowded=original_vehicle_crowded)

    pos = _short_position(position, occlusion)
    preserve = (
        "Preserve road layout, camera viewpoint, building facades, storefronts, signs, "
        "windows, doors, traffic signs, and place identity."
    )

    if occlusion == "vehicle":
        vehicle_constraint = (
            LIGHTX2V_DUAL_VEHICLE_CONSTRAINT
            if route == "dual"
            else LIGHTX2V_LOCAL_VEHICLE_CONSTRAINT
        )
        occluder_text = f"Add {vehicle_constraint} on {pos}."
    elif occlusion == "person":
        occluder_text = f"Add {LIGHTX2V_PERSON_CONSTRAINT} on {pos}."
    else:
        occluder_text = f"Add exactly one realistic street participant on {pos}."

    if route == "dual":
        weather_instr = WEATHER_INSTRUCTION_PHRASES.get(weather or "", f"Apply {weather} weather only.")
        dual_vehicle_focus = (
            "Make this a hard but structure-preserving VPR-positive edit. A realistic vehicle should create the main "
            "occlusion while the requested weather changes appearance across the scene. The vehicle must interrupt a "
            "continuous lower-scene cue such as lane markings, a curb line, a parking-lane boundary, or a crosswalk "
            "edge. Preserve the exact camera viewpoint, road topology, building silhouettes, facade layout, main sign "
            "identity, skyline, and all place-defining structures. Do not replace buildings, warp facades, create "
            "traffic jams, use black masks, or cover the full landmark facade."
        )
        if weather == "overcast" and occlusion == "vehicle":
            dual_vehicle_focus = (
                f"{dual_vehicle_focus} Under the heavy overcast lighting, make the vehicle read clearly as a darker "
                "mid-ground mass against the flatter street background, and let it block a slightly longer continuous "
                "lane or curbside strip than in the rain/night variants, while still preserving the landmark facade."
            )
        elif weather == "rain" and occlusion == "vehicle":
            dual_vehicle_focus = (
                f"{dual_vehicle_focus} Under rain, extend the wet reflective region around the vehicle so the vehicle "
                "plus adjacent lane, curb, and storefront-lower-boundary cues form one continuous difficult area, but do not blur geometry."
            )
        elif weather in {"night", "rainy_night"} and occlusion == "vehicle":
            dual_vehicle_focus = (
                f"{dual_vehicle_focus} Under night lighting, make the vehicle read clearly against the darker road and "
                "use headlight glare and wet reflections to make the lower curb/lane/facade-boundary band harder without turning the whole image into unreadable darkness."
            )
        elif weather == "snow" and occlusion == "vehicle":
            dual_vehicle_focus = (
                f"{dual_vehicle_focus} Under snow, add visible slush and compacted tire-track contrast around the vehicle "
                "so the lower road, curb, and parking-strip region becomes harder, while keeping building geometry recoverable."
            )
        return (
            f"Preserve all scene geometry, buildings, storefronts, signs, existing vehicles, "
            f"road layout and camera viewpoint.\n\n"
            f"{weather_instr}\n\n"
            f"{occluder_text}\n\n"
            f"{dual_vehicle_focus}\n\n"
            f"Do not modify scene structure or object layout."
        )

    return base_prompt or "Skip augmentation."


def predict_bad_image(
    route: str,
    prompt: str,
    reason: str = "",
    weather: str | None = None,
    occlusion: str | None = None,
) -> dict:
    weather = normalize_weather(weather)
    text = f"{prompt} {reason}".lower()
    flags = []

    if route in {"local", "dual"} and not occlusion:
        flags.append("missing_occlusion")
    if route in {"global", "dual"} and not weather:
        flags.append("missing_weather")
    if route in {"local", "dual"} and any(x in text for x in ["taxi rear", "rear of the", "specific sign"]):
        flags.append("over_specific_occlusion_target")
    if any(x in text for x in ["facade close-up", "no visible sidewalk", "no visible road", "no plausible sidewalk"]):
        flags.append("occlusion_implausible")
    if route in {"local", "dual"} and occlusion == "person" and has_vehicle_surface(text):
        flags.append("person_selected_despite_vehicle_surface")
    if route in {"local", "dual"} and occlusion == "vehicle" and not has_vehicle_surface(text):
        flags.append("vehicle_surface_not_explicit")
    if route in {"local", "dual"} and occlusion == "person" and not has_person_surface(text):
        flags.append("person_surface_not_explicit")
    if "dense green tree" in text and route in {"local", "dual"}:
        flags.append("foreground_foliage_may_confuse_occluder")
    if route == "dual" and occlusion == "vehicle" and weather == "night":
        flags.append("night_vehicle_high_hallucination_risk")

    risk_score = min(10, len(flags) * 3)
    return {
        "risk_score": risk_score,
        "risk_flags": flags,
        "skip_recommendation": risk_score >= 6,
    }
