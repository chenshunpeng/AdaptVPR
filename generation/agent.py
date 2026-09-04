import base64
import hashlib
import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from generation.router import Route, normalize_decision
from verification.evaluator import DualTraitEvaluator
from generation.iclight import ICLightGenerator
from generation.lightx2v import Lightx2vGenerator
from generation.reflection_controller import ReflectionController
from generation.llm_client import build_llm_client
from generation.output_layout import final_output_path
from prompts.rules import (
    WEATHERS,
    NEGATIVE_PROMPT,
    build_structured_prompt,
    choose_occlusion_from_context,
    ensure_global_iclight_constraints,
    global_negative_prompt,
    normalize_weather,
    predict_bad_image,
)


@dataclass
class AgentResult:
    """Final result returned by the public single-image agent API."""

    final_image: Image.Image
    route: str
    rounds_used: int
    final_prompt: str
    final_score_geo: float
    final_score_div: float
    passed: bool
    skipped: bool = False
    failed: bool = False
    eligible_for_training: bool = False
    stop_reason: str = ""


DEFAULT_MAX_GENERATIONS = int(
    os.getenv(
        "ADAPTVPR_MAX_GENERATIONS",
        os.getenv("ADAPTVPR_MAX_ROUNDS", "4"),
    )
)
VLM_ROUTER_RETRIES = int(os.getenv("ADAPTVPR_VLM_ROUTER_RETRIES", "0"))
VLM_MODEL = os.getenv(
    "ADAPTVPR_PLANNER_MODEL",
    "qwen3-vl-4b-instruct-remote",
)
def _route_ratios_from_env() -> dict[str, float]:
    ratios = {
        "skip": 0.25,
        "global": 0.25,
        "local": 0.25,
        "dual": 0.25,
    }
    raw = os.getenv("ADAPTVPR_TARGET_ROUTE_RATIOS", "").strip()
    if not raw:
        return ratios
    parsed: dict[str, float] = {}
    for item in raw.split(","):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        key = key.strip().lower()
        if key not in ratios:
            continue
        try:
            parsed[key] = max(0.0, float(value))
        except ValueError:
            continue
    if not parsed:
        return ratios
    ratios.update(parsed)
    total = sum(ratios.values())
    if total <= 0:
        return ratios
    return {key: value / total for key, value in ratios.items()}


TARGET_ROUTE_RATIOS = _route_ratios_from_env()
TARGET_LIGHTX2V_RATIO = TARGET_ROUTE_RATIOS["dual"] + TARGET_ROUTE_RATIOS["local"]
WEATHER_THRESHOLD = float(os.getenv("ADAPTVPR_WEATHER_THRESHOLD", "0.55"))
OCCLUSION_THRESHOLD = float(os.getenv("ADAPTVPR_OCCLUSION_THRESHOLD", "0.58"))
DEFICIT_WEIGHT = float(os.getenv("ADAPTVPR_DEFICIT_WEIGHT", "1.0"))
CAPABILITY_WEIGHT = float(os.getenv("ADAPTVPR_CAPABILITY_WEIGHT", "0.25"))
MIN_RATIO = float(os.getenv("ADAPTVPR_MIN_ROUTE_RATIO", "0.20"))
MAX_RATIO = float(os.getenv("ADAPTVPR_MAX_ROUTE_RATIO", "0.30"))
MIN_RATIO_DEFICIT_MULTIPLIER = float(os.getenv("ADAPTVPR_MIN_ROUTE_DEFICIT_MULTIPLIER", "2.0"))
GLOBAL_WEATHER_TARGET_RATIOS = {
    "overcast": 0.20,
    "fog": 0.20,
    "rain": 0.20,
    "snow": 0.20,
    "night": 0.20,
}
GLOBAL_WEATHER_ORDER = ("overcast", "fog", "rain", "snow", "night")
GLOBAL_SAFE_WEATHERS = ("overcast", "fog")
GLOBAL_MAX_SINGLE_WEATHER_RATIO = float(os.getenv("ADAPTVPR_GLOBAL_MAX_WEATHER_RATIO", "0.50"))
GLOBAL_ICLIGHT_HIGHRES_DENOISE = float(os.getenv("ADAPTVPR_GLOBAL_ICLIGHT_DENOISE", "0.30"))
GLOBAL_RAIN_ICLIGHT_HIGHRES_DENOISE = float(os.getenv("ADAPTVPR_GLOBAL_RAIN_ICLIGHT_DENOISE", "0.22"))
SYSTEM_PROMPT = """You are a strict VPR image augmentation capability scorer.
Return only one JSON object. Do not use markdown.
Do not choose or output the final route. The final route is selected by a quota
scheduler outside the VLM.

Score only the independent capabilities:
- weather_score in [0,1]: whether the image can safely receive a global weather
  or lighting edit while preserving VPR identity.
- occlusion_score in [0,1]: whether the image can safely receive exactly one
  realistic vehicle/person occluder on a legal ground-plane surface.

weather_score should be high when road geometry, building boundaries, sky/lighting,
and place-defining structures remain clear under weather/time edits. Lower it for
close facades, dense repetitive facades, vegetation-dominated scenes, weak road
visibility, sky fragments, or weather edits likely to damage facade texture.

occlusion_score should be high only when there is a clear, real,
perspective-consistent legal vehicle/person placement surface. Vehicle is preferred:
visible traffic lane, road lane, curbside lane, parking bay, parking lane, or
roadside parking area. Person is fallback: sidewalk, curb, crosswalk, roadside
pavement, or road-edge pavement. Lower it when an occluder would sit on the image
border, wall, sky, building facade, or would cover the main facade, storefront,
key sign, or road layout.

Set bad_image=true for close-up building/detail fragments, walls, doors, windows,
sign/storefront crops, sky fragments, scenes with no clear street space, views too
close or narrow, or images where neither weather editing nor occlusion can be done
realistically.

Allowed weather values are exactly: "rain", "snow", "night", "overcast", "fog".
Allowed occlusion values are exactly: "person", "vehicle".

The generated prompt must be English, realistic, and preserve road layout, camera viewpoint,
building geometry, lane markings, traffic signs, and place identity. Do not add dense crowds,
traffic jams, text, logos, black rectangles, black masks, edge shadows, or unrealistic objects."""


def image_to_b64(path: Path, max_side: int = 1024) -> str:
    image = Image.open(path).convert("RGB")
    return pil_image_to_b64(image, max_side=max_side)


def pil_image_to_b64(image: Image.Image, max_side: int = 1024) -> str:
    image = image.convert("RGB").copy()
    image.thumbnail((max_side, max_side))
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_json(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        raise json.JSONDecodeError("empty VLM response", text, 0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def is_systemic_api_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    systemic_markers = (
        "quota is not enough",
        "insufficient_quota",
        "arrearage",
        "overdue-payment",
        "access denied",
        "account is in good standing",
        "internalservererror",
        "error code: 502",
        "bad gateway",
        "service unavailable",
        "upstream",
    )
    return any(marker in text for marker in systemic_markers)


class SceneAugmentAgent:

    def __init__(
        self,
        llm_client=None,
        mock: bool = False,
        planning_only: bool = False,
        max_generations: int | None = None,
        planning_enabled: bool = True,
        reflection_enabled: bool = True,
        base_seed: int = 0,
    ):
        print("[Agent] Initializing modules...")

        needs_llm = planning_enabled or reflection_enabled or planning_only
        if llm_client is None and needs_llm:
            llm_client = build_llm_client()

        self.llm_client = llm_client
        self.planning_enabled = planning_enabled
        self.reflection_enabled = reflection_enabled
        self.base_seed = int(base_seed)
        configured_max = DEFAULT_MAX_GENERATIONS if max_generations is None else max_generations
        self.max_generations = max(1, int(configured_max))
        self.route_counts = {"skip": 0, "global": 0, "local": 0, "dual": 0}
        self.weather_counts = {weather: 0 for weather in sorted(WEATHERS)}
        self.global_weather_counts = {weather: 0 for weather in GLOBAL_WEATHER_ORDER}
        self.global_weather_pass_counts = {weather: 0 for weather in GLOBAL_WEATHER_ORDER}
        self.occlusion_counts = {"vehicle": 0, "person": 0}
        self.negative_prompt = NEGATIVE_PROMPT
        self.iclight = None
        self.lightx2v = None
        self.evaluator = None
        if not planning_only:
            self.iclight = ICLightGenerator(api_url="" if mock else None)
            self.lightx2v = Lightx2vGenerator(api_url="" if mock else None)
            self.evaluator = DualTraitEvaluator(mock=mock)
            if reflection_enabled:
                self.reflection_controller = ReflectionController(
                    llm_client=self.llm_client,
                    llm_model=VLM_MODEL,
                    evaluator=self.evaluator,
                    generate=self._generate_with_seed,
                )
            else:
                self.reflection_controller = None
        else:
            self.reflection_controller = None

        print("[Agent] Initialization complete")

    @staticmethod
    def _capability_user_prompt(file_name: str, city: str) -> str:
        return f"""Analyze this VPR street-view image and score augmentation capabilities.
Return this JSON schema:
{{
  "file_name": "{file_name}",
  "city": "{city}",
  "weather_score": 0.0-1.0,
  "occlusion_score": 0.0-1.0,
  "bad_image": true|false,
  "weather": "rain|snow|night|overcast|fog|null",
  "occlusion": "person|vehicle|null",
  "position": "precise plausible edit position",
  "prompt": "English generation prompt",
  "reason": "short reason",
  "skip_reason": "why skip is needed, or empty string",
  "street_scene_quality": "good|partial|bad",
  "occlusion_feasibility": "high|medium|low|none",
  "weather_feasibility": "high|medium|low|none",
  "road_visibility": "clear|partial|none",
  "sky_visibility": "clear|partial|none",
  "vegetation_level": "low|medium|high",
  "facade_density": "low|medium|high",
  "close_building": "yes|no",
  "distant_landmarks_readable": "high|medium|low",
  "global_weather_risk": "low|medium|high",
  "safe_global_weathers": ["rain|snow|night|overcast|fog"]
}}"""

    def _request_capabilities(
        self, image: Image.Image, *, file_name: str, city: str
    ) -> tuple[dict | None, Exception | None]:
        if not self.planning_enabled or self.llm_client is None:
            raise RuntimeError("Qwen planning is disabled for this agent")
        user_prompt = self._capability_user_prompt(file_name, city)
        raw_decision = None
        last_error = None
        response = ""
        for attempt in range(1, VLM_ROUTER_RETRIES + 2):
            try:
                response = self.llm_client.chat_with_images(
                    system=SYSTEM_PROMPT,
                    user=user_prompt,
                    images=[pil_image_to_b64(image)],
                    json_mode=True,
                    temperature=0.2,
                    model=VLM_MODEL,
                )
                raw_decision = extract_json(response)
                break
            except json.JSONDecodeError as exc:
                last_error = exc
                preview = (response or "").replace("\n", " ")[:160]
                print(
                    f"[Agent] VLM route JSON parse failed attempt={attempt}/{VLM_ROUTER_RETRIES + 1} "
                    f"file={file_name} error={exc} response_preview={preview!r}",
                    flush=True,
                )
            except Exception as exc:
                last_error = exc
                if is_systemic_api_error(exc):
                    print(
                        f"[Agent] systemic VLM/API error; stop planner to avoid mass router_failed "
                        f"file={file_name} error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    raise
                print(
                    f"[Agent] VLM route request failed attempt={attempt}/{VLM_ROUTER_RETRIES + 1} "
                    f"file={file_name} error={type(exc).__name__}: {exc}",
                    flush=True,
                )
        return raw_decision, last_error

    def _finalize_planned_decision(
        self,
        raw_decision: dict,
        *,
        image_path: Path,
        city: str,
        source_path: str,
        schedule_and_count: bool,
    ) -> dict:
        scheduled = (
            self._schedule_route_from_capabilities(raw_decision)
            if schedule_and_count
            else raw_decision
        )
        decision = normalize_decision(scheduled, image_path=image_path)
        for key in (
            "bad_image",
            "quota_eligible_routes",
            "quota_deficits",
            "quota_route_scores",
            "quota_reason",
        ):
            if key in scheduled:
                decision[key] = scheduled[key]

        decision["city"] = city
        decision["source_path"] = source_path
        if decision.get("router_failed"):
            return decision

        # Paper-level invariant: b_bad=true is a hard Skip. This is applied
        # after normalization as well, so cached/external entries cannot bypass
        # the capability scheduler and reach a generation route.
        decision = self._enforce_bad_image_skip(decision)
        decision = self._apply_original_vehicle_crowding_policy(decision)
        decision = self._apply_bad_image_policy(decision)
        decision = self._apply_scene_aware_global_policy(decision)

        if schedule_and_count:
            self._record_decision_counts(decision)
        return decision

    def _plan_image_object(
        self,
        image: Image.Image,
        *,
        image_path: Path,
        city: str,
        source_path: str,
        entry: dict | None = None,
    ) -> dict:
        if entry:
            if entry.get("router_failed") or entry.get("route") == "router_failed":
                return dict(entry)
            raw_decision = dict(entry)
            raw_decision["bad_image"] = self._boolish(entry.get("bad_image"))
            schedule_and_count = False
        else:
            raw_decision, last_error = self._request_capabilities(
                image, file_name=image_path.name, city=city
            )
            if raw_decision is None:
                decision = self._router_failed_decision(image_path, last_error)
                decision["city"] = city
                decision["source_path"] = source_path
                return decision
            schedule_and_count = True
        return self._finalize_planned_decision(
            raw_decision,
            image_path=image_path,
            city=city,
            source_path=source_path,
            schedule_and_count=schedule_and_count,
        )

    def plan_image(self, image_path: str | Path, entry: dict = None) -> dict:
        image_path = Path(image_path)
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        return self._plan_image_object(
            image,
            image_path=image_path,
            city=image_path.parent.name,
            source_path=str(image_path),
            entry=entry,
        )

    def _detect_original_vehicle_crowding(self, decision: dict) -> tuple[bool, list[str]]:
        if decision.get("route") != Route.LOCAL.value or decision.get("occlusion") != "vehicle":
            return False, []

        text = " ".join(
            str(decision.get(key, "") or "").lower()
            for key in (
                "position",
                "reason",
                "skip_reason",
                "street_scene_quality",
                "road_visibility",
                "occlusion_feasibility",
            )
        )
        strong_patterns = {
            "existing traffic",
            "existing vehicles",
            "existing vehicle flow",
            "behind existing traffic",
            "between existing vehicles",
            "traffic constrained",
            "foreground car clutter",
            "vehicle clutter",
            "road is cluttered",
            "traffic jam",
            "dense vehicle",
            "dense traffic",
            "many vehicles",
            "several vehicles",
            "multiple vehicles",
            "parked cars",
            "bus dominance",
        }
        weak_patterns = {
            "existing car",
            "curbside parking",
            "parking lane",
            "roadside parking",
        }
        reasons = [pattern for pattern in strong_patterns if pattern in text]
        weak_hits = [pattern for pattern in weak_patterns if pattern in text]

        if reasons:
            return True, sorted(set(reasons))
        # Avoid enabling removal for a single named car/van; require at least two weak original-vehicle cues.
        if len(set(weak_hits)) >= 2:
            return True, sorted(set(weak_hits))
        return False, sorted(set(weak_hits))

    def _apply_original_vehicle_crowding_policy(self, decision: dict) -> dict:
        crowded, reasons = self._detect_original_vehicle_crowding(decision)
        decision["original_vehicle_crowded"] = crowded
        if reasons:
            decision["original_vehicle_crowding_reasons"] = reasons
        if not crowded:
            return decision

        decision["local_vehicle_policy"] = "remove_vehicle_clutter_from_original_scene"
        decision["prompt"] = build_structured_prompt(
            route=decision["route"],
            weather=decision.get("weather"),
            occlusion=decision.get("occlusion"),
            position=decision.get("position", ""),
            base_prompt=decision.get("reason", ""),
            original_vehicle_crowded=True,
        )
        return decision

    def _router_failed_decision(self, image_path: Path, error: Exception | None = None) -> dict:
        reason = "vlm_empty_or_invalid_json"
        error_type = ""
        if error is not None:
            error_type = type(error).__name__
            reason = f"{reason}: {error}"
        return {
            "file_name": image_path.name,
            "city": image_path.parent.name,
            "route": "router_failed",
            "weather": None,
            "occlusion": None,
            "weather_score": 0,
            "occlusion_score": 0,
            "selected_model": "None",
            "position": "none",
            "prompt": reason,
            "reason": reason,
            "skip_reason": "",
            "street_scene_quality": "bad",
            "occlusion_feasibility": "none",
            "weather_feasibility": "none",
            "router_failed": True,
            "router_status": "needs_retry",
            "router_error_type": error_type,
            "router_error": reason,
        }

    def run_path(
        self,
        image_path: str | Path,
        output_root: str | Path,
        entry: dict = None,
        collect_bad: bool = True,
        frozen_prompt: bool = False,
        sample_id: str | None = None,
    ) -> dict:
        image_path = Path(image_path)
        output_root = Path(output_root)
        if frozen_prompt:
            if entry is None:
                raise ValueError("frozen_prompt=True requires an input entry")
            from generation.inputs import normalize_frozen_prompt_entry

            decision = normalize_frozen_prompt_entry(entry, image_path)
        else:
            decision = self.plan_image(image_path, entry=entry)
            decision.setdefault("sample_id", sample_id or image_path.stem)
            decision.setdefault("prompt_source", "qwen_planner" if entry is None else "external_entry")
            decision.setdefault("frozen_prompt", False)
        ref = Image.open(image_path).convert("RGB")

        route = decision["route"]
        prompt = decision["prompt"]

        if route == "router_failed":
            stem = self._sample_stem(image_path, decision)
            failed_dir = output_root / "router_failed"
            failed_dir.mkdir(parents=True, exist_ok=True)
            failed_path = failed_dir / f"{stem}__router_failed.json"
            record = dict(decision)
            record.update({
                "output_path": str(failed_path),
                "final_reflect_path": str(failed_path),
                "final_prompt": prompt,
                "reflection_rounds": [],
                "passed": False,
                "s_geo": 0.0,
                "s_div": 0.0,
                "rounds_used": 0,
                "generated": False,
                "eligible_for_training": False,
                "status": "router_failed",
                "router_failed": True,
                "router_status": "needs_retry",
            })
            failed_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            return record

        # Skip route: generate no image and store only the JSON record.
        if route in ("skip", "pass"):
            stem = self._sample_stem(image_path, decision)
            skip_dir = output_root / "skip"
            skip_dir.mkdir(parents=True, exist_ok=True)
            skip_path = skip_dir / f"{stem}__skip.json"
            record = dict(decision)
            record.update({
                "output_path": str(skip_path),
                "final_reflect_path": str(skip_path),
                "final_prompt": prompt,
                "reflection_rounds": [],
                "passed": False,
                "skipped": True,
                "failed": False,
                "generated": False,
                "eligible_for_training": False,
                "status": "skipped",
                "s_geo": 1.0,
                "s_div": 0.0,
                "rounds_used": 0,
            })
            skip_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            return record

        # Initial generation is followed by at most three source-anchored,
        # categorically distinct reflections for Local/Dual.
        reflection_enabled = bool(getattr(self, "reflection_enabled", True))
        route_max_generations = (
            1
            if route == Route.GLOBAL.value or not reflection_enabled
            else self.max_generations
        )
        seed_material = (
            f"{getattr(self, 'base_seed', 0)}|{decision.get('sample_id')}|{image_path}"
        )
        base_seed = 1 + int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16) % 2_000_000_000
        print(
            f"[Agent] {image_path.name} generation=1/{route_max_generations} route={route}",
            flush=True,
        )
        initial_image = self._generate(ref, prompt, route, decision=decision, seed=base_seed)
        if initial_image.size != ref.size:
            initial_image = initial_image.resize(ref.size, Image.Resampling.LANCZOS)
        initial_path = self._output_path(output_root, image_path, decision, 1)
        initial_path.parent.mkdir(parents=True, exist_ok=True)
        initial_image.save(initial_path, quality=95)
        initial_result = self.evaluator.evaluate(ref, initial_image, entry=decision)
        initial_eval = self._eval_to_dict(initial_result)
        rounds = [{
            "round": 1,
            "seed": base_seed,
            "prompt": prompt,
            "effective_prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "image_path": str(initial_path),
            "eval": initial_eval,
            "acceptance_decision": {
                "accepted": bool(initial_result.passed),
                "acceptance_rule": "dual_trait_scores_only",
                "reasons": [] if initial_result.passed else ["dual_trait_verifier_failed"],
            },
        }]
        input_prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        print(
            f"  s_geo={initial_result.s_geo:.3f} s_div={initial_result.s_div:.3f} "
            f"passed={initial_result.passed}",
            flush=True,
        )

        final_image = initial_image
        final_eval = initial_eval
        final_path = initial_path
        final_prompt = prompt
        score_accepted = bool(initial_result.passed)
        reflection_stop_reason = None
        if (
            not initial_result.passed
            and route in {Route.LOCAL.value, Route.DUAL.value}
            and route_max_generations > 1
        ):
            reflection_dir = output_root / "rounds" / self._sample_stem(image_path, decision)
            reflected = self.reflection_controller.run(
                source=ref,
                initial_candidate=initial_image,
                initial_prompt=prompt,
                initial_eval=initial_eval,
                decision=decision,
                output_dir=reflection_dir,
                base_seed=base_seed,
                max_reflections=min(3, route_max_generations - 1),
            )
            rounds.extend(reflected["attempts"])
            reflection_stop_reason = reflected["stop_reason"]
            score_accepted = bool(reflected["passed"])
            if reflected["attempts"]:
                final_prompt = reflected["attempts"][-1]["effective_prompt"]
            if reflected["passed"]:
                final_image = reflected["final_image"]
                final_eval = reflected["final_eval"]
            else:
                final_image = reflected["last_candidate"]
                final_eval = reflected["last_eval"]
            if reflected["attempts"] and reflected["attempts"][-1].get("image_path"):
                final_path = Path(reflected["attempts"][-1]["image_path"])

        # Only accepted images belong in the normal route directories. Failed
        # candidates are retained for audit under rejected/<route>/ and cannot
        # be collected accidentally by a route-directory glob.
        passed = score_accepted
        stem = self._sample_stem(image_path, decision)
        final_save_path = final_output_path(
            output_root,
            stem,
            route,
            decision.get("weather"),
            decision.get("occlusion"),
            passed=passed,
        )
        final_save_dir = final_save_path.parent
        final_save_dir.mkdir(parents=True, exist_ok=True)
        if final_image.size != ref.size:
            final_image = final_image.resize(ref.size, Image.Resampling.LANCZOS)
        final_image.save(final_save_path, quality=95)

        if route == Route.GLOBAL.value and final_eval["passed"]:
            weather = decision.get("weather")
            if weather in self.global_weather_pass_counts:
                self.global_weather_pass_counts[weather] += 1

        stop_reason = (
            "passed"
            if passed
            else "global_fast_reject"
            if route == Route.GLOBAL.value
            else reflection_stop_reason or "max_generations_exhausted"
        )
        record = dict(decision)
        record.update({
            "output_path": str(final_save_path),
            "final_reflect_path": str(final_path),
            "final_prompt": final_prompt,
            "reflection_rounds": rounds,
            "generated": True,
            "reflection_enabled": reflection_enabled,
            "input_prompt": prompt,
            "input_prompt_sha256": input_prompt_sha256,
            "effective_prompt_sha256": hashlib.sha256(final_prompt.encode("utf-8")).hexdigest(),
            "prompt_unchanged": final_prompt == prompt,
            "passed": passed,
            "acceptance_rule": "dual_trait_scores_only",
            "failed": not passed,
            "status": "passed" if passed else "failed",
            "eligible_for_training": passed,
            "stop_reason": stop_reason,
            "max_generations": route_max_generations,
            "s_geo": final_eval["s_geo"],
            "s_div": final_eval["s_div"],
            "geo_ok": final_eval["geo_ok"],
            "div_ok": final_eval["div_ok"],
            "rounds_used": rounds[-1]["round"],
        })
        record["training_candidates"] = []
        if passed:
            record["training_candidates"].append({
                "route": route,
                "weather": decision.get("weather"),
                "occlusion": decision.get("occlusion"),
                "output_path": str(final_save_path),
                "source_path": str(image_path),
                "passed": True,
                "eligible_for_training": True,
                "s_geo": final_eval["s_geo"],
                "s_div": final_eval["s_div"],
                "geo_ok": final_eval["geo_ok"],
                "div_ok": final_eval["div_ok"],
            })

        if collect_bad and not passed:
            self._collect_bad_case(output_root, image_path, final_path, record)

        return record

    def run(self, input_image: Image.Image, save_dir: str = None, entry: dict = None) -> AgentResult:
        print("\n" + "=" * 50)
        print("[Agent] ===== Processing started =====")

        print("[Agent] Step 1: scene understanding and routing...")
        image_digest = hashlib.sha256(input_image.tobytes()).hexdigest()
        decision = self._plan_image_object(
            input_image,
            image_path=Path("unknown") / "input.jpg",
            city="unknown",
            source_path=f"memory://sha256/{image_digest}",
            entry=entry,
        )
        route = decision["route"]
        prompt = decision["prompt"]

        if route in {Route.SKIP.value, "pass"}:
            return AgentResult(
                final_image=input_image.copy(),
                route=Route.SKIP.value,
                rounds_used=0,
                final_prompt=prompt,
                final_score_geo=1.0,
                final_score_div=0.0,
                passed=False,
                skipped=True,
                eligible_for_training=False,
                stop_reason="bad_image_direct_skip" if decision.get("bad_image") else "skip",
            )

        route_max_generations = 1 if route == Route.GLOBAL.value else self.max_generations
        base_seed = 1 + int(hashlib.sha256(input_image.tobytes()).hexdigest()[:8], 16) % 2_000_000_000
        gen_image = self._generate(input_image, prompt, route, decision=decision, seed=base_seed)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            gen_image.save(f"{save_dir}/round_1.jpg")
        eval_entry = dict(decision)
        eval_result = self.evaluator.evaluate(input_image, gen_image, entry=eval_entry)
        final_eval = self._eval_to_dict(eval_result)
        passed = bool(eval_result.passed)
        round_num = 1
        stop_reason = "passed" if passed else "global_fast_reject"

        if not passed and route in {Route.LOCAL.value, Route.DUAL.value} and route_max_generations > 1:
            reflected = self.reflection_controller.run(
                source=input_image,
                initial_candidate=gen_image,
                initial_prompt=prompt,
                initial_eval=final_eval,
                decision=decision,
                output_dir=Path(save_dir) if save_dir else None,
                base_seed=base_seed,
                max_reflections=min(3, route_max_generations - 1),
            )
            passed = bool(reflected["passed"])
            round_num = 1 + len(reflected["attempts"])
            stop_reason = reflected["stop_reason"]
            gen_image = reflected["final_image"] if passed else reflected["last_candidate"]
            final_eval = reflected["final_eval"] if passed else reflected["last_eval"]
            if reflected["attempts"]:
                prompt = reflected["attempts"][-1]["effective_prompt"]

        result = AgentResult(
            final_image=gen_image,
            route=route,
            rounds_used=round_num,
            final_prompt=prompt,
            final_score_geo=final_eval["s_geo"],
            final_score_div=final_eval["s_div"],
            passed=passed,
            failed=not passed,
            eligible_for_training=passed,
            stop_reason=stop_reason,
        )

        print("\n[Agent] ===== Processing complete =====")
        print(f"[Agent] route={result.route} | rounds={result.rounds_used} | "
              f"s_geo={result.final_score_geo:.3f} | s_div={result.final_score_div:.3f}")

        return result

    def _generate(
        self,
        ref_image: Image.Image,
        prompt: str,
        route: str,
        decision: dict | None = None,
        seed: int | None = None,
    ) -> Image.Image:
        generation_seed = 42 if seed is None else int(seed)
        if route == Route.GLOBAL.value:
            if not (decision or {}).get("frozen_prompt"):
                prompt = ensure_global_iclight_constraints(prompt)
            weather = (decision or {}).get("weather")
            highres_denoise = (
                GLOBAL_RAIN_ICLIGHT_HIGHRES_DENOISE
                if weather == "rain"
                else GLOBAL_ICLIGHT_HIGHRES_DENOISE
            )
            return self.iclight.generate(
                ref_image,
                prompt,
                negative_prompt=global_negative_prompt(),
                highres_denoise=highres_denoise,
                seed=generation_seed,
            )
        elif route == Route.LOCAL.value:
            return self.lightx2v.generate_local(
                ref_image, prompt, negative_prompt=self.negative_prompt, seed=generation_seed
            )
        elif route == Route.DUAL.value:
            return self.lightx2v.generate_dual(
                ref_image, prompt, negative_prompt=self.negative_prompt, seed=generation_seed
            )
        else:
            raise ValueError(f"Unknown route: {route}")

    def _generate_with_seed(
        self,
        ref_image: Image.Image,
        prompt: str,
        route: str,
        decision: dict,
        seed: int,
    ) -> Image.Image:
        return self._generate(ref_image, prompt, route, decision=decision, seed=seed)

    def _apply_bad_image_policy(self, decision: dict) -> dict:
        decision = self._enforce_bad_image_skip(decision)
        if decision.get("route") == Route.SKIP.value:
            return decision
        route = decision["route"]
        risk = predict_bad_image(
            route=route,
            prompt=decision.get("prompt", ""),
            reason=decision.get("reason", ""),
            weather=decision.get("weather"),
            occlusion=decision.get("occlusion"),
        )
        decision["risk_score"] = risk["risk_score"]
        decision["risk_flags"] = risk["risk_flags"]
        decision["skip_recommendation"] = risk["skip_recommendation"]
        print(
            f"[Agent] Route: {route} | model: {decision.get('selected_model')} | "
            f"risk: {risk['risk_score']} {risk['risk_flags']}"
        )
        if risk["skip_recommendation"] and route in {Route.LOCAL.value, Route.DUAL.value}:
            hard_skip_flags = {
                "occlusion_implausible",
                "foreground_foliage_may_confuse_occluder",
            }
            weather_feasibility = str(decision.get("weather_feasibility", "")).lower()
            should_skip = bool(hard_skip_flags.intersection(risk["risk_flags"]))
            if should_skip or weather_feasibility not in {"high", "medium"}:
                print("[Agent] High bad-image risk detected; falling back to Skip")
                decision["original_route"] = route
                decision["route"] = Route.SKIP.value
                decision["weather"] = None
                decision["occlusion"] = None
                decision["selected_model"] = "None"
                decision["skip_reason"] = self._skip_reason_from_risk(risk["risk_flags"])
                decision["prompt"] = build_structured_prompt(
                    route=decision["route"],
                    weather=None,
                    occlusion=None,
                    base_prompt=decision["skip_reason"],
                )
                return decision

            print("[Agent] LightX2V risk is high but weather editing remains suitable; falling back to Global")
            decision["original_route"] = route
            decision["downgrade_reason"] = "lightx2v_risk_high_weather_feasible"
            decision["route"] = Route.GLOBAL.value
            decision["weather"] = self._choose_weather(decision.get("weather"))
            decision["occlusion"] = None
            decision["selected_model"] = "IC-Light"
            decision["prompt"] = build_structured_prompt(
                route=decision["route"],
                weather=decision["weather"],
                occlusion=None,
                base_prompt=decision.get("prompt", ""),
            )
        return decision

    def _enforce_bad_image_skip(self, decision: dict) -> dict:
        """Apply the paper's b_bad => Skip invariant without quota competition."""
        bad_image = self._boolish(decision.get("bad_image"))
        if str(decision.get("street_scene_quality", "") or "").lower() == "bad":
            bad_image = True
        decision["bad_image"] = bad_image
        if not bad_image:
            return decision

        previous_route = str(decision.get("route") or "")
        decision["original_route"] = previous_route
        decision["route"] = Route.SKIP.value
        decision["weather"] = None
        decision["occlusion"] = None
        decision["selected_model"] = "None"
        decision["quota_eligible_routes"] = [Route.SKIP.value]
        decision["quota_route_scores"] = {Route.SKIP.value: 1.0}
        decision["quota_reason"] = "hard_bad_image_skip_before_quota"
        decision["skip_reason"] = (
            str(decision.get("skip_reason") or "").strip()
            or "bad_image_requires_direct_skip"
        )
        decision["prompt"] = build_structured_prompt(
            route=Route.SKIP.value,
            weather=None,
            occlusion=None,
            base_prompt=decision["skip_reason"],
        )
        return decision

    def _record_route_count(self, route: str) -> None:
        if route in self.route_counts:
            self.route_counts[route] += 1

    def _record_decision_counts(self, decision: dict) -> None:
        route = decision.get("route")
        self._record_route_count(route)
        weather = decision.get("weather")
        if route in {Route.GLOBAL.value, Route.DUAL.value} and weather in self.weather_counts:
            self.weather_counts[weather] += 1
        if route == Route.GLOBAL.value and weather in self.global_weather_counts:
            self.global_weather_counts[weather] += 1
        occlusion = decision.get("occlusion")
        if route in {Route.LOCAL.value, Route.DUAL.value} and occlusion in self.occlusion_counts:
            self.occlusion_counts[occlusion] += 1

    def _route_deficits(self) -> dict:
        planned = sum(self.route_counts.values()) + 1
        return {
            "skip": TARGET_ROUTE_RATIOS["skip"] * planned - self.route_counts["skip"],
            "global": TARGET_ROUTE_RATIOS["global"] * planned - self.route_counts["global"],
            "dual": TARGET_ROUTE_RATIOS["dual"] * planned - self.route_counts["dual"],
            "local": TARGET_ROUTE_RATIOS["local"] * planned - self.route_counts["local"],
            "lightx2v": TARGET_LIGHTX2V_RATIO * planned - (self.route_counts["local"] + self.route_counts["dual"]),
        }

    def _score01(self, value) -> float:
        try:
            score = float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0
        if score > 1.0:
            score = score / 10.0
        return max(0.0, min(1.0, score))

    def _boolish(self, value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    def _route_capability(self, route: str, weather_score: float, occlusion_score: float) -> float:
        if route == Route.GLOBAL.value:
            return weather_score
        if route == Route.LOCAL.value:
            return occlusion_score
        if route == Route.DUAL.value:
            return min(weather_score, occlusion_score)
        if route == Route.SKIP.value:
            return 1.0 - max(weather_score, occlusion_score)
        return 0.0

    def _schedule_route_from_capabilities(self, raw: dict) -> dict:
        weather_score = self._score01(raw.get("weather_score"))
        occlusion_score = self._score01(raw.get("occlusion_score"))
        bad_image = self._boolish(raw.get("bad_image"))
        street_quality = str(raw.get("street_scene_quality", "") or "").lower()
        if street_quality == "bad":
            bad_image = True

        # A bad image is not a quota candidate. It must bypass capability and
        # deficit scoring and enter Skip directly.
        if bad_image:
            scheduled = dict(raw)
            scheduled.update({
                "bad_image": True,
                "route": Route.SKIP.value,
                "weather": None,
                "occlusion": None,
                "weather_score": weather_score,
                "occlusion_score": occlusion_score,
                "quota_eligible_routes": [Route.SKIP.value],
                "quota_deficits": {},
                "quota_route_scores": {Route.SKIP.value: 1.0},
                "quota_reason": "hard_bad_image_skip_before_quota",
                "skip_reason": str(raw.get("skip_reason") or raw.get("reason") or "bad_image_requires_direct_skip"),
            })
            scheduled["prompt"] = build_structured_prompt(
                route=Route.SKIP.value,
                weather=None,
                occlusion=None,
                base_prompt=scheduled["skip_reason"],
            )
            print("[Agent] bad_image=true; hard routing directly to Skip", flush=True)
            return scheduled

        eligible: list[str] = []
        if weather_score >= WEATHER_THRESHOLD:
            eligible.append(Route.GLOBAL.value)
        if occlusion_score >= OCCLUSION_THRESHOLD:
            eligible.append(Route.LOCAL.value)
        if weather_score >= WEATHER_THRESHOLD and occlusion_score >= OCCLUSION_THRESHOLD:
            eligible.append(Route.DUAL.value)
        if not eligible:
            eligible.append(Route.SKIP.value)

        planned = sum(self.route_counts.values()) + 1
        deficits = self._route_deficits()
        filtered = [
            route
            for route in eligible
            if self.route_counts.get(route, 0) / planned <= MAX_RATIO
        ]
        candidates = filtered or eligible

        def route_score(route: str) -> float:
            current_ratio = self.route_counts.get(route, 0) / planned
            deficit_weight = DEFICIT_WEIGHT
            if current_ratio < MIN_RATIO:
                deficit_weight *= MIN_RATIO_DEFICIT_MULTIPLIER
            return (
                deficit_weight * deficits.get(route, 0.0)
                + CAPABILITY_WEIGHT * self._route_capability(route, weather_score, occlusion_score)
            )

        route = max(candidates, key=route_score)
        scheduled = dict(raw)
        scheduled["route"] = route
        scheduled["weather_score"] = weather_score
        scheduled["occlusion_score"] = occlusion_score
        scheduled["quota_eligible_routes"] = eligible
        scheduled["quota_deficits"] = {key: round(value, 4) for key, value in deficits.items() if key in TARGET_ROUTE_RATIOS}
        scheduled["quota_route_scores"] = {key: round(route_score(key), 4) for key in candidates}
        scheduled["quota_reason"] = (
            f"quota_scheduler route={route} eligible={eligible} "
            f"weather_score={weather_score:.3f} occlusion_score={occlusion_score:.3f}"
        )

        if route == Route.SKIP.value:
            scheduled["weather"] = None
            scheduled["occlusion"] = None
            scheduled["skip_reason"] = scheduled.get("skip_reason") or scheduled.get("reason") or "quota_scheduler_skip"
        elif route == Route.GLOBAL.value:
            scheduled["weather"] = self._choose_weather(scheduled.get("weather"))
            scheduled["occlusion"] = None
        elif route == Route.LOCAL.value:
            scheduled["weather"] = None
            scheduled["occlusion"] = self._choose_occlusion(scheduled)
        elif route == Route.DUAL.value:
            scheduled["weather"] = self._choose_weather(scheduled.get("weather"))
            scheduled["occlusion"] = self._choose_occlusion(scheduled)

        scheduled["prompt"] = build_structured_prompt(
            route=route,
            weather=scheduled.get("weather"),
            occlusion=scheduled.get("occlusion"),
            position=str(scheduled.get("position", "") or scheduled.get("target_region", "")).strip(),
            base_prompt=scheduled.get("skip_reason") if route == Route.SKIP.value else scheduled.get("reason", ""),
        )
        print(
            f"[Agent] quota scheduler: route={route} eligible={eligible} "
            f"scores={{'weather': {weather_score:.3f}, 'occlusion': {occlusion_score:.3f}}} "
            f"counts={self.route_counts}",
            flush=True,
        )
        return scheduled

    def _apply_scene_aware_global_policy(self, decision: dict) -> dict:
        if decision.get("route") != Route.GLOBAL.value:
            return decision

        road_visibility = str(decision.get("road_visibility", "") or "").lower()
        sky_visibility = str(decision.get("sky_visibility", "") or "").lower()
        vegetation_level = str(decision.get("vegetation_level", "") or "").lower()
        facade_density = str(decision.get("facade_density", "") or "").lower()
        close_building = str(decision.get("close_building", "") or "").lower()
        landmarks = str(decision.get("distant_landmarks_readable", "") or "").lower()
        vlm_risk = str(decision.get("global_weather_risk", "") or "").lower()
        safe_weathers = [
            str(item).lower().strip()
            for item in decision.get("safe_global_weathers", [])
            if str(item).lower().strip() in GLOBAL_WEATHER_TARGET_RATIOS
        ]

        text = " ".join(
            str(decision.get(key, "") or "").lower()
            for key in (
                "city",
                "position",
                "prompt",
                "reason",
                "skip_reason",
                "street_scene_quality",
                "weather_feasibility",
            )
        )
        vegetation_terms = (
            "tree", "trees", "vegetation", "foliage", "bush", "bushes", "park",
            "greenery", "roadside scene dominated by vegetation", "dense green",
        )
        billboard_terms = ("billboard", "advertisement", "advertising", "signboard", "roadside sign")
        dense_facade_terms = (
            "dense facade", "repetitive facade", "high-density facade", "storefront row",
            "narrow building facade", "close facade", "building fills", "facade-dominated",
        )
        weak_road_terms = (
            "weak road", "limited road", "peripheral road", "partial road", "no clear road",
            "road barely visible", "road is partly visible",
        )

        def has_scene_term(terms: tuple[str, ...]) -> bool:
            for term in terms:
                if " " in term or "-" in term:
                    if term in text:
                        return True
                elif re.search(rf"\b{re.escape(term)}\b", text):
                    return True
            return False

        vegetation_risk = vegetation_level == "high" or has_scene_term(vegetation_terms + billboard_terms)
        dense_facade_risk = (
            facade_density == "high"
            or close_building in {"yes", "true", "1"}
            or has_scene_term(dense_facade_terms)
        )
        weak_road_risk = road_visibility in {"partial", "none", "low"} or has_scene_term(weak_road_terms)
        weak_landmark_risk = landmarks == "low" and weak_road_risk
        scene_high_risk = (
            vlm_risk == "high"
            or (vegetation_risk and weak_road_risk)
            or (dense_facade_risk and weak_road_risk)
            or weak_landmark_risk
        )
        scene_medium_risk = (
            vlm_risk == "medium"
            or vegetation_risk
            or dense_facade_risk
            or weak_road_risk
            or sky_visibility == "none"
        )
        restriction_reasons = []
        if vlm_risk in {"medium", "high"}:
            restriction_reasons.append(f"global_weather_risk={vlm_risk}")
        if vegetation_risk:
            restriction_reasons.append("vegetation_or_billboard_risk")
        if dense_facade_risk:
            restriction_reasons.append("dense_or_close_facade_risk")
        if weak_road_risk:
            restriction_reasons.append("weak_road_visibility")
        if weak_landmark_risk:
            restriction_reasons.append("weak_landmark_with_weak_road")
        if sky_visibility == "none":
            restriction_reasons.append("no_sky_visibility")

        if scene_high_risk and road_visibility == "none":
            original_route = decision.get("route")
            decision["original_route"] = decision.get("original_route", original_route)
            decision["route"] = Route.SKIP.value
            decision["weather"] = None
            decision["occlusion"] = None
            decision["selected_model"] = "None"
            decision["skip_reason"] = "scene_aware_global_skip: high-risk global scene with no clear road geometry"
            decision["global_scene_policy"] = {
                "action": "skip",
                "risk": "high",
                "reason": "no_clear_road_geometry_for_safe_global_weather",
            }
            decision["prompt"] = build_structured_prompt(
                route=decision["route"],
                weather=None,
                occlusion=None,
                base_prompt=decision["skip_reason"],
            )
            return decision

        scene_restricted = scene_high_risk
        pool = list(GLOBAL_SAFE_WEATHERS if scene_restricted else GLOBAL_WEATHER_ORDER)
        if safe_weathers:
            filtered = [weather for weather in pool if weather in safe_weathers]
            if filtered:
                pool = filtered

        generated_before = {weather: int(self.global_weather_counts.get(weather, 0)) for weather in GLOBAL_WEATHER_ORDER}
        before = {
            weather: int(self.global_weather_pass_counts.get(weather, 0))
            for weather in GLOBAL_WEATHER_ORDER
        }
        planned = sum(before.values()) + 1
        generated_planned = sum(generated_before.values()) + 1
        capped_pool = [
            weather
            for weather in pool
            if generated_planned <= 1
            or self.global_weather_counts.get(weather, 0) / max(generated_planned - 1, 1)
            <= GLOBAL_MAX_SINGLE_WEATHER_RATIO
        ]
        if capped_pool:
            pool = capped_pool

        deficits = {
            weather: GLOBAL_WEATHER_TARGET_RATIOS[weather] * planned - self.global_weather_pass_counts.get(weather, 0)
            for weather in pool
        }
        max_deficit = max(deficits.values())
        candidates = [weather for weather, deficit in deficits.items() if deficit == max_deficit]
        selected_weather = random.choice(candidates)
        after = dict(before)
        after[selected_weather] = after.get(selected_weather, 0) + 1

        original_weather = decision.get("weather")
        decision["weather"] = selected_weather
        decision["occlusion"] = None
        decision["selected_model"] = "IC-Light"
        decision["selected_weather"] = selected_weather
        decision["weather_selection_reason"] = (
            "scene_restricted_safe_pool_quota" if scene_restricted else "target_ratio_quota"
        )
        decision["scene_policy_restricted_weather"] = scene_restricted
        decision["scene_policy_restriction_reasons"] = restriction_reasons
        decision["global_weather_pass_counts_before"] = before
        decision["global_weather_pass_counts_projected_after"] = after
        decision["global_weather_counts_before"] = generated_before
        decision["global_weather_target_ratios"] = GLOBAL_WEATHER_TARGET_RATIOS
        decision["global_weather_candidate_pool"] = pool
        decision["global_weather_deficits"] = {key: round(value, 4) for key, value in deficits.items()}
        decision["global_weather_quota_basis"] = "passed_global_weather_counts"
        decision["global_iclight_highres_denoise"] = (
            GLOBAL_RAIN_ICLIGHT_HIGHRES_DENOISE
            if selected_weather == "rain"
            else GLOBAL_ICLIGHT_HIGHRES_DENOISE
        )
        decision["global_scene_policy"] = {
            "action": "restrict_weather" if scene_restricted else "quota_select_weather",
            "risk": "high" if scene_high_risk else "medium" if scene_medium_risk else "low",
            "original_weather": original_weather,
            "selected_weather": selected_weather,
            "allowed_weathers": pool,
            "road_visibility": road_visibility,
            "sky_visibility": sky_visibility,
            "vegetation_level": vegetation_level,
            "facade_density": facade_density,
            "close_building": close_building,
            "distant_landmarks_readable": landmarks,
        }
        if original_weather != selected_weather:
            decision["scene_weather_original"] = original_weather
            decision["scene_weather_restricted_to"] = selected_weather if scene_restricted else None

        decision["prompt"] = build_structured_prompt(
            route=decision["route"],
            weather=decision["weather"],
            occlusion=None,
            position=decision.get("position", ""),
            base_prompt=decision.get("reason", ""),
        )
        print(
            f"[Agent] global weather quota: {original_weather} -> {selected_weather} "
            f"reason={decision['weather_selection_reason']} pass_counts={before}->{after} "
            f"generated_counts={generated_before} "
            f"file={decision.get('file_name')}",
            flush=True,
        )

        return decision

    def _choose_occlusion(self, decision: dict) -> str | None:
        return choose_occlusion_from_context(
            decision.get("position"),
            decision.get("prompt"),
            decision.get("reason"),
            decision.get("skip_reason"),
            decision.get("street_scene_quality"),
            decision.get("occlusion_feasibility"),
            requested=decision.get("occlusion"),
        )

    def _choose_weather(self, requested: str | None) -> str:
        return normalize_weather(requested) or "rain"

    def _skip_reason_from_risk(self, flags: list[str]) -> str:
        if "occlusion_implausible" in flags:
            return "no_valid_street_surface_for_occlusion"
        if "foreground_foliage_may_confuse_occluder" in flags:
            return "occlusion_would_be_unreliable_or_edge_artifact"
        if "night_vehicle_high_hallucination_risk" in flags:
            return "vehicle_occlusion_high_hallucination_risk_and_weather_not_feasible"
        return "local_or_dual_risk_too_high"

    def _eval_to_dict(self, eval_result) -> dict:
        return {
            "passed": eval_result.passed,
            "s_geo": eval_result.s_geo,
            "s_div": eval_result.s_div,
            "geo_ok": eval_result.geo_ok,
            "div_ok": eval_result.div_ok,
            "feedback": eval_result.feedback,
        }

    def _output_path(self, output_root: Path, image_path: Path, decision: dict, round_num: int) -> Path:
        return output_root / "rounds" / self._sample_stem(image_path, decision) / f"r{round_num}.jpg"

    def _sample_stem(self, image_path: Path, decision: dict) -> str:
        raw = str(decision.get("sample_id") or image_path.stem)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
        return safe or image_path.stem

    def _collect_bad_case(self, output_root: Path, image_path: Path, final_path: Path, record: dict) -> None:
        bad_dir = output_root / "bad_cases"
        bad_img_dir = bad_dir / "images"
        bad_img_dir.mkdir(parents=True, exist_ok=True)
        bad_path = bad_img_dir / f"{self._sample_stem(image_path, record)}__{final_path.name}"
        shutil.copy2(final_path, bad_path)
        bad_record = dict(record)
        bad_record["bad_case_image"] = str(bad_path)
        with (bad_dir / "bad_cases.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(bad_record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    agent = SceneAugmentAgent(mock=True)
    test_image = Image.new("RGB", (512, 384), color=(100, 110, 90))
    result = agent.run(test_image, save_dir="/tmp/agent_test")
    result.final_image.save("/tmp/agent_final.jpg")
    print("\n[Main] Final image saved: /tmp/agent_final.jpg")
    print(f"[Main] Route: {result.route}")
    print(f"[Main] Rounds used: {result.rounds_used}")
    print(f"[Main] Final prompt: {result.final_prompt}")
