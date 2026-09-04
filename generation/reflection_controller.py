"""Source-anchored, visibly distinct reflection for Local and Dual routes."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
from io import BytesIO
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image


def _image_b64(image: Image.Image, max_side: int = 1024) -> str:
    preview = image.convert("RGB").copy()
    preview.thumbnail((max_side, max_side))
    buffer = BytesIO()
    preview.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _extract_json(text: str) -> dict:
    value = (text or "").strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _image_hash(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _interround_l1(left: Image.Image, right: Image.Image) -> float:
    size = (384, 384)
    left_array = np.asarray(left.convert("RGB").resize(size), dtype=np.float32)
    right_array = np.asarray(right.convert("RGB").resize(size), dtype=np.float32)
    return float(np.mean(np.abs(left_array - right_array)))


class ReflectionController:
    """Run at most three source-anchored reflection rounds.

    The failed image is supplied to the VLM for diagnosis, but every replacement
    is generated from the untouched source.  This prevents permanent-structure
    errors from accumulating across rounds.
    """

    def __init__(
        self,
        *,
        llm_client,
        llm_model: str,
        evaluator,
        generate: Callable[[Image.Image, str, str, dict, int], Image.Image],
    ) -> None:
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.evaluator = evaluator
        self.generate = generate

    @staticmethod
    def _seed(base_seed: int, round_index: int) -> int:
        # Initial generation remains reproducible; every reflection samples an
        # independent seed and persists it in the round record.
        del base_seed, round_index
        return secrets.randbelow(2147483646) + 1

    @staticmethod
    def _local_allowed_occluders(decision: dict, previous_prompt: str) -> list[str]:
        """Preserve the Local prompt's actual transient-occluder vocabulary."""
        text = f"{decision.get('occlusion', '')} {previous_prompt}".lower()
        allowed = []
        vocabulary = (
            ("vehicle", ("vehicle", "car", "van", "taxi", "bus", "truck")),
            ("pedestrian", ("person", "people", "pedestrian")),
            ("cyclist", ("cyclist", "bicycle", "bike")),
            ("scooter rider", ("scooter", "motorbike", "motorcycle")),
            ("roadside vegetation", ("tree", "shrub", "bush", "vegetation")),
        )
        for label, keywords in vocabulary:
            if any(keyword in text for keyword in keywords):
                allowed.append(label)
        if not allowed:
            allowed.append("pedestrian" if decision.get("occlusion") == "person" else "vehicle")
        return allowed

    @staticmethod
    def _local_contract(decision: dict, previous_prompt: str, round_index: int) -> str:
        position = str(decision.get("position") or "a legal interior road region")
        allowed = ", ".join(ReflectionController._local_allowed_occluders(decision, previous_prompt))
        common = (
            "Generate from the untouched source image. Preserve the exact camera, "
            "vanishing point, road topology, curbs, lane markings, every facade, "
            "window, door, sign, roof, tree, pole, sidewalk, sky, lighting, and "
            "background. Keep each new transient occluder fully inside the frame with at least "
            "12% image-width clearance from both side borders and 9% image-height "
            "clearance above the bottom border. Align wheels or feet to the legal ground plane "
            "with correct perspective and contact shadows. Add no signs, weather effects, "
            "permanent structures, or unrelated objects. "
            f"Allowed occluder family inferred from the original prompt: {allowed}."
        )
        if round_index == 1:
            action = (
                f"ROUND 1: repair the failed Local edit at {position}. Select one or two fully visible "
                "members of the allowed family, and change their scale or placement in direct response "
                "to the verifier feedback while preserving the original occlusion intent."
            )
        elif round_index == 2:
            action = (
                f"ROUND 2: use a visibly different subtype or composition from the allowed family at "
                f"legal regions compatible with {position}. Change count, scale, and placement from "
                "round 1; use one to three separated occluders only when the scene supports them."
            )
        else:
            action = (
                f"ROUND 3: apply the strongest still-photorealistic alternative from the allowed family "
                f"at {position}. Use a different subtype, count, or legal interior region than round 2, "
                "and fix every remaining verifier issue without changing permanent scene structure."
            )
        return f"{common}\n{action}\nThe result must be categorically different from the previous candidate at 384-pixel preview width."

    @staticmethod
    def _dual_contract(decision: dict, round_index: int) -> str:
        position = str(decision.get("position") or "a legal interior road region")
        weather = str(decision.get("weather") or "rain")
        occlusion = str(decision.get("occlusion") or "vehicle")
        if occlusion == "person":
            common = (
                "Generate from the untouched source image. Preserve the exact camera, "
                "vanishing point, road topology, curbs, lane boundaries, every facade, "
                "window, door, sign, roof, tree, pole, and sidewalk. Keep every new "
                "pedestrian fully inside the frame with clear margins from every border. "
                "Every pedestrian must be a natural full-body person with visible feet "
                "on legal sidewalk, crosswalk, curb, or roadside pavement, with correct "
                "scale, perspective, lighting, and contact shadow. Do not add vehicles. "
                "Never rebuild, shift, add, or remove permanent scene structure."
            )
            if round_index == 1:
                action = (
                    f"ROUND 1: at {position}, add exactly one clearly visible full-body pedestrian "
                    "at natural street scale, away from image borders and not a dark silhouette. "
                    f"Make {weather} unmistakable at a moderate-heavy level. Rain requires visible "
                    "streaks, continuous wet pavement, and several coherent reflections."
                )
            elif round_index == 2:
                action = (
                    f"ROUND 2: replace the prior person layout with exactly two separated full-body "
                    f"pedestrians at legal pavement regions compatible with {position}. Give them "
                    "different clothing colors, poses, and positions with no overlap. Increase "
                    f"{weather} by one obvious physical step; for rain add denser streaks, multiple "
                    "shallow puddles, and a broad reflection band."
                )
            else:
                action = (
                    f"ROUND 3: replace the two-person layout with exactly one prominent full-body "
                    f"pedestrian in a different legal interior pavement region compatible with {position}. "
                    "Use a clearly different pose, clothing color, scale, and position while remaining "
                    f"photorealistic. Apply the strongest still-photorealistic {weather} cue bundle; "
                    "rain requires dense visible rainfall, continuous glossy pavement, reflections, and puddles."
                )
            return (
                f"{common}\n{action}\nThe result must be categorically different from the previous "
                "candidate within one second of viewing."
            )

        common = (
            "Generate from the untouched source image. Preserve the exact camera, "
            "vanishing point, road topology, curbs, lane boundaries, every facade, "
            "window, door, sign, roof, tree, pole, and sidewalk. Keep every changed "
            "vehicle fully inside the frame with at least 12% image-width clearance "
            "from both side borders and at least 9% image-height clearance above the "
            "bottom border. All wheels must align to the road plane with correct "
            "perspective and contact shadows. Never rebuild, shift, add, or remove "
            "permanent scene structure."
        )
        if round_index == 1:
            action = (
                f"ROUND 1: at {position}, add exactly one fully visible white delivery van or "
                "medium box truck occupying 24-28% of image width and 20-25% of image height. "
                f"Make {weather} unmistakable at a moderate-heavy level. Rain requires visible "
                "streaks, continuous wet pavement, and several coherent reflections."
            )
        elif round_index == 2:
            action = (
                f"ROUND 2: replace the prior layout with exactly two separated fully visible vehicles "
                f"compatible with {position}: one compact delivery van occupying 20-24% of image "
                "width and one taxi or passenger car occupying 13-17%. Separate their centers by at "
                f"least one full vehicle width. Increase {weather} by one obvious physical step; for "
                "rain add denser streaks, multiple shallow puddles, a broad reflection band, and tire spray."
            )
        else:
            action = (
                f"ROUND 3: replace the two-vehicle layout with exactly one dominant long box truck "
                f"or shuttle bus compatible with {position}, occupying 30-35% of image width and "
                f"24-29% of image height in a different interior road region. Apply the strongest "
                f"still-photorealistic {weather} cue bundle; rain requires dense visible rainfall, "
                "continuous glossy pavement, elongated reflections, puddles, and tire spray."
            )
        return f"{common}\n{action}\nThe result must be categorically different from the previous candidate within one second of viewing."

    def _dual_rewrite(
        self,
        source: Image.Image,
        failed: Image.Image,
        previous_prompt: str,
        feedback: dict,
        decision: dict,
        round_index: int,
    ) -> tuple[str, dict, str]:
        contract = self._dual_contract(decision, round_index)
        occluder = "pedestrian" if decision.get("occlusion") == "person" else "vehicle"
        request = f"""Image 1 is the untouched source. Image 2 is the previous failed Dual candidate.
Reflection round: {round_index}
Previous prompt:
{previous_prompt}

Machine and gate feedback:
{json.dumps(feedback, ensure_ascii=False, indent=2)}

Analyze concrete visible failures in Image 2 and write new directives that fix them. Do not repeat
the previous strategy. The {occluder} count, appearance, scale, position, and weather magnitude
must visibly change without switching to a different occluder family.
The following round contract is authoritative and overrides conflicts:
{contract}

Return strict JSON:
{{
  "concrete_visible_failures": ["failure"],
  "structure_repairs": ["repair"],
  "occluder_change": "specific type/count/scale/placement change",
  "weather_change": "specific change",
  "rewritten_reflection_prompt": "complete new directives"
}}"""
        raw = ""
        analysis: dict
        try:
            raw = self.llm_client.chat_with_images(
                system="Analyze the failed Dual augmentation and rewrite its prompt. Return JSON only.",
                user=request,
                images=[_image_b64(source), _image_b64(failed)],
                json_mode=True,
                temperature=0.2,
                model=self.llm_model,
            )
            analysis = _extract_json(raw)
        except Exception as exc:
            analysis = {
                "parse_error": f"{type(exc).__name__}: {exc}",
                "concrete_visible_failures": ["VLM response failed; use the authoritative round contract."],
                "rewritten_reflection_prompt": "",
            }
        rewritten = str(analysis.get("rewritten_reflection_prompt") or "").strip()
        failures = analysis.get("concrete_visible_failures") or []
        failure_text = "\n".join(f"- {item}" for item in failures)
        prompt = (
            f"VLM FEEDBACK-DRIVEN DUAL REFLECTION ROUND {round_index}.\n"
            f"Concrete failed-candidate feedback:\n{failure_text or '- use machine feedback'}\n"
            f"VLM rewritten directives:\n{rewritten or 'Correct every machine-gate failure.'}\n"
            f"Authoritative non-negotiable contract:\n{contract}"
        )
        return prompt, analysis, raw

    def _local_rewrite(
        self,
        source: Image.Image,
        failed: Image.Image,
        previous_prompt: str,
        feedback: dict,
        decision: dict,
        round_index: int,
    ) -> tuple[str, dict, str]:
        """Rewrite a Local prompt from verifier and VLM diagnostic feedback.

        The VLM diagnoses the previous failed candidate, while the deterministic
        round contract remains authoritative so a malformed response cannot
        weaken the route, geometry, border, or visible-change constraints.
        """

        contract = self._local_contract(decision, previous_prompt, round_index)
        request = f"""Image 1 is the untouched source. Image 2 is the previous failed Local candidate.
Reflection round: {round_index}
Previous prompt:
{previous_prompt}

Verifier and VLM diagnostic feedback:
{json.dumps(feedback, ensure_ascii=False, indent=2)}

Analyze the concrete visible failures in Image 2 and rewrite the Local prompt to
fix the reported geometry, diversity, occluder-legality, border, structure, and
visible-change problems. Do not add weather or alter permanent scene structure.
The following round contract is authoritative and overrides conflicts:
{contract}

Return strict JSON:
{{
  "concrete_visible_failures": ["failure"],
  "feedback_repairs": ["repair tied to the supplied feedback"],
  "occluder_change": "specific type/count/scale/placement change",
  "rewritten_reflection_prompt": "complete new directives"
}}"""
        raw = ""
        analysis: dict
        try:
            raw = self.llm_client.chat_with_images(
                system="Analyze the failed Local augmentation and rewrite its prompt. Return JSON only.",
                user=request,
                images=[_image_b64(source), _image_b64(failed)],
                json_mode=True,
                temperature=0.2,
                model=self.llm_model,
            )
            analysis = _extract_json(raw)
        except Exception as exc:
            analysis = {
                "parse_error": f"{type(exc).__name__}: {exc}",
                "concrete_visible_failures": [
                    "VLM response failed; use the authoritative round contract."
                ],
                "rewritten_reflection_prompt": "",
            }
        rewritten = str(analysis.get("rewritten_reflection_prompt") or "").strip()
        failures = analysis.get("concrete_visible_failures") or []
        repairs = analysis.get("feedback_repairs") or []
        failure_text = "\n".join(f"- {item}" for item in failures)
        repair_text = "\n".join(f"- {item}" for item in repairs)
        prompt = (
            f"VLM FEEDBACK-DRIVEN LOCAL REFLECTION ROUND {round_index}.\n"
            f"Concrete failed-candidate feedback:\n{failure_text or '- use verifier feedback'}\n"
            f"Feedback-specific repairs:\n{repair_text or '- correct every reported failure'}\n"
            f"VLM rewritten directives:\n{rewritten or 'Correct every verifier and diagnostic failure.'}\n"
            f"Authoritative non-negotiable contract:\n{contract}"
        )
        return prompt, analysis, raw

    def _vision_json_with_raw(
        self, system: str, user: str, images: list[Image.Image]
    ) -> tuple[dict, str]:
        raw = self.llm_client.chat_with_images(
            system=system,
            user=user,
            images=[_image_b64(image) for image in images],
            json_mode=True,
            temperature=0.0,
            model=self.llm_model,
        )
        return _extract_json(raw), raw

    def _vision_json(self, system: str, user: str, images: list[Image.Image]) -> dict:
        result, _ = self._vision_json_with_raw(system, user, images)
        return result

    def _route_legality_diagnostic(
        self, source: Image.Image, candidate: Image.Image, route: str,
        round_index: int, decision: dict, prompt: str,
    ) -> dict:
        if route == "local":
            allowed = ", ".join(self._local_allowed_occluders(decision, prompt))
            try:
                result = self._vision_json(
                    "Audit only authorized transient Local edits. Return JSON only.",
                    f"Compare source and candidate for Local reflection round={round_index}. Allowed "
                    f"occluders are: {allowed}. New permanent structures, unrelated objects, cropped "
                    "occluders, floating objects, and objects on walls or sky are forbidden. Return "
                    '{"accepted":true|false,"new_occluders":["type"],"violations":["..."]}.',
                    [source, candidate],
                )
                return {**result, "accepted": bool(result.get("accepted")), "allowed_occluders": allowed}
            except Exception as exc:
                return {"accepted": False, "allowed_occluders": allowed, "error": f"{type(exc).__name__}: {exc}"}
        if route == "dual" and decision.get("occlusion") == "person":
            expected = 2 if round_index == 2 else 1
            try:
                result = self._vision_json(
                    "Audit only authorized Dual weather and pedestrian edits. Return JSON only.",
                    f"Compare source and candidate for Dual reflection round={round_index}. Exactly "
                    f"{expected} new full-body pedestrian(s) are authorized together with the requested "
                    "weather edit. New vehicles, cropped people, silhouettes, floating people, people "
                    "on walls or sky, and permanent structure changes are forbidden. Return "
                    '{"accepted":true|false,"new_person_count":0,"violations":["..."]}.',
                    [source, candidate],
                )
                accepted = (
                    bool(result.get("accepted"))
                    and int(result.get("new_person_count", -1)) == expected
                )
                return {**result, "accepted": accepted, "expected_person_count": expected}
            except Exception as exc:
                return {
                    "accepted": False,
                    "expected_person_count": expected,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        expected = 2 if round_index == 2 else 1
        try:
            result = self._vision_json(
                "Audit only authorized transient edits. Return JSON only.",
                f"Compare source and candidate for route={route}, round={round_index}. Exactly {expected} new "
                "fully visible vehicle(s) are authorized. Permanent structure changes, extra vehicles, cropped "
                "vehicles, floating vehicles, and vehicles intersecting walls or sidewalks are forbidden. Return "
                '{"accepted":true|false,"new_vehicle_count":0,"violations":["..."]}.',
                [source, candidate],
            )
            accepted = bool(result.get("accepted")) and int(result.get("new_vehicle_count", -1)) == expected
            return {**result, "accepted": accepted, "expected_vehicle_count": expected}
        except Exception as exc:
            return {"accepted": False, "expected_vehicle_count": expected, "error": f"{type(exc).__name__}: {exc}"}

    def _border_diagnostic(self, candidate: Image.Image, route: str) -> dict:
        try:
            result = self._vision_json(
                "Audit transient-occluder border integrity. Return JSON only.",
                f"For route={route}, every newly prominent transient occluder must be fully visible, not cropped, "
                "not attached to an image edge, "
                "with clear margins at both sides and above the bottom. Return "
                '{"accepted":true|false,"cropped":true|false,"edge_attached":true|false,"violations":["..."]}.',
                [candidate],
            )
            accepted = bool(result.get("accepted")) and not result.get("cropped") and not result.get("edge_attached")
            return {**result, "accepted": accepted}
        except Exception as exc:
            return {"accepted": False, "error": f"{type(exc).__name__}: {exc}"}

    def _structure_diagnostic(self, source: Image.Image, candidate: Image.Image) -> dict:
        request = (
            "Image 1 is the untouched source and Image 2 is an augmented candidate. Audit permanent scene "
            "structure only. Ignore intended vehicles, pedestrians, precipitation, snow, wet pavement, "
            "reflections, illumination, and minor texture variation. A permanent element merely hidden behind "
            "a new vehicle is occluded, not changed. Detect only material changes to camera/viewpoint, road "
            "topology or width, curb/lane boundaries, buildings/facades, windows/doors/signs/rooflines, trees, "
            "poles, or sidewalks. Return one JSON object with exactly four keys named "
            "permanent_change_detected, permanent_changes, confidence, and reason. Decide every value from the "
            "two images; do not copy placeholder or default values from the request. permanent_change_detected "
            "must be a JSON boolean. permanent_changes must be a list of localized visible changes. confidence "
            "must be a calibrated number from 0 to 1, and reason must briefly cite the comparison. The boolean "
            "and evidence list must agree: a detected change requires at least one concrete item, while no "
            "detected change requires an empty list."
        )

        def normalize(result: dict, raw: str, stage: str) -> dict:
            detected = result.get("permanent_change_detected")
            changes_value = result.get("permanent_changes")
            errors = []
            if type(detected) is not bool:
                errors.append("permanent_change_detected_must_be_json_boolean")
            if not isinstance(changes_value, list):
                errors.append("permanent_changes_must_be_list")
                changes = []
            else:
                changes = [str(item).strip() for item in changes_value if str(item).strip()]
            try:
                confidence = float(result.get("confidence"))
            except (TypeError, ValueError):
                confidence = -1.0
                errors.append("confidence_must_be_number")
            if not 0.0 <= confidence <= 1.0:
                errors.append("confidence_out_of_range")
            consistent = type(detected) is bool and detected == bool(changes)
            if not consistent:
                errors.append("boolean_evidence_contradiction")
            return {
                "stage": stage,
                "permanent_change_detected": detected,
                "permanent_changes": changes,
                "confidence": confidence,
                "reason": result.get("reason"),
                "consistent": consistent,
                "conclusive": consistent and confidence >= 0.80 and not errors,
                "validation_errors": errors,
                "raw": raw,
            }

        audits = []
        call_errors = []
        try:
            result, raw = self._vision_json_with_raw(
                "Audit permanent scene structure using concrete visual evidence. Return JSON only.",
                request,
                [source, candidate],
            )
            audits.append(normalize(result, raw, "primary"))
        except Exception as exc:
            call_errors.append(f"primary:{type(exc).__name__}: {exc}")

        primary = audits[0] if audits else None
        if primary and primary["conclusive"]:
            return {
                "accepted": not primary["permanent_change_detected"],
                "resolution": "primary_conclusive",
                "requires_geometric_review": False,
                "audits": audits,
            }

        try:
            result, raw = self._vision_json_with_raw(
                "Independently adjudicate an uncertain permanent-structure audit. Return JSON only.",
                f"{request} A previous audit was inconclusive because its confidence or schema consistency was "
                "insufficient. Re-check the images independently without assuming its conclusion. Cite a concrete visible "
                "permanent change only if it can be localized in both images.",
                [source, candidate],
            )
            audits.append(normalize(result, raw, "adjudication"))
        except Exception as exc:
            call_errors.append(f"adjudication:{type(exc).__name__}: {exc}")

        review = audits[-1] if len(audits) >= 2 else None
        if review and review["conclusive"]:
            return {
                "accepted": not review["permanent_change_detected"],
                "resolution": "adjudication_conclusive",
                "requires_geometric_review": False,
                "audits": audits,
                "call_errors": call_errors,
            }
        if audits and not call_errors:
            return {
                "accepted": True,
                "resolution": "deferred_to_geometry",
                "requires_geometric_review": True,
                "audits": audits,
            }
        return {
            "accepted": False,
            "resolution": "structure_audit_unavailable",
            "requires_geometric_review": False,
            "audits": audits,
            "call_errors": call_errors,
        }

    def _visible_change_diagnostic(
        self, previous: Image.Image, candidate: Image.Image, round_index: int,
        route: str, decision: dict,
    ) -> dict:
        votes = []
        if route == "local":
            criteria = (
                "occluder type and count",
                "occluder scale and position",
                "overall one-second visible change",
            )
        elif route == "dual" and decision.get("occlusion") == "person":
            criteria = (
                "pedestrian count and appearance",
                "pedestrian scale and position",
                "overall one-second visible change",
            )
        else:
            criteria = (
                "vehicle count and type",
                "vehicle scale and position",
                "overall one-second visible change",
            )
        for criterion in criteria:
            try:
                result = self._vision_json(
                    "Judge whether two augmentation candidates are obviously different. Return JSON only.",
                    f"Focus on {criterion}. Round {round_index} must be immediately and categorically different "
                    'from the previous candidate. Return {"different":true|false,"reason":"..."}.',
                    [previous, candidate],
                )
                votes.append({**result, "different": bool(result.get("different")), "criterion": criterion})
            except Exception as exc:
                votes.append({"different": False, "criterion": criterion, "error": f"{type(exc).__name__}: {exc}"})
        positives = sum(bool(vote["different"]) for vote in votes)
        return {"accepted": positives == 3, "positive_votes": positives, "required_votes": 3, "votes": votes}

    @staticmethod
    def _acceptance_decision(initial_eval: dict, evaluation: dict, interround: float, diagnostics: list[dict]) -> dict:
        """Apply the published two-score acceptance rule.

        The VLM/visual checks are diagnostics for prompt reflection.  They are
        deliberately non-binding: accepting a generated pair depends only on
        the dual-trait verifier recorded in ``evaluation.passed``.
        """
        reasons = []
        if not evaluation.get("passed"):
            reasons.append("dual_trait_verifier_failed")
        diagnostic_warnings = []
        if interround < 10.0:
            diagnostic_warnings.append("human_visible_pixel_change_below_10")
        if not all(item.get("accepted") for item in diagnostics):
            diagnostic_warnings.append("one_or_more_vlm_diagnostics_failed")
        if float(evaluation.get("s_geo", 0.0)) + 0.02 < float(initial_eval.get("s_geo", 0.0)):
            diagnostic_warnings.append("geometry_regressed_from_initial")
        return {
            "accepted": not reasons,
            "reasons": reasons,
            "acceptance_rule": "dual_trait_scores_only",
            "diagnostic_warnings": diagnostic_warnings,
            "deltas": {
                "s_geo": round(float(evaluation.get("s_geo", 0.0)) - float(initial_eval.get("s_geo", 0.0)), 6),
                "s_div": round(float(evaluation.get("s_div", 0.0)) - float(initial_eval.get("s_div", 0.0)), 6),
                "interround_l1": round(interround, 6),
            },
        }

    def run(
        self,
        *,
        source: Image.Image,
        initial_candidate: Image.Image,
        initial_prompt: str,
        initial_eval: dict,
        decision: dict,
        output_dir: Path | None,
        base_seed: int,
        max_reflections: int = 3,
    ) -> dict:
        route = str(decision.get("route"))
        if route not in {"local", "dual"}:
            raise ValueError(f"Reflection is only supported for Local/Dual, got {route!r}")

        used_hashes = {_prompt_hash(initial_prompt)}
        previous_candidate = initial_candidate
        previous_prompt = initial_prompt
        feedback: dict = {"initial_eval": initial_eval}
        attempts = []
        accepted_image = None
        accepted_eval = None
        last_candidate = initial_candidate
        last_eval = initial_eval

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        for round_index in range(1, min(3, max_reflections) + 1):
            analysis = None
            raw_analysis = None
            if route == "dual":
                prompt, analysis, raw_analysis = self._dual_rewrite(
                    source, previous_candidate, previous_prompt, feedback, decision, round_index
                )
            else:
                prompt, analysis, raw_analysis = self._local_rewrite(
                    source, previous_candidate, previous_prompt, feedback, decision, round_index
                )

            prompt_sha256 = _prompt_hash(prompt)
            if prompt_sha256 in used_hashes:
                raise RuntimeError(
                    f"Repeated reflection prompt before generation: round={round_index} sha256={prompt_sha256}"
                )
            used_hashes.add(prompt_sha256)

            seed = self._seed(base_seed, round_index)
            candidate = self.generate(source, prompt, route, decision, seed)
            if candidate.size != source.size:
                candidate = candidate.resize(source.size, Image.Resampling.LANCZOS)
            candidate_path = None
            if output_dir is not None:
                candidate_path = output_dir / f"reflection_{round_index}.jpg"
                candidate.save(candidate_path, quality=95)

            eval_result = self.evaluator.evaluate(source, candidate, entry=decision)
            evaluation = {
                "passed": bool(eval_result.passed),
                "s_geo": float(eval_result.s_geo),
                "s_div": float(eval_result.s_div),
                "geo_ok": bool(eval_result.geo_ok),
                "div_ok": bool(eval_result.div_ok),
                "feedback": eval_result.feedback,
            }
            interround = _interround_l1(previous_candidate, candidate)
            legality = self._route_legality_diagnostic(
                source, candidate, route, round_index, decision, prompt
            )
            border = self._border_diagnostic(candidate, route)
            structure = self._structure_diagnostic(source, candidate)
            visible = self._visible_change_diagnostic(
                previous_candidate, candidate, round_index, route, decision
            )
            acceptance = self._acceptance_decision(
                initial_eval, evaluation, interround, [legality, border, structure, visible]
            )

            attempt = {
                "round": round_index + 1,
                "reflection_index": round_index,
                "seed": seed,
                "generation_mode": "source_anchored_categorical_reflection",
                "prompt": prompt,
                "effective_prompt": prompt,
                "prompt_sha256": prompt_sha256,
                "image_path": str(candidate_path) if candidate_path else None,
                "candidate_sha256": _image_hash(candidate),
                "eval": evaluation,
                "feedback_for_vlm_prompt_rewrite": feedback,
                "feedback_used_for_current_prompt_rewrite": feedback,
                "vlm_prompt_analysis": analysis,
                "vlm_prompt_raw": raw_analysis,
                "route_legality_diagnostic": legality,
                "local_border_integrity_diagnostic": border,
                "source_structure_lock_diagnostic": structure,
                "human_visible_change_diagnostic": visible,
                "acceptance_decision": acceptance,
            }
            attempts.append(attempt)
            last_candidate = candidate
            last_eval = evaluation
            if acceptance["accepted"]:
                accepted_image = candidate
                accepted_eval = evaluation
                break

            feedback = {
                "evaluation": evaluation,
                "acceptance_decision": acceptance,
                "route_legality_diagnostic": legality,
                "local_border_integrity_diagnostic": border,
                "source_structure_lock_diagnostic": structure,
                "human_visible_change_diagnostic": visible,
            }
            attempt["feedback_for_next_prompt_rewrite"] = feedback
            previous_candidate = candidate
            previous_prompt = prompt

        return {
            "passed": accepted_image is not None,
            "final_image": accepted_image,
            "final_eval": accepted_eval,
            "last_candidate": last_candidate,
            "last_eval": last_eval,
            "attempts": attempts,
            "stop_reason": (
                "passed_dual_trait_verifier"
                if accepted_image is not None
                else "max_reflections_without_dual_trait_pass"
            ),
        }
