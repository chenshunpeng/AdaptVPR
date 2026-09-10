import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

dotenv = types.ModuleType("dotenv")
dotenv.load_dotenv = lambda *_args, **_kwargs: None
sys.modules.setdefault("dotenv", dotenv)
sys.modules.setdefault("numpy", types.ModuleType("numpy"))

from generation.agent import SceneAugmentAgent, _ratios_from_env, scheduler_manifest
from generation.inputs import parse_condition
from prompts.rules import build_structured_prompt
from verification.evaluator import DualTraitEvaluator


class CapabilityLLM:
    def __init__(self):
        self.calls = []

    def chat_with_images(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps(
            {
                "file_name": "input.jpg",
                "city": "unknown",
                "weather_score": 0.83,
                "occlusion_score": 0.22,
                "bad_image": False,
                "weather": "overcast",
                "occlusion": None,
                "position": "clear street scene",
                "prompt": "Apply realistic overcast illumination.",
                "reason": "weather editing is safe",
                "skip_reason": "",
                "street_scene_quality": "good",
                "occlusion_feasibility": "low",
                "weather_feasibility": "high",
                "road_visibility": "clear",
                "sky_visibility": "clear",
                "vegetation_level": "low",
                "facade_density": "low",
                "close_building": "no",
                "distant_landmarks_readable": "high",
                "global_weather_risk": "low",
                "safe_global_weathers": ["overcast"],
            }
        )


class AgentPlanningTest(unittest.TestCase):
    def test_agent_uses_single_authoritative_planner_and_reflection_paths(self):
        agent = SceneAugmentAgent(
            llm_client=CapabilityLLM(),
            mock=True,
            planning_only=True,
        )
        self.assertFalse(hasattr(agent, "planner"))
        self.assertFalse(hasattr(agent, "refiner"))
        self.assertIsNone(agent.reflection_controller)

    def test_dual_scheduler_weather_and_prompt_are_not_overwritten(self):
        agent = SceneAugmentAgent.__new__(SceneAugmentAgent)
        scheduled = {
            "file_name": "dual.jpg",
            "city": "Bangkok",
            "route": "dual",
            "weather": "snow",
            "occlusion": "vehicle",
            "weather_score": 0.9,
            "occlusion_score": 0.9,
            "bad_image": False,
            "position": "right traffic lane near the curb",
            "reason": "both edits are feasible",
            "skip_reason": "",
            "street_scene_quality": "good",
            "occlusion_feasibility": "high",
            "weather_feasibility": "high",
        }
        expected_prompt = build_structured_prompt(
            route="dual",
            weather="snow",
            occlusion="vehicle",
            position=scheduled["position"],
            base_prompt=scheduled["reason"],
        )
        with patch.object(
            agent, "_schedule_route_from_capabilities", return_value=scheduled
        ) as scheduler, patch.object(
            agent, "_record_decision_counts"
        ) as recorder, patch(
            "generation.agent.random.Random",
            side_effect=AssertionError("Dual weather must not be reassigned after scheduling"),
        ):
            decision = agent._finalize_planned_decision(
                dict(scheduled),
                image_path=Path("Bangkok/dual.jpg"),
                city="Bangkok",
                source_path="/different/machine/Bangkok/dual.jpg",
                schedule_and_count=True,
            )
        scheduler.assert_called_once()
        recorder.assert_called_once()
        self.assertEqual(decision["weather"], "snow")
        self.assertEqual(decision["prompt"], expected_prompt)

    def test_skip_semantics_are_consistent_across_public_apis_and_evaluator(self):
        decision = {
            "route": "skip",
            "prompt": "No generation required.",
            "bad_image": False,
        }
        source = Image.new("RGB", (32, 32), "white")

        run_agent = SceneAugmentAgent.__new__(SceneAugmentAgent)
        run_agent._plan_image_object = lambda *_args, **_kwargs: dict(decision)
        run_result = run_agent.run(source)
        self.assertFalse(run_result.passed)
        self.assertTrue(run_result.skipped)

        path_agent = SceneAugmentAgent.__new__(SceneAugmentAgent)
        path_agent.plan_image = lambda *_args, **_kwargs: dict(decision)
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "input.jpg"
            source.save(image_path)
            path_result = path_agent.run_path(image_path, Path(tmp_dir) / "output")
        self.assertFalse(path_result["passed"])
        self.assertTrue(path_result["skipped"])

        eval_result = DualTraitEvaluator(mock=True).evaluate(source, source, route="skip")
        self.assertFalse(eval_result.passed)
        self.assertTrue(eval_result.skipped)

    def test_paper_local_taxonomy_conditions_remain_frozen_prompt_compatible(self):
        self.assertEqual(
            parse_condition("Curbside / parked-vehicle occlusion"),
            (None, "vehicle"),
        )
        self.assertEqual(
            parse_condition("Road-traffic occlusion"),
            (None, "vehicle"),
        )
        self.assertEqual(
            parse_condition("Other local occlusions"),
            (None, "person"),
        )

    def test_public_run_uses_capability_scheduler_without_fixed_seven_scores(self):
        llm = CapabilityLLM()
        agent = SceneAugmentAgent.__new__(SceneAugmentAgent)
        agent.llm_client = llm
        agent.planning_enabled = True
        agent.max_generations = 4
        agent.route_counts = {"skip": 0, "global": 0, "local": 0, "dual": 0}
        agent.weather_counts = {
            "fog": 0,
            "night": 0,
            "overcast": 0,
            "rain": 0,
            "rainy_night": 0,
            "snow": 0,
        }
        agent.global_weather_counts = {
            "overcast": 0,
            "fog": 0,
            "rain": 0,
            "snow": 0,
            "night": 0,
        }
        agent.global_weather_pass_counts = dict(agent.global_weather_counts)
        agent.occlusion_counts = {"vehicle": 0, "person": 0}
        agent.evaluator = types.SimpleNamespace(
            evaluate=lambda *_args, **_kwargs: types.SimpleNamespace(
                passed=True,
                s_geo=0.91,
                s_div=0.18,
                geo_ok=True,
                div_ok=True,
                feedback={},
            )
        )

        source = Image.new("RGB", (32, 32), "white")
        with patch.object(
            agent,
            "_schedule_route_from_capabilities",
            wraps=agent._schedule_route_from_capabilities,
        ) as scheduler, patch.object(agent, "_generate", return_value=source.copy()):
            result = agent.run(source)

        scheduler.assert_called_once()
        scheduled_input = scheduler.call_args.args[0]
        self.assertEqual(scheduled_input["weather_score"], 0.83)
        self.assertEqual(scheduled_input["occlusion_score"], 0.22)
        self.assertEqual(result.route, "global")
        self.assertEqual(agent.route_counts["global"], 1)
        self.assertEqual(len(llm.calls), 1)
        self.assertIn("weather_score in [0,1]", llm.calls[0]["system"])

    def test_scheduler_golden_sequence_preserves_balanced_generation_routes(self):
        agent = SceneAugmentAgent.__new__(SceneAugmentAgent)
        agent.route_counts = {"skip": 0, "global": 0, "local": 0, "dual": 0}
        agent.weather_counts = {
            "fog": 0,
            "night": 0,
            "overcast": 0,
            "rain": 0,
            "snow": 0,
        }
        agent.global_weather_counts = dict(agent.weather_counts)
        agent.global_weather_pass_counts = dict(agent.weather_counts)
        agent.occlusion_counts = {"vehicle": 0, "person": 0}
        raw = {
            "weather_score": 0.90,
            "occlusion_score": 0.90,
            "bad_image": False,
            "weather": "overcast",
            "occlusion": "vehicle",
            "position": "right traffic lane",
            "reason": "both edits are feasible",
        }

        routes = []
        for _ in range(12):
            decision = agent._schedule_route_from_capabilities(dict(raw))
            routes.append(decision["route"])
            agent._record_decision_counts(decision)

        self.assertEqual(routes, ["global", "local", "dual"] * 4)
        self.assertEqual(
            agent.route_counts,
            {"skip": 0, "global": 4, "local": 4, "dual": 4},
        )
        self.assertEqual(
            scheduler_manifest()["target_route_ratios"],
            {"skip": 0.25, "global": 0.25, "local": 0.25, "dual": 0.25},
        )

    def test_scheduler_ratio_configuration_is_normalized(self):
        with patch.dict(os.environ, {"ADAPTVPR_TEST_RATIOS": "a:1,b:3"}):
            ratios = _ratios_from_env(
                "ADAPTVPR_TEST_RATIOS",
                {"a": 0.5, "b": 0.5},
            )
        self.assertEqual(ratios, {"a": 0.25, "b": 0.75})


if __name__ == "__main__":
    unittest.main()
