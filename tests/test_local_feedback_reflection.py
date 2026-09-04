import json
import sys
import types
import unittest
from unittest.mock import patch

from PIL import Image

# This unit exercises prompt wiring only; numeric image comparison is not used.
sys.modules.setdefault("numpy", types.ModuleType("numpy"))

from generation.reflection_controller import ReflectionController


class RecordingLLM:
    def __init__(self):
        self.calls = []

    def chat_with_images(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps(
            {
                "concrete_visible_failures": ["geometry score is below threshold"],
                "feedback_repairs": ["preserve the exact curb and lane geometry"],
                "vehicle_change": "move one van fully inside the road region",
                "rewritten_reflection_prompt": "Preserve exact geometry and move one van inward.",
            }
        )


class LocalFeedbackReflectionTest(unittest.TestCase):
    def test_dual_person_contract_never_switches_to_vehicle(self):
        decision = {
            "occlusion": "person",
            "weather": "rain",
            "position": "visible right-side sidewalk",
        }

        contracts = [
            ReflectionController._dual_contract(decision, round_index)
            for round_index in (1, 2, 3)
        ]

        self.assertIn("exactly one clearly visible full-body pedestrian", contracts[0])
        self.assertIn("exactly two separated full-body pedestrians", contracts[1])
        self.assertIn("exactly one prominent full-body pedestrian", contracts[2])
        for contract in contracts:
            self.assertIn("Do not add vehicles", contract)
            self.assertNotIn("delivery van", contract)
            self.assertNotIn("box truck", contract)
            self.assertNotIn("shuttle bus", contract)

        vehicle_contract = ReflectionController._dual_contract(
            {**decision, "occlusion": "vehicle"}, 1
        )
        self.assertIn("delivery van", vehicle_contract)

    def test_dual_person_legality_counts_people_not_vehicles(self):
        controller = ReflectionController(
            llm_client=RecordingLLM(),
            llm_model="qwen-test",
            evaluator=None,
            generate=lambda *args: None,
        )
        source = Image.new("RGB", (32, 32), "white")
        candidate = Image.new("RGB", (32, 32), "black")
        with patch.object(
            controller,
            "_vision_json",
            return_value={"accepted": True, "new_person_count": 2, "violations": []},
        ) as audit:
            result = controller._route_legality_diagnostic(
                source,
                candidate,
                "dual",
                2,
                {"occlusion": "person"},
                "rain plus two pedestrians",
            )

        self.assertTrue(result["accepted"])
        self.assertEqual(result["expected_person_count"], 2)
        self.assertIn("new_person_count", audit.call_args.args[1])

    def test_reflection_rounds_use_independent_random_seeds(self):
        with patch(
            "generation.reflection_controller.secrets.randbelow",
            side_effect=[100, 200],
        ) as random_seed:
            first = ReflectionController._seed(42, 1)
            second = ReflectionController._seed(42, 2)

        self.assertEqual((first, second), (101, 201))
        self.assertEqual(random_seed.call_count, 2)
        random_seed.assert_any_call(2147483646)

    def test_local_contract_preserves_prompt_occluder_families(self):
        previous_prompt = (
            "Add pedestrians, one cyclist, a scooter rider, and roadside vegetation "
            "as transient occluders."
        )
        decision = {"occlusion": "person", "position": "visible roadside pavement"}

        allowed = ReflectionController._local_allowed_occluders(decision, previous_prompt)
        contract = ReflectionController._local_contract(decision, previous_prompt, 2)

        self.assertEqual(
            allowed,
            ["pedestrian", "cyclist", "scooter rider", "roadside vegetation"],
        )
        self.assertIn("Allowed occluder family", contract)
        self.assertIn("roadside vegetation", contract)
        self.assertNotIn("exactly two separated fully visible vehicles", contract)

    def test_local_rewrite_receives_verifier_feedback_and_failed_image(self):
        llm = RecordingLLM()
        controller = ReflectionController(
            llm_client=llm,
            llm_model="qwen-test",
            evaluator=None,
            generate=lambda *args: None,
        )
        source = Image.new("RGB", (32, 32), "white")
        failed = Image.new("RGB", (32, 32), "black")
        feedback = {
            "evaluation": {
                "s_geo": 0.41,
                "s_div": 0.12,
                "feedback": {"geo_issue": {"prompt_instruction": "KEEP_CURB_MARKER"}},
            },
            "local_border_integrity_diagnostic": {"accepted": False, "violations": ["CROPPED_MARKER"]},
        }

        prompt, analysis, raw = controller._local_rewrite(
            source,
            failed,
            "old prompt",
            feedback,
            {"position": "right traffic lane"},
            1,
        )

        self.assertEqual(len(llm.calls), 1)
        call = llm.calls[0]
        self.assertEqual(len(call["images"]), 2)
        self.assertIn("KEEP_CURB_MARKER", call["user"])
        self.assertIn("CROPPED_MARKER", call["user"])
        self.assertIn("Verifier and VLM diagnostic feedback", call["user"])
        self.assertIn("old prompt", call["user"])
        self.assertIn("Preserve exact geometry and move one van inward.", prompt)
        self.assertIn("Authoritative non-negotiable contract", prompt)
        self.assertEqual(analysis["vehicle_change"], "move one van fully inside the road region")
        self.assertTrue(raw)


if __name__ == "__main__":
    unittest.main()
