import unittest
import sys
import types

sys.modules.setdefault("numpy", types.ModuleType("numpy"))

from generation.reflection_controller import ReflectionController


class AcceptancePolicyTest(unittest.TestCase):
    def test_vlm_diagnostics_do_not_override_passing_dual_trait_scores(self):
        decision = ReflectionController._acceptance_decision(
            initial_eval={"s_geo": 0.9},
            evaluation={"passed": True, "s_geo": 0.75, "s_div": 0.2},
            interround=2.0,
            diagnostics=[{"accepted": False}],
        )

        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["acceptance_rule"], "dual_trait_scores_only")
        self.assertIn("one_or_more_vlm_diagnostics_failed", decision["diagnostic_warnings"])

    def test_failing_dual_trait_scores_are_rejected(self):
        decision = ReflectionController._acceptance_decision(
            initial_eval={"s_geo": 0.5},
            evaluation={"passed": False, "s_geo": 0.9, "s_div": 0.01},
            interround=20.0,
            diagnostics=[{"accepted": True}],
        )

        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reasons"], ["dual_trait_verifier_failed"])

if __name__ == "__main__":
    unittest.main()
