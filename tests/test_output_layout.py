import unittest
from pathlib import Path

from generation.output_layout import final_output_path


class OutputLayoutTest(unittest.TestCase):
    def test_accepted_image_stays_in_route_directory(self):
        path = final_output_path(
            Path("output"), "sample", "dual", "rain", "vehicle", passed=True
        )
        self.assertEqual(path, Path("output/dual/sample__dual_rain_vehicle__final.jpg"))

    def test_rejected_image_is_isolated_from_route_directory(self):
        path = final_output_path(
            Path("output"), "sample", "local", None, "vehicle", passed=False
        )
        self.assertEqual(
            path, Path("output/rejected/local/sample__local_vehicle__rejected.jpg")
        )
        self.assertNotEqual(path.parent, Path("output/local"))


if __name__ == "__main__":
    unittest.main()
