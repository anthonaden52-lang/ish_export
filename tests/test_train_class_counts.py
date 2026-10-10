import unittest

import pandas as pd

from train import _class_counts


class TrainingClassCountTests(unittest.TestCase):
    def test_counts_both_classes_for_release_audit(self):
        counts = _class_counts(pd.Series([0, 1, 1, 0, 1]))

        self.assertEqual(counts, {"positive": 3, "negative": 2})

    def test_empty_calibration_window_has_zero_class_counts(self):
        counts = _class_counts(pd.Series([], dtype=int))

        self.assertEqual(counts, {"positive": 0, "negative": 0})


if __name__ == "__main__":
    unittest.main()
