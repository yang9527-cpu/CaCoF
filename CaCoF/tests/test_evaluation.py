"""Evaluation must preserve named-class metrics across label integer types."""

import unittest

import numpy as np

from main import evaluate


class EvaluationTests(unittest.TestCase):
    def test_integer_dtype_does_not_change_confusion_counts(self):
        probabilities = np.eye(37)
        labels = np.arange(37)
        expected = {"ACC": 100.0, "MacroF1": 100.0, "Brier": 0.0}
        for dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.int64):
            with self.subTest(dtype=dtype):
                self.assertEqual(evaluate(labels.astype(dtype), probabilities), expected)

    def test_macro_f1_keeps_absent_class_coordinates(self):
        probabilities = np.array([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        result = evaluate(np.array([0, 1, 1]), probabilities)
        self.assertAlmostEqual(result["ACC"], 200 / 3)
        self.assertAlmostEqual(result["MacroF1"], 400 / 9)
        self.assertAlmostEqual(result["Brier"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
