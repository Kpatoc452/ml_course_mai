"""Проверки математической корректности учебного вывода (unittest)."""

import unittest
from graphlib import CycleError

import numpy as np
import pandas as pd

from bayes import DiscreteBayesNet


class BayesTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"a": [0, 0, 0, 1], "b": [0, 0, 1, 1]})
        self.model = DiscreteBayesNet({"a": [0, 1], "b": [0, 1]}, {"b": ["a"], "a": []}).fit(self.data)

    def test_hand_calculated_chain(self):
        # P(a=1)=2/6, P(b=1|a=0)=2/5, P(b=1|a=1)=2/3.
        p_b = (4 / 6) * (2 / 5) + (2 / 6) * (2 / 3)
        self.assertAlmostEqual(self.model.query("b")[1], p_b)
        self.assertAlmostEqual(self.model.query("a", {"b": 1})[1], (2 / 6) * (2 / 3) / p_b)
        self.assertAlmostEqual(self.model.evidence_probability({"b": 1}), p_b)
        self.assertAlmostEqual(self.model.joint.probability.sum(), 1)

    def test_unseen_parent_combination(self):
        model = DiscreteBayesNet({"a": [0, 1, 2], "b": [0, 1]}, {"a": [], "b": ["a"]}).fit(self.data)
        np.testing.assert_allclose(model.query("b", {"a": 2}), [0.5, 0.5])

    def test_observed_target_and_invalid_evidence(self):
        np.testing.assert_allclose(self.model.query("a", {"a": 1}), [0, 1])
        with self.assertRaises(ValueError):
            self.model.query("b", {"a": 99})

    def test_invalid_data_and_cycle(self):
        with self.assertRaises(CycleError):
            DiscreteBayesNet({"a": [0, 1], "b": [0, 1]}, {"a": ["b"], "b": ["a"]})
        with self.assertRaises(ValueError):
            self.model.fit(self.data.assign(a=np.nan))

    def test_removed_edge_independence(self):
        model = DiscreteBayesNet({"a": [0, 1], "b": [0, 1]}, {"a": [], "b": []}).fit(self.data)
        np.testing.assert_allclose(model.query("b"), model.query("b", {"a": 1}))

    def test_collider_explaining_away(self):
        data = pd.DataFrame([(a, b, int(a or b)) for a in [0, 1] for b in [0, 1]] * 100,
                            columns=["a", "b", "c"])
        model = DiscreteBayesNet({v: [0, 1] for v in data}, {"a": [], "b": [], "c": ["a", "b"]}).fit(data)
        self.assertAlmostEqual(model.query("a")[1], model.query("a", {"b": 1})[1])
        self.assertGreater(model.query("a", {"c": 1})[1], model.query("a", {"b": 1, "c": 1})[1])


if __name__ == "__main__":
    unittest.main()
