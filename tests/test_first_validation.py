import unittest

import numpy as np

from dh_spectral.first_validation import (
    cosine_similarity_matrix,
    observation_vectors,
    pearson_correlation_matrix,
    system_photon_budgets,
)


class FirstValidationTests(unittest.TestCase):
    def test_equal_total_photon_budget(self) -> None:
        budgets = system_photon_budgets(20000.0)
        for system_budget in budgets.values():
            self.assertAlmostEqual(sum(system_budget), 20000.0)

    def test_similarity_diagonal_is_one(self) -> None:
        vectors = np.array([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0], [1.0, 2.0, 0.0]])
        np.testing.assert_allclose(np.diag(pearson_correlation_matrix(vectors)), 1.0)
        np.testing.assert_allclose(np.diag(cosine_similarity_matrix(vectors)), 1.0)

    def test_dual_observation_concatenates_two_half_power_channels(self) -> None:
        plus = np.zeros((2, 3, 3))
        minus = np.zeros((2, 3, 3))
        double = np.zeros((2, 3, 3))
        plus[:, 1, 1] = 1.0
        minus[:, 0, 0] = 1.0
        double[:, 2, 2] = 1.0
        banks = {
            "double_helix": (double,),
            "single_helix": (plus,),
            "dual_single_helix": (plus, minus),
        }
        vectors = observation_vectors(banks, np.array([0, 1]))
        self.assertEqual(vectors["dual_single_helix"].shape, (2, 18))
        np.testing.assert_allclose(vectors["dual_single_helix"].sum(axis=1), 1.0)

