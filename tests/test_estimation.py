import unittest

import numpy as np

from dh_spectral.estimation import crlb_from_fisher, estimate_wavelength, poisson_gaussian_fisher


def _toy_bank() -> np.ndarray:
    bank = np.zeros((3, 9, 9))
    bank[0, 4, 2] = 1.0
    bank[1, 4, 4] = 1.0
    bank[2, 4, 6] = 1.0
    return bank


class EstimationTests(unittest.TestCase):
    def test_template_estimator_recovers_exact_member(self) -> None:
        bank = _toy_bank()
        estimate, confidence, _ = estimate_wavelength((bank[1],), (bank,), np.array([500, 550, 600]))
        self.assertEqual(estimate, 550)
        self.assertGreater(confidence, 0.9)

    def test_fisher_and_crlb_are_finite(self) -> None:
        bank = _toy_bank()
        fisher = poisson_gaussian_fisher(bank, np.array([500.0, 550.0, 600.0]), 1000, 0.1, 1.0)
        self.assertTrue(np.all(np.isfinite(fisher)))
        self.assertTrue(np.all(crlb_from_fisher(fisher) > 0))
