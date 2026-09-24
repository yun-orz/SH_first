import unittest

import numpy as np

from dh_spectral.complementarity import (
    _interpolate_psf,
    _quadratic_refined_estimate,
    _weighted_template_costs,
    channel_complementarity_metrics,
    candidate_library_specs,
    dual_fisher_curve,
    fair_photon_budgets,
    validate_psf_bank,
)
from dh_spectral.estimation import crlb_from_fisher
from dh_spectral.forward import add_camera_noise


class ComplementarityTests(unittest.TestCase):
    def test_candidate_library_has_expected_size(self) -> None:
        self.assertEqual(len(candidate_library_specs()), 90)

    def test_total_photon_budget_is_conserved(self) -> None:
        budgets = fair_photon_budgets(20_000.0)
        for values in budgets.values():
            self.assertAlmostEqual(sum(values), 20_000.0)

    def test_dual_fisher_is_sum_of_independent_channels(self) -> None:
        first = np.array([1.0, 2.0, 3.0])
        second = np.array([4.0, 5.0, 6.0])
        np.testing.assert_allclose(dual_fisher_curve(first, second), first + second)

    def test_identical_channels_have_unit_redundancy(self) -> None:
        rng = np.random.default_rng(7)
        bank = rng.random((5, 9, 9))
        bank /= bank.sum(axis=(1, 2), keepdims=True)
        metrics = channel_complementarity_metrics(bank, bank)
        self.assertAlmostEqual(metrics["channel_redundancy"], 1.0, places=6)
        self.assertAlmostEqual(metrics["mirror_aligned_redundancy"], 1.0, places=6)
        self.assertAlmostEqual(metrics["spectral_derivative_similarity"], 1.0, places=6)

    def test_psf_bank_normalization_validation(self) -> None:
        bank = np.ones((3, 5, 5), dtype=float) / 25.0
        validate_psf_bank(bank)
        with self.assertRaises(ValueError):
            validate_psf_bank(bank * 0.9)

    def test_crlb_definition(self) -> None:
        fisher = np.array([1.0, 4.0, 25.0])
        np.testing.assert_allclose(crlb_from_fisher(fisher), np.array([1.0, 0.5, 0.2]))

    def test_noise_is_reproducible_with_same_seed(self) -> None:
        image = np.ones((9, 9), dtype=float)
        first = add_camera_noise(image, 1000.0, 0.2, 1.5, 16, np.random.default_rng(123))
        second = add_camera_noise(image, 1000.0, 0.2, 1.5, 16, np.random.default_rng(123))
        np.testing.assert_array_equal(first, second)

    def test_off_grid_interpolation_and_quadratic_refinement(self) -> None:
        wavelengths = np.array([500.0, 505.0, 510.0])
        bank = np.zeros((3, 2, 2), dtype=float)
        bank[:, 0, 0] = (1.0, 0.5, 0.0)
        bank[:, 1, 1] = (0.0, 0.5, 1.0)
        interpolated = _interpolate_psf(bank, wavelengths, 502.5)
        np.testing.assert_allclose(interpolated, np.array([[0.75, 0.0], [0.0, 0.25]]))
        costs = (wavelengths - 503.0) ** 2
        self.assertAlmostEqual(_quadratic_refined_estimate(wavelengths, costs, 505.0), 503.0)

    def test_weighted_cost_recovers_exact_template(self) -> None:
        bank = np.zeros((3, 3, 3), dtype=float)
        bank[0, 0, 0] = 1.0
        bank[1, 1, 1] = 1.0
        bank[2, 2, 2] = 1.0
        photons = 1000.0
        background = 0.2
        observation = photons * bank[1] + background
        costs = _weighted_template_costs(
            (observation,), (bank,), (photons,), background, 1.5
        )
        self.assertEqual(int(np.argmin(costs)), 1)


if __name__ == "__main__":
    unittest.main()
