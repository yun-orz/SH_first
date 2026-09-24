import unittest

import numpy as np

from dh_spectral.forward import add_camera_noise, photon_budgets, render_incoherent_scene


class ForwardTests(unittest.TestCase):
    def test_scene_energy_is_preserved_away_from_boundaries(self) -> None:
        scene = np.zeros((33, 33))
        scene[16, 16] = 1.0
        psf = np.ones((5, 5)) / 25.0
        rendered = render_incoherent_scene(scene, psf)
        np.testing.assert_allclose(rendered.sum(), 1.0, atol=1e-12)

    def test_fair_photon_budget_has_equal_total(self) -> None:
        budgets = photon_budgets(1000, "fair_photon", 0.46, 0.46, 0.75)
        self.assertEqual(sum(budgets["dual"]), budgets["double"][0])
        self.assertEqual(budgets["double"][0], 1000)

    def test_noise_is_reproducible(self) -> None:
        image = np.ones((9, 9))
        first = add_camera_noise(image, 1000, 0.1, 1.0, 16, np.random.default_rng(7))
        second = add_camera_noise(image, 1000, 0.1, 1.0, 16, np.random.default_rng(7))
        np.testing.assert_array_equal(first, second)
