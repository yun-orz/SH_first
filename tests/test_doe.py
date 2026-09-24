import unittest

import numpy as np

from dh_spectral.doe import height_to_phase, phase_to_height, pupil_grid, spiral_zone_phase
from dh_spectral.materials import fused_silica_index


class DOETests(unittest.TestCase):
    def test_phase_height_round_trip_at_design_wavelength(self) -> None:
        wavelength_m = 550e-9
        phase = np.linspace(0.0, 2.0 * np.pi, 100, endpoint=False)
        index = float(fused_silica_index(550.0))
        height = phase_to_height(phase, wavelength_m, index, levels=16, continuous=True)
        recovered = height_to_phase(height, wavelength_m, "fused_silica")
        np.testing.assert_allclose(recovered, phase, atol=1e-12)

    def test_conjugate_spiral_phases_are_opposites(self) -> None:
        _, _, radius, angle = pupil_grid(64, 2e-3)
        plus = spiral_zone_phase(radius, angle, 1e-3, 8, 0, 1, 1)
        minus = spiral_zone_phase(radius, angle, 1e-3, 8, 0, 1, -1)
        np.testing.assert_allclose(np.mod(plus + minus, 2 * np.pi), 0.0, atol=1e-12)
