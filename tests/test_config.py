import unittest
from pathlib import Path

from dh_spectral.config import load_config


class ConfigTests(unittest.TestCase):
    def test_baseline_config_loads(self) -> None:
        config = load_config(Path("configs/baseline.yaml"))
        self.assertEqual(config.optics.wavelengths_nm[0], 400.0)
        self.assertEqual(config.optics.wavelengths_nm[-1], 700.0)
        self.assertEqual(len(config.optics.wavelengths_nm), 61)
