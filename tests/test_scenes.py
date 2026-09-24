import unittest

import numpy as np

from dh_spectral.scenes import checkerboard_emitters, crop_patch


class SceneTests(unittest.TestCase):
    def test_checkerboard_labels_alternate(self) -> None:
        first, second, coordinates = checkerboard_emitters(65, 16, 16)
        self.assertGreater(len(coordinates), 1)
        self.assertEqual(int(first.sum() + second.sum()), len(coordinates))
        self.assertEqual(coordinates[0][2], 0)
        self.assertEqual(coordinates[1][2], 1)

    def test_crop_patch_shape(self) -> None:
        image = np.zeros((65, 65))
        self.assertEqual(crop_patch(image, 32, 32, 33).shape, (33, 33))
