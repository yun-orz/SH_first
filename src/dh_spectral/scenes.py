from __future__ import annotations

import numpy as np


def checkerboard_emitters(
    canvas_size: int,
    spacing_px: int,
    margin_px: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
    """生成两种波长交替排列的点阵场景及点位真值。"""
    first = np.zeros((canvas_size, canvas_size), dtype=float)
    second = np.zeros_like(first)
    coordinates: list[tuple[int, int, int]] = []
    positions = range(margin_px, canvas_size - margin_px, spacing_px)
    for row, y in enumerate(positions):
        for column, x in enumerate(positions):
            label = (row + column) % 2
            (first if label == 0 else second)[y, x] = 1.0
            coordinates.append((y, x, label))
    if not coordinates:
        raise ValueError("画布、边距和点间距组合没有产生有效点")
    return first, second, coordinates


def crop_patch(image: np.ndarray, center_y: int, center_x: int, size: int) -> np.ndarray:
    half = size // 2
    return image[center_y - half : center_y + half + 1, center_x - half : center_x + half + 1]

