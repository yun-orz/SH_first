from __future__ import annotations

import numpy as np

from .materials import refractive_index


def pupil_grid(grid_size: int, diameter_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinate = (np.arange(grid_size) - grid_size / 2 + 0.5) * diameter_m / grid_size
    x, y = np.meshgrid(coordinate, coordinate)
    radius = np.hypot(x, y)
    angle = np.arctan2(y, x)
    return x, y, radius, angle


def circular_aperture(radius: np.ndarray, pupil_radius_m: float, fill: float = 1.0) -> np.ndarray:
    return (radius <= pupil_radius_m * fill).astype(float)


def spiral_zone_phase(
    radius: np.ndarray,
    angle: np.ndarray,
    pupil_radius_m: float,
    zone_count: int,
    charge_start: int,
    charge_step: int,
    handedness: int = 1,
) -> np.ndarray:
    """生成等面积环带螺旋相位，作为 DOE 优化的解析初值。"""
    normalized_area = np.clip((radius / pupil_radius_m) ** 2, 0.0, 1.0 - np.finfo(float).eps)
    zone = np.floor(normalized_area * zone_count).astype(int)
    charge = charge_start + charge_step * zone
    return np.mod(handedness * charge * angle, 2.0 * np.pi)


def lens_phase(radius: np.ndarray, wavelength_m: float, focal_length_m: float) -> np.ndarray:
    return np.mod(-np.pi * radius**2 / (wavelength_m * focal_length_m), 2.0 * np.pi)


def phase_to_height(
    phase: np.ndarray,
    design_wavelength_m: float,
    refractive_index_design: float,
    levels: int,
    continuous: bool,
) -> np.ndarray:
    wrapped = np.mod(phase, 2.0 * np.pi)
    if not continuous:
        wrapped = np.floor(wrapped / (2.0 * np.pi) * levels) / levels * (2.0 * np.pi)
    return wrapped * design_wavelength_m / (2.0 * np.pi * (refractive_index_design - 1.0))


def height_to_phase(height_m: np.ndarray, wavelength_m: float, material: str) -> np.ndarray:
    index = float(refractive_index(material, wavelength_m * 1e9))
    return 2.0 * np.pi * (index - 1.0) * height_m / wavelength_m


def build_height_mask(
    radius: np.ndarray,
    angle: np.ndarray,
    pupil_radius_m: float,
    focal_length_m: float,
    design_wavelength_m: float,
    material: str,
    zone_count: int,
    charge_start: int,
    charge_step: int,
    levels: int,
    continuous: bool,
    handedness: int = 1,
) -> np.ndarray:
    spiral = spiral_zone_phase(
        radius,
        angle,
        pupil_radius_m,
        zone_count,
        charge_start,
        charge_step,
        handedness,
    )
    total_phase = spiral + lens_phase(radius, design_wavelength_m, focal_length_m)
    index_design = float(refractive_index(material, design_wavelength_m * 1e9))
    return phase_to_height(total_phase, design_wavelength_m, index_design, levels, continuous)

