from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from .doe import circular_aperture, height_to_phase


def center_crop(image: np.ndarray, size: int) -> np.ndarray:
    start_y = (image.shape[-2] - size) // 2
    start_x = (image.shape[-1] - size) // 2
    return image[..., start_y : start_y + size, start_x : start_x + size]


def _resize_to_sensor(image: np.ndarray, source_pitch_m: float, sensor_pitch_m: float) -> np.ndarray:
    factor = source_pitch_m / sensor_pitch_m
    if 0.75 <= factor <= 1.25:
        return image
    return zoom(image, factor, order=1, mode="constant", prefilter=False)


def fresnel_psf(
    height_m: np.ndarray,
    radius_m: np.ndarray,
    pupil_radius_m: float,
    wavelength_m: float,
    sensor_distance_m: float,
    pupil_sample_pitch_m: float,
    sensor_pixel_pitch_m: float,
    material: str,
    aperture_fill: float,
    crop_size: int,
) -> np.ndarray:
    """用单 FFT Fresnel 传播计算归一化强度 PSF。"""
    aperture = circular_aperture(radius_m, pupil_radius_m, aperture_fill)
    phase = height_to_phase(height_m, wavelength_m, material)
    quadratic = np.pi * radius_m**2 / (wavelength_m * sensor_distance_m)
    pupil_field = aperture * np.exp(1j * (phase + quadratic))
    sensor_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_field)))
    intensity = np.abs(sensor_field) ** 2
    natural_pitch = wavelength_m * sensor_distance_m / (height_m.shape[0] * pupil_sample_pitch_m)
    intensity = _resize_to_sensor(intensity, natural_pitch, sensor_pixel_pitch_m)
    intensity = center_crop(intensity, crop_size)
    total = intensity.sum()
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("PSF 能量无效")
    return intensity / total


def psf_centroid(psf: np.ndarray) -> tuple[float, float]:
    y, x = np.indices(psf.shape, dtype=float)
    total = psf.sum()
    return float((x * psf).sum() / total), float((y * psf).sum() / total)


def second_moment_features(psf: np.ndarray) -> dict[str, float]:
    x0, y0 = psf_centroid(psf)
    y, x = np.indices(psf.shape, dtype=float)
    dx, dy = x - x0, y - y0
    total = psf.sum()
    cxx = float((psf * dx * dx).sum() / total)
    cyy = float((psf * dy * dy).sum() / total)
    cxy = float((psf * dx * dy).sum() / total)
    angle = 0.5 * np.arctan2(2.0 * cxy, cxx - cyy)
    eigenvalues = np.linalg.eigvalsh(np.array([[cxx, cxy], [cxy, cyy]]))
    return {
        "centroid_x": x0,
        "centroid_y": y0,
        "orientation_rad": float(angle),
        "major_sigma": float(np.sqrt(max(eigenvalues[-1], 0.0))),
        "minor_sigma": float(np.sqrt(max(eigenvalues[0], 0.0))),
    }

