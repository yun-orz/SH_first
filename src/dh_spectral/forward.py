from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def normalize_scene(scene: np.ndarray) -> np.ndarray:
    scene = np.asarray(scene, dtype=float)
    if np.any(scene < 0):
        raise ValueError("场景强度不能为负")
    total = scene.sum()
    if total <= 0:
        raise ValueError("场景总能量必须大于零")
    return scene / total


def render_incoherent_scene(scene: np.ndarray, psf: np.ndarray) -> np.ndarray:
    return fftconvolve(normalize_scene(scene), psf, mode="same")


def render_spectral_scene(scene_cube: np.ndarray, psf_bank: np.ndarray) -> np.ndarray:
    if scene_cube.shape[0] != psf_bank.shape[0]:
        raise ValueError("光谱场景和 PSF 库的波长维长度不同")
    result = np.zeros(scene_cube.shape[1:], dtype=float)
    for scene_slice, psf in zip(scene_cube, psf_bank):
        result += fftconvolve(scene_slice, psf, mode="same")
    return result


def add_camera_noise(
    normalized_image: np.ndarray,
    signal_photons: float,
    background_per_pixel: float,
    read_noise_e: float,
    bit_depth: int,
    rng: np.random.Generator,
) -> np.ndarray:
    image = np.asarray(normalized_image, dtype=float)
    image = np.clip(image, 0.0, None)
    if image.sum() > 0:
        image = image / image.sum()
    expected = image * signal_photons + background_per_pixel
    noisy = rng.poisson(expected).astype(float)
    if read_noise_e > 0:
        noisy += rng.normal(0.0, read_noise_e, noisy.shape)
    maximum = float(2**bit_depth - 1)
    return np.clip(np.rint(noisy), 0.0, maximum)


def photon_budgets(total_photons: float, mode: str, transmission: float, reflection: float, efficiency: float) -> dict[str, tuple[float, ...]]:
    """返回双通道和单通道的探测光子预算。"""
    if mode == "fair_photon":
        return {"dual": (total_photons / 2.0, total_photons / 2.0), "double": (total_photons,)}
    if mode == "hardware_realistic":
        return {
            "dual": (total_photons * transmission * efficiency, total_photons * reflection * efficiency),
            "double": (total_photons * efficiency,),
        }
    raise ValueError(f"未知比较模式: {mode}")

