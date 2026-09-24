from __future__ import annotations

import numpy as np


def fused_silica_index(wavelength_nm: float | np.ndarray) -> np.ndarray:
    """Malitson Sellmeier 模型，波长单位为 nm。"""
    wavelength_um = np.asarray(wavelength_nm, dtype=float) / 1000.0
    wavelength_sq = wavelength_um**2
    b = (0.6961663, 0.4079426, 0.8974794)
    c = (0.0684043**2, 0.1162414**2, 9.896161**2)
    n_sq = 1.0
    for coefficient, pole in zip(b, c):
        n_sq = n_sq + coefficient * wavelength_sq / (wavelength_sq - pole)
    return np.sqrt(n_sq)


def refractive_index(material: str, wavelength_nm: float | np.ndarray) -> np.ndarray:
    if material.lower() in {"fused_silica", "silica", "sio2"}:
        return fused_silica_index(wavelength_nm)
    raise ValueError(f"暂不支持材料: {material}")

