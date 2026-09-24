from __future__ import annotations

import numpy as np


def _normalized_residual(observation: np.ndarray, template: np.ndarray) -> float:
    obs = np.clip(np.asarray(observation, dtype=float), 0.0, None)
    tpl = np.clip(np.asarray(template, dtype=float), 0.0, None)
    obs = obs / max(obs.sum(), np.finfo(float).eps)
    tpl = tpl / max(tpl.sum(), np.finfo(float).eps)
    return float(np.mean((obs - tpl) ** 2))


def estimate_wavelength(
    observations: tuple[np.ndarray, ...],
    template_banks: tuple[np.ndarray, ...],
    wavelengths_nm: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """统一的多通道模板最小二乘估计，返回估计值、置信度和代价。"""
    if len(observations) != len(template_banks):
        raise ValueError("观测通道数与模板库数量不一致")
    costs = np.zeros(len(wavelengths_nm), dtype=float)
    for index in range(len(wavelengths_nm)):
        costs[index] = sum(
            _normalized_residual(observation, bank[index])
            for observation, bank in zip(observations, template_banks)
        )
    order = np.argsort(costs)
    best = int(order[0])
    second = float(costs[order[1]]) if len(order) > 1 else np.inf
    confidence = (second - float(costs[best])) / max(second, np.finfo(float).eps)
    return float(wavelengths_nm[best]), float(confidence), costs


def poisson_gaussian_fisher(
    psf_bank: np.ndarray,
    wavelengths_nm: np.ndarray,
    photons: float,
    background_per_pixel: float,
    read_noise_e: float,
) -> np.ndarray:
    """使用方差近似 μ+σ² 计算波长 Fisher 信息。"""
    mean = photons * psf_bank + background_per_pixel
    derivative = np.gradient(mean, wavelengths_nm, axis=0, edge_order=2)
    variance = np.maximum(mean + read_noise_e**2, np.finfo(float).eps)
    return np.sum(derivative**2 / variance, axis=(-2, -1))


def multichannel_fisher(
    banks: tuple[np.ndarray, ...],
    wavelengths_nm: np.ndarray,
    photon_budgets: tuple[float, ...],
    background_per_pixel: float,
    read_noise_e: float,
) -> np.ndarray:
    result = np.zeros(len(wavelengths_nm), dtype=float)
    for bank, photons in zip(banks, photon_budgets):
        result += poisson_gaussian_fisher(
            bank, wavelengths_nm, photons, background_per_pixel, read_noise_e
        )
    return result


def crlb_from_fisher(fisher: np.ndarray) -> np.ndarray:
    return 1.0 / np.sqrt(np.maximum(fisher, np.finfo(float).eps))

