from __future__ import annotations

import numpy as np


def wavelength_metrics(truth_nm: np.ndarray, estimate_nm: np.ndarray, outlier_threshold_nm: float = 10.0) -> dict[str, float]:
    error = np.asarray(estimate_nm) - np.asarray(truth_nm)
    return {
        "bias_nm": float(np.mean(error)),
        "mae_nm": float(np.mean(np.abs(error))),
        "rmse_nm": float(np.sqrt(np.mean(error**2))),
        "std_nm": float(np.std(error, ddof=1)) if len(error) > 1 else 0.0,
        "outlier_rate": float(np.mean(np.abs(error) > outlier_threshold_nm)),
    }


def adjacent_template_correlation(bank: np.ndarray) -> np.ndarray:
    flattened = bank.reshape(bank.shape[0], -1)
    flattened = flattened - flattened.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(flattened, axis=1)
    numerator = np.sum(flattened[:-1] * flattened[1:], axis=1)
    return numerator / np.maximum(norms[:-1] * norms[1:], np.finfo(float).eps)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, repeats: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    samples = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)

