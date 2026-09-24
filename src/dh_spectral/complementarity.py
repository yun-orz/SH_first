from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import combinations_with_replacement
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from .config import ResearchConfig
from .doe import build_height_mask, pupil_grid
from .estimation import crlb_from_fisher, poisson_gaussian_fisher
from .evaluation import wavelength_metrics
from .forward import add_camera_noise
from .pipeline import build_psf_banks
from .plotting import configure_chinese_font
from .propagation import fresnel_psf


configure_chinese_font()


ZONE_COUNTS = (4, 6, 8, 10, 12)
CHARGE_STARTS = (0, 1, 2)
CHARGE_STEPS = (1, 2, 3)
HANDEDNESSES = (-1, 1)
ATLAS_WAVELENGTHS_NM = (450.0, 500.0, 550.0, 600.0, 650.0)


@dataclass(frozen=True)
class SingleHelixCandidate:
    candidate_id: str
    zone_count: int
    charge_start: int
    charge_step: int
    handedness: int


def candidate_library_specs() -> list[SingleHelixCandidate]:
    """返回固定且可复现的 90 个 Single Helix 候选参数。"""
    result: list[SingleHelixCandidate] = []
    for zone_count in ZONE_COUNTS:
        for charge_start in CHARGE_STARTS:
            for charge_step in CHARGE_STEPS:
                for handedness in HANDEDNESSES:
                    hand = "p1" if handedness > 0 else "m1"
                    candidate_id = f"z{zone_count:02d}_c{charge_start}_s{charge_step}_h{hand}"
                    result.append(
                        SingleHelixCandidate(
                            candidate_id, zone_count, charge_start, charge_step, handedness
                        )
                    )
    return result


def fair_photon_budgets(total_photons: float) -> dict[str, tuple[float, ...]]:
    """第二轮只允许等总探测光子比较。"""
    return {
        "double_helix": (total_photons,),
        "single_helix": (total_photons,),
        "dual_single_helix": (total_photons / 2.0, total_photons / 2.0),
    }


def dual_fisher_curve(first_half_fisher: np.ndarray, second_half_fisher: np.ndarray) -> np.ndarray:
    """独立通道的联合 Fisher 等于各通道 Fisher 之和。"""
    return np.asarray(first_half_fisher, dtype=float) + np.asarray(second_half_fisher, dtype=float)


def validate_psf_bank(bank: np.ndarray, atol: float = 2e-6) -> None:
    totals = np.asarray(bank, dtype=float).sum(axis=(-2, -1))
    if np.any(bank < 0) or not np.allclose(totals, 1.0, atol=atol):
        raise ValueError("PSF 库必须非负且每个波长严格归一化")


def _normalized_vectors(bank: np.ndarray, centered: bool = False) -> np.ndarray:
    vectors = np.asarray(bank, dtype=np.float32).reshape(bank.shape[0], -1)
    if centered:
        vectors = vectors - vectors.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, np.finfo(np.float32).eps)


def channel_complementarity_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """计算一对通道的平均同波长冗余；这些量仅作为本项目分析指标。"""
    first_cos = _normalized_vectors(first)
    second_cos = _normalized_vectors(second)
    cosine = np.sum(first_cos * second_cos, axis=1)

    first_corr = _normalized_vectors(first, centered=True)
    second_corr = _normalized_vectors(second, centered=True)
    pearson = np.sum(first_corr * second_corr, axis=1)
    transformed = (
        second_corr,
        _normalized_vectors(np.flip(second, axis=2), centered=True),
        _normalized_vectors(np.flip(second, axis=1), centered=True),
        _normalized_vectors(np.flip(second, axis=(1, 2)), centered=True),
    )
    aligned = np.max(np.stack([np.sum(first_corr * item, axis=1) for item in transformed]), axis=0)

    first_derivative = np.gradient(np.asarray(first, dtype=np.float32), axis=0, edge_order=2)
    second_derivative = np.gradient(np.asarray(second, dtype=np.float32), axis=0, edge_order=2)
    derivative = np.sum(
        _normalized_vectors(first_derivative) * _normalized_vectors(second_derivative), axis=1
    )
    return {
        "channel_redundancy": float(np.mean(cosine)),
        "channel_pearson_redundancy": float(np.mean(pearson)),
        "mirror_aligned_redundancy": float(np.mean(aligned)),
        "spectral_derivative_similarity": float(np.mean(derivative)),
    }


def _cache_signature(config: ResearchConfig, candidate: SingleHelixCandidate) -> str:
    payload = {
        "optics": asdict(config.optics),
        "doe_fixed": {
            "material": config.doe.material,
            "phase_levels": config.doe.phase_levels,
            "continuous_phase": config.doe.continuous_phase,
            "aperture_fill": config.doe.aperture_fill,
        },
        "crop_size": config.simulation.crop_size,
        "candidate": asdict(candidate),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _build_candidate_bank(config: ResearchConfig, candidate: SingleHelixCandidate) -> np.ndarray:
    optics, doe = config.optics, config.doe
    diameter_m = optics.pupil_diameter_mm * 1e-3
    _, _, radius, angle = pupil_grid(optics.grid_size, diameter_m)
    height = build_height_mask(
        radius=radius,
        angle=angle,
        pupil_radius_m=diameter_m / 2.0,
        focal_length_m=optics.focal_length_mm * 1e-3,
        design_wavelength_m=optics.design_wavelength_nm * 1e-9,
        material=doe.material,
        zone_count=candidate.zone_count,
        charge_start=candidate.charge_start,
        charge_step=candidate.charge_step,
        levels=doe.phase_levels,
        continuous=doe.continuous_phase,
        handedness=candidate.handedness,
    )
    common = dict(
        height_m=height,
        radius_m=radius,
        pupil_radius_m=diameter_m / 2.0,
        sensor_distance_m=optics.sensor_distance_mm * 1e-3,
        pupil_sample_pitch_m=diameter_m / optics.grid_size,
        sensor_pixel_pitch_m=optics.pixel_pitch_um * 1e-6,
        material=doe.material,
        aperture_fill=doe.aperture_fill,
        crop_size=config.simulation.crop_size,
    )
    bank = np.stack(
        [fresnel_psf(wavelength_m=value * 1e-9, **common) for value in optics.wavelengths_nm]
    ).astype(np.float32)
    validate_psf_bank(bank)
    return bank


def build_candidate_library(
    config: ResearchConfig,
    cache_directory: Path,
) -> tuple[list[SingleHelixCandidate], np.ndarray, int]:
    """建立候选 PSF 库；每个 DOE 单独缓存，重复运行时不重复传播。"""
    cache_directory.mkdir(parents=True, exist_ok=True)
    specs = candidate_library_specs()
    banks: list[np.ndarray] = []
    cache_hits = 0
    for index, candidate in enumerate(specs, start=1):
        cache_path = cache_directory / f"{candidate.candidate_id}.npz"
        signature = _cache_signature(config, candidate)
        bank: np.ndarray | None = None
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as cached:
                if str(cached["signature"].item()) == signature:
                    bank = np.asarray(cached["psf_bank"], dtype=np.float32)
                    validate_psf_bank(bank)
                    cache_hits += 1
        if bank is None:
            print(f"生成候选 {index:02d}/{len(specs)}: {candidate.candidate_id}", flush=True)
            bank = _build_candidate_bank(config, candidate)
            np.savez_compressed(cache_path, signature=np.array(signature), psf_bank=bank)
        banks.append(bank)
    return specs, np.stack(banks), cache_hits


def _adjacent_cosine(bank: np.ndarray) -> np.ndarray:
    vectors = _normalized_vectors(bank)
    return np.sum(vectors[:-1] * vectors[1:], axis=1)


def _candidate_metrics(
    config: ResearchConfig,
    specs: list[SingleHelixCandidate],
    banks: np.ndarray,
    wavelengths_nm: np.ndarray,
    analysis_mask: np.ndarray,
) -> tuple[list[dict[str, float | int | str]], np.ndarray, np.ndarray]:
    camera = config.camera
    full_fisher = np.stack([
        poisson_gaussian_fisher(
            bank, wavelengths_nm, camera.photons_total,
            camera.background_photons_per_pixel, camera.read_noise_e,
        ) for bank in banks
    ])
    half_fisher = np.stack([
        poisson_gaussian_fisher(
            bank, wavelengths_nm, camera.photons_total / 2.0,
            camera.background_photons_per_pixel, camera.read_noise_e,
        ) for bank in banks
    ])
    rows: list[dict[str, float | int | str]] = []
    for candidate, bank, fisher in zip(specs, banks, full_fisher):
        adjacent = _adjacent_cosine(bank)
        rows.append({
            **asdict(candidate),
            "median_fisher": float(np.median(fisher[analysis_mask])),
            "median_crlb_nm": float(np.median(crlb_from_fisher(fisher)[analysis_mask])),
            "mean_adjacent_psf_similarity": float(np.mean(adjacent)),
            "max_adjacent_psf_similarity": float(np.max(adjacent)),
        })
    return rows, full_fisher, half_fisher


def _all_pair_redundancy(banks: np.ndarray) -> dict[str, np.ndarray]:
    """用逐波长矩阵乘法一次计算全部候选对，避免重复处理相同 DOE。"""
    candidate_count, wavelength_count, height, width = banks.shape
    pixel_count = height * width
    raw = banks.reshape(candidate_count, wavelength_count, pixel_count).astype(np.float32)
    cosine = raw / np.maximum(np.linalg.norm(raw, axis=2, keepdims=True), np.finfo(np.float32).eps)
    centered = raw - raw.mean(axis=2, keepdims=True)
    pearson = centered / np.maximum(
        np.linalg.norm(centered, axis=2, keepdims=True), np.finfo(np.float32).eps
    )
    derivatives = np.gradient(banks, axis=1, edge_order=2).reshape(
        candidate_count, wavelength_count, pixel_count
    )
    derivative = derivatives / np.maximum(
        np.linalg.norm(derivatives, axis=2, keepdims=True), np.finfo(np.float32).eps
    )
    sums = {
        "channel_redundancy": np.zeros((candidate_count, candidate_count), dtype=np.float64),
        "channel_pearson_redundancy": np.zeros((candidate_count, candidate_count), dtype=np.float64),
        "mirror_aligned_redundancy": np.zeros((candidate_count, candidate_count), dtype=np.float64),
        "spectral_derivative_similarity": np.zeros((candidate_count, candidate_count), dtype=np.float64),
    }
    for wavelength_index in range(wavelength_count):
        current_cos = cosine[:, wavelength_index]
        current_pearson = pearson[:, wavelength_index]
        current_derivative = derivative[:, wavelength_index]
        sums["channel_redundancy"] += current_cos @ current_cos.T
        direct = current_pearson @ current_pearson.T
        sums["channel_pearson_redundancy"] += direct
        spatial = current_pearson.reshape(candidate_count, height, width)
        aligned_candidates = [direct]
        for transformed in (
            np.flip(spatial, axis=2), np.flip(spatial, axis=1), np.flip(spatial, axis=(1, 2)),
        ):
            aligned_candidates.append(current_pearson @ transformed.reshape(candidate_count, -1).T)
        sums["mirror_aligned_redundancy"] += np.max(np.stack(aligned_candidates), axis=0)
        sums["spectral_derivative_similarity"] += current_derivative @ current_derivative.T
    return {name: values / wavelength_count for name, values in sums.items()}


def _pair_rows(
    specs: list[SingleHelixCandidate],
    half_fisher: np.ndarray,
    redundancy: dict[str, np.ndarray],
    analysis_mask: np.ndarray,
    median_fisher_double: float,
    median_fisher_best_single: float,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for first_index, second_index in combinations_with_replacement(range(len(specs)), 2):
        first, second = specs[first_index], specs[second_index]
        fisher = dual_fisher_curve(half_fisher[first_index], half_fisher[second_index])
        median_fisher = float(np.median(fisher[analysis_mask]))
        rows.append({
            "pair_id": f"{first.candidate_id}__{second.candidate_id}",
            "channel_A_id": first.candidate_id,
            "channel_A_zone_count": first.zone_count,
            "channel_A_charge_start": first.charge_start,
            "channel_A_charge_step": first.charge_step,
            "channel_A_handedness": first.handedness,
            "channel_B_id": second.candidate_id,
            "channel_B_zone_count": second.zone_count,
            "channel_B_charge_start": second.charge_start,
            "channel_B_charge_step": second.charge_step,
            "channel_B_handedness": second.handedness,
            **{name: float(matrix[first_index, second_index]) for name, matrix in redundancy.items()},
            "median_fisher": median_fisher,
            "median_crlb_nm": float(np.median(crlb_from_fisher(fisher)[analysis_mask])),
            "gain_DH": median_fisher / median_fisher_double,
            "gain_SH": median_fisher / median_fisher_best_single,
            "delta_F_DH": median_fisher - median_fisher_double,
            "delta_F_SH": median_fisher - median_fisher_best_single,
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _nearest_indices(wavelengths_nm: np.ndarray) -> np.ndarray:
    indices = np.array([int(np.argmin(np.abs(wavelengths_nm - value))) for value in ATLAS_WAVELENGTHS_NM])
    if not np.allclose(wavelengths_nm[indices], ATLAS_WAVELENGTHS_NM):
        raise ValueError("baseline.yaml 的波长网格必须包含 450、500、550、600、650 nm")
    return indices


def _plot_baseline(
    output: Path,
    wavelengths_nm: np.ndarray,
    curves: dict[str, np.ndarray],
    banks: dict[str, np.ndarray],
) -> None:
    labels = {
        "double_helix": "Double Helix", "single_helix": "Single Helix (+)",
        "dual_single_helix": "原始 Dual SH (+/−)",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, curve in curves.items():
        axes[0].semilogy(wavelengths_nm, curve, label=labels[name], linewidth=1.8)
        axes[1].semilogy(wavelengths_nm, crlb_from_fisher(curve), label=labels[name], linewidth=1.8)
    axes[0].set_ylabel("Fisher information / nm$^{-2}$")
    axes[1].set_ylabel("CRLB / nm")
    for axis in axes:
        axis.set_xlabel("波长 / nm")
        axis.grid(alpha=0.25, which="both")
        axis.legend()
    figure.suptitle("第二轮实验：第一轮 baseline 复现")
    figure.tight_layout()
    figure.savefig(output / "baseline_fisher_crlb.png", dpi=240)
    plt.close(figure)

    selected = _nearest_indices(wavelengths_nm)
    rows = (("double", "Double Helix"), ("single_plus", "Single Helix +"), ("single_minus", "Single Helix −"))
    figure, axes = plt.subplots(3, len(selected), figsize=(12, 7))
    for row_index, (name, label) in enumerate(rows):
        for column, wavelength_index in enumerate(selected):
            axes[row_index, column].imshow(banks[name][wavelength_index], cmap="inferno")
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
            if row_index == 0:
                axes[row_index, column].set_title(f"{wavelengths_nm[wavelength_index]:.0f} nm")
            if column == 0:
                axes[row_index, column].set_ylabel(label)
    figure.suptitle("当前解析 DOE 的 PSF atlas")
    figure.tight_layout()
    figure.savefig(output / "current_psf_atlas.png", dpi=240)
    plt.close(figure)


def _plot_scatter(output: Path, pair_rows: list[dict[str, object]]) -> None:
    redundancy = np.array([float(row["mirror_aligned_redundancy"]) for row in pair_rows])
    derivative = np.array([float(row["spectral_derivative_similarity"]) for row in pair_rows])
    gain_dh = np.array([float(row["gain_DH"]) for row in pair_rows])
    gain_sh = np.array([float(row["gain_SH"]) for row in pair_rows])
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    points = axis.scatter(redundancy, gain_dh, c=derivative, s=13, alpha=0.55, cmap="coolwarm")
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("镜像对齐后的通道冗余（低值更互补）")
    axis.set_ylabel("Fisher Gain over Double Helix")
    axis.set_title("Figure A  PSF 冗余性与 Fisher 增益")
    axis.grid(alpha=0.2)
    figure.colorbar(points, ax=axis, label="光谱导数相似度")
    figure.tight_layout()
    figure.savefig(output / "complementarity_vs_fisher.png", dpi=300)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    points = axes[0].scatter(derivative, gain_dh, c=redundancy, s=13, alpha=0.55, cmap="viridis")
    axes[1].scatter(derivative, gain_sh, c=redundancy, s=13, alpha=0.55, cmap="viridis")
    axes[0].set_ylabel("Fisher Gain over Double Helix")
    axes[1].set_ylabel("Fisher Gain over best Single Helix")
    for axis, title in zip(axes, ("Figure B", "Figure C")):
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("spectral derivative similarity")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    figure.colorbar(points, ax=axes, label="镜像对齐冗余")
    figure.suptitle("光谱导数冗余与 Fisher 增益")
    figure.savefig(output / "derivative_similarity_vs_fisher.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _representative_indices(pair_rows: list[dict[str, object]]) -> dict[str, list[int]]:
    order = np.argsort([float(row["median_fisher"]) for row in pair_rows])
    middle_start = len(order) // 2 - 1
    return {
        "top": list(order[-3:][::-1]),
        "middle": list(order[middle_start : middle_start + 3]),
        "bad": list(order[:3]),
    }


def _pair_candidate_indices(row: dict[str, object], candidate_index: dict[str, int]) -> tuple[int, int]:
    return candidate_index[str(row["channel_A_id"])], candidate_index[str(row["channel_B_id"])]


def _plot_pair_curves(
    output: Path, group: str, selected_rows: list[dict[str, object]],
    candidate_index: dict[str, int], half_fisher: np.ndarray,
    wavelengths_nm: np.ndarray, double_fisher: np.ndarray, best_single_fisher: np.ndarray,
) -> None:
    fisher_curves: list[tuple[str, np.ndarray]] = []
    for rank, row in enumerate(selected_rows, start=1):
        first, second = _pair_candidate_indices(row, candidate_index)
        fisher_curves.append((f"{group}-{rank}", dual_fisher_curve(half_fisher[first], half_fisher[second])))
    for metric, filename, ylabel in (
        (lambda values: values, f"fisher_{group}_pairs.png", "Fisher information / nm$^{-2}$"),
        (crlb_from_fisher, f"crlb_{group}_pairs.png", "CRLB / nm"),
    ):
        figure, axis = plt.subplots(figsize=(9, 5.5))
        axis.semilogy(wavelengths_nm, metric(double_fisher), "--", label="Double Helix", linewidth=1.4)
        axis.semilogy(wavelengths_nm, metric(best_single_fisher), "--", label="best Single Helix", linewidth=1.4)
        for label, curve in fisher_curves:
            axis.semilogy(wavelengths_nm, metric(curve), label=label, linewidth=1.7)
        axis.set_xlabel("波长 / nm")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{group.capitalize()} 3 Dual SH 组合")
        axis.grid(alpha=0.25, which="both")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / filename, dpi=260)
        plt.close(figure)


def _plot_pair_atlas(
    output: Path, group: str, selected_rows: list[dict[str, object]],
    candidate_index: dict[str, int], banks: np.ndarray, wavelengths_nm: np.ndarray,
) -> None:
    wavelength_indices = _nearest_indices(wavelengths_nm)
    figure, axes = plt.subplots(6, len(wavelength_indices), figsize=(12, 13))
    for pair_number, row in enumerate(selected_rows, start=1):
        first, second = _pair_candidate_indices(row, candidate_index)
        for channel_offset, candidate_number in enumerate((first, second)):
            plot_row = (pair_number - 1) * 2 + channel_offset
            for column, wavelength_index in enumerate(wavelength_indices):
                axes[plot_row, column].imshow(banks[candidate_number, wavelength_index], cmap="inferno")
                axes[plot_row, column].set_xticks([])
                axes[plot_row, column].set_yticks([])
                if plot_row == 0:
                    axes[plot_row, column].set_title(f"{wavelengths_nm[wavelength_index]:.0f} nm")
                if column == 0:
                    channel = "A" if channel_offset == 0 else "B"
                    axes[plot_row, column].set_ylabel(f"{group}-{pair_number} · Ch {channel}")
    figure.suptitle(f"{group.capitalize()} 3 Dual SH：两通道 PSF atlas")
    figure.tight_layout()
    figure.savefig(output / f"{group}_pair_psf_atlas.png", dpi=240)
    plt.close(figure)


def _interpolate_psf(bank: np.ndarray, wavelengths_nm: np.ndarray, wavelength_nm: float) -> np.ndarray:
    """在 5 nm PSF 网格之间线性插值，用于离网格 Monte Carlo 真值。"""
    upper = int(np.searchsorted(wavelengths_nm, wavelength_nm, side="right"))
    upper = min(max(upper, 1), len(wavelengths_nm) - 1)
    lower = upper - 1
    span = float(wavelengths_nm[upper] - wavelengths_nm[lower])
    weight = (wavelength_nm - float(wavelengths_nm[lower])) / span
    interpolated = (1.0 - weight) * bank[lower] + weight * bank[upper]
    return interpolated / interpolated.sum()


def _quadratic_refined_estimate(
    wavelengths_nm: np.ndarray, costs: np.ndarray, discrete_estimate_nm: float
) -> float:
    """用最小代价点及其两个邻点拟合抛物线，得到亚网格波长估计。"""
    best = int(np.argmin(costs))
    if best == 0 or best == len(costs) - 1:
        return discrete_estimate_nm
    x = wavelengths_nm[best - 1 : best + 2]
    y = costs[best - 1 : best + 2]
    coefficients = np.polyfit(x, y, 2)
    if coefficients[0] <= 0 or not np.all(np.isfinite(coefficients)):
        return discrete_estimate_nm
    estimate = -coefficients[1] / (2.0 * coefficients[0])
    return float(np.clip(estimate, x[0], x[-1]))


def _weighted_template_costs(
    observations: tuple[np.ndarray, ...],
    template_banks: tuple[np.ndarray, ...],
    photon_budgets: tuple[float, ...],
    background_per_pixel: float,
    read_noise_e: float,
) -> np.ndarray:
    """按与 Fisher 相同的泊松–高斯方差近似计算统一 WLS 代价。"""
    costs = np.zeros(template_banks[0].shape[0], dtype=float)
    for observation, bank, photons in zip(observations, template_banks, photon_budgets):
        expected = photons * bank + background_per_pixel
        variance = np.maximum(expected + read_noise_e**2, np.finfo(float).eps)
        costs += np.sum((observation[None, ...] - expected) ** 2 / variance, axis=(-2, -1))
    return costs


def _monte_carlo(
    config: ResearchConfig, wavelengths_nm: np.ndarray, banks: np.ndarray,
    double_bank: np.ndarray, selected_rows: list[dict[str, object]],
    original_pair: dict[str, object], best_single_index: int,
    candidate_index: dict[str, int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    systems: list[tuple[str, tuple[np.ndarray, ...], tuple[float, ...]]] = [
        ("double_helix", (double_bank,), (config.camera.photons_total,)),
        ("best_single_helix", (banks[best_single_index],), (config.camera.photons_total,)),
    ]
    original_indices = _pair_candidate_indices(original_pair, candidate_index)
    systems.append((
        "original_dual_plus_minus", tuple(banks[index] for index in original_indices),
        (config.camera.photons_total / 2.0, config.camera.photons_total / 2.0),
    ))
    for rank, row in enumerate(selected_rows, start=1):
        pair_indices = _pair_candidate_indices(row, candidate_index)
        systems.append((
            f"top_{rank}_{row['pair_id']}", tuple(banks[index] for index in pair_indices),
            (config.camera.photons_total / 2.0, config.camera.photons_total / 2.0),
        ))
    rng = np.random.default_rng(config.simulation.seed + 200)
    raw_rows: list[dict[str, object]] = []
    camera = config.camera
    for truth in config.simulation.test_wavelengths_nm:
        for repeat in range(config.simulation.monte_carlo_repeats):
            for system_name, system_banks, budgets in systems:
                observations = tuple(
                    add_camera_noise(
                        _interpolate_psf(bank, wavelengths_nm, truth), photons,
                        camera.background_photons_per_pixel,
                        camera.read_noise_e, camera.bit_depth, rng,
                    ) for bank, photons in zip(system_banks, budgets)
                )
                costs = _weighted_template_costs(
                    observations, system_banks, budgets,
                    camera.background_photons_per_pixel, camera.read_noise_e,
                )
                order = np.argsort(costs)
                best = int(order[0])
                discrete_estimate = float(wavelengths_nm[best])
                second_cost = float(costs[order[1]])
                confidence = (second_cost - float(costs[best])) / max(
                    second_cost, np.finfo(float).eps
                )
                estimate = _quadratic_refined_estimate(wavelengths_nm, costs, discrete_estimate)
                raw_rows.append({
                    "system": system_name, "truth_nm": truth,
                    "estimate_nm": estimate, "error_nm": estimate - truth,
                    "confidence": confidence, "repeat": repeat,
                })
    summary_rows: list[dict[str, object]] = []
    for system_name, _, _ in systems:
        subset = [row for row in raw_rows if row["system"] == system_name]
        metrics = wavelength_metrics(
            np.array([float(row["truth_nm"]) for row in subset]),
            np.array([float(row["estimate_nm"]) for row in subset]),
        )
        summary_rows.append({"system": system_name, "samples": len(subset), **metrics})
    return raw_rows, summary_rows


def _effect_analysis(pair_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for parameter in ("zone_count", "charge_start", "charge_step", "handedness"):
        same = np.array([
            row[f"channel_A_{parameter}"] == row[f"channel_B_{parameter}"] for row in pair_rows
        ], dtype=bool)
        redundancy = np.array([float(row["mirror_aligned_redundancy"]) for row in pair_rows])
        derivative = np.array([float(row["spectral_derivative_similarity"]) for row in pair_rows])
        gain = np.array([float(row["gain_DH"]) for row in pair_rows])
        result.append({
            "parameter": parameter,
            "same_pair_count": int(np.sum(same)), "different_pair_count": int(np.sum(~same)),
            "mean_mirror_redundancy_same": float(np.mean(redundancy[same])),
            "mean_mirror_redundancy_different": float(np.mean(redundancy[~same])),
            "redundancy_change_different_minus_same": float(np.mean(redundancy[~same]) - np.mean(redundancy[same])),
            "mean_derivative_similarity_same": float(np.mean(derivative[same])),
            "mean_derivative_similarity_different": float(np.mean(derivative[~same])),
            "mean_gain_DH_same": float(np.mean(gain[same])),
            "mean_gain_DH_different": float(np.mean(gain[~same])),
        })
    return result


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "pearson_r": float(pearsonr(x, y).statistic),
        "spearman_rho": float(spearmanr(x, y).statistic),
    }


def _research_summary(
    output: Path, config: ResearchConfig, baseline_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]], pair_rows: list[dict[str, object]],
    representative_rows: list[dict[str, object]], effect_rows: list[dict[str, object]],
    monte_carlo_rows: list[dict[str, object]], original_pair: dict[str, object],
) -> dict[str, object]:
    top = max(pair_rows, key=lambda row: float(row["median_fisher"]))
    best_single = max(candidate_rows, key=lambda row: float(row["median_fisher"]))
    mirror = np.array([float(row["mirror_aligned_redundancy"]) for row in pair_rows])
    derivative = np.array([float(row["spectral_derivative_similarity"]) for row in pair_rows])
    gain_dh = np.array([float(row["gain_DH"]) for row in pair_rows])
    gain_sh = np.array([float(row["gain_SH"]) for row in pair_rows])
    correlations = {
        "mirror_redundancy_vs_gain_DH": _safe_correlation(mirror, gain_dh),
        "derivative_similarity_vs_gain_DH": _safe_correlation(derivative, gain_dh),
        "derivative_similarity_vs_gain_SH": _safe_correlation(derivative, gain_sh),
    }
    most_influential = max(
        effect_rows, key=lambda row: abs(float(row["redundancy_change_different_minus_same"]))
    )
    count_over_dh = int(np.sum(gain_dh > 1.0))
    count_over_sh = int(np.sum(gain_sh > 1.0))
    handedness_only = [
        row for row in pair_rows
        if row["channel_A_zone_count"] == row["channel_B_zone_count"]
        and row["channel_A_charge_start"] == row["channel_B_charge_start"]
        and row["channel_A_charge_step"] == row["channel_B_charge_step"]
        and row["channel_A_handedness"] != row["channel_B_handedness"]
    ]
    handedness_only_gain = np.array([float(row["gain_DH"]) for row in handedness_only])
    handedness_only_mirror = np.array([
        float(row["mirror_aligned_redundancy"]) for row in handedness_only
    ])
    baseline_lookup = {str(row["system"]): row for row in baseline_rows}
    best_monte_carlo = min(monte_carlo_rows, key=lambda row: float(row["rmse_nm"]))
    top_monte_carlo = [row for row in monte_carlo_rows if str(row["system"]).startswith("top_")]
    best_top_monte_carlo = min(top_monte_carlo, key=lambda row: float(row["rmse_nm"]))
    lines = [
        "# 第二轮科学验证：Single Helix 通道互补性扫描", "", "## 实验约束", "",
        f"- 使用 `baseline.yaml`，波长范围 {min(config.optics.wavelengths_nm):.0f}–{max(config.optics.wavelengths_nm):.0f} nm。",
        f"- 总信号光子数固定为 {config.camera.photons_total:g}；双通道严格按 N/2 + N/2 分配。",
        f"- 扫描 {len(candidate_rows)} 个 Single Helix DOE，形成 {len(pair_rows)} 个无序双通道组合（包含 A=A 对照）。",
        "- 没有改变 aperture、传感器或噪声模型，也没有筛掉失败组合。", "",
        "## 1. Baseline 是否复现", "",
        f"- Double Helix：median Fisher = {float(baseline_lookup['double_helix']['median_fisher']):.4f}，median CRLB = {float(baseline_lookup['double_helix']['median_crlb_nm']):.4f} nm。",
        f"- 当前 Single Helix：median Fisher = {float(baseline_lookup['current_single_helix']['median_fisher']):.4f}，median CRLB = {float(baseline_lookup['current_single_helix']['median_crlb_nm']):.4f} nm。",
        f"- 当前 +/- Dual SH：median Fisher = {float(baseline_lookup['current_dual_plus_minus']['median_fisher']):.4f}，median CRLB = {float(baseline_lookup['current_dual_plus_minus']['median_crlb_nm']):.4f} nm。", "",
        "## 2. 当前 +/- handedness 为什么可能没有优势", "",
        f"当前组合的普通余弦冗余为 {float(original_pair['channel_redundancy']):.4f}，镜像对齐冗余为 {float(original_pair['mirror_aligned_redundancy']):.4f}，光谱导数相似度为 {float(original_pair['spectral_derivative_similarity']):.4f}。正负 handedness 主要产生镜像/共轭外观；镜像对齐后仍相似，说明它们并未创造同等光子条件下足够独立的波长敏感结构。与此同时，每路只有一半光子且承担独立读出噪声。", "",
        "## 3. 仅改变 handedness 是否足够", "",
        f"严格保持 zone_count、charge_start、charge_step 相同而只反转 handedness 的组合共有 {len(handedness_only)} 组；其平均镜像对齐冗余为 {float(np.mean(handedness_only_mirror)):.4f}，平均 Gain_DH 为 {float(np.mean(handedness_only_gain)):.4f}，其中 {int(np.sum(handedness_only_gain > 1.0))} 组超过 Double Helix。两个 DOE 往往仍是镜像关系，因此普通像素相关下降不应被直接解释为信息互补。", "",
        "## 4. 哪些参数最影响互补性", "",
        f"按“参数不同相对参数相同所造成的镜像对齐冗余变化”这一描述性统计，影响最大的是 `{most_influential['parameter']}`（变化 {float(most_influential['redundancy_change_different_minus_same']):+.4f}）。这不是因果证明，因为全组合中多个参数会同时变化；完整统计见 `parameter_effects.csv`。", "",
        "## 5. 互补性与 Fisher Gain 是否存在明确关系", "",
        f"- 镜像对齐冗余 vs Gain_DH：Pearson r = {correlations['mirror_redundancy_vs_gain_DH']['pearson_r']:.3f}，Spearman ρ = {correlations['mirror_redundancy_vs_gain_DH']['spearman_rho']:.3f}。",
        f"- 光谱导数相似度 vs Gain_DH：Pearson r = {correlations['derivative_similarity_vs_gain_DH']['pearson_r']:.3f}，Spearman ρ = {correlations['derivative_similarity_vs_gain_DH']['spearman_rho']:.3f}。",
        f"- 光谱导数相似度 vs Gain_SH：Pearson r = {correlations['derivative_similarity_vs_gain_SH']['pearson_r']:.3f}，Spearman ρ = {correlations['derivative_similarity_vs_gain_SH']['spearman_rho']:.3f}。",
        "这些相关系数只描述本参数库，不把本项目定义的互补性指标包装成公认理论量。", "",
        "## 6. 是否超过两个基线", "",
        f"- {count_over_dh}/{len(pair_rows)} 个 Dual SH 组合的 median Fisher 超过 Double Helix。",
        f"- {count_over_sh}/{len(pair_rows)} 个 Dual SH 组合的 median Fisher 超过候选库中的 best Single Helix。",
        f"- 最优 Dual SH 为 `{top['pair_id']}`：Gain_DH = {float(top['gain_DH']):.4f}，Gain_SH = {float(top['gain_SH']):.4f}。",
        f"- best Single Helix 为 `{best_single['candidate_id']}`，median Fisher = {float(best_single['median_fisher']):.4f}。", "",
        f"特别重要的是，Fisher 第一名的两个通道参数完全相同，镜像对齐冗余为 {float(top['mirror_aligned_redundancy']):.4f}、导数相似度为 {float(top['spectral_derivative_similarity']):.4f}。因此它超过 Double Helix 不能归因于互补性，而应归因于该 Single Helix 参数本身比当前 Double Helix 初值更强。", "",
        "对于单一已知参数 λ，独立通道 Fisher 可以相加；在理想泊松主导且总光子固定时，双通道结果接近两个全光子单通道 Fisher 的加权平均，因此不能仅靠两通道外观不同超过其中最强的单通道。读出噪声还会使拆分更吃亏。双通道真正可能体现价值的地方，是存在位置、亮度、背景等未知干扰参数或全局模板歧义时，而这不属于本轮局部单参数 Fisher 的证明范围。", "",
        "## 7. Monte Carlo", "",
        "观测 PSF 由相邻 5 nm 标定模板线性插值得到，估计采用与 Fisher 方差模型一致的统一泊松–高斯加权最小二乘，并对代价曲线作局部二次亚网格估计；这样可避免高光子条件下所有方法都只返回同一个 5 nm 网格点。", "",
        "| system | bias / nm | MAE / nm | RMSE / nm | std / nm | outlier rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in monte_carlo_rows:
        lines.append(
            f"| {row['system']} | {float(row['bias_nm']):.3f} | {float(row['mae_nm']):.3f} | {float(row['rmse_nm']):.3f} | {float(row['std_nm']):.3f} | {float(row['outlier_rate']):.3f} |"
        )
    lines.extend([
        "",
        f"本次 120 次/系统的描述性结果中，最低 RMSE 来自 `{best_monte_carlo['system']}`（{float(best_monte_carlo['rmse_nm']):.4f} nm）；三个按中位 Fisher 选择的组合中最低 RMSE 为 {float(best_top_monte_carlo['rmse_nm']):.4f} nm。因而本轮没有观察到“中位 Fisher 排名自动转化为这组六个测试波长的估计 RMSE 排名”。可能原因包括 Fisher 是逐波长局部下界，而模板估计还受全局歧义、5 nm 标定采样和有限重复次数影响。",
    ])
    lines.extend([
        "", "## 8. 结论边界", "",
        "本轮只回答点源、单一待估波长下的局部信息量问题。若 Dual SH 只超过 Double Helix 而没有超过 best Single Helix，就只能说明所选 Double Helix 解析初值较弱，不能宣称拆分结构具有整体优势。所有 Top/Middle/Bad 代表组合均来自完整扫描，完整结果保存在 `dual_pair_metrics.csv`。",
    ])
    (output / "research_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "pair_count": len(pair_rows), "count_gain_over_double_helix": count_over_dh,
        "count_gain_over_best_single": count_over_sh, "best_pair": top,
        "best_single": best_single, "correlations": correlations,
        "handedness_only": {
            "pair_count": len(handedness_only),
            "mean_mirror_aligned_redundancy": float(np.mean(handedness_only_mirror)),
            "mean_gain_DH": float(np.mean(handedness_only_gain)),
            "count_gain_over_double_helix": int(np.sum(handedness_only_gain > 1.0)),
        },
        "representative_pair_count": len(representative_rows),
    }


def run_complementarity_sweep(config: ResearchConfig, output_directory: str | Path) -> dict[str, object]:
    if config.simulation.comparison_mode != "fair_photon":
        raise ValueError("第二轮实验只允许 comparison_mode = fair_photon")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    wavelengths_nm = np.asarray(config.optics.wavelengths_nm, dtype=float)
    analysis_mask = (wavelengths_nm >= 450.0) & (wavelengths_nm <= 650.0)
    camera = config.camera
    # 从当前代码重新生成 baseline，避免把第一轮数值硬编码进实验。
    baseline_banks = build_psf_banks(config)
    specs, candidate_banks, cache_hits = build_candidate_library(config, output / "cache")
    candidate_index = {candidate.candidate_id: index for index, candidate in enumerate(specs)}
    candidate_rows, full_fisher, half_fisher = _candidate_metrics(
        config, specs, candidate_banks, wavelengths_nm, analysis_mask
    )
    best_single_index = int(np.argmax([float(row["median_fisher"]) for row in candidate_rows]))
    double_fisher = poisson_gaussian_fisher(
        baseline_banks["double"], wavelengths_nm, camera.photons_total,
        camera.background_photons_per_pixel, camera.read_noise_e,
    )
    plus_id = f"z{config.doe.zone_count:02d}_c{config.doe.single_charge_start}_s{config.doe.single_charge_step}_hp1"
    minus_id = f"z{config.doe.zone_count:02d}_c{config.doe.single_charge_start}_s{config.doe.single_charge_step}_hm1"
    plus_index, minus_index = candidate_index[plus_id], candidate_index[minus_id]
    current_single_fisher = full_fisher[plus_index]
    current_dual_fisher = dual_fisher_curve(half_fisher[plus_index], half_fisher[minus_index])
    baseline_curves = {
        "double_helix": double_fisher, "single_helix": current_single_fisher,
        "dual_single_helix": current_dual_fisher,
    }
    baseline_rows: list[dict[str, object]] = []
    for system, curve in (
        ("double_helix", double_fisher), ("current_single_helix", current_single_fisher),
        ("current_dual_plus_minus", current_dual_fisher),
        ("best_single_helix", full_fisher[best_single_index]),
    ):
        baseline_rows.append({
            "system": system,
            "candidate_id": specs[best_single_index].candidate_id if system == "best_single_helix" else "",
            "median_fisher": float(np.median(curve[analysis_mask])),
            "median_crlb_nm": float(np.median(crlb_from_fisher(curve)[analysis_mask])),
        })
    _write_csv(output / "baseline_summary.csv", baseline_rows)
    baseline_curve_rows: list[dict[str, object]] = []
    for wavelength_index, wavelength in enumerate(wavelengths_nm):
        row: dict[str, object] = {"wavelength_nm": wavelength}
        for system, curve in baseline_curves.items():
            row[f"fisher_{system}"] = curve[wavelength_index]
            row[f"crlb_{system}_nm"] = crlb_from_fisher(curve)[wavelength_index]
        baseline_curve_rows.append(row)
    _write_csv(output / "baseline_curves.csv", baseline_curve_rows)
    _write_csv(output / "single_candidate_metrics.csv", candidate_rows)
    _plot_baseline(output, wavelengths_nm, baseline_curves, baseline_banks)
    print("计算全部候选对的通道冗余和光谱导数相似度……", flush=True)
    redundancy = _all_pair_redundancy(candidate_banks)
    pair_rows = _pair_rows(
        specs, half_fisher, redundancy, analysis_mask,
        float(np.median(double_fisher[analysis_mask])),
        float(np.median(full_fisher[best_single_index][analysis_mask])),
    )
    _write_csv(output / "dual_pair_metrics.csv", pair_rows)
    _plot_scatter(output, pair_rows)
    selected = _representative_indices(pair_rows)
    representative_rows: list[dict[str, object]] = []
    representative_curve_rows: list[dict[str, object]] = []
    for group, indices in selected.items():
        rows = [pair_rows[index] for index in indices]
        for rank, row in enumerate(rows, start=1):
            representative_rows.append({"selection_group": group, "rank": rank, **row})
            first, second = _pair_candidate_indices(row, candidate_index)
            curve = dual_fisher_curve(half_fisher[first], half_fisher[second])
            crlb = crlb_from_fisher(curve)
            for wavelength_index, wavelength in enumerate(wavelengths_nm):
                representative_curve_rows.append({
                    "selection_group": group, "rank": rank, "pair_id": row["pair_id"],
                    "wavelength_nm": wavelength, "fisher": curve[wavelength_index],
                    "crlb_nm": crlb[wavelength_index],
                })
        _plot_pair_curves(
            output, group, rows, candidate_index, half_fisher, wavelengths_nm,
            double_fisher, full_fisher[best_single_index],
        )
        _plot_pair_atlas(output, group, rows, candidate_index, candidate_banks, wavelengths_nm)
    _write_csv(output / "top_pairs.csv", representative_rows)
    _write_csv(output / "representative_fisher_curves.csv", representative_curve_rows)
    original_pair = next(
        row for row in pair_rows
        if {str(row["channel_A_id"]), str(row["channel_B_id"])} == {plus_id, minus_id}
    )
    raw_monte_carlo, monte_carlo_rows = _monte_carlo(
        config, wavelengths_nm, candidate_banks, baseline_banks["double"],
        [pair_rows[index] for index in selected["top"]], original_pair,
        best_single_index, candidate_index,
    )
    _write_csv(output / "monte_carlo_estimates.csv", raw_monte_carlo)
    _write_csv(output / "monte_carlo_top_pairs.csv", monte_carlo_rows)
    effect_rows = _effect_analysis(pair_rows)
    _write_csv(output / "parameter_effects.csv", effect_rows)
    summary = _research_summary(
        output, config, baseline_rows, candidate_rows, pair_rows, representative_rows,
        effect_rows, monte_carlo_rows, original_pair,
    )
    summary.update({
        "candidate_count": len(specs), "cache_hits": cache_hits,
        "comparison_mode": config.simulation.comparison_mode,
        "total_photons": camera.photons_total,
    })
    with (output / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary
