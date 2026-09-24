from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from .config import ResearchConfig
from .estimation import crlb_from_fisher, multichannel_fisher
from .pipeline import build_psf_banks
from .plotting import configure_chinese_font
from .propagation import psf_centroid, second_moment_features


configure_chinese_font()


SYSTEM_ORDER = ("double_helix", "single_helix", "dual_single_helix")
SYSTEM_LABELS = {
    "double_helix": "单 DOE 双螺旋",
    "single_helix": "单通道单螺旋",
    "dual_single_helix": "双通道单螺旋",
}
SELECTED_WAVELENGTHS_NM = (450.0, 500.0, 550.0, 600.0, 650.0)


def _nearest_indices(wavelengths_nm: np.ndarray, selected_nm: tuple[float, ...]) -> np.ndarray:
    indices = np.array([int(np.argmin(np.abs(wavelengths_nm - value))) for value in selected_nm])
    actual = wavelengths_nm[indices]
    if not np.allclose(actual, np.asarray(selected_nm), atol=1e-9):
        raise ValueError(f"配置波长网格缺少指定波长，最近值为: {actual.tolist()}")
    return indices


def system_banks(banks: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, ...]]:
    """把底层 PSF 库映射为三种待比较系统。"""
    return {
        "double_helix": (banks["double"],),
        "single_helix": (banks["single_plus"],),
        "dual_single_helix": (banks["single_plus"], banks["single_minus"]),
    }


def system_photon_budgets(total_photons: float) -> dict[str, tuple[float, ...]]:
    """三系统使用相同总探测信号光子；双通道在两路间均分。"""
    return {
        "double_helix": (total_photons,),
        "single_helix": (total_photons,),
        "dual_single_helix": (total_photons / 2.0, total_photons / 2.0),
    }


def system_channel_fractions() -> dict[str, tuple[float, ...]]:
    return system_photon_budgets(1.0)


def observation_vectors(
    banks_by_system: dict[str, tuple[np.ndarray, ...]],
    selected_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """构造各波长的无噪声系统观测向量；双通道按通道拼接。"""
    fractions = system_channel_fractions()
    result: dict[str, np.ndarray] = {}
    for system in SYSTEM_ORDER:
        rows = []
        for index in selected_indices:
            row = np.concatenate(
                [fraction * bank[index].ravel() for bank, fraction in zip(banks_by_system[system], fractions[system])]
            )
            rows.append(row)
        result[system] = np.stack(rows)
    return result


def pearson_correlation_matrix(vectors: np.ndarray) -> np.ndarray:
    centered = vectors - vectors.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = centered / np.maximum(norms, np.finfo(float).eps)
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, np.finfo(float).eps)
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def _rotation_angle_deg(system: str, channels: tuple[np.ndarray, ...]) -> float:
    """按系统物理含义定义旋转角，而不是混用同一个图像矩。"""
    if system == "double_helix":
        angle = np.rad2deg(second_moment_features(channels[0])["orientation_rad"])
        return float(np.mod(angle, 180.0))

    if system == "single_helix":
        x0, y0 = psf_centroid(channels[0])
        center_x = (channels[0].shape[1] - 1) / 2.0
        center_y = (channels[0].shape[0] - 1) / 2.0
        return float(np.mod(np.rad2deg(np.arctan2(y0 - center_y, x0 - center_x)), 360.0))

    plus_x, plus_y = psf_centroid(channels[0])
    minus_x, minus_y = psf_centroid(channels[1])
    return float(np.mod(np.rad2deg(np.arctan2(plus_y - minus_y, plus_x - minus_x)), 360.0))


def _intensity_features(distribution: np.ndarray) -> dict[str, float]:
    normalized = np.clip(distribution.astype(float), 0.0, None)
    normalized /= max(float(normalized.sum()), np.finfo(float).eps)
    y, x = np.indices(normalized.shape[-2:], dtype=float)
    x0, y0 = psf_centroid(normalized)
    radius = np.hypot(x - x0, y - y0)
    return {
        "peak_fraction": float(normalized.max()),
        "effective_area_px": float(1.0 / max(np.sum(normalized**2), np.finfo(float).eps)),
        "encircled_energy_r5": float(normalized[radius <= 5.0].sum()),
    }


def extract_psf_features(
    banks_by_system: dict[str, tuple[np.ndarray, ...]],
    wavelengths_nm: np.ndarray,
    selected_indices: np.ndarray,
) -> list[dict[str, float | str]]:
    records: list[dict[str, float | str]] = []
    fractions = system_channel_fractions()
    for system in SYSTEM_ORDER:
        for index in selected_indices:
            channels = tuple(bank[index] for bank in banks_by_system[system])
            system_angle = _rotation_angle_deg(system, channels)
            for channel_number, (channel, fraction) in enumerate(zip(channels, fractions[system]), start=1):
                centroid_x, centroid_y = psf_centroid(channel)
                intensity = _intensity_features(channel)
                records.append(
                    {
                        "system": system,
                        "channel": f"ch{channel_number}",
                        "wavelength_nm": float(wavelengths_nm[index]),
                        "centroid_x_px": centroid_x,
                        "centroid_y_px": centroid_y,
                        "rotation_angle_deg": system_angle,
                        "photon_fraction": fraction,
                        **intensity,
                    }
                )
    return records


def compute_fisher_curves(
    config: ResearchConfig,
    banks_by_system: dict[str, tuple[np.ndarray, ...]],
    wavelengths_nm: np.ndarray,
) -> dict[str, np.ndarray]:
    budgets = system_photon_budgets(config.camera.photons_total)
    return {
        system: multichannel_fisher(
            banks_by_system[system],
            wavelengths_nm,
            budgets[system],
            config.camera.background_photons_per_pixel,
            config.camera.read_noise_e,
        )
        for system in SYSTEM_ORDER
    }


def _draw_box(axis: plt.Axes, xy: tuple[float, float], text: str, color: str, width: float = 0.18) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x - width / 2.0, y - 0.07), width, 0.14,
        boxstyle="round,pad=0.015", facecolor=color, edgecolor="#263238", linewidth=1.2,
    )
    axis.add_patch(patch)
    axis.text(x, y, text, ha="center", va="center", fontsize=9)


def _draw_arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color="#37474f", linewidth=1.3))


def plot_system_schematic(output: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(11, 7))
    colors = {"source": "#eceff1", "doe": "#ffe0b2", "sensor": "#bbdefb", "fusion": "#c8e6c9"}
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")

    _draw_box(axes[0], (0.12, 0.5), "窄带点源\nλ", colors["source"])
    _draw_box(axes[0], (0.43, 0.5), "DOE-DH\n同口径", colors["doe"])
    _draw_box(axes[0], (0.76, 0.5), "单色传感器\n100%光子", colors["sensor"], width=0.22)
    _draw_arrow(axes[0], (0.22, 0.5), (0.33, 0.5))
    _draw_arrow(axes[0], (0.53, 0.5), (0.64, 0.5))
    axes[0].set_title("Baseline 1：单 DOE Double Helix PSF", loc="left", fontweight="bold")

    _draw_box(axes[1], (0.12, 0.5), "窄带点源\nλ", colors["source"])
    _draw_box(axes[1], (0.43, 0.5), "DOE-SH+\n同口径", colors["doe"])
    _draw_box(axes[1], (0.76, 0.5), "单色传感器\n100%光子", colors["sensor"], width=0.22)
    _draw_arrow(axes[1], (0.22, 0.5), (0.33, 0.5))
    _draw_arrow(axes[1], (0.53, 0.5), (0.64, 0.5))
    axes[1].set_title("Baseline 2：单通道 Single Helix PSF", loc="left", fontweight="bold")

    _draw_box(axes[2], (0.08, 0.5), "窄带点源\nλ", colors["source"], width=0.14)
    _draw_box(axes[2], (0.26, 0.5), "50:50\n分光", "#e1bee7", width=0.12)
    _draw_box(axes[2], (0.48, 0.7), "DOE-SH+", colors["doe"], width=0.14)
    _draw_box(axes[2], (0.48, 0.3), "DOE-SH−", colors["doe"], width=0.14)
    _draw_box(axes[2], (0.67, 0.7), "Sensor 1\n50%光子", colors["sensor"], width=0.16)
    _draw_box(axes[2], (0.67, 0.3), "Sensor 2\n50%光子", colors["sensor"], width=0.16)
    _draw_box(axes[2], (0.9, 0.5), "双通道\n联合观测", colors["fusion"], width=0.16)
    _draw_arrow(axes[2], (0.16, 0.5), (0.19, 0.5))
    _draw_arrow(axes[2], (0.32, 0.53), (0.4, 0.67))
    _draw_arrow(axes[2], (0.32, 0.47), (0.4, 0.33))
    _draw_arrow(axes[2], (0.55, 0.7), (0.59, 0.7))
    _draw_arrow(axes[2], (0.55, 0.3), (0.59, 0.3))
    _draw_arrow(axes[2], (0.75, 0.68), (0.82, 0.55))
    _draw_arrow(axes[2], (0.75, 0.32), (0.82, 0.45))
    axes[2].set_title("Proposed：双通道共轭 Single Helix PSF（总光子不变）", loc="left", fontweight="bold")

    figure.suptitle("Figure 1  三种光谱编码系统的最小验证结构", fontsize=15, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output / "figure1_system_schematic.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "figure1_system_schematic.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_wavelength_psfs(
    output: Path,
    banks: dict[str, np.ndarray],
    wavelengths_nm: np.ndarray,
    selected_indices: np.ndarray,
) -> None:
    rows = (
        ("double", 1.0, "Double Helix"),
        ("single_plus", 1.0, "Single Helix"),
        ("single_plus", 0.5, "Dual SH · Channel 1"),
        ("single_minus", 0.5, "Dual SH · Channel 2"),
    )
    distributions = [fraction * banks[name][index] for name, fraction, _ in rows for index in selected_indices]
    vmax = float(max(image.max() for image in distributions))
    figure, axes = plt.subplots(len(rows), len(selected_indices), figsize=(13, 9), constrained_layout=True)
    last_image = None
    for row, (name, fraction, label) in enumerate(rows):
        for column, index in enumerate(selected_indices):
            last_image = axes[row, column].imshow(
                fraction * banks[name][index], cmap="inferno", vmin=0.0, vmax=vmax, interpolation="nearest"
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(f"{wavelengths_nm[index]:.0f} nm")
            if column == 0:
                axes[row, column].set_ylabel(label, fontsize=10)
    figure.colorbar(last_image, ax=axes, fraction=0.018, pad=0.015, label="总光子中的单像素强度份额")
    figure.suptitle("Figure 2  五个代表波长下的 PSF 强度分布（统一总光子标度）", fontsize=15, fontweight="bold")
    figure.savefig(output / "figure2_wavelength_psfs.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "figure2_wavelength_psfs.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_similarity_matrices(
    output: Path,
    correlations: dict[str, np.ndarray],
    cosine_similarities: dict[str, np.ndarray],
) -> None:
    labels = [f"{value:.0f}" for value in SELECTED_WAVELENGTHS_NM]
    figure, axes = plt.subplots(3, 2, figsize=(10, 13))
    for row, system in enumerate(SYSTEM_ORDER):
        for column, (matrix, metric_name) in enumerate(
            ((correlations[system], "Pearson correlation"), (cosine_similarities[system], "Cosine similarity"))
        ):
            if column == 0:
                image = axes[row, column].imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
            else:
                image = axes[row, column].imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
            axes[row, column].set_xticks(range(len(labels)), labels=labels)
            axes[row, column].set_yticks(range(len(labels)), labels=labels)
            axes[row, column].set_xlabel("波长 / nm")
            axes[row, column].set_ylabel("波长 / nm")
            axes[row, column].set_title(f"{SYSTEM_LABELS[system]} · {metric_name}")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    color = "white" if matrix[i, j] < 0.55 else "black"
                    axes[row, column].text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=color, fontsize=7)
            figure.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.04)
    figure.suptitle("Figure 3  波长间 PSF 可分辨性矩阵（越低越易区分）", fontsize=15, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output / "figure3_similarity_matrices.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "figure3_similarity_matrices.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_fisher_information(
    output: Path,
    wavelengths_nm: np.ndarray,
    fisher: dict[str, np.ndarray],
) -> None:
    colors = {"double_helix": "#d32f2f", "single_helix": "#ef6c00", "dual_single_helix": "#1565c0"}
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for system in SYSTEM_ORDER:
        axis.semilogy(wavelengths_nm, fisher[system], label=SYSTEM_LABELS[system], color=colors[system], linewidth=2)
    for value in SELECTED_WAVELENGTHS_NM:
        axis.axvline(value, color="#90a4ae", linewidth=0.6, alpha=0.35)
    axis.set_xlabel("波长 / nm")
    axis.set_ylabel("Fisher information / nm$^{-2}$")
    axis.set_title("Figure 4  相同总光子预算下的波长 Fisher 信息", fontweight="bold")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "figure4_fisher_information.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "figure4_fisher_information.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_crlb(
    output: Path,
    wavelengths_nm: np.ndarray,
    fisher: dict[str, np.ndarray],
    total_photons: float,
) -> None:
    colors = {"double_helix": "#d32f2f", "single_helix": "#ef6c00", "dual_single_helix": "#1565c0"}
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for system in SYSTEM_ORDER:
        axis.semilogy(
            wavelengths_nm, crlb_from_fisher(fisher[system]),
            label=SYSTEM_LABELS[system], color=colors[system], linewidth=2,
        )
    axis.axvspan(450.0, 650.0, color="#cfd8dc", alpha=0.22, label="主要验证波段")
    axis.set_xlabel("波长 / nm")
    axis.set_ylabel("波长估计标准差下界 / nm")
    axis.set_title(f"Figure 5  CRLB（总信号光子数 = {total_photons:g}）", fontweight="bold")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "figure5_crlb_equal_photons.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "figure5_crlb_equal_photons.pdf", bbox_inches="tight")
    plt.close(figure)


def _off_diagonal_mean(matrix: np.ndarray) -> float:
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(matrix[mask].mean())


def save_individual_psfs_and_profiles(
    output: Path,
    banks: dict[str, np.ndarray],
    wavelengths_nm: np.ndarray,
    selected_indices: np.ndarray,
) -> None:
    """保存独立 PSF 图片和径向强度曲线，便于逐波长复核。"""
    image_directory = output / "psf_images"
    image_directory.mkdir(parents=True, exist_ok=True)
    rows = (
        ("double_helix", "double", 1.0),
        ("single_helix", "single_plus", 1.0),
        ("dual_single_helix_ch1", "single_plus", 0.5),
        ("dual_single_helix_ch2", "single_minus", 0.5),
    )
    distributions = [fraction * banks[name][index] for _, name, fraction in rows for index in selected_indices]
    common_vmax = float(max(distribution.max() for distribution in distributions))
    profile_records: list[dict[str, float | str]] = []
    for system_label, bank_name, fraction in rows:
        for index in selected_indices:
            distribution = fraction * banks[bank_name][index]
            wavelength = float(wavelengths_nm[index])
            plt.imsave(
                image_directory / f"{system_label}_{wavelength:.0f}nm.png",
                distribution,
                cmap="inferno",
                vmin=0.0,
                vmax=common_vmax,
            )
            centroid_x, centroid_y = psf_centroid(distribution)
            y, x = np.indices(distribution.shape, dtype=float)
            radius = np.hypot(x - centroid_x, y - centroid_y)
            integer_radius = np.floor(radius).astype(int)
            for radial_bin in range(int(integer_radius.max()) + 1):
                mask = integer_radius == radial_bin
                if np.any(mask):
                    profile_records.append(
                        {
                            "system_channel": system_label,
                            "wavelength_nm": wavelength,
                            "radius_px": float(radial_bin + 0.5),
                            "mean_intensity_fraction": float(distribution[mask].mean()),
                            "annular_energy_fraction": float(distribution[mask].sum()),
                        }
                    )
    with (output / "radial_intensity_profiles.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profile_records[0]))
        writer.writeheader()
        writer.writerows(profile_records)


def _objective_judgement(summary: dict[str, dict[str, float]]) -> str:
    dual = summary["dual_single_helix"]
    best_baseline_fisher = max(summary["double_helix"]["median_fisher"], summary["single_helix"]["median_fisher"])
    best_baseline_similarity = min(
        summary["double_helix"]["mean_offdiag_cosine"],
        summary["single_helix"]["mean_offdiag_cosine"],
    )
    fisher_gain = dual["median_fisher"] / max(best_baseline_fisher, np.finfo(float).eps)
    similarity_gain = best_baseline_similarity - dual["mean_offdiag_cosine"]
    if fisher_gain > 1.05 and similarity_gain > 0.01:
        return "初步值得继续：双通道在等总光子下同时提高 Fisher 信息并降低波长间相似度；仍需制造误差与实测验证。"
    if fisher_gain < 0.95 and similarity_gain <= 0.0:
        return "当前设计未显示优势：等总光子下双通道 Fisher 信息更低且相似度没有改善；应先优化 DOE，不建议直接加工。"
    return "证据混合：双通道只在部分指标或波段有改善，尚不能声称优于基线；值得做小规模 DOE 优化后再决定是否加工。"


def run_first_validation(config: ResearchConfig, output_directory: str | Path) -> dict[str, object]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    wavelengths_nm = np.asarray(config.optics.wavelengths_nm, dtype=float)
    selected_indices = _nearest_indices(wavelengths_nm, SELECTED_WAVELENGTHS_NM)
    banks = build_psf_banks(config)
    banks_by_system = system_banks(banks)

    vectors = observation_vectors(banks_by_system, selected_indices)
    correlations = {system: pearson_correlation_matrix(vectors[system]) for system in SYSTEM_ORDER}
    cosine_similarities = {system: cosine_similarity_matrix(vectors[system]) for system in SYSTEM_ORDER}
    fisher = compute_fisher_curves(config, banks_by_system, wavelengths_nm)
    features = extract_psf_features(banks_by_system, wavelengths_nm, selected_indices)

    with (output / "psf_features.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(features[0]))
        writer.writeheader()
        writer.writerows(features)

    np.savez_compressed(
        output / "psf_intensity_data.npz",
        wavelengths_nm=np.asarray(SELECTED_WAVELENGTHS_NM),
        double_helix=banks["double"][selected_indices],
        single_helix=banks["single_plus"][selected_indices],
        dual_channel_1=0.5 * banks["single_plus"][selected_indices],
        dual_channel_2=0.5 * banks["single_minus"][selected_indices],
    )
    np.savez_compressed(
        output / "separability_data.npz",
        wavelengths_nm=np.asarray(SELECTED_WAVELENGTHS_NM),
        **{f"pearson_{system}": correlations[system] for system in SYSTEM_ORDER},
        **{f"cosine_{system}": cosine_similarities[system] for system in SYSTEM_ORDER},
        **{f"fisher_{system}": fisher[system] for system in SYSTEM_ORDER},
        **{f"crlb_std_{system}": crlb_from_fisher(fisher[system]) for system in SYSTEM_ORDER},
    )

    validation_mask = (wavelengths_nm >= 450.0) & (wavelengths_nm <= 650.0)
    metric_summary: dict[str, dict[str, float]] = {}
    for system in SYSTEM_ORDER:
        metric_summary[system] = {
            "mean_offdiag_pearson": _off_diagonal_mean(correlations[system]),
            "mean_offdiag_cosine": _off_diagonal_mean(cosine_similarities[system]),
            "median_fisher": float(np.median(fisher[system][validation_mask])),
            "median_crlb_std_nm": float(np.median(crlb_from_fisher(fisher[system][validation_mask]))),
            "worst_crlb_std_nm": float(np.max(crlb_from_fisher(fisher[system][validation_mask]))),
        }

    summary: dict[str, object] = {
        "experiment": "DOE 光谱编码最小物理可行性验证",
        "selected_wavelengths_nm": list(SELECTED_WAVELENGTHS_NM),
        "fairness": {
            "same_aperture_mm": config.optics.pupil_diameter_mm,
            "same_sensor_pixel_pitch_um": config.optics.pixel_pitch_um,
            "same_total_signal_photons": config.camera.photons_total,
            "same_background_photons_per_pixel": config.camera.background_photons_per_pixel,
            "same_read_noise_e": config.camera.read_noise_e,
            "dual_channel_photon_split": [0.5, 0.5],
            "note": "双通道总信号光子与两个基线相同，但读取两幅图，因此承担两次背景与读出噪声。",
        },
        "metrics": metric_summary,
        "objective_judgement": _objective_judgement(metric_summary),
        "limitations": [
            "解析环带螺旋 DOE 仅为物理可运行初值，尚未针对可分辨性优化。",
            "PSF 在固定裁剪窗口内按相同总探测光子归一化，本实验比较编码形状而非实际衍射效率。",
            "Dual-SH 使用两幅同参数传感器观测；总像素读出数高于单通道，后续需补充等总像素比较。",
        ],
    }
    with (output / "validation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    plot_system_schematic(output)
    save_individual_psfs_and_profiles(output, banks, wavelengths_nm, selected_indices)
    plot_wavelength_psfs(output, banks, wavelengths_nm, selected_indices)
    plot_similarity_matrices(output, correlations, cosine_similarities)
    plot_fisher_information(output, wavelengths_nm, fisher)
    plot_crlb(output, wavelengths_nm, fisher, config.camera.photons_total)
    return summary
