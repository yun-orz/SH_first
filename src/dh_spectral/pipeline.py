from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import ResearchConfig
from .doe import build_height_mask, pupil_grid
from .estimation import crlb_from_fisher, estimate_wavelength, multichannel_fisher
from .evaluation import adjacent_template_correlation, wavelength_metrics
from .forward import add_camera_noise, photon_budgets
from .forward import render_spectral_scene
from .plotting import configure_chinese_font
from .propagation import fresnel_psf, second_moment_features
from .scenes import checkerboard_emitters, crop_patch


configure_chinese_font()


def build_psf_banks(config: ResearchConfig) -> dict[str, np.ndarray]:
    optics, doe, simulation = config.optics, config.doe, config.simulation
    diameter_m = optics.pupil_diameter_mm * 1e-3
    focal_m = optics.focal_length_mm * 1e-3
    sensor_distance_m = optics.sensor_distance_mm * 1e-3
    design_wavelength_m = optics.design_wavelength_nm * 1e-9
    _, _, radius, angle = pupil_grid(optics.grid_size, diameter_m)
    common = dict(
        radius=radius,
        angle=angle,
        pupil_radius_m=diameter_m / 2.0,
        focal_length_m=focal_m,
        design_wavelength_m=design_wavelength_m,
        material=doe.material,
        zone_count=doe.zone_count,
        levels=doe.phase_levels,
        continuous=doe.continuous_phase,
    )
    heights = {
        "single_plus": build_height_mask(
            **common,
            charge_start=doe.single_charge_start,
            charge_step=doe.single_charge_step,
            handedness=1,
        ),
        "single_minus": build_height_mask(
            **common,
            charge_start=doe.single_charge_start,
            charge_step=doe.single_charge_step,
            handedness=-1,
        ),
        "double": build_height_mask(
            **common,
            charge_start=doe.double_charge_start,
            charge_step=doe.double_charge_step,
            handedness=1,
        ),
    }
    propagation_common = dict(
        radius_m=radius,
        pupil_radius_m=diameter_m / 2.0,
        sensor_distance_m=sensor_distance_m,
        pupil_sample_pitch_m=diameter_m / optics.grid_size,
        sensor_pixel_pitch_m=optics.pixel_pitch_um * 1e-6,
        material=doe.material,
        aperture_fill=doe.aperture_fill,
        crop_size=simulation.crop_size,
    )
    banks: dict[str, np.ndarray] = {}
    for name, height in heights.items():
        banks[name] = np.stack(
            [
                fresnel_psf(
                    height_m=height,
                    wavelength_m=wavelength_nm * 1e-9,
                    **propagation_common,
                )
                for wavelength_nm in optics.wavelengths_nm
            ]
        )
    banks["height_single_plus_m"] = heights["single_plus"]
    banks["height_single_minus_m"] = heights["single_minus"]
    banks["height_double_m"] = heights["double"]
    return banks


def _nearest_index(wavelengths: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(wavelengths - value)))


def _plot_atlas(output: Path, wavelengths: np.ndarray, banks: dict[str, np.ndarray]) -> None:
    selected = np.linspace(0, len(wavelengths) - 1, 5, dtype=int)
    names = ("single_plus", "single_minus", "double")
    labels = ("单螺旋 +", "单螺旋 −", "双螺旋基线")
    figure, axes = plt.subplots(len(names), len(selected), figsize=(12, 7))
    for row, (name, label) in enumerate(zip(names, labels)):
        for column, index in enumerate(selected):
            axes[row, column].imshow(banks[name][index], cmap="inferno")
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(f"{wavelengths[index]:.0f} nm")
            if column == 0:
                axes[row, column].set_ylabel(label)
    figure.suptitle("多波长 PSF 图谱")
    figure.tight_layout()
    figure.savefig(output / "psf_atlas.png", dpi=180)
    plt.close(figure)


def _plot_calibration(output: Path, wavelengths: np.ndarray, banks: dict[str, np.ndarray]) -> None:
    features = {
        name: [second_moment_features(psf) for psf in banks[name]]
        for name in ("single_plus", "single_minus", "double")
    }
    plus_x = np.array([item["centroid_x"] for item in features["single_plus"]])
    plus_y = np.array([item["centroid_y"] for item in features["single_plus"]])
    minus_x = np.array([item["centroid_x"] for item in features["single_minus"]])
    minus_y = np.array([item["centroid_y"] for item in features["single_minus"]])
    displacement = np.hypot(plus_x - minus_x, plus_y - minus_y)
    direction = np.unwrap(np.arctan2(plus_y - minus_y, plus_x - minus_x))
    double_angle = np.unwrap(
        np.array([item["orientation_rad"] for item in features["double"]]),
        period=np.pi,
    )
    double_extent = np.array([item["major_sigma"] for item in features["double"]])

    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes[0, 0].plot(wavelengths, displacement)
    axes[0, 0].set_ylabel("双通道质心距离 / pixel")
    axes[0, 1].plot(wavelengths, np.rad2deg(direction))
    axes[0, 1].set_ylabel("双通道位移方向 / degree")
    axes[1, 0].plot(wavelengths, np.rad2deg(double_angle))
    axes[1, 0].set_ylabel("双螺旋主轴方向 / degree")
    axes[1, 1].plot(wavelengths, double_extent)
    axes[1, 1].set_ylabel("双螺旋主轴尺度 / pixel")
    for axis in axes[-1]:
        axis.set_xlabel("波长 / nm")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.suptitle("可解释编码量标定曲线")
    figure.tight_layout()
    figure.savefig(output / "calibration_curves.png", dpi=180)
    plt.close(figure)


def _fisher_analysis(config: ResearchConfig, banks: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    camera = config.camera
    wavelengths = np.asarray(config.optics.wavelengths_nm)
    budgets = photon_budgets(
        camera.photons_total,
        config.simulation.comparison_mode,
        camera.beamsplitter_transmission,
        camera.beamsplitter_reflection,
        camera.doe_efficiency,
    )
    dual = multichannel_fisher(
        (banks["single_plus"], banks["single_minus"]),
        wavelengths,
        budgets["dual"],
        camera.background_photons_per_pixel,
        camera.read_noise_e,
    )
    double = multichannel_fisher(
        (banks["double"],),
        wavelengths,
        budgets["double"],
        camera.background_photons_per_pixel,
        camera.read_noise_e,
    )
    return dual, double


def _plot_fisher(output: Path, wavelengths: np.ndarray, dual: np.ndarray, double: np.ndarray) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].semilogy(wavelengths, dual, label="双通道单螺旋")
    axes[0].semilogy(wavelengths, double, label="单通道双螺旋")
    axes[0].set_ylabel("Fisher 信息 / nm^-2")
    axes[1].semilogy(wavelengths, crlb_from_fisher(dual), label="双通道单螺旋")
    axes[1].semilogy(wavelengths, crlb_from_fisher(double), label="单通道双螺旋")
    axes[1].set_ylabel("CRLB / nm")
    for axis in axes:
        axis.set_xlabel("波长 / nm")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("等资源光谱信息量比较")
    figure.tight_layout()
    figure.savefig(output / "fisher_crlb.png", dpi=180)
    plt.close(figure)


def _monte_carlo(config: ResearchConfig, banks: dict[str, np.ndarray]) -> list[dict[str, float | str | int]]:
    wavelengths = np.asarray(config.optics.wavelengths_nm)
    camera, simulation = config.camera, config.simulation
    budgets = photon_budgets(
        camera.photons_total,
        simulation.comparison_mode,
        camera.beamsplitter_transmission,
        camera.beamsplitter_reflection,
        camera.doe_efficiency,
    )
    rng = np.random.default_rng(simulation.seed)
    records: list[dict[str, float | str | int]] = []
    for truth in simulation.test_wavelengths_nm:
        index = _nearest_index(wavelengths, truth)
        for repeat in range(simulation.monte_carlo_repeats):
            dual_observations = tuple(
                add_camera_noise(
                    banks[name][index], photons, camera.background_photons_per_pixel,
                    camera.read_noise_e, camera.bit_depth, rng,
                )
                for name, photons in zip(("single_plus", "single_minus"), budgets["dual"])
            )
            dual_estimate, dual_confidence, _ = estimate_wavelength(
                dual_observations,
                (banks["single_plus"], banks["single_minus"]),
                wavelengths,
            )
            double_observation = add_camera_noise(
                banks["double"][index], budgets["double"][0],
                camera.background_photons_per_pixel, camera.read_noise_e,
                camera.bit_depth, rng,
            )
            double_estimate, double_confidence, _ = estimate_wavelength(
                (double_observation,), (banks["double"],), wavelengths
            )
            records.extend(
                [
                    {"system": "dual_single_helix", "truth_nm": truth, "estimate_nm": dual_estimate,
                     "confidence": dual_confidence, "repeat": repeat},
                    {"system": "single_double_helix", "truth_nm": truth, "estimate_nm": double_estimate,
                     "confidence": double_confidence, "repeat": repeat},
                ]
            )
    return records


def run_baseline(config: ResearchConfig, output_directory: str | Path) -> dict[str, object]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    wavelengths = np.asarray(config.optics.wavelengths_nm)
    banks = build_psf_banks(config)

    np.savez_compressed(
        output / "psf_bank.npz",
        wavelengths_nm=wavelengths,
        single_plus=banks["single_plus"],
        single_minus=banks["single_minus"],
        double=banks["double"],
        height_single_plus_m=banks["height_single_plus_m"],
        height_single_minus_m=banks["height_single_minus_m"],
        height_double_m=banks["height_double_m"],
    )
    _plot_atlas(output, wavelengths, banks)
    _plot_calibration(output, wavelengths, banks)
    fisher_dual, fisher_double = _fisher_analysis(config, banks)
    _plot_fisher(output, wavelengths, fisher_dual, fisher_double)
    records = _monte_carlo(config, banks)
    with (output / "monte_carlo.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    metrics: dict[str, dict[str, float]] = {}
    for system in ("dual_single_helix", "single_double_helix"):
        subset = [record for record in records if record["system"] == system]
        metrics[system] = wavelength_metrics(
            np.array([float(item["truth_nm"]) for item in subset]),
            np.array([float(item["estimate_nm"]) for item in subset]),
        )
    summary: dict[str, object] = {
        "config": asdict(config),
        "metrics": metrics,
        "median_crlb_nm": {
            "dual_single_helix": float(np.median(crlb_from_fisher(fisher_dual))),
            "single_double_helix": float(np.median(crlb_from_fisher(fisher_double))),
        },
        "max_adjacent_template_correlation": {
            "single_plus": float(np.max(adjacent_template_correlation(banks["single_plus"]))),
            "single_minus": float(np.max(adjacent_template_correlation(banks["single_minus"]))),
            "double": float(np.max(adjacent_template_correlation(banks["double"]))),
        },
        "warning": "解析 DOE 仅为优化初值；加工前必须代入真实工艺约束并完成实测标定。",
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def run_scene_sweep(config: ResearchConfig, output_directory: str | Path) -> list[dict[str, float | int | str]]:
    """扫描点间距和每发射点光子数，得到双通道收益适用范围图。"""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    banks = build_psf_banks(config)
    wavelengths = np.asarray(config.optics.wavelengths_nm)
    scene_cfg, camera = config.scene, config.camera
    wavelength_indices = tuple(_nearest_index(wavelengths, value) for value in scene_cfg.wavelengths_nm)
    rng = np.random.default_rng(config.simulation.seed + 1)
    records: list[dict[str, float | int | str]] = []

    for spacing in scene_cfg.spacings_px:
        first, second, coordinates = checkerboard_emitters(
            scene_cfg.canvas_size, spacing, scene_cfg.patch_size // 2
        )
        scene_cube = np.stack((first, second))
        noiseless = {
            name: render_spectral_scene(
                scene_cube,
                np.stack((banks[name][wavelength_indices[0]], banks[name][wavelength_indices[1]])),
            )
            for name in ("single_plus", "single_minus", "double")
        }
        for photons_per_emitter in scene_cfg.photons_per_emitter:
            incident_total = photons_per_emitter * len(coordinates)
            budgets = photon_budgets(
                incident_total,
                config.simulation.comparison_mode,
                camera.beamsplitter_transmission,
                camera.beamsplitter_reflection,
                camera.doe_efficiency,
            )
            for repeat in range(scene_cfg.repeats):
                observations = {
                    "single_plus": add_camera_noise(
                        noiseless["single_plus"], budgets["dual"][0], camera.background_photons_per_pixel,
                        camera.read_noise_e, camera.bit_depth, rng,
                    ),
                    "single_minus": add_camera_noise(
                        noiseless["single_minus"], budgets["dual"][1], camera.background_photons_per_pixel,
                        camera.read_noise_e, camera.bit_depth, rng,
                    ),
                    "double": add_camera_noise(
                        noiseless["double"], budgets["double"][0], camera.background_photons_per_pixel,
                        camera.read_noise_e, camera.bit_depth, rng,
                    ),
                }
                for y, x, label in coordinates:
                    truth = scene_cfg.wavelengths_nm[label]
                    plus_patch = crop_patch(observations["single_plus"], y, x, scene_cfg.patch_size)
                    minus_patch = crop_patch(observations["single_minus"], y, x, scene_cfg.patch_size)
                    double_patch = crop_patch(observations["double"], y, x, scene_cfg.patch_size)
                    dual_estimate, _, _ = estimate_wavelength(
                        (plus_patch, minus_patch),
                        (banks["single_plus"], banks["single_minus"]), wavelengths,
                    )
                    double_estimate, _, _ = estimate_wavelength(
                        (double_patch,), (banks["double"],), wavelengths,
                    )
                    records.extend(
                        [
                            {"system": "dual_single_helix", "spacing_px": spacing,
                             "photons_per_emitter": photons_per_emitter, "repeat": repeat,
                             "truth_nm": truth, "estimate_nm": dual_estimate},
                            {"system": "single_double_helix", "spacing_px": spacing,
                             "photons_per_emitter": photons_per_emitter, "repeat": repeat,
                             "truth_nm": truth, "estimate_nm": double_estimate},
                        ]
                    )

    with (output / "scene_sweep.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    _plot_applicability_map(output, config, records)
    return records


def _plot_applicability_map(
    output: Path,
    config: ResearchConfig,
    records: list[dict[str, float | int | str]],
) -> None:
    spacings = config.scene.spacings_px
    photons = config.scene.photons_per_emitter
    gain = np.zeros((len(photons), len(spacings)))
    for row, photon_count in enumerate(photons):
        for column, spacing in enumerate(spacings):
            errors: dict[str, list[float]] = {"dual_single_helix": [], "single_double_helix": []}
            for record in records:
                if record["spacing_px"] == spacing and record["photons_per_emitter"] == photon_count:
                    errors[str(record["system"])].append(
                        float(record["estimate_nm"]) - float(record["truth_nm"])
                    )
            dual_rmse = float(np.sqrt(np.mean(np.square(errors["dual_single_helix"]))))
            double_rmse = float(np.sqrt(np.mean(np.square(errors["single_double_helix"]))))
            gain[row, column] = double_rmse - dual_rmse

    limit = max(float(np.max(np.abs(gain))), 1.0)
    figure, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(gain, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(spacings)), labels=spacings)
    axis.set_yticks(range(len(photons)), labels=[f"{value:g}" for value in photons])
    axis.set_xlabel("点间距 / pixel（越小越复杂）")
    axis.set_ylabel("每个发射点的入射光子数")
    axis.set_title("适用范围：双螺旋 RMSE − 双通道 RMSE（正值表示双通道占优）")
    for row in range(gain.shape[0]):
        for column in range(gain.shape[1]):
            axis.text(column, row, f"{gain[row, column]:.1f}", ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, label="RMSE收益 / nm")
    figure.tight_layout()
    figure.savefig(output / "applicability_map.png", dpi=180)
    plt.close(figure)
