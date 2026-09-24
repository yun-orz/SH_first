from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class OpticsConfig:
    grid_size: int
    pupil_diameter_mm: float
    focal_length_mm: float
    sensor_distance_mm: float
    pixel_pitch_um: float
    design_wavelength_nm: float
    wavelengths_nm: tuple[float, ...]


@dataclass(frozen=True)
class DOEConfig:
    material: str
    zone_count: int
    single_charge_start: int
    single_charge_step: int
    double_charge_start: int
    double_charge_step: int
    phase_levels: int
    continuous_phase: bool
    aperture_fill: float


@dataclass(frozen=True)
class CameraConfig:
    photons_total: float
    background_photons_per_pixel: float
    read_noise_e: float
    bit_depth: int
    quantum_efficiency: float
    beamsplitter_transmission: float
    beamsplitter_reflection: float
    doe_efficiency: float


@dataclass(frozen=True)
class SimulationConfig:
    crop_size: int
    monte_carlo_repeats: int
    seed: int
    test_wavelengths_nm: tuple[float, ...]
    comparison_mode: str


@dataclass(frozen=True)
class SceneConfig:
    canvas_size: int
    patch_size: int
    spacings_px: tuple[int, ...]
    photons_per_emitter: tuple[float, ...]
    repeats: int
    wavelengths_nm: tuple[float, float]


@dataclass(frozen=True)
class ResearchConfig:
    optics: OpticsConfig
    doe: DOEConfig
    camera: CameraConfig
    simulation: SimulationConfig
    scene: SceneConfig


def _wavelength_grid(raw: Any) -> tuple[float, ...]:
    if isinstance(raw, list):
        return tuple(float(value) for value in raw)
    start, stop, step = float(raw["start"]), float(raw["stop"]), float(raw["step"])
    count = int(round((stop - start) / step))
    return tuple(start + index * step for index in range(count + 1))


def load_config(path: str | Path) -> ResearchConfig:
    """读取并校验 YAML 配置。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    optics_raw = dict(raw["optics"])
    optics_raw["wavelengths_nm"] = _wavelength_grid(optics_raw["wavelengths_nm"])
    simulation_raw = dict(raw["simulation"])
    simulation_raw["test_wavelengths_nm"] = tuple(float(v) for v in simulation_raw["test_wavelengths_nm"])
    scene_raw = dict(raw["scene"])
    scene_raw["spacings_px"] = tuple(int(v) for v in scene_raw["spacings_px"])
    scene_raw["photons_per_emitter"] = tuple(float(v) for v in scene_raw["photons_per_emitter"])
    scene_raw["wavelengths_nm"] = tuple(float(v) for v in scene_raw["wavelengths_nm"])

    config = ResearchConfig(
        optics=OpticsConfig(**optics_raw),
        doe=DOEConfig(**raw["doe"]),
        camera=CameraConfig(**raw["camera"]),
        simulation=SimulationConfig(**simulation_raw),
        scene=SceneConfig(**scene_raw),
    )
    validate_config(config)
    return config


def validate_config(config: ResearchConfig) -> None:
    if config.optics.grid_size < 64 or config.optics.grid_size % 2:
        raise ValueError("grid_size 必须是不小于 64 的偶数")
    if config.simulation.crop_size % 2 == 0:
        raise ValueError("crop_size 必须是奇数")
    if config.simulation.crop_size > config.optics.grid_size:
        raise ValueError("crop_size 不能大于 grid_size")
    if config.doe.zone_count < 2:
        raise ValueError("zone_count 至少为 2")
    if not config.doe.continuous_phase and config.doe.phase_levels < 2:
        raise ValueError("量化相位至少需要 2 级")
    if config.simulation.comparison_mode not in {"fair_photon", "hardware_realistic"}:
        raise ValueError("comparison_mode 只能为 fair_photon 或 hardware_realistic")
    if config.scene.canvas_size % 2 == 0 or config.scene.patch_size % 2 == 0:
        raise ValueError("场景画布和局部模板尺寸必须为奇数")
    if config.scene.patch_size != config.simulation.crop_size:
        raise ValueError("scene.patch_size 必须等于 simulation.crop_size")
    if len(config.scene.wavelengths_nm) != 2:
        raise ValueError("当前棋盘场景必须指定两个中心波长")
