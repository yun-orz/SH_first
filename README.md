# 双通道单螺旋光谱编码研究平台

这是一个面向本科科研的可复现 Python 基线，用同一套物理参数和估计算法比较：

- 分光后两块共轭 DOE 形成的双通道单螺旋编码；
- 一块 DOE 形成的单通道双螺旋编码。

平台已经实现物理高度 DOE、多波长材料色散、标量衍射、非相干场景卷积、泊松与读出噪声、等光子/真实分光两种口径、统一模板最大似然估计、Fisher 信息和 CRLB。当前解析 DOE 是“可运行初值”，不能未经实测标定直接送厂加工。

## 快速开始

```powershell
python -m pip install -e .
dh-spectral baseline --config configs/baseline.yaml --output outputs/baseline
python -m unittest discover -s tests -v
dh-spectral scene-sweep --config configs/baseline.yaml --output outputs/scene_sweep
dh-spectral optimize-seed --config configs/baseline.yaml --output outputs/optimization
```

不安装包也可以运行：

```powershell
$env:PYTHONPATH="src"
python -m dh_spectral.cli baseline --config configs/baseline.yaml --output outputs/baseline
```

基线命令会生成：

- `psf_atlas.png`：两类系统在代表波长处的 PSF；
- `calibration_curves.png`：质心、角度、瓣间距等可解释编码量；
- `fisher_crlb.png`：等资源下的 Fisher 信息和 CRLB；
- `monte_carlo.csv`：未知波长的重复估计结果；
- `summary.json`：主要配置和性能摘要；
- `psf_bank.npz`：后续场景仿真和实验标定可直接使用的 PSF 库。

`scene-sweep` 会生成交替波长点阵，扫描点间距和每点光子数，输出 `scene_sweep.csv` 与 `applicability_map.png`。热图正值表示双通道的波长 RMSE 更低；这是定位适用范围的第一版可复现实验，而不是预设双通道必然获胜。

`optimize-seed` 对环带数与拓扑荷步进做可复现网格搜索，输出候选排名和推荐解析初值。它用于筛掉明显不合适的初值，不替代基于真实加工约束的连续相位优化。

## 重要解释

本项目把 DOE 相位先转换为物理高度，再用材料色散计算各波长相位。双通道每路只获得分光后的光子；`fair_photon` 模式则把两路总光子数与单通道严格对齐。解析环带螺旋相位用于建立基线，真实研究应根据实测 PSF 和加工约束继续优化。

详细推进与实验记录格式见 [docs/research_protocol.md](docs/research_protocol.md) 和 [docs/equipment_checklist.md](docs/equipment_checklist.md)。
