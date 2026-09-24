# 第二轮科学验证：Single Helix 通道互补性扫描

## 实验约束

- 使用 `baseline.yaml`，波长范围 400–700 nm。
- 总信号光子数固定为 20000；双通道严格按 N/2 + N/2 分配。
- 扫描 90 个 Single Helix DOE，形成 4095 个无序双通道组合（包含 A=A 对照）。
- 没有改变 aperture、传感器或噪声模型，也没有筛掉失败组合。

## 1. Baseline 是否复现

- Double Helix：median Fisher = 36.5141，median CRLB = 0.1655 nm。
- 当前 Single Helix：median Fisher = 39.1628，median CRLB = 0.1598 nm。
- 当前 +/- Dual SH：median Fisher = 34.8569，median CRLB = 0.1694 nm。

## 2. 当前 +/- handedness 为什么可能没有优势

当前组合的普通余弦冗余为 0.4026，镜像对齐冗余为 0.6908，光谱导数相似度为 0.0519。正负 handedness 主要产生镜像/共轭外观；镜像对齐后仍相似，说明它们并未创造同等光子条件下足够独立的波长敏感结构。与此同时，每路只有一半光子且承担独立读出噪声。

## 3. 仅改变 handedness 是否足够

严格保持 zone_count、charge_start、charge_step 相同而只反转 handedness 的组合共有 45 组；其平均镜像对齐冗余为 0.7474，平均 Gain_DH 为 0.8611，其中 16 组超过 Double Helix。两个 DOE 往往仍是镜像关系，因此普通像素相关下降不应被直接解释为信息互补。

## 4. 哪些参数最影响互补性

按“参数不同相对参数相同所造成的镜像对齐冗余变化”这一描述性统计，影响最大的是 `charge_step`（变化 -0.2615）。这不是因果证明，因为全组合中多个参数会同时变化；完整统计见 `parameter_effects.csv`。

## 5. 互补性与 Fisher Gain 是否存在明确关系

- 镜像对齐冗余 vs Gain_DH：Pearson r = 0.044，Spearman ρ = 0.131。
- 光谱导数相似度 vs Gain_DH：Pearson r = 0.006，Spearman ρ = 0.176。
- 光谱导数相似度 vs Gain_SH：Pearson r = 0.006，Spearman ρ = 0.176。
这些相关系数只描述本参数库，不把本项目定义的互补性指标包装成公认理论量。

## 6. 是否超过两个基线

- 985/4095 个 Dual SH 组合的 median Fisher 超过 Double Helix。
- 0/4095 个 Dual SH 组合的 median Fisher 超过候选库中的 best Single Helix。
- 最优 Dual SH 为 `z06_c2_s2_hm1__z06_c2_s2_hm1`：Gain_DH = 1.2222，Gain_SH = 0.8461。
- best Single Helix 为 `z06_c2_s2_hp1`，median Fisher = 52.7425。

特别重要的是，Fisher 第一名的两个通道参数完全相同，镜像对齐冗余为 1.0000、导数相似度为 1.0000。因此它超过 Double Helix 不能归因于互补性，而应归因于该 Single Helix 参数本身比当前 Double Helix 初值更强。

对于单一已知参数 λ，独立通道 Fisher 可以相加；在理想泊松主导且总光子固定时，双通道结果接近两个全光子单通道 Fisher 的加权平均，因此不能仅靠两通道外观不同超过其中最强的单通道。读出噪声还会使拆分更吃亏。双通道真正可能体现价值的地方，是存在位置、亮度、背景等未知干扰参数或全局模板歧义时，而这不属于本轮局部单参数 Fisher 的证明范围。

## 7. Monte Carlo

观测 PSF 由相邻 5 nm 标定模板线性插值得到，估计采用与 Fisher 方差模型一致的统一泊松–高斯加权最小二乘，并对代价曲线作局部二次亚网格估计；这样可避免高光子条件下所有方法都只返回同一个 5 nm 网格点。

| system | bias / nm | MAE / nm | RMSE / nm | std / nm | outlier rate |
|---|---:|---:|---:|---:|---:|
| double_helix | 0.029 | 0.261 | 0.312 | 0.312 | 0.000 |
| best_single_helix | 0.027 | 0.228 | 0.283 | 0.283 | 0.000 |
| original_dual_plus_minus | 0.053 | 0.224 | 0.284 | 0.280 | 0.000 |
| top_1_z06_c2_s2_hm1__z06_c2_s2_hm1 | 0.037 | 0.284 | 0.358 | 0.358 | 0.000 |
| top_2_z04_c2_s2_hm1__z08_c2_s2_hp1 | 0.028 | 0.312 | 0.370 | 0.371 | 0.000 |
| top_3_z06_c2_s2_hp1__z06_c2_s2_hp1 | 0.032 | 0.308 | 0.362 | 0.362 | 0.000 |

本次 120 次/系统的描述性结果中，最低 RMSE 来自 `best_single_helix`（0.2832 nm）；三个按中位 Fisher 选择的组合中最低 RMSE 为 0.3584 nm。因而本轮没有观察到“中位 Fisher 排名自动转化为这组六个测试波长的估计 RMSE 排名”。可能原因包括 Fisher 是逐波长局部下界，而模板估计还受全局歧义、5 nm 标定采样和有限重复次数影响。

## 8. 结论边界

本轮只回答点源、单一待估波长下的局部信息量问题。若 Dual SH 只超过 Double Helix 而没有超过 best Single Helix，就只能说明所选 Double Helix 解析初值较弱，不能宣称拆分结构具有整体优势。所有 Top/Middle/Bad 代表组合均来自完整扫描，完整结果保存在 `dual_pair_metrics.csv`。
