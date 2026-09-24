from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import ResearchConfig
from .estimation import crlb_from_fisher
from .evaluation import adjacent_template_correlation
from .pipeline import _fisher_analysis, build_psf_banks


def run_seed_search(
    config: ResearchConfig,
    output_directory: str | Path,
    zone_counts: tuple[int, ...],
    charge_steps: tuple[int, ...],
) -> list[dict[str, float | int]]:
    """对解析单螺旋初值做小规模网格搜索，不替代后续连续相位优化。"""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int]] = []
    for zones in zone_counts:
        for step in charge_steps:
            candidate = replace(config, doe=replace(config.doe, zone_count=zones, single_charge_step=step))
            banks = build_psf_banks(candidate)
            fisher_dual, fisher_double = _fisher_analysis(candidate, banks)
            dual_crlb = crlb_from_fisher(fisher_dual)
            ambiguity = max(
                float(np.quantile(adjacent_template_correlation(banks["single_plus"]), 0.95)),
                float(np.quantile(adjacent_template_correlation(banks["single_minus"]), 0.95)),
            )
            # 分数越大越好：奖励信息量，惩罚相邻模板过度相似和极差波长点。
            score = float(
                np.median(np.log10(np.maximum(fisher_dual, np.finfo(float).eps)))
                - 2.0 * ambiguity
                - np.log10(max(float(np.quantile(dual_crlb, 0.9)), np.finfo(float).eps))
            )
            rows.append(
                {
                    "zone_count": zones,
                    "single_charge_step": step,
                    "score": score,
                    "median_dual_crlb_nm": float(np.median(dual_crlb)),
                    "p90_dual_crlb_nm": float(np.quantile(dual_crlb, 0.9)),
                    "median_double_crlb_nm": float(np.median(crlb_from_fisher(fisher_double))),
                    "p95_adjacent_correlation": ambiguity,
                }
            )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    with (output / "seed_search.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    recommendation = {
        "recommended_seed": rows[0],
        "warning": "这里只搜索解析初值参数；送厂前仍需加入真实加工约束和制造误差。",
    }
    with (output / "recommended_seed.json").open("w", encoding="utf-8") as handle:
        json.dump(recommendation, handle, ensure_ascii=False, indent=2)
    return rows

