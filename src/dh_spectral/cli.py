from __future__ import annotations

import argparse
import json

from .complementarity import run_complementarity_sweep
from .config import load_config
from .first_validation import run_first_validation
from .optimization import run_seed_search
from .pipeline import run_baseline, run_scene_sweep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="双通道单螺旋光谱编码研究平台")
    subparsers = parser.add_subparsers(dest="command", required=True)
    baseline = subparsers.add_parser("baseline", help="运行多波长点源公平比较基线")
    baseline.add_argument("--config", required=True, help="YAML 配置文件")
    baseline.add_argument("--output", required=True, help="结果输出目录")
    scene = subparsers.add_parser("scene-sweep", help="扫描点阵复杂度和光子数并绘制适用范围图")
    scene.add_argument("--config", required=True, help="YAML 配置文件")
    scene.add_argument("--output", required=True, help="结果输出目录")
    optimize = subparsers.add_parser("optimize-seed", help="搜索解析 DOE 初值参数")
    optimize.add_argument("--config", required=True, help="YAML 配置文件")
    optimize.add_argument("--output", required=True, help="结果输出目录")
    optimize.add_argument("--zones", default="4,6,8", help="候选环带数，逗号分隔")
    optimize.add_argument("--charge-steps", default="1,2", help="候选拓扑荷步进，逗号分隔")
    validation = subparsers.add_parser("validate-idea", help="运行三系统最小物理可行性验证")
    validation.add_argument("--config", required=True, help="YAML 配置文件")
    validation.add_argument("--output", required=True, help="结果输出目录")
    complementarity = subparsers.add_parser(
        "complementarity-sweep", help="运行 Single Helix 参数互补性与 Fisher 增益扫描"
    )
    complementarity.add_argument("--config", required=True, help="YAML 配置文件")
    complementarity.add_argument("--output", required=True, help="结果输出目录")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "baseline":
        summary = run_baseline(load_config(args.config), args.output)
        print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    elif args.command == "scene-sweep":
        records = run_scene_sweep(load_config(args.config), args.output)
        print(json.dumps({"records": len(records), "output": args.output}, ensure_ascii=False, indent=2))
    elif args.command == "optimize-seed":
        zones = tuple(int(value) for value in args.zones.split(","))
        steps = tuple(int(value) for value in args.charge_steps.split(","))
        rows = run_seed_search(load_config(args.config), args.output, zones, steps)
        print(json.dumps(rows[0], ensure_ascii=False, indent=2))
    elif args.command == "validate-idea":
        summary = run_first_validation(load_config(args.config), args.output)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "complementarity-sweep":
        summary = run_complementarity_sweep(load_config(args.config), args.output)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
