# -*- coding: utf-8 -*-
"""
compare_runs.py —— 比较多次 `sweep_seeds.py` 的结果（跨数据集 / 跨数据量）

为什么单独写这个：`sweep_seeds.py` 只能在**同一份数据集**内比较配置
（h48 vs h96），因为它把测试集绑在 `features.ART` 上。而「2000 情境 vs
2 万情境」是两份不同的数据集，各自的测试集不同 —— 直接比两边的
top-1 等于拿 A 卷分数比 B 卷分数。

正确做法（本脚本假定你这么做了）：
  1. 各数据集分别训练，但都用 `sweep_seeds.py --eval-art <公共留出集>`
     在同一套留出情境上评估；
  2. 各数据集用**同一组 seed**，于是可以做按 seed 配对的 t 检验。

用法：
    python compare_runs.py ^
        --runs "旧2000=artifacts_sweep/sweep.json" ^
               "新2000=artifacts_sweep_v2/sweep.json" ^
               "新20000=artifacts_sweep_20k/sweep.json"
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from sweep_seeds import paired_ttest

METRICS = [
    ("mean_utility", "平均效用", "{:.4f}", True),
    ("regret",       "后悔值",   "{:.4f}", False),   # 越低越好
    ("top1",         "top-1",    "{:.2%}", True),
    ("top3",         "top-3",    "{:.2%}", True),
    ("category_hit", "品类命中", "{:.2%}", True),
]


def load(spec: str):
    name, _, path = spec.partition("=")
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"找不到 {p}")
    d = json.loads(p.read_text(encoding="utf-8"))
    # sweep.json 的 results 是 {config: [rows...]}；本脚本只支持单配置
    cfgs = d["configs"]
    if len(cfgs) != 1:
        raise SystemExit(f"{p} 含多个配置 {cfgs}，请用单配置的 sweep 结果比较")
    rows = sorted(d["results"][cfgs[0]], key=lambda r: r["seed"])
    return name.strip(), d, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help='形如 "新20000=artifacts_sweep_20k/sweep.json"')
    args = ap.parse_args()

    runs = [load(s) for s in args.runs]
    names = [r[0] for r in runs]

    # 种子集合必须一致，否则配对无效
    seeds0 = [r["seed"] for r in runs[0][2]]
    for name, _, rows in runs[1:]:
        if [r["seed"] for r in rows] != seeds0:
            raise SystemExit(f"{name} 的 seed 集合与 {names[0]} 不一致，无法配对比较")

    print("=" * 84)
    print(f"多数据集比较   每个数据集 {len(seeds0)} 个 seed: {seeds0}")
    print("=" * 84)
    for name, meta, rows in runs:
        print(f"\n[{name}]  epochs={meta.get('epochs')}  "
              f"平均训练用时={np.mean([r['seconds'] for r in rows]):.0f}s")
        for key, zh, fmt, _ in METRICS:
            v = [r[key] for r in rows]
            print(f"  {zh:<8} {fmt.format(np.mean(v))} ± {fmt.format(np.std(v, ddof=1))}")

    print()
    print("=" * 84)
    print("配对 t 检验（按 seed 配对；相邻数据集两两比较）")
    print("=" * 84)
    for i in range(1, len(runs)):
        a_name, _, a_rows = runs[i]
        b_name, _, b_rows = runs[i - 1]
        print(f"\n{a_name}  vs  {b_name}")
        for key, zh, fmt, higher_better in METRICS:
            a = [r[key] for r in a_rows]
            b = [r[key] for r in b_rows]
            dmu, t, p = paired_ttest(a, b)
            verdict = "显著" if (not math.isnan(p) and p < 0.05) else "不显著"
            wins = sum(1 for x, y in zip(a, b)
                       if (x > y if higher_better else x < y))
            print(f"  {zh:<8} 均值差 {dmu:+.4f}   t={t:+.2f}  p≈{p:.3f}  "
                  f"{wins}/{len(a)} seed 更好  -> {verdict}")

    print()
    print("提示：p 值用正态近似，n=5 时只能判断量级；结论请看效应量 + 获胜 seed 数。")


if __name__ == "__main__":
    main()
