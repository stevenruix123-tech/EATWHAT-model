# -*- coding: utf-8 -*-
"""
sweep.py —— 容量扫描：dish_id_emb 维度 x DishScorer 隐层宽度

回答一个问题：**当前模型还有容量提升空间吗？**

方法：固定数据（同一份 samples.csv）、固定轮数、固定随机种子，
只改两个容量旋钮，逐轮记录验证集 top-1 / top-3 和耗时。

- id_dim  ：dish_id_emb 的维度（当前硬编码 8）—— 「每道菜的个性化自由度」
- hidden  ：DishScorer MLP 隐层宽度（当前 48）

产出：结果表 results/summary.md + results/summary.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RES.mkdir(exist_ok=True)

EPOCHS = 60
SEED = 7
GRID = [(idd, h) for idd in (4, 8, 16, 32, 64) for h in (48, 96)]


def run_one(id_dim: int, hidden: int) -> dict | None:
    tag = f"id{id_dim}_h{hidden}"
    out = RES / f"{tag}.json"
    if out.exists():
        print(f"[跳过] {tag} 已有结果", flush=True)
        return json.loads(out.read_text(encoding="utf-8"))
    print(f"\n===== id_dim={id_dim}  hidden={hidden} =====", flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(
        [sys.executable, "-u", "cap_run.py", "--id-dim", str(id_dim),
         "--hidden", str(hidden), "--epochs", str(EPOCHS), "--seed", str(SEED),
         "--tag", tag],
        cwd=str(HERE), capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    for line in (r.stdout or "").splitlines()[-4:]:
        print("   ", line, flush=True)
    if r.returncode != 0 or not out.exists():
        print(f"[失败] {tag} rc={r.returncode}", flush=True)
        print((r.stderr or "")[-800:], flush=True)
        return None
    j = json.loads(out.read_text(encoding="utf-8"))
    j["_outer_seconds"] = round(time.perf_counter() - t0, 1)
    return j


def main() -> int:
    # 先清掉冒烟测试留下的文件，避免污染结论
    (RES / "SMOKE.json").unlink(missing_ok=True)

    results = []
    for id_dim, hidden in GRID:
        j = run_one(id_dim, hidden)
        if j:
            results.append(j)
            (RES / "summary.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    if not results:
        print("没有任何结果")
        return 1

    base = next((r for r in results if r["id_dim"] == 8 and r["hidden"] == 48), None)
    lines = []
    lines.append("# 容量扫描：dish_id_emb 维度 × 隐层宽度\n")
    lines.append(f"- 数据固定：317 道菜 / 18 品类 / 2000 情境 / 7340 行（同一份 samples.csv）")
    lines.append(f"- 训练：{EPOCHS} 轮，seed={SEED}，其余超参与 train.py 一致")
    lines.append(f"- 指标：验证集 top-1 / top-3（对照 gold，口径同 train._eval_top1）")
    lines.append(f"- 调参旋钮：`id_dim`=dish_id_emb 维度（当前 8）、`hidden`=DishScorer 隐层（当前 48）\n")

    lines.append("## 结果\n")
    lines.append("| id_dim | hidden | 参数量 | 最佳轮 | val top-1 | val top-3 | 每轮(s) | 最快达到 top-1≥50% 的轮 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: (x["id_dim"], x["hidden"])):
        pe = r.get("per_epoch", [])
        first50 = next((e["epoch"] for e in pe if e.get("val_top1", 0) >= 0.50), None)
        t1 = r.get("best_epoch_val_top1")
        t3 = r.get("best_epoch_val_top3")
        lines.append(
            f"| {r['id_dim']} | {r['hidden']} | {r['n_params']:,} | {r['best_epoch']} | "
            f"{t1:.2%} | {t3:.2%} | {r['epoch_total_s_mean']:.2f} | "
            f"{first50 if first50 is not None else '—'} |")

    if base:
        lines.append("\n## 与当前配置（id_dim=8, hidden=48）的对比\n")
        lines.append("| 配置 | 参数量 | val top-1 | Δ top-1 | val top-3 | Δ top-3 | 每轮(s) |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in sorted(results, key=lambda x: (x["id_dim"], x["hidden"])):
            t1 = r.get("best_epoch_val_top1") or 0
            t3 = r.get("best_epoch_val_top3") or 0
            b1 = base.get("best_epoch_val_top1") or 0
            b3 = base.get("best_epoch_val_top3") or 0
            lines.append(
                f"| id{r['id_dim']}_h{r['hidden']} | {r['n_params']:,} | {t1:.2%} | "
                f"{t1-b1:+.2%} | {t3:.2%} | {t3-b3:+.2%} | {r['epoch_total_s_mean']:.2f} |")

    # 峰值也能反映「容量够不够」：若大模型峰值更高，说明当前欠拟合
    lines.append("\n## 逐轮峰值（排除选轮口径影响）\n")
    lines.append("| 配置 | 峰值 val top-1 | 峰值 val top-3 | 末轮 val top-1 | 是否过拟合(峰值-末轮) |")
    lines.append("|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: (x["id_dim"], x["hidden"])):
        pk1 = r.get("peak_val_top1") or 0
        fn1 = r.get("final_val_top1") or 0
        lines.append(f"| id{r['id_dim']}_h{r['hidden']} | {pk1:.2%} | "
                     f"{r.get('peak_val_top3', 0):.2%} | {fn1:.2%} | {pk1-fn1:+.2%} |")

    (RES / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n结果已写入 {RES/'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
