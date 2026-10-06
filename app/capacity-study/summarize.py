# -*- coding: utf-8 -*-
"""汇总已完成的容量扫描结果（不启动新训练）。"""
import json
import statistics
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"

rows = []
for p in sorted(RES.glob("id*_h*.json")):
    j = json.loads(p.read_text(encoding="utf-8"))
    pe = j.get("per_epoch", [])
    t1s = [e.get("val_top1") for e in pe if e.get("val_top1") is not None]
    t3s = [e.get("val_top3") for e in pe if e.get("val_top3") is not None]
    # 取后 20 轮的平均值：更稳，避免单轮波动被当成真实差异
    tail1 = t1s[-20:] if len(t1s) >= 20 else t1s
    tail3 = t3s[-20:] if len(t3s) >= 20 else t3s
    first50 = next((e["epoch"] for e in pe if (e.get("val_top1") or 0) >= 0.50), None)
    rows.append(dict(
        id_dim=j["id_dim"], hidden=j["hidden"], n_params=j["n_params"],
        best_ep=j["best_epoch"],
        best1=j.get("best_epoch_val_top1") or 0, best3=j.get("best_epoch_val_top3") or 0,
        peak1=j.get("peak_val_top1") or 0, peak3=j.get("peak_val_top3") or 0,
        final1=j.get("final_val_top1") or 0, final3=j.get("final_val_top3") or 0,
        tail1=statistics.mean(tail1) if tail1 else 0,
        tail3=statistics.mean(tail3) if tail3 else 0,
        sd1=statistics.pstdev(tail1) if len(tail1) > 1 else 0,
        sd3=statistics.pstdev(tail3) if len(tail3) > 1 else 0,
        ep_s=j["epoch_total_s_mean"], fwd=j["fwd_s_mean"], bwd=j["bwd_s_mean"],
        wall=j["wall_s"], first50=first50, epochs=j["epochs"],
    ))

rows.sort(key=lambda r: (r["id_dim"], r["hidden"]))
base = next((r for r in rows if r["id_dim"] == 8 and r["hidden"] == 48), None)

print(f"已完成配置数: {len(rows)} / 12   (每个 {rows[0]['epochs']} 轮)")
print()
hdr = (f"{'配置':<12}{'参数量':>9}{'最佳轮':>7}"
       f"{'最佳top1':>10}{'最佳top3':>10}"
       f"{'末20轮top1':>12}{'±sd':>7}{'末20轮top3':>12}{'±sd':>7}"
       f"{'每轮s':>8}{'fwd':>7}{'bwd':>7}")
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"id{r['id_dim']:<3}h{r['hidden']:<6}{r['n_params']:>9,}{r['best_ep']:>7}"
          f"{r['best1']:>9.2%}{r['best3']:>10.2%}"
          f"{r['tail1']:>12.2%}{r['sd1']:>7.2%}{r['tail3']:>12.2%}{r['sd3']:>7.2%}"
          f"{r['ep_s']:>8.2f}{r['fwd']:>7.2f}{r['bwd']:>7.2f}")

if base:
    print()
    print(f"以当前配置 id8_h48 为基线（末20轮 top1={base['tail1']:.2%} ± {base['sd1']:.2%}）")
    print(f"{'配置':<12}{'参数量':>10}{'Δ参数量':>10}{'Δtop1':>9}{'Δtop3':>9}{'Δ每轮s':>9}")
    for r in rows:
        print(f"id{r['id_dim']:<3}h{r['hidden']:<6}{r['n_params']:>10,}"
              f"{r['n_params']-base['n_params']:>+10,}"
              f"{r['tail1']-base['tail1']:>+9.2%}{r['tail3']-base['tail3']:>+9.2%}"
              f"{r['ep_s']-base['ep_s']:>+9.2f}")

print()
print("首次达到 val top-1 ≥ 50% 的轮次（越小说明收敛越快）:")
for r in rows:
    print(f"  id{r['id_dim']:<3}h{r['hidden']:<5} -> {r['first50'] if r['first50'] is not None else '—'}")

# 与基线的差是否超过噪声（用末20轮 sd 粗判）
if base:
    print()
    print(f"噪声参考：基线末20轮 top1 的 sd = {base['sd1']:.2%}（约 {base['sd1']*283:.1f} 个情境）")
    sig = [r for r in rows if abs(r["tail1"] - base["tail1"]) > 2 * max(base["sd1"], 1e-9)]
    print(f"超出 2×sd 的配置: {len(sig)} 个 -> "
          + (", ".join(f"id{r['id_dim']}_h{r['hidden']}" for r in sig) if sig else "无"))
