# -*- coding: utf-8 -*-
"""等价性验证：新的向量化 utility_all() 必须与旧的逐行 utility() 逐元素一致。

用旧实现（备份文件，含原始标量 utility）对新实现求最大绝对误差。
这是重构的安全网 —— 效用函数是这个项目的「结果信号」，算错了整个数据集就废了。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\123\Desktop\桌面应用\文档")
sys.path.insert(0, str(BASE / "meal-decision"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


new = load("bd_new", BASE / "meal-decision" / "build_dataset.py")
old = load("bd_old", BASE / "meal-decision" / "_build_dataset_scalar_备份.py")

# 两个模块读同一份 dishes.csv，但各自有独立的 rng，用同一组固定情境对比
rng = np.random.default_rng(7)
scenarios = []
for i in range(400):
    ts = rng.choice(new.TIME_SLOTS)
    scenarios.append(dict(
        scenario_id=i, time_slot=str(ts),
        hunger=int(rng.integers(1, 6)), available_time=int(rng.integers(15, 151)),
        budget=str(rng.choice(new.BUDGETS)), mood=str(rng.choice(new.MOODS)),
        weather=str(rng.choice(new.WEATHERS)), companion=str(rng.choice(new.COMPANIONS)),
        health_goal=str(rng.choice(new.HEALTH)), spicy_tol=str(rng.choice(new.SPICY_TOL)),
        taboo=str(rng.choice(new.TABOOS)),
    ))

da_new = new._dish_arrays(new.dishes)
# 旧备份里还没有 _dish_arrays（那是随向量化一起加的），这里就地为旧模块构造一份
da_old = {
    "name": old.dishes.dish_name.to_numpy(),
    "category": old.dishes.category.to_numpy(),
    "time_cost": old.dishes.time_cost.to_numpy(dtype=float),
    "price": old.dishes.price.to_numpy(dtype=float),
    "spicy": old.dishes.spicy.to_numpy(dtype=float),
    "heavy": old.dishes.heavy.to_numpy(dtype=float),
    "healthy": old.dishes.healthy.to_numpy(dtype=float),
    "temperature": old.dishes.temperature.to_numpy(dtype=float),
    "carb": old.dishes.carb.to_numpy(dtype=float),
    "protein": old.dishes.protein.to_numpy(dtype=float),
    "taboo": old.dishes.taboo.fillna("").astype(str).to_numpy(),
    "is_veg": old.dishes.is_veg.fillna(0).astype(int).to_numpy(),
}
assert list(da_new["name"]) == list(da_old["name"]), "两个模块菜品顺序不一致"

worst = 0.0
worst_at = None
worst_final = 0.0
n_neg_old = 0
mismatch_rank = 0
for sc in scenarios:
    v_new = new.utility_all(da_new, sc)
    v_old = np.array([old.utility(d, sc) for _, d in old.dishes.iterrows()])
    n_neg_old += int((v_old < 0).sum())

    err = np.abs(v_new - v_old)
    m = float(err.max())
    if m > worst:
        worst = m
        worst_at = (sc, int(err.argmax()), float(v_new[err.argmax()]), float(v_old[err.argmax()]))

    # 真正决定数据集的量：_final_scale 之后的效用，以及排序/最优菜是否一致
    f_new = new._final_scale(v_new)
    f_old = old._final_scale(v_old)
    worst_final = max(worst_final, float(np.abs(f_new - f_old).max()))
    if int(f_new.argmax()) != int(f_old.argmax()):
        mismatch_rank += 1

print(f"对比情境数        : {len(scenarios)}")
print(f"每次对比菜品数    : {len(new.dishes)}")
print(f"总对比元素数      : {len(scenarios)*len(new.dishes):,}")
print()
print("[原始分]（未归一化，旧版允许为负，新版截断到 0）")
print(f"  最大绝对误差    : {worst:.3e}")
print(f"  旧版为负的元素数: {n_neg_old:,}  (占比 {n_neg_old/(len(scenarios)*len(new.dishes)):.1%})")
if worst_at:
    sc, i, a, b = worst_at
    print(f"  最大误差处      : 情境{sc['scenario_id']} / 菜「{da_new['name'][i]}」 新={a:.6f} 旧={b:.6f}")
print()
print("[归一化后]（_final_scale，这才是写进 samples.csv 的信号）")
print(f"  最大绝对误差    : {worst_final:.3e}")
print(f"  最优菜不一致次数: {mismatch_rank} / {len(scenarios)}")
print()
ok = worst_final < 1e-9 and mismatch_rank == 0
print(f"结论              : {'✅ 最终信号完全一致' if ok else '❌ 最终信号存在差异，需检查'}")

# 顺带量一下提速
import time
t = time.time()
for sc in scenarios:
    new.utility_all(da_new, sc)
dt_new = time.time() - t
t = time.time()
for sc in scenarios[:20]:
    for _, d in old.dishes.iterrows():
        old.utility(d, sc)
dt_old_20 = time.time() - t
print(f"\n向量化耗时        : {dt_new:.3f}s / {len(scenarios)} 情境")
print(f"旧实现耗时        : {dt_old_20:.3f}s / 20 情境 -> 推算 {dt_old_20*len(scenarios)/20:.1f}s")
print(f"提速              : 约 {dt_old_20*len(scenarios)/20/dt_new:.0f}x")
