# -*- coding: utf-8 -*-
"""
build_dataset.py —— 生成「今天该吃什么」决策数据集

思路（三层）：
  1. 菜品库 dishes.csv      ：46 道菜，8 个品类，每道菜带客观属性
  2. 情境采样 + 效用函数     ：对「情境 x 菜品」打一个 0~1 的分数 utility
  3. 训练表 samples.csv      ：情境特征 + 实际选择 + 该菜品的真实效用

关键点：真实效用 utility 就是「结果信号」。
        它连续、有噪声、且只有被选中的那道菜能被观测到 —— 这正是决策数据的形态。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------------------
SEED = 20240617
N_SCENARIOS = 2000      # 唯一情境数
SAMPLES_PER_SCENARIO = 3  # 每个情境重复采样的次数（模拟同一个情境下的多次真实选择）
OUT = Path(__file__).resolve().parent / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------------------
# 1. 菜品库
# ----------------------------------------------------------------------------
# 字段顺序: 菜名, 品类, 耗时(分钟), 价格(元), 辣度0-3, 厚重感1-5, 健康度1-5,
#           温度(0凉/1温/2热), 碳水1-5, 蛋白质1-5, 忌口标签

DISH_COLS = ["dish_name", "category", "time_cost", "price", "spicy", "heavy",
             "healthy", "temperature", "carb", "protein", "taboo"]

# 菜品库从 artifacts/dishes.csv 读取 —— 单一数据源。
# （旧版把菜品硬编码在本文件里，与 dishes.csv 是两份数据，改一处漏一处必然不同步。
#  现在 dishes.csv 是唯一来源，features.py 也读同一个文件。）
DISH_CSV = OUT / "dishes.csv"
if not DISH_CSV.exists():
    raise SystemExit(
        f"找不到菜品库 {DISH_CSV}。\n"
        f"请先准备菜品库（至少含 {DISH_COLS} 十一列），或运行仓库根目录的 "
        f"_merge_dishes.py 生成。"
    )

dishes = pd.read_csv(DISH_CSV)
_missing = [c for c in DISH_COLS if c not in dishes.columns]
if _missing:
    raise SystemExit(f"菜品库缺少必需列：{_missing}；实际列：{list(dishes.columns)}")

dishes["taboo"] = dishes.taboo.fillna("").astype(str)
if "is_veg" not in dishes.columns:
    # 兼容没有 is_veg 列的老库：按 taboo 与菜名粗判
    _meat = ("鸡", "牛", "肉", "鱼", "排", "羊", "虾", "蟹", "鹅", "鸭", "猪", "培根")
    dishes["is_veg"] = [
        1 if (str(t) == "" and not any(m in n for m in _meat)) else 0
        for n, t in zip(dishes.dish_name, dishes.taboo)
    ]
dishes["is_veg"] = dishes.is_veg.fillna(0).astype(int)

if "dish_id" not in dishes.columns:
    dishes.insert(0, "dish_id", range(len(dishes)))
dishes = dishes.reset_index(drop=True)

CATEGORIES = sorted(dishes.category.unique().tolist())
CAT_TO_ID = {c: i for i, c in enumerate(CATEGORIES)}

# 每个品类的平均属性（用于情境特征里的品类级热度倾向，防止模型走捷径）
cat_table = dishes.groupby("category").agg(
    cat_mean_price=("price", "mean"),
    cat_mean_time=("time_cost", "mean"),
    cat_mean_healthy=("healthy", "mean"),
    cat_mean_heavy=("heavy", "mean"),
).reset_index()


# ----------------------------------------------------------------------------
# 2. 情境采样
# ----------------------------------------------------------------------------
TIME_SLOTS = ["早餐", "午餐", "下午加餐", "晚餐", "夜宵"]
MOODS = ["开心", "一般", "疲惫", "压力大", "低落", "兴奋"]
WEATHERS = ["晴", "阴", "雨", "雪", "热", "冷"]
COMPANIONS = ["一个人", "同事", "朋友", "家人", "约会"]
HEALTH = ["不关心", "随便", "想健康", "严格控制"]
BUDGETS = ["省着点", "正常", "想吃好点", "不差钱"]
SPICY_TOL = ["不吃辣", "微辣", "中辣", "重辣"]
TABOOS = ["", "海鲜", "乳制品", "麸质", "海鲜,乳制品", "素食"]

TIME_SLOT_HOUR = {"早餐": 7.5, "午餐": 12.0, "下午加餐": 15.5, "晚餐": 19.0, "夜宵": 22.5}

# 权重：轻微随机化，模拟「每个人偏好不同」
def sample_scenario(i: int) -> dict:
    t_slot = rng.choice(TIME_SLOTS, p=[0.12, 0.30, 0.08, 0.34, 0.16])
    hunger = int(rng.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.35, 0.30, 0.15]))
    avail = int(np.clip(rng.normal(50, 25), 15, 150))
    budget = rng.choice(BUDGETS, p=[0.18, 0.45, 0.27, 0.10])
    mood = rng.choice(MOODS, p=[0.20, 0.30, 0.20, 0.12, 0.08, 0.10])
    weather = rng.choice(WEATHERS, p=[0.30, 0.20, 0.18, 0.06, 0.14, 0.12])
    companion = rng.choice(COMPANIONS, p=[0.30, 0.20, 0.22, 0.18, 0.10])
    health = rng.choice(HEALTH, p=[0.34, 0.26, 0.30, 0.10])
    spicy = rng.choice(SPICY_TOL, p=[0.22, 0.30, 0.30, 0.18])
    taboo = rng.choice(TABOOS, p=[0.60, 0.08, 0.08, 0.08, 0.08, 0.08])
    # 注意：这里刻意不放「随机个人偏好权重」。
    # 效用必须是情境的确定性函数，否则会引入模型永远无法从特征中学到的噪声。
    return dict(scenario_id=i, time_slot=str(t_slot), hunger=hunger,
                available_time=avail, budget=str(budget), mood=str(mood),
                weather=str(weather), companion=str(companion),
                health_goal=str(health), spicy_tol=str(spicy), taboo=str(taboo))


# ----------------------------------------------------------------------------
# 3. 效用函数（这是「结果信号」的核心）
# ----------------------------------------------------------------------------
SPICY_TOL_LEVEL = {"不吃辣": 0, "微辣": 1, "中辣": 2, "重辣": 3}
BUDGET_LEVEL = {"省着点": 0, "正常": 1, "想吃好点": 2, "不差钱": 3}
BUDGET_PRICE = {0: 18.0, 1: 35.0, 2: 60.0, 3: 95.0}
SLOT_PREF = {  # 每个时段对各属性的理想区间
    "早餐":     dict(heavy=(1, 2),  temp=(2, 2), time=(20, 8)),
    "午餐":     dict(heavy=(3, 4),  temp=(1, 2), time=(40, 20)),
    "下午加餐": dict(heavy=(1, 2),  temp=(0, 1), time=(20, 8)),
    "晚餐":     dict(heavy=(3, 4),  temp=(1, 2), time=(55, 25)),
    "夜宵":     dict(heavy=(2, 3),  temp=(2, 2), time=(40, 20)),
}


def _trapezoid(x: float, lo: float, hi: float, soft: float) -> float:
    """x 落在 [lo,hi] 得 1 分，超出后线性衰减到 soft 处为 0。"""
    if x < lo:
        return max(0.0, 1.0 - (lo - x) / max(soft, 1e-6))
    if x > hi:
        return max(0.0, 1.0 - (x - hi) / max(soft, 1e-6))
    return 1.0


def utility(row: pd.Series, sc: dict) -> float:
    """给定情境 sc，计算某道菜的效用分。第 0 项时段适配度是品类级加成，
    其余是菜品级属性。返回原始分（未归一化），由 _scale 做量程变换。"""
    slot = sc["time_slot"]
    pref = SLOT_PREF[slot]
    mood, weather, comp = sc["mood"], sc["weather"], sc["companion"]
    health = sc["health_goal"]
    score = 0.0

    # (0) 时段适配度：现实里有些搭配几乎不会发生（早餐吃火锅/夜宵吃草）
    if slot == "早餐":
        score += {"早餐粥点": 1.25, "面食粉面": 0.95, "轻食健康": 0.85}.get(row.category, 0.0)
        score -= {"火锅麻辣烫": 1.6, "烧烤小吃": 0.85, "家常炒菜": 0.55}.get(row.category, 0.0)
    elif slot == "夜宵":
        score += {"烧烤小吃": 1.15, "火锅麻辣烫": 0.75, "面食粉面": 0.55}.get(row.category, 0.0)
        score -= {"轻食健康": 1.25, "早餐粥点": 0.75}.get(row.category, 0.0)
    elif slot == "下午加餐":
        score += {"轻食健康": 1.15, "早餐粥点": 0.5}.get(row.category, 0.0)
    else:  # 午/晚餐
        score += {"家常炒菜": 0.35, "火锅麻辣烫": 0.30, "西式简餐": 0.20}.get(row.category, 0.0)

    # (1) 时段契合：厚重感 + 温度
    lo, hi = pref["heavy"]
    score += 1.60 * _trapezoid(row.heavy, lo, hi, 2.2)
    lo, hi = pref["temp"]
    score += 1.00 * _trapezoid(row.temperature, lo, hi, 1.5)

    # (2) 时间够不够
    t_need = row.time_cost * (1.15 if comp in ("朋友", "家人", "约会") else 1.0)
    score += 1.60 * _trapezoid(sc["available_time"] - t_need, 0, 999, 30.0)

    # (3) 预算
    b = BUDGET_LEVEL[sc["budget"]]
    p_ideal = BUDGET_PRICE[b]
    lo, hi = (p_ideal * 0.5, p_ideal * 1.15) if b < 3 else (p_ideal * 0.6, 999)
    score += 1.20 * _trapezoid(row.price, lo, hi, p_ideal * 1.3)

    # (4) 饥饿度 -> 分量
    score += 1.10 * _trapezoid(row.heavy + 0.5 * row.carb, sc["hunger"] - 0.5,
                               sc["hunger"] + 2.0, 3.0)
    if sc["hunger"] >= 4 and row.heavy <= 2:
        score -= 0.50

    # (5) 心情
    if mood in ("疲惫", "压力大"):
        score += 0.45 * _trapezoid(row.heavy + row.carb, 6, 10, 5.0)
        score += 0.25 * row.carb / 5.0
    if mood == "低落":
        score += 0.45 * (row.spicy / 3.0)
        score += 0.25 * _trapezoid(row.heavy, 3, 5, 2.0)
    if mood == "开心":
        score += 0.18 * (1.0 - abs(row.heavy - 3.5) / 3.5)
    if mood == "兴奋":
        score += 0.20 * (row.protein / 5.0)

    # (6) 天气
    if weather == "冷":
        score += 0.65 * _trapezoid(row.temperature, 2, 2, 1.2)
    elif weather == "热":
        score += 0.65 * _trapezoid(row.temperature, 0, 1, 1.2)
    elif weather == "雨":
        score += 0.42 * _trapezoid(row.time_cost, 0, 22, 26.0)   # 懒得出门
        score += 0.22 * _trapezoid(row.temperature, 1, 2, 1.5)
    elif weather == "雪":
        # 雪天：既不想出门（要快），又想吃热乎的（这是测试集专属情境）
        score += 0.70 * _trapezoid(row.time_cost, 0, 30, 34.0)
        score += 0.85 * _trapezoid(row.temperature, 2, 2, 1.8)

    # (7) 一起吃
    if comp in ("朋友", "家人", "约会"):
        score += 0.72 * _trapezoid(row.heavy, 3, 5, 2.0)
        score += 0.55 * _trapezoid(row.time_cost, 25, 999, 30.0)   # 愿意等 / 慢慢吃
        if comp == "约会":
            score += 0.32 * _trapezoid(row.temperature, 2, 2, 1.5)
        if comp == "朋友":
            score += 0.22 * (row.spicy / 3.0)
    elif comp == "同事":
        score += 0.40 * _trapezoid(row.time_cost, 0, 20, 25.0)
        score += 0.18 * _trapezoid(row.spicy, 0, 1, 2.0)           # 避免气味尴尬

    # (8) 健康诉求：先当「筛选条件」，再当「加分项」
    #     这是本效用函数里最关键的一条设计：如果只给健康菜加分，油腻面食
    #     能靠「温热、厚重、顶饱」等其它维度追平沙拉，结果就完全不符合现实。
    #     真实的减脂/控糖行为是先把一大类选项排除掉，再在剩下的里面挑。
    hw = {"不关心": 0.00, "随便": 0.00, "想健康": 0.55, "严格控制": 1.00}[health]
    if health == "想健康":
        if row.healthy <= 2:
            score -= 1.30 * (3 - row.healthy) / 2.0
        if row.heavy >= 5:
            score -= 0.85
    elif health == "严格控制":
        if row.healthy <= 2:
            return 0.0            # 直接排除：这一档根本不会被考虑
        if row.heavy >= 4:
            score -= 1.30
        elif row.heavy == 3:
            score -= 0.35
    if hw > 0:
        score += 1.20 * hw * row.healthy / 5.0
        score -= 0.55 * hw * max(0.0, row.heavy - 3) / 2.0

    # (9) 辣度
    tol = SPICY_TOL_LEVEL[sc["spicy_tol"]]
    eff = min(row.spicy, tol)
    if row.spicy > tol:
        score -= 1.15 * (row.spicy - tol)          # 超过承受能力：硬伤
    score += 0.32 * (eff / 3.0)

    # (10) 忌口：一票否决（过敏原按标签求交集）
    if sc["taboo"] and row.taboo:
        if set(str(sc["taboo"]).split(",")) & set(str(row.taboo).split(",")):
            return 0.0
    # 情境的「素食」诉求必须单独判，因为它是饮食约束、不是过敏原，
    # 不写在 dish.taboo 里（见下一段）。否则素食情境会误放行所有荤菜。
    if sc["taboo"] and "素食" in str(sc["taboo"]).split(",") and int(row.get("is_veg", 0)) != 1:
        return 0.0

    # (11) 素食：用菜品库里显式的 is_veg 列，而不是靠菜名关键字猜。
    #      旧版按 ("鸡","牛","肉","鱼","排","羊","三文鱼","火腿","虾") 匹配菜名，
    #      会漏掉「酿豆腐」（含肉馅）、「蚝仔烙」、「猪脚饭」这类名字里没有上述字的荤菜，
    #      也会误伤「素鸡」「素蟹粉」这类名字带荤字的素菜。
    if sc["taboo"] == "素食" and int(row.get("is_veg", 0)) != 1:
        return 0.0

    # (12) 夜宵惩罚健康菜（深夜吃草有点惨）
    if slot == "夜宵" and row.healthy >= 5 and row.heavy <= 2:
        score -= 0.35

    # (13) 下午加餐：不接受大餐
    if slot == "下午加餐" and row.heavy >= 4:
        score -= 0.70

    return float(score)


# 原始分的合理量程上限（实测约 12.4），决定效用的分辨率
SCORE_RANGE = 13.5

# 情境筛选阈值：最优与次优的差距。改这里就够了。
CLEAR_GAP = 0.006      # 差距大于此值 = 选择明确，保留
TIE_GAP = 0.0015       # 差距小于此值 = 近似并列，随机保留一半（真人也会随手选）


def _final_scale(raw: np.ndarray) -> np.ndarray:
    """把 0~13 的原始分映射到 (0,1) 区间。
    刻意用「近线性」标度而不是指数压缩：指数会把 0.5 分的真实差距压成 0.001，
    导致最优和次优几乎并列、监督信号失去分辨力。经验上线性标度最利于学习。"""
    return np.clip(np.maximum(raw, 0.0) / SCORE_RANGE, 0.0, 0.995)


# ----------------------------------------------------------------------------
# 4. 生成数据（扫描式情境筛选：留下「选择足够明确」且口味多样的情境）
# ----------------------------------------------------------------------------
def build():
    print("生成候选情境并计算效用矩阵 ...")
    candidates = [sample_scenario(i) for i in range(N_SCENARIOS * 5)]
    n_dish = len(dishes)
    U = np.zeros((len(candidates), n_dish))
    for ci, sc in enumerate(candidates):
        U[ci] = [utility(d, sc) for _, d in dishes.iterrows()]
    V = _final_scale(U)
    top2 = np.sort(V, axis=1)[:, -2:][:, ::-1]
    gap = top2[:, 0] - top2[:, 1]
    best_idx = V.argmax(axis=1)
    best_cat = dishes.category.to_numpy()[best_idx]
    best_slot = np.array([c["time_slot"] for c in candidates])

    # 筛选：并列（近似无差别）只保留一半（真人也有随手选的时候），
    #       其余要求差距 >0.006（约等于原始分 0.08，是明确的选择偏好）
    keep_clear = gap > CLEAR_GAP
    keep_tie = gap <= TIE_GAP
    tie_keep = rng.random(len(candidates)) < 0.5
    allowed = keep_clear | (keep_tie & tie_keep)

    # 口味均衡：品类 x 时段 组合最多 160 个情境，避免盖饭/面食一家独大
    CAP = 160
    counts: dict = {}
    keep = []
    for ci in range(len(candidates)):
        if not allowed[ci]:
            continue
        key = (best_cat[ci], best_slot[ci])
        if counts.get(key, 0) >= CAP:
            continue
        counts[key] = counts.get(key, 0) + 1
        keep.append(ci)
    keep = np.array(keep)
    order = rng.permutation(len(keep))
    keep = keep[order][:N_SCENARIOS]

    scenarios = []
    for new_id, ci in enumerate(keep):
        sc = dict(candidates[int(ci)])
        sc["scenario_id"] = new_id
        scenarios.append(sc)
    print(f"  候选 {len(candidates)} -> 保留 {len(scenarios)} 个情境")

    rows, gold_rows = [], []
    for sc in scenarios:
        utils = _final_scale(np.array([utility(d, sc) for _, d in dishes.iterrows()]))
        order_i = np.argsort(-utils)
        gold = dishes.iloc[order_i[0]]
        gold_rows.append({**sc, "gold_dish_id": int(gold.dish_id),
                          "gold_dish": gold.dish_name,
                          "gold_category": gold.category,
                          "gold_utility": float(utils[order_i[0]]),
                          "runner_up_dish": dishes.iloc[order_i[1]].dish_name,
                          "runner_up_utility": float(utils[order_i[1]])})
        # 每个情境重复采样：88% 选最优，12% 在次优里挑一个（模拟真人随手选）
        for _ in range(SAMPLES_PER_SCENARIO):
            if rng.random() < 0.88:
                pick = order_i[0]
            else:
                pick = order_i[int(rng.integers(1, min(5, len(order_i))))]
            d = dishes.iloc[pick]
            u = float(utils[pick])
            # 结果信号 = 真实效用 + 感知噪声
            observed = float(np.clip(u + rng.normal(0, 0.035), 0.0, 1.0))
            rows.append({**sc,
                         "dish_id": int(d.dish_id), "dish_name": d.dish_name,
                         "category": d.category, "chosen": 1,
                         "source": "chosen",
                         "utility": round(u, 4),
                         "observed_signal": round(observed, 4)})

        # ---- 考虑集评分（关键补强）----
        # 纯 bandit 数据每个情境只观测 1 道菜，模型永远不知道同品类其他菜的真实
        # 效用，于是「品类内选哪道」几乎学不动（实测 top-1 只有 14%，接近随机）。
        # 现实中人常会顺手给「考虑过的几道菜」打分（收藏、点评、心里记一笔），
        # 这部分信号成本极低但信息量极大。这里按 35% 的情境生成，取该品类前 3 名。
        if rng.random() < 0.35:
            cat_best = order_i[0]
            same_cat = [j for j in order_i
                        if dishes.iloc[j].category == dishes.iloc[cat_best].category][:3]
            for j in same_cat:
                if j == order_i[0]:
                    continue                     # 最优那道已经在 chosen 里了
                d = dishes.iloc[j]
                u = float(utils[j])
                observed = float(np.clip(u + rng.normal(0, 0.035), 0.0, 1.0))
                rows.append({**sc,
                             "dish_id": int(d.dish_id), "dish_name": d.dish_name,
                             "category": d.category, "chosen": 0,
                             "source": "considered",
                             "utility": round(u, 4),
                             "observed_signal": round(observed, 4)})

    samples = pd.DataFrame(rows)
    gold_df = pd.DataFrame(gold_rows)
    sc_df = pd.DataFrame(scenarios)

    # ---------------- 划分：把「雪天」整个留作测试集（分布偏移测试） ----------------
    snow_scen = set(sc_df.loc[sc_df.weather == "雪", "scenario_id"])
    other_scen = np.array(sorted(set(sc_df.scenario_id) - snow_scen))
    rng.shuffle(other_scen)
    n_val = int(len(other_scen) * 0.15)
    val_scen = set(other_scen[:n_val])
    train_scen = set(other_scen[n_val:])
    # 测试集 = 全部雪天 + 200 个随机非雪天情境（做对照）
    ctrl = set(rng.choice(sorted(train_scen), size=min(200, len(train_scen)), replace=False))
    test_scen = snow_scen | ctrl

    def split_of(sid):
        if sid in test_scen:
            return "test"
        if sid in val_scen:
            return "val"
        return "train"

    samples["split"] = samples.scenario_id.map(split_of)
    gold_df["split"] = gold_df.scenario_id.map(split_of)
    # 训练/验证集里剔除雪天，保证测试集是真正的「没见过」情境
    samples.loc[(samples.split == "test") & (samples.weather == "雪"), "shift"] = "snow"
    samples["shift"] = samples["shift"].fillna("normal")

    dishes.to_csv(OUT / "dishes.csv", index=False, encoding="utf-8-sig")
    cat_table.to_csv(OUT / "categories.csv", index=False, encoding="utf-8-sig")
    samples.to_csv(OUT / "samples.csv", index=False, encoding="utf-8-sig")
    gold_df.to_csv(OUT / "scenarios.csv", index=False, encoding="utf-8-sig")

    # 额外产出：测试集按情境聚合的对照表
    gold_df[gold_df.split == "test"].to_csv(
        OUT / "test_scenarios.csv", index=False, encoding="utf-8-sig")

    meta = {
        "seed": SEED,
        "n_scenarios": N_SCENARIOS,
        "samples_per_scenario": SAMPLES_PER_SCENARIO,
        "n_dishes": len(dishes),
        "categories": CATEGORIES,
        "n_samples": len(samples),
        "split_counts": samples.split.value_counts().to_dict(),
        "scenario_split_counts": gold_df.split.value_counts().to_dict(),
        "shift_counts": samples[samples.split == "test"]["shift"].value_counts().to_dict(),
        "context_fields": ["time_slot", "hunger", "available_time", "budget", "mood",
                           "weather", "companion", "health_goal", "spicy_tol", "taboo"],
        "dish_fields": ["category", "time_cost", "price", "spicy", "heavy",
                        "healthy", "temperature", "carb", "protein"],
        "source_counts": samples["source"].value_counts().to_dict(),
    }
    (OUT / "dataset_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return samples, gold_df, dishes, meta


if __name__ == "__main__":
    samples, gold, dish_df, meta = build()
    print("=" * 62)
    print("数据集生成完成 ->", OUT)
    print("=" * 62)
    print(f"菜品数        : {meta['n_dishes']}   (品类 {len(meta['categories'])} 个)")
    print(f"唯一情境数    : {meta['n_scenarios']}")
    print(f"总样本行数    : {meta['n_samples']}")
    print(f"按 split      : {meta['split_counts']}")
    print(f"按情境 split  : {meta['scenario_split_counts']}")
    print(f"测试集偏移    : {meta['shift_counts']}")
    print("-" * 62)
    print("各品类被选为最优的次数（标签分布）：")
    print(gold.gold_category.value_counts().to_string())
    print("-" * 62)
    print("【结果信号质量诊断】")
    u = gold.gold_utility.values
    g = (gold.gold_utility - gold.runner_up_utility).values
    print(f"  效用分 min/中位/max : {u.min():.3f} / {np.median(u):.3f} / {u.max():.3f}")
    print(f"  分数 >= 0.99 的比例  : {(u >= 0.99).mean():.1%}   (越低越好，说明没有饱和)")
    print(f"  最优-次优 差值 中位  : {np.median(g):.4f}")
    print(f"  并列(<0.002)比例     : {(g < 0.002).mean():.1%}")
    print(f"  清晰(>0.008)比例     : {(g > 0.008).mean():.1%}")
    print("-" * 62)
    print("样例（前 6 行）：")
    cols = ["scenario_id", "time_slot", "hunger", "available_time", "mood", "weather",
            "companion", "health_goal", "spicy_tol", "dish_name", "category",
            "utility", "observed_signal", "split", "shift"]
    print(samples[cols].head(6).to_string(index=False))
