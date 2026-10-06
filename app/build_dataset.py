# -*- coding: utf-8 -*-
"""
build_dataset.py —— 生成「今天该吃什么」决策数据集

思路（三层）：
  1. 菜品库 dishes.csv      ：317 道菜，18 个品类，每道菜带客观属性
  2. 情境采样 + 效用函数     ：对「情境 x 菜品」打一个 0~1 的分数 utility
  3. 训练表 samples.csv      ：情境特征 + 实际选择 + 该菜品的真实效用

关键点：真实效用 utility 就是「结果信号」。
        它连续、有噪声、且只有被选中的那道菜能被观测到 —— 这正是决策数据的形态。
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------------------
SEED = 20240617
N_SCENARIOS = 2000      # 唯一情境数
SAMPLES_PER_SCENARIO = 3  # 每个情境重复采样的次数（模拟同一个情境下的多次真实选择）
# 候选情境池大小。原来取 N_SCENARIOS*5=10000，那时只有 8 个品类、
# 「选择明确」的情境占比高，够用。菜品库扩到 18 品类 / 317 道菜之后，
# 同品类内相似菜变多、并列变多，通过筛选的候选变少，池子必须放大才不会
# 出现「keep 不够 2000 -> 情境 id 回绕 -> 同一情境重复计入」。取 10 倍。
CANDIDATE_POOL = N_SCENARIOS * 10
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


# 每个时段的「品类级适配度」：正数=该时段常吃，负数=该时段几乎不会吃。
# 必须覆盖 dishes.csv 里所有 18 个品类 —— 漏掉的品类在本项恒得 0 分，
# 会在时段竞争里系统性吃亏（实测漏配时「糖水甜品」当最优只有 4 次）。
# 数值沿用原 8 类的标定，新增 10 类按同一量纲外推。
SLOT_CAT_BONUS = {
    "早餐": {
        "早餐粥点": 1.25, "面食粉面": 0.95, "轻食健康": 0.85,
        "早茶点心": 1.10, "广式粥粉面": 0.60, "面包甜点": 0.50,
        "火锅麻辣烫": -1.60, "烧烤小吃": -0.85, "家常炒菜": -0.55,
        "本帮江浙菜": -0.45, "烧腊卤味": -0.35, "老火汤羹": -0.35,
        "广府家常菜": -0.30, "煲仔饭碟头饭": -0.20, "长三角粉面": 0.50,
        "中式快餐盖饭": -0.35, "糖水甜品": 0.30, "西式简餐": -0.10,
    },
    "午餐": {
        "家常炒菜": 0.35, "火锅麻辣烫": 0.30, "西式简餐": 0.20,
        "广府家常菜": 0.40, "本帮江浙菜": 0.35, "烧腊卤味": 0.40,
        "煲仔饭碟头饭": 0.45, "广式粥粉面": 0.35, "长三角粉面": 0.40,
        "面食粉面": 0.30, "老火汤羹": 0.20, "早茶点心": 0.10,
        "面包甜点": 0.15, "轻食健康": 0.20, "烧烤小吃": 0.05,
        "早餐粥点": -0.10, "糖水甜品": -0.20, "中式快餐盖饭": 0.45,
    },
    "下午加餐": {
        "轻食健康": 1.15, "早餐粥点": 0.50, "糖水甜品": 1.00,
        "早茶点心": 0.50, "面包甜点": 0.55,
        "火锅麻辣烫": -1.00, "家常炒菜": -0.50, "烧烤小吃": -0.20,
        "本帮江浙菜": -0.45, "广府家常菜": -0.45, "烧腊卤味": -0.40,
        "煲仔饭碟头饭": -0.55, "西式简餐": -0.25, "老火汤羹": -0.10,
        "广式粥粉面": -0.05, "长三角粉面": -0.05, "面食粉面": -0.05,
        "中式快餐盖饭": -0.45,
    },
    "晚餐": {
        "家常炒菜": 0.35, "火锅麻辣烫": 0.30, "西式简餐": 0.20,
        "广府家常菜": 0.40, "本帮江浙菜": 0.35, "烧腊卤味": 0.40,
        "煲仔饭碟头饭": 0.40, "广式粥粉面": 0.30, "长三角粉面": 0.35,
        "面食粉面": 0.25, "老火汤羹": 0.25, "早茶点心": 0.05,
        "面包甜点": 0.10, "轻食健康": 0.15, "烧烤小吃": 0.10,
        "早餐粥点": -0.15, "糖水甜品": -0.10, "中式快餐盖饭": 0.40,
    },
    "夜宵": {
        "烧烤小吃": 1.15, "火锅麻辣烫": 0.75, "面食粉面": 0.55,
        "广式粥粉面": 0.60, "煲仔饭碟头饭": 0.35, "糖水甜品": 0.40,
        "长三角粉面": 0.35,
        "轻食健康": -1.25, "早餐粥点": -0.75, "本帮江浙菜": -0.40,
        "广府家常菜": -0.30, "早茶点心": -0.35, "老火汤羹": -0.30,
        "面包甜点": -0.30, "烧腊卤味": -0.20, "西式简餐": -0.15,
        "家常炒菜": -0.10, "中式快餐盖饭": 0.30,
    },
}


def _check_slot_coverage():
    """守住上面那条注释：任何品类在任何时段都不能漏配。

    漏配不会报错，只会让该品类在本项恒得 0 分、在时段竞争里系统性吃亏，
    属于「静默出错」。所以在导入时就断言住。
    """
    missing = {s: sorted(set(CATEGORIES) - set(SLOT_CAT_BONUS[s])) for s in SLOT_CAT_BONUS}
    missing = {s: v for s, v in missing.items() if v}
    if missing:
        raise SystemExit(f"SLOT_CAT_BONUS 漏配品类（会导致该品类在时段竞争里吃亏）：{missing}")


_check_slot_coverage()


def _trapezoid(x: float, lo: float, hi: float, soft: float) -> float:
    """x 落在 [lo,hi] 得 1 分，超出后线性衰减到 soft 处为 0。"""
    if x < lo:
        return max(0.0, 1.0 - (lo - x) / max(soft, 1e-6))
    if x > hi:
        return max(0.0, 1.0 - (x - hi) / max(soft, 1e-6))
    return 1.0


def _dish_arrays(df: pd.DataFrame) -> dict:
    """把菜品表拆成 numpy 数组，供向量化效用计算使用（只做一次）。"""
    return {
        "name": df.dish_name.to_numpy(),
        "category": df.category.to_numpy(),
        "time_cost": df.time_cost.to_numpy(dtype=float),
        "price": df.price.to_numpy(dtype=float),
        "spicy": df.spicy.to_numpy(dtype=float),
        "heavy": df.heavy.to_numpy(dtype=float),
        "healthy": df.healthy.to_numpy(dtype=float),
        "temperature": df.temperature.to_numpy(dtype=float),
        "carb": df.carb.to_numpy(dtype=float),
        "protein": df.protein.to_numpy(dtype=float),
        "taboo": df.taboo.fillna("").astype(str).to_numpy(),
        "is_veg": df.is_veg.fillna(0).astype(int).to_numpy(),
    }


def _trap(x, lo, hi, soft):
    """_trapezoid 的向量化版本：x 落在 [lo,hi] 得 1 分，超出后线性衰减到 soft 处为 0。
    与标量版逐一对照过，结果一致。"""
    x = np.asarray(x, dtype=float)
    soft = max(float(soft), 1e-6)
    out = np.ones_like(x)
    low = np.divide(lo - x, soft, out=np.zeros_like(x), where=(x < lo))
    out = np.where(x < lo, np.maximum(0.0, 1.0 - low), out)
    high = np.divide(x - hi, soft, out=np.zeros_like(x), where=(x > hi))
    out = np.where(x > hi, np.maximum(0.0, 1.0 - high), out)
    return out


def utility_all(da: dict, sc: dict) -> np.ndarray:
    """给定情境 sc，一次性算出**所有菜**的效用分。

    第 (0) 项时段适配度是品类级加成，其余是菜品级属性。
    返回原始分（未归一化），由 _final_scale 做量程变换。
    本函数是旧版逐行 utility() 的向量化等价实现：对菜品维度向量化，
    对情境仍是标量（情境数少、且每条分支都依赖情境取值）。
    """
    slot = sc["time_slot"]
    pref = SLOT_PREF[slot]
    mood, weather, comp = sc["mood"], sc["weather"], sc["companion"]
    health = sc["health_goal"]

    heavy = da["heavy"]
    carb = da["carb"]
    temp = da["temperature"]
    spicy = da["spicy"]
    healthy = da["healthy"]
    protein = da["protein"]
    time_cost = da["time_cost"]
    price = da["price"]

    score = np.zeros_like(time_cost)

    # (0) 时段适配度：现实里有些搭配几乎不会发生（早餐吃火锅/夜宵吃草）
    #     表定义在 SLOT_CAT_BONUS，按情境时段取一组 {品类: 加减分}。
    cat = da["category"]
    for _cname, _cval in SLOT_CAT_BONUS[slot].items():
        score += _cval * (cat == _cname)

    # (1) 时段契合：厚重感 + 温度
    lo, hi = pref["heavy"]
    score += 1.60 * _trap(heavy, lo, hi, 2.2)
    lo, hi = pref["temp"]
    score += 1.00 * _trap(temp, lo, hi, 1.5)

    # (2) 时间够不够
    t_need = time_cost * (1.15 if comp in ("朋友", "家人", "约会") else 1.0)
    score += 1.60 * _trap(sc["available_time"] - t_need, 0, 999, 30.0)

    # (3) 预算
    b = BUDGET_LEVEL[sc["budget"]]
    p_ideal = BUDGET_PRICE[b]
    lo, hi = (p_ideal * 0.5, p_ideal * 1.15) if b < 3 else (p_ideal * 0.6, 999)
    score += 1.20 * _trap(price, lo, hi, p_ideal * 1.3)

    # (4) 饥饿度 -> 分量
    score += 1.10 * _trap(heavy + 0.5 * carb, sc["hunger"] - 0.5, sc["hunger"] + 2.0, 3.0)
    if sc["hunger"] >= 4:
        score -= 0.50 * (heavy <= 2)

    # (5) 心情
    if mood in ("疲惫", "压力大"):
        score += 0.45 * _trap(heavy + carb, 6, 10, 5.0)
        score += 0.25 * carb / 5.0
    if mood == "低落":
        score += 0.45 * (spicy / 3.0)
        score += 0.25 * _trap(heavy, 3, 5, 2.0)
    if mood == "开心":
        score += 0.18 * (1.0 - np.abs(heavy - 3.5) / 3.5)
    if mood == "兴奋":
        score += 0.20 * (protein / 5.0)

    # (6) 天气
    if weather == "冷":
        score += 0.65 * _trap(temp, 2, 2, 1.2)
    elif weather == "热":
        score += 0.65 * _trap(temp, 0, 1, 1.2)
    elif weather == "雨":
        score += 0.42 * _trap(time_cost, 0, 22, 26.0)   # 懒得出门
        score += 0.22 * _trap(temp, 1, 2, 1.5)
    elif weather == "雪":
        # 雪天：既不想出门（要快），又想吃热乎的（这是测试集专属情境）
        score += 0.70 * _trap(time_cost, 0, 30, 34.0)
        score += 0.85 * _trap(temp, 2, 2, 1.8)

    # (7) 一起吃
    if comp in ("朋友", "家人", "约会"):
        score += 0.72 * _trap(heavy, 3, 5, 2.0)
        score += 0.55 * _trap(time_cost, 25, 999, 30.0)   # 愿意等 / 慢慢吃
        if comp == "约会":
            score += 0.32 * _trap(temp, 2, 2, 1.5)
        if comp == "朋友":
            score += 0.22 * (spicy / 3.0)
    elif comp == "同事":
        score += 0.40 * _trap(time_cost, 0, 20, 25.0)
        score += 0.18 * _trap(spicy, 0, 1, 2.0)           # 避免气味尴尬

    # (8) 健康诉求：先当「筛选条件」，再当「加分项」
    #     这是本效用函数里最关键的一条设计：如果只给健康菜加分，油腻面食
    #     能靠「温热、厚重、顶饱」等其它维度追平沙拉，结果就完全不符合现实。
    #     真实的减脂/控糖行为是先把一大类选项排除掉，再在剩下的里面挑。
    hw = {"不关心": 0.00, "随便": 0.00, "想健康": 0.55, "严格控制": 1.00}[health]
    excluded = np.zeros(len(score), dtype=bool)
    if health == "想健康":
        score -= 1.30 * np.where(healthy <= 2, (3 - healthy) / 2.0, 0.0)
        score -= 0.85 * (heavy >= 5)
    elif health == "严格控制":
        excluded |= (healthy <= 2)      # 直接排除：这一档根本不会被考虑
        score -= 1.30 * (heavy >= 4)
        score -= 0.35 * (heavy == 3)
    if hw > 0:
        score += 1.20 * hw * healthy / 5.0
        score -= 0.55 * hw * np.maximum(0.0, heavy - 3) / 2.0

    # (9) 辣度
    tol = SPICY_TOL_LEVEL[sc["spicy_tol"]]
    eff = np.minimum(spicy, tol)
    score -= 1.15 * np.maximum(0.0, spicy - tol)   # 超过承受能力：硬伤
    score += 0.32 * (eff / 3.0)

    # (10) 忌口：一票否决。
    #     情境的 taboo 是逗号分隔的多值（如 "海鲜,乳制品" / "素食"）。
    #     「素食」是饮食约束而非过敏原，它不写在 dish.taboo 里，
    #     而是由 dishes.csv 的 is_veg 列表达；过敏原则按标签求交集。
    sc_tab = [t for t in str(sc["taboo"]).split(",") if t]
    if sc_tab:
        allergens = [t for t in sc_tab if t != "素食"]
        if allergens:
            pat = "|".join(allergens)
            hit = np.array([bool(t) and bool(re.search(pat, t)) for t in da["taboo"]])
            excluded |= hit
        if "素食" in sc_tab:
            excluded |= (da["is_veg"] != 1)

    # (12) 夜宵惩罚健康菜（深夜吃草有点惨）
    if slot == "夜宵":
        score -= 0.35 * ((healthy >= 5) & (heavy <= 2))

    # (13) 下午加餐：不接受大餐
    if slot == "下午加餐":
        score -= 0.70 * (heavy >= 4)

    score = np.maximum(score, 0.0)
    return np.where(excluded, 0.0, score)


def _cat_bonus(cat_arr, table: dict) -> np.ndarray:
    """按品类给加成/惩罚。等价于旧版的 {...}.get(row.category, 0.0)。"""
    out = np.zeros(len(cat_arr), dtype=float)
    for name, val in table.items():
        out += np.where(cat_arr == name, val, 0.0)
    return out


def utility(row, sc: dict) -> float:
    """单道菜的效用（兼容旧调用签名；内部走向量化实现）。"""
    return float(utility_all(_dish_arrays(pd.DataFrame([row])), sc)[0])



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
    candidates = [sample_scenario(i) for i in range(CANDIDATE_POOL)]
    n_dish = len(dishes)
    da = _dish_arrays(dishes)          # 向量化效用计算用的数组（只构造一次）
    U = np.zeros((len(candidates), n_dish))
    for ci, sc in enumerate(candidates):
        U[ci] = utility_all(da, sc)
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

    # 口味均衡：品类 x 时段 组合最多 CAP 个情境，避免大品类一家独大。
    # 原值 160 是按「8 个品类、每类 6 道菜」标定的；菜品库扩到 18 个品类后
    # 该上限几乎不再生效（大品类靠菜多、时段宽，稳定霸榜），小品类会被挤到
    # 只剩个位数样本 —— 实测「早茶点心」只当上 2 次最优。
    # 这里按品类数等比调小，让每个「品类 x 时段」都有机会进数据集。
    CAP = 90
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
    if len(keep) < N_SCENARIOS:
        # 不够就会在下一行切片时静默留下重复情境（id 回绕），
        # 表现为「唯一情境数」虚高、同一条情境重复计入训练集。直接报错。
        raise SystemExit(
            f"通过筛选的情境只有 {len(keep)} 个，不足 N_SCENARIOS={N_SCENARIOS}。\n"
            f"请调大 CANDIDATE_POOL（当前 {CANDIDATE_POOL}）或调大 CAP（当前 {CAP}）。"
        )
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
        utils = _final_scale(utility_all(da, sc))
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

    # ---- 真实效用矩阵落盘（第二层排序目标的唯一正确来源）----
    # 每个情境下**所有菜**的真实效用。第 2 层要在「候选品类内」学排序，
    # 就必须知道品类内每道菜各自的效用。
    # 之前只存了 gold 一行，训练时只能拿「某一行观测到的效用」当整组的排序目标，
    # 等于给品类内每道菜发了同一个目标值 —— 排序目标因此退化（详见 train.group_targets）。
    util_matrix = np.stack([
        _final_scale(utility_all(da, sc)) for sc in scenarios
    ]).astype(np.float32)
    np.save(OUT / "scenario_utility.npy", util_matrix)
    # 行序与 scenarios.csv 严格一致，另存一份 id 便于交叉校验
    np.save(OUT / "scenario_ids.npy", np.array([sc["scenario_id"] for sc in scenarios],
                                               dtype=np.int64))

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
