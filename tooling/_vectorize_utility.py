# -*- coding: utf-8 -*-
"""把 build_dataset.utility 的逐行实现替换为向量化实现。

旧实现每次调用接收一个 pd.Series，对单道菜算分。317 道菜 x 10000 情境
= 317 万次调用，实测约 95us/次 -> 全量 10 分钟。
向量化后对「所有菜一次算完」，语义与旧实现严格一致。
"""
import io

SRC = r"C:\Users\123\Desktop\桌面应用\文档\meal-decision\build_dataset.py"

NEW = '''def _dish_arrays(df: pd.DataFrame) -> dict:
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
    cat = da["category"]
    if slot == "早餐":
        score += _cat_bonus(cat, {"早餐粥点": 1.25, "面食粉面": 0.95, "轻食健康": 0.85})
        score -= _cat_bonus(cat, {"火锅麻辣烫": 1.6, "烧烤小吃": 0.85, "家常炒菜": 0.55})
    elif slot == "夜宵":
        score += _cat_bonus(cat, {"烧烤小吃": 1.15, "火锅麻辣烫": 0.75, "面食粉面": 0.55})
        score -= _cat_bonus(cat, {"轻食健康": 1.25, "早餐粥点": 0.75})
    elif slot == "下午加餐":
        score += _cat_bonus(cat, {"轻食健康": 1.15, "早餐粥点": 0.5})
    else:  # 午/晚餐
        score += _cat_bonus(cat, {"家常炒菜": 0.35, "火锅麻辣烫": 0.30, "西式简餐": 0.20})

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


'''

p = SRC
lines = io.open(p, encoding="utf-8").read().split("\n")
start, end = 143, 278          # 0-based: 旧 utility() 从第 144 行到第 279 行
assert lines[start].startswith("def utility(row: pd.Series"), lines[start][:60]
assert lines[end + 1].startswith("SCORE_RANGE ="), lines[end + 1][:60]

lines[start:end + 1] = NEW.split("\n")
src = "\n".join(lines)

# 需要 re 与 re.search
if "\nimport re\n" not in src:
    src = src.replace("import json\nimport math", "import json\nimport math\nimport re", 1)

io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("utility() 已向量化为 utility_all() + utility() 兼容包装")
