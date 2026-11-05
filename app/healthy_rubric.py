# -*- coding: utf-8 -*-
"""
healthy_rubric.py —— `healthy` 标注口径修正（交接文档 §7.2）

【问题】
`healthy` 是人工按「做法清淡」估出来的：凡是清煮/白煮/炖，一律给 4。
可锅里煮的是五花肉还是鸡胸，标注时没区分。于是出现两条连锁错误：

  1. 汆白肉（白煮**五花肉**）heavy=3 却 healthy=4，松仁小肚（加工肉制品）
     heavy=3 healthy=3，都躲过了「严格控制」的 healthy<=2 硬筛选；
  2. 进了候选池后，`1.20 * healthy/5` 给它们 +0.96 的健康加分，
     而 heavy=3 只扣 0.35 —— 净得 +0.61，直接排第一。
     实测「减脂期午餐」推荐出 汆白肉 / 松仁小肚 / 炒蚕蛹，全是东北菜。

【口径】
`healthy` 描述的是「对减脂/控糖人群的友好度」，主因是**食材的脂肪含量与
加工度**，其次才是烹法。所以按下面三组规则给 healthy 压上限：

  A 组（cap 2）高脂部位 / 加工肉 / 油炸 / 糖油 —— 这类菜无论怎么煮都不该算健康
  B 组（cap 3）内脏、红烧/干煸/回锅/油焖/熏腊 这类「油或糖参与较重」的烹法
  C 组        与 heavy 的单调一致性（heavy=4 不可能 healthy=5）

规则只**压上限**、不抬高，且宁可漏判也不误判：命中率优先于覆盖率。
含关键字但不是那回事的菜单独列进 EXEMPT（肠粉是米制品、炸酱面不是炸的）。

【为什么改标签而不是改效用函数】
`healthy` 是模型输入特征（features.DISH_CAT_COLS），改标签 = 改特征，
模型能学到；而菜名不是特征，拿菜名做规则模型学不到，只会加剧过拟合。
效用函数那边只改**属性可表达**的部分（见 build_dataset 的「严格控制」分支）。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# ---- A 组：命中即 healthy 上限 2 -------------------------------------------
# 每项 (正则, 说明)。宁可少列，也不要放进会误伤的词。
RULES_A = [
    (r"五花|白肉", "五花肉/白肉，纯脂肪部位"),
    (r"肥肠|大肠|小肚|皮冻|皮肚", "肠/肚类高脂部位或加工品"),
    (r"扣肉|梅菜扣肉", "扣肉，五花肉+糖油"),
    (r"培根|腊肠|腊肉|香肠|红肠|血肠|午餐肉|咸肉|熏肉", "腌制/熏制加工肉"),
    (r"猪蹄|蹄髈|肘子|炸猪排|炸鸡|炸鱼|薯条|薯片|油条|天妇罗|春卷", "油炸或高脂皮"),
    (r"炸(?!酱)", "油炸（炸酱面除外，见 EXEMPT）"),
    (r"奶油|黄油|芝士|奶酪|冰淇淋|慕斯|泡芙|蛋糕|拔丝|蜜汁|糖醋", "糖油/乳脂类"),
]

# ---- B 组：命中即 healthy 上限 3 -------------------------------------------
RULES_B = [
    (r"肝|腰花|腰子|肺|猪肚|牛肚|肚丝|鸡胗|鸭胗|肫|血旺|毛血", "内脏类"),
    (r"腩", "牛腩等肥瘦相间的部位"),
    (r"红烧|干煸|回锅|油焖|爆炒|水煮肉|毛血旺", "重油/重糖烹法"),
    # 火腿通常是配料/汤底，单列 B 组而不是 A（压到 2 会让「火腿冬瓜汤」这类
    # 清汤菜从 4 直落 2，属于误伤）
    (r"火腿|熏", "火腿等腌制配料 / 熏制工艺"),
]

# ---- 例外：含上述关键字但并非高脂 -----------------------------------------
EXEMPT_GLOBAL = [
    r"炸酱面",      # 「炸酱」是炒酱，不是油炸
    r"肠粉",        # 米浆蒸制，不是肠
    r"猪肠粉",      # 同上
    r"粉肠",        # 广式粥品里的粉肠（米粉/小肠），按 3 处理已偏保守
    r"希腊",        # 「希腊酸奶」的「腊」不是熏腊
]

# ---- C 组：与 heavy 的单调一致性 -------------------------------------------
# heavy 是「厚重感」，和 healthy 描述同一件事的两端，不应该出现
# 「非常厚重 + 非常健康」。只压到明显矛盾的区间，不动 3 分档。
HEAVY_CAP = {1: 5, 2: 5, 3: 5, 4: 3, 5: 2}


def _hits(name: str, rules) -> list[str]:
    return [why for pat, why in rules if re.search(pat, name)]


def correct(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (修正后的 df 副本, 改动明细)。幂等：重复调用不会继续压低。"""
    out = df.copy()
    rows = []
    exempt_pats = [re.compile(p) for p in EXEMPT_GLOBAL]
    for i, name in enumerate(out.dish_name.astype(str)):
        old = int(out.at[i, "healthy"])
        heavy = int(out.at[i, "heavy"])
        if any(p.search(name) for p in exempt_pats):
            continue
        caps = []
        why_a = _hits(name, RULES_A)
        if why_a:
            caps.append((2, "A:" + why_a[0]))
        why_b = _hits(name, RULES_B)
        if why_b:
            caps.append((3, "B:" + why_b[0]))
        caps.append((HEAVY_CAP.get(heavy, 5), f"C:heavy={heavy}"))
        new = min([old] + [c for c, _ in caps])
        if new != old:
            reason = min(caps, key=lambda t: t[0])[1]
            rows.append(dict(dish_name=name, category=out.at[i, "category"],
                             heavy=heavy, healthy_old=old, healthy_new=new,
                             reason=reason))
            out.at[i, "healthy"] = new
    return out, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="修正 dishes.csv 的 healthy 标注口径")
    ap.add_argument("--dishes", default=str(Path(__file__).resolve().parent /
                                            "artifacts" / "dishes.csv"))
    ap.add_argument("--apply", action="store_true", help="真正写回（默认只预览）")
    args = ap.parse_args()

    src = Path(args.dishes)
    df = pd.read_csv(src)
    fixed, diff = correct(df)

    print(f"菜品库: {src}  ({len(df)} 道菜)")
    if diff.empty:
        print("没有需要修正的条目。")
        return
    print(f"需要修正: {len(diff)} 道\n")
    print(diff.sort_values(["healthy_new", "healthy_old"], ascending=[True, False])
          .to_string(index=False))
    print()
    print("修正后 healthy 分布：")
    print(fixed.healthy.value_counts().sort_index().to_string())
    print()
    print("各品类 healthy>=3 的道数（「严格控制」可用池）：")
    before = df[df.healthy >= 3].category.value_counts()
    after = fixed[fixed.healthy >= 3].category.value_counts()
    cmp = pd.DataFrame({"修正前": before, "修正后": after}).fillna(0).astype(int)
    cmp["变化"] = cmp["修正后"] - cmp["修正前"]
    print(cmp.sort_values("变化").to_string())
    print(f"\n严格控制可用池: {int((df.healthy >= 3).sum())} -> "
          f"{int((fixed.healthy >= 3).sum())} 道")

    if args.apply:
        fixed.to_csv(src, index=False, encoding="utf-8-sig")
        print(f"\n已写回 -> {src}")
    else:
        print("\n（预览模式，未写盘。加 --apply 落盘。）")


if __name__ == "__main__":
    main()
