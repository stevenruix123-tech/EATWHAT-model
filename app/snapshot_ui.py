# -*- coding: utf-8 -*-
"""
snapshot_ui.py —— 把真实界面渲染成一张静态快照（用于验收视觉，不改动服务）

做法：
  1. 从运行中的服务取回真实 HTML（界面本身）
  2. 用与前端完全相同的 draw() 逻辑，把 compute 的推荐结果预先渲染进 HTML
  3. 输出 artifacts/ui_snapshot.html，再用 html_shot 截图为 ui_shot.png

这样看到的就是真实界面在「已出结果」状态下的样子。
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import server as S
from features import ART

OUT_HTML = ART / "ui_snapshot.html"


def fetch_page(url: str = "http://127.0.0.1:8848/") -> str:
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.read().decode("utf-8")


def fetch_meta(url: str = "http://127.0.0.1:8848/api/meta") -> dict:
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def render_block(d: dict) -> str:
    """与前端 draw() 等价的静态 HTML（视觉一致即可）。"""
    cu = " &nbsp;&nbsp; ".join(
        f'<span class="bar" style="width:{max(6, round(c["prob"] * 130))}px"></span>'
        f'<b>{c["name"]}</b> {c["prob"] * 100:.0f}%' for c in d["cuisine_top"])
    t = d["items"][0]
    alts = "".join(
        f'<div class="alt"><div><b>第{x["rank"]}名 · {x["dish_name"]}</b>'
        f'<span class="tag"> · {x["category"]} · {x["price"]}元 · {x["reason"]}</span></div></div>'
        for x in d["items"][1:])
    rows = "".join(
        f'<tr><td>{s["dish_name"]}</td><td>{s["score"]:.3f}</td></tr>'
        for s in d["all_scores"])
    truth = ""
    if d.get("truth"):
        tr = d["truth"]
        truth = (f'<div class="truth">对照：规则下的最优是「{tr["best_dish"]}」'
                 f'(效用 {tr["best_possible"]})，本次首推效用 {tr["model_top1_utility"]}。'
                 f'效用是 0~1 的连续值。</div>')
    return (f'<div class="cuisine">第一层判断： {cu}</div>'
            f'<div class="pick"><div class="name">{t["dish_name"]}</div>'
            f'<div class="meta">{t["category"]} · {t["price"]} 元 · {t["time_cost"]} 分钟</div>'
            f'<div class="why">为什么：{t["reason"]}</div></div>{alts}'
            f'<details open><summary>看模型给每道菜的原始打分（不美化）</summary>'
            f'<div class="fine">分数是模型学出来的效用预测值，<b>只能在同一品类内横向比较</b>，'
            f'跨品类没有可比性（这是两层结构的特点）。同一品类内分数越接近，'
            f'说明这几道菜本来就差不多，随便挑一个都行。</div>'
            f'<table class="raw"><tr><th>菜</th><th>分数</th></tr>{rows}</table></details>'
            f'{truth}')


def main():
    S.load_model()
    d = S.predict({"time_slot": "夜宵", "hunger": 5, "available_time": 25,
                   "budget": "正常", "mood": "疲惫", "weather": "冷",
                   "companion": "一个人", "health_goal": "不关心",
                   "spicy_tol": "中辣", "taboo": "", "topk": 3})
    html = fetch_page()

    # 快照是存成 file:// 的静态文件，页面里的 fetch('/api/meta') 在 file:// 下
    # 必然失败 —— 而预设按钮、左侧表单、顶部统计全是在 boot() 里从 META 渲染的，
    # fetch 一失败 boot() 就中断，快照会只剩一个空的「你的情况」卡片。
    # 做法：把 /api/meta 的真实内容内联进去，并让 boot() 直接用内联数据，
    # 不去发请求。这样静态快照也能渲染出完整界面。
    meta = fetch_meta()
    meta_js = json.dumps(meta, ensure_ascii=False).replace("</", "<\\/")
    banner = f"<script>const __SNAP_META__ = {meta_js};</script>\n"
    html = html.replace("<script>", banner + "<script>", 1)
    html = html.replace("META = await (await fetch('/api/meta')).json();",
                        "META = __SNAP_META__;")

    # 预设选中态与滑块位置：直接改成「深夜加班」的取值，让截图和结果一致
    html = html.replace("const v2 = el('span'); const v1 = el('span');", "")
    html = html.replace("ST = {...META.presets['工作日午餐']};",
                        "ST = {...META.presets['深夜加班']};")
    # 预填结果面板
    html = html.replace('<div id="out"><div class="empty">左边点几下，然后按「给我推荐」</div></div>',
                        f'<div id="out">{render_block(d)}</div>')
    # 阻止脚本在截图时再次请求（保留表单渲染）
    html = html.replace("document.getElementById('go').onclick = run;",
                        "document.getElementById('go').onclick = ()=>{};")
    OUT_HTML.write_text(html, encoding="utf-8")
    print("written ->", OUT_HTML)
    print("首推:", d["items"][0]["dish_name"], "| 前三:",
          [x["dish_name"] for x in d["items"]])


if __name__ == "__main__":
    main()
