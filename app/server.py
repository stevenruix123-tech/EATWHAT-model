# -*- coding: utf-8 -*-
"""
server.py —— 「今天该吃什么」本地 Web 界面

零额外依赖：HTTP 服务用 Python 标准库，模型常驻内存，单次推荐毫秒级。

启动：
    python server.py                 # 默认 http://127.0.0.1:8848
    python server.py --port 9000     # 换端口
    python server.py --no-browser    # 不自动打开浏览器

接口：
    GET  /            -> 界面
    GET  /api/meta    -> 可选值、菜品库、模型信息
    POST /api/predict -> 传入情境，返回推荐（JSON）
"""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd

from features import (ART, CATEGORIES, CTX_VOCAB, DISH_CAT_IDX, DISH_NAMES, N_DISH)
import train as T
import recommend as R

HOST_DEFAULT = "127.0.0.1"
PORT_DEFAULT = 8848

FIELDS = [
    ("time_slot", "时段", CTX_VOCAB["time_slot"]),
    ("hunger", "饥饿程度", ["1", "2", "3", "4", "5"]),
    ("available_time", "可用时间(分钟)", []),
    ("budget", "预算", CTX_VOCAB["budget"]),
    ("mood", "心情", CTX_VOCAB["mood"]),
    ("weather", "天气", CTX_VOCAB["weather"]),
    ("companion", "和谁吃", CTX_VOCAB["companion"]),
    ("health_goal", "健康诉求", CTX_VOCAB["health_goal"]),
    ("spicy_tol", "吃辣能力", CTX_VOCAB["spicy_tol"]),
    ("taboo", "忌口（可留空）", CTX_VOCAB["taboo"]),
]

_MODEL: dict = {}


def load_model():
    cm, dm = R.load_models()
    _MODEL["cm"], _MODEL["dm"] = cm, dm
    samples = pd.read_csv(ART / "samples.csv")
    _MODEL["stats"] = {
        "n_dishes": N_DISH,
        "n_categories": len(CATEGORIES),
        "n_scenarios": int(samples.scenario_id.nunique()),
        "n_rows": int(len(samples)),
    }


def predict(payload: dict) -> dict:
    """核心推理：返回结构化的推荐结果（供前端渲染）。"""
    cm, dm = _MODEL["cm"], _MODEL["dm"]

    sc = {}
    for field, _, _ in FIELDS:
        v = payload.get(field)
        if field == "hunger":
            sc[field] = int(v if v is not None else 3)
        elif field == "available_time":
            sc[field] = int(v if v is not None else 45)
        elif field == "taboo":
            sc[field] = "" if v is None else str(v).strip()
        else:
            v = "" if v is None else str(v)
            if v not in CTX_VOCAB[field]:
                v = CTX_VOCAB[field][0]
            sc[field] = v

    exclude = {str(x) for x in payload.get("exclude", []) if str(x)}
    topk = max(1, min(int(payload.get("topk", 3)), 10))

    row = pd.DataFrame([sc])
    c, n = T.encode_context(row)

    # ---- 结构分支 ----
    # 单阶段模型（cm is None）：菜品打分器用「整目录 824 路 softmax」训练，
    #   分数跨品类可比，直接在全部菜里排序（交接文档 §7.4）。
    #   此时没有第一层过滤，品类分布只作展示，由菜品打分的品类聚合得到。
    # 两层模型：先选品类，再在候选品类内排序。
    single = cm is None
    if single:
        cand = np.arange(N_DISH)
        CUISINE_T = 2.0
        ids_all, dc_all, dn_all = T.tile_dishes(cand[None, :], 1)
        pre = dm.forward(c, n, ids_all, dc_all, dn_all)[0].astype(float)
        top30 = np.argsort(-pre, kind="stable")[:30]
        agg = np.zeros(len(CATEGORIES))
        for i in top30:
            agg[DISH_CAT_IDX[int(i), 0]] += np.exp(pre[i] - pre[top30[0]])
        cat_p = agg / max(agg.sum(), 1e-12)
        cat_order = np.argsort(-cat_p)
        win = int(cat_order[0])
    else:
        # 用温度缩放软化：模型原始 logits 偏尖锐（会给出 99%、100% 这种过度自信的
        # 数字），而它的实际品类命中率约 70%（见 artifacts/report.md 的分指标拆解）。
        # 展示值应当反映真实把握，所以把温度调大压平分布。
        CUISINE_T = 2.0
        logits = cm.forward(c, n)[0]
        z = (logits - logits.max()) / CUISINE_T
        cat_p = np.exp(z)
        cat_p = cat_p / cat_p.sum()
        cat_order = np.argsort(-cat_p)
        win = int(cat_order[0])
        cand = np.where(DISH_CAT_IDX[:, 0] == win)[0]

    # ---- 逐道打分（单阶段=全库，两层=候选品类内）----
    ids, dc, dn = T.tile_dishes(cand[None, :], 1)
    scores = dm.forward(c, n, ids, dc, dn)[0].astype(float)
    ex_mask = np.array([DISH_NAMES[int(i)] in exclude for i in cand])
    if ex_mask.all():                      # 排除过头了，退回全库
        ex_mask = np.zeros_like(ex_mask, dtype=bool)
    scores = np.where(ex_mask, -1e9, scores)

    # 相对概率（展示用）。
    # 模型的原始分数没有概率校准，不能直接全库 softmax：
    #   - 两层模型：候选是「一个品类」，约 43 道，且分数挤在很窄的区间；
    #   - 单阶段模型：候选是全部 824 道，很多菜同样合适，全库 softmax 会让
    #     每个概率都趋近 0（实测前 3 名只有 1%、1%、0%，展示上毫无意义）。
    # 做法：只在「头部候选」内部算相对匹配度 —— 头部 10 道的动态范围当温度，
    # 这样显示的是「前几名之间差多少」，而不是「占全库的多少」。
    head = np.argsort(-scores)[:max(10, topk)]
    hs = scores[head]
    span = float(hs[0] - hs[-1]) if len(hs) > 1 else 0.0
    T_SCALE = max(span / 3.0, 1e-6)
    ph = np.exp(np.clip((hs - hs[0]) / T_SCALE, -30, 0))
    ph = ph / ph.sum()
    p = np.zeros(len(scores))
    p[head] = ph

    order = np.argsort(-scores)[:topk]
    items = []
    for rank, j in enumerate(order, 1):
        did = int(cand[j])
        d = R.DISH_TABLE.iloc[did]
        items.append({
            "rank": rank,
            "dish_name": DISH_NAMES[did],
            "category": str(d.category),
            "prob": round(float(p[j]), 4),
            # 注意：这里以前还有个 "match"（和 prob 完全相同的历史遗留字段），2026-11-09 删了。
            "score": round(float(scores[j]) if scores[j] > -1e8 else 0.0, 4),
            "reason": R.why(d, sc),
            "price": int(d.price),
            "time_cost": int(d.time_cost),
            "spicy": int(d.spicy),
            "healthy": int(d.healthy),
            "heavy": int(d.heavy),
            "temperature": int(d.temperature),
        })

    # 供展示：实时算这道菜「在规则下的真实效用」（模型训练时看不到的上帝视角）
    truth = None
    try:
        U = T.utility_vector(row.iloc[0])
        top_id = DISH_NAMES.index(items[0]["dish_name"])
        truth = {
            "model_top1_utility": round(float(U[top_id]), 4),
            "best_possible": round(float(U.max()), 4),
            "best_dish": DISH_NAMES[int(U.argmax())],
        }
    except Exception:
        truth = None

    return {
        "scenario": sc,
        # 单阶段（整目录 softmax）时品类分布只是展示，不参与筛选 —— 前端据此换措辞
        "stage": "single" if single else "two_layer",
        "cuisine_top": [
            {"name": CATEGORIES[int(i)], "prob": round(float(cat_p[i]), 4)}
            for i in cat_order[:3]
        ],
        "candidates": int(len(cand)),
        "excluded": int(ex_mask.sum()),
        "items": items,
        # 头部候选的原始打分（不做任何美化，供用户自己判断差距）。
        # **只发前 20 条**：单阶段下候选有 824 道，全发会让响应 96% 是这张表、
        # 页面还要多画 824 行。总数看 `candidates`（就是参与打分的菜数）。
        "all_scores": sorted(
            [{"dish_name": DISH_NAMES[int(cand[k])], "score": round(float(scores[k]), 3)}
             for k in range(len(cand))],
            key=lambda x: -x["score"])[:20],
        "truth": truth,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "MealDecision/1.0"

    def log_message(self, fmt, *args):        # 静音默认访问日志，控制台只留启动信息
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/meta":
            self._json({
                "fields": [{"key": k, "label": lbl, "options": opts}
                           for k, lbl, opts in FIELDS],
                "dishes": [
                    {"id": i, "name": DISH_NAMES[i],
                     "category": str(R.DISH_TABLE.iloc[i].category)}
                    for i in range(N_DISH)
                ],
                "presets": R.PRESETS,
                "stats": _MODEL["stats"],
                # 模型结构：有 cuisine.* 才是两层。前端靠它在首屏就显示正确的措辞，
                # 不能等第一次出结果（draw）才纠正 —— 否则首屏会谎报结构。
                "stage": "single" if _MODEL.get("cm") is None else "two_layer",
            })
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/api/predict":
            self._json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            self._json(predict(payload))
        except Exception as e:                       # 出错也要返回可读信息
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>今天该吃什么</title>
<style>
  :root{
    --bg:#f7f7f5; --card:#fff; --ink:#1c1c1e; --muted:#6b7280; --line:#e5e7eb;
    --brand:#e8590c; --brand-soft:#fff4ec; --bar:#f59e0b;
  }
  @media (prefers-color-scheme: dark){
    :root{ --bg:#141416; --card:#1e1e21; --ink:#f2f2f4; --muted:#9ca3af;
           --line:#2e2e33; --brand:#ff7a2f; --brand-soft:#2a1c12; }
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
  header{padding:22px 20px 8px;max-width:1080px;margin:0 auto}
  h1{margin:0;font-size:24px;letter-spacing:.5px}
  .sub{color:var(--muted);font-size:13px;margin-top:6px}
  .wrap{max-width:1080px;margin:0 auto;padding:10px 20px 60px;
    display:grid;grid-template-columns:1.15fr 1fr;gap:22px;align-items:start}
  @media (max-width:880px){ .wrap{grid-template-columns:1fr} }
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;
    padding:16px 18px 20px}
  .card h2{margin:0 0 10px;font-size:15px;font-weight:600;color:var(--muted);
    letter-spacing:.4px}
  .field{margin-bottom:14px}
  .field > label{display:block;font-size:13px;color:var(--muted);margin-bottom:5px}
  /* 宽屏下表单两列：主按钮「给我推荐」不用滚才能看到（1280×900 下 #go 顶部 1014px -> ~720px）。
     窄屏（手机 / 分屏）仍是一列。宽字段加 .wide 跨两列。 */
  @media (min-width:720px){
    #form{display:grid;grid-template-columns:1fr 1fr;column-gap:16px;align-items:start}
    #form .field{margin-bottom:7px}
    #form .field > label{margin-bottom:4px}
    #form .field.wide{grid-column:1 / -1}
    /* 半宽列里把胶囊收小一档，5~6 个选项才不至于只剩一个落单到第二行 */
    #form .seg{gap:4px}
    #form .seg button{padding:4px 8px;font-size:12px}
  }
  .seg{display:flex;flex-wrap:wrap;gap:5px}
  .seg button{border:1px solid var(--line);background:transparent;color:var(--ink);
    padding:6px 11px;border-radius:999px;font-size:13px;cursor:pointer;transition:.12s}
  .seg button:hover{border-color:var(--brand)}
  .seg button.on{background:var(--brand);border-color:var(--brand);color:#fff}
  input[type=range]{width:100%;accent-color:var(--brand)}
  .rowv{display:flex;justify-content:space-between;font-size:13px;color:var(--muted)}
  .big{font-size:17px;color:var(--ink);font-weight:600}
  .go{width:100%;margin-top:8px;padding:12px;border:0;border-radius:10px;
    background:var(--brand);color:#fff;font-size:15px;font-weight:600;cursor:pointer}
  .go:hover{filter:brightness(1.06)}
  .presets{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:12px}
  .presets button{border:1px dashed var(--line);background:transparent;color:var(--muted);
    padding:4px 9px;border-radius:999px;font-size:12px;cursor:pointer}
  .presets button:hover{color:var(--brand);border-color:var(--brand)}
  .cuisine{font-size:13px;color:var(--muted);margin-bottom:12px}
  .bar{display:inline-block;height:7px;border-radius:4px;background:var(--bar);
    vertical-align:middle;margin:0 7px 2px 0}
  .pick{border:1px solid var(--brand);background:var(--brand-soft);border-radius:12px;
    padding:14px 16px;margin-bottom:12px}
  .pick .name{font-size:21px;font-weight:700;letter-spacing:.3px}
  .pick .meta{font-size:12.5px;color:var(--muted);margin-top:3px}
  .pick .why{font-size:13px;margin-top:8px;color:var(--ink);opacity:.85}
  .pick .nope{margin-top:10px;border:1px solid var(--brand);background:transparent;
    color:var(--brand);padding:5px 11px;border-radius:8px;font-size:12.5px;cursor:pointer}
  .pick .nope:hover{background:var(--brand);color:#fff}
  .loading{color:var(--muted);font-size:14px;padding:30px 4px;text-align:center}
  .error{color:#b42318;font-size:13.5px;line-height:1.75;padding:16px 4px}
  .error code{background:var(--bg);padding:1px 5px;border-radius:4px;font-size:12.5px}
  :focus-visible{outline:2px solid var(--brand);outline-offset:2px}
  .alt{display:flex;justify-content:space-between;gap:10px;padding:9px 2px;
    border-bottom:1px dashed var(--line);font-size:14px}
  .alt:last-child{border-bottom:0}
  .alt .tag{color:var(--muted);font-size:12.5px}
  table.raw{width:100%;border-collapse:collapse;margin-top:8px;font-size:12.5px;
    font-variant-numeric:tabular-nums}
  table.raw th{text-align:left;color:var(--muted);font-weight:500;padding:3px 4px;
    border-bottom:1px solid var(--line)}
  table.raw td{padding:3px 4px;border-bottom:1px dashed var(--line)}
  table.raw td.cat{color:var(--muted);white-space:nowrap}
  table.raw td:last-child,table.raw th:last-child{text-align:right;color:var(--muted)}
  .pct{color:var(--muted);font-variant-numeric:tabular-nums;font-size:13px}
  .truth{margin-top:14px;padding-top:12px;border-top:1px solid var(--line);
    font-size:12.5px;color:var(--muted)}
  .empty{color:var(--muted);font-size:14px;padding:26px 4px;text-align:center}
  .spin{display:inline-block;width:13px;height:13px;border:2px solid rgba(255,255,255,.4);
    border-top-color:#fff;border-radius:50%;animation:s .7s linear infinite;
    vertical-align:-2px;margin-right:7px}
  @keyframes s{to{transform:rotate(360deg)}}
  .fine{font-size:12px;color:var(--muted);margin-top:10px}
  details{margin-top:12px}
  summary{cursor:pointer;font-size:13px;color:var(--muted)}
  .picker{margin-top:10px}
  .picker input[type=search]{width:100%;padding:8px 10px;border:1px solid var(--line);
    border-radius:9px;background:transparent;color:var(--ink);font-size:13px;font-family:inherit}
  .picker input[type=search]:focus{outline:none;border-color:var(--brand)}
  .hint{font-size:12px;color:var(--muted);padding:6px 2px}
  .chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
  .chips:empty{display:none}
  .chips button{border:1px solid var(--line);background:transparent;color:var(--muted);
    padding:4px 9px;border-radius:8px;font-size:12px;cursor:pointer}
  .chips button.rm{background:var(--brand);color:#fff;border-color:var(--brand)}
</style>
</head>
<body>
<header>
  <h1>今天该吃什么</h1>
  <div class="sub">决策模型 · <span id="stage-label">两层结构：先判断品类，再在品类内排序</span> ·
    <span id="stats"></span></div>
</header>

<div class="wrap">
  <section class="card">
    <h2>你的情况</h2>
    <div class="presets" id="presets"></div>
    <div id="form"></div>
    <button class="go" id="go">给我推荐</button>
    <details>
      <summary>不想吃的东西（搜菜名，点一下排除）</summary>
      <div class="picker">
        <input type="search" id="q" aria-label="按菜名搜索要排除的菜"
               placeholder="输入菜名筛选，如 牛油 / 面 / 汤" autocomplete="off">
        <div class="chips" id="sel"></div>
        <div class="chips" id="chips"></div>
      </div>
    </details>
    <div class="fine">模型只看这些情境特征和菜品属性，没有联网、没有调用大模型。</div>
  </section>

  <section class="card">
    <h2>推荐结果</h2>
    <div id="out" aria-live="polite"><div class="empty">左边点几下，然后按「给我推荐」</div></div>
  </section>
</div>

<script>
let META=null, ST={}, EX=new Set();
// 忌口面板的 DOM 与候选表：boot() 里初始化；draw() 里的「不想吃这道」也要复用
let Q=null, SUG=null, SEL=null, NAMES=[];
const pct = p => Math.round(p*100) + '%';

function el(t,c,h){const e=document.createElement(t); if(c)e.className=c;
  if(h!==undefined)e.innerHTML=h; return e;}

// 分段按钮的选中态：视觉 class 和 aria-pressed 必须一起改（否则读屏读不出选了什么）
function markOn(seg, btn){
  [...seg.children].forEach(x=>{
    x.classList.remove('on'); x.setAttribute('aria-pressed','false');
  });
  btn.classList.add('on'); btn.setAttribute('aria-pressed','true');
}

// 分段按钮组：role=group + aria-label，读屏才知道这一组在问什么
function segBox(label){
  const seg = el('div','seg');
  seg.setAttribute('role','group');
  seg.setAttribute('aria-label', label);
  return seg;
}

// 把 EX 渲染成可点掉的标签（点一下恢复）
function renderSel(){
  if(!SEL) return;
  SEL.innerHTML = '';
  [...EX].forEach(n=>{
    const b = el('button','rm', n + ' ×');
    b.title = '点一下恢复';
    b.onclick = ()=>{ EX.delete(n); renderSel(); renderSug(); };
    SEL.appendChild(b);
  });
}

// 输入即筛：最多 8 条，已排除的不再出现
function renderSug(){
  if(!SUG) return;
  SUG.innerHTML = '';
  const kw = Q.value.trim();
  if(!kw){ SUG.appendChild(el('div','hint','输入菜名筛选，最多显示 8 条')); return; }
  const hit = NAMES.filter(n=>n.includes(kw) && !EX.has(n)).slice(0,8);
  if(!hit.length){ SUG.appendChild(el('div','hint','没有匹配的菜，换个词试试')); return; }
  hit.forEach(n=>{
    const b = el('button',null,n);
    b.onclick = ()=>{ EX.add(n); renderSel(); renderSug(); };
    SUG.appendChild(b);
  });
}

async function boot(){
  META = await (await fetch('/api/meta')).json();
  ST = {...META.presets['工作日午餐']};
  // 首屏就把结构写对：单阶段 / 两层（否则要等第一次出结果才纠正，首屏会谎报）
  const sl0 = document.getElementById('stage-label');
  if (sl0) sl0.textContent = META.stage === 'single'
    ? '单阶段：整目录 softmax，直接在全部菜里排序'
    : '两层结构：先判断品类，再在品类内排序';
  const s = META.stats;
  document.getElementById('stats').textContent =
    `${s.n_dishes} 道菜 / ${s.n_categories} 个品类 · 训练自 ${s.n_scenarios} 个情境、${s.n_rows} 条记录`;

  // 预设
  const pv = document.getElementById('presets');
  Object.keys(META.presets).forEach(name=>{
    const b = el('button',null,name);
    b.onclick = ()=>{ ST = {...META.presets[name]}; render(); run(); };
    pv.appendChild(b);
  });

  // 表单
  const form = document.getElementById('form');
  META.fields.forEach(f=>{
    const box = el('div','field');
    box.dataset.key = f.key;          // render() 按 key 找字段，不再靠下标对齐
    // 选项多的字段横跨两列，免得半宽列里最后一个胶囊孤零零掉到第二行
    if(f.key==='time_slot' || f.key==='taboo') box.classList.add('wide');
    box.appendChild(el('label',null,f.label));
    if(f.key==='available_time'){
      const r = el('input'); r.type='range'; r.min=5; r.max=150; r.step=5;
      r.value = ST[f.key] ?? 45;
      const rv = el('div','rowv');
      const v1 = el('span'); const v2 = el('span','big', r.value+' 分钟');
      rv.appendChild(v1); rv.appendChild(v2);
      r.oninput = ()=>{ ST[f.key]=+r.value; v2.textContent = r.value+' 分钟'; };
      box.appendChild(r); box.appendChild(rv);
    } else if(f.key==='hunger'){
      const seg = segBox(f.label);
      f.options.forEach(o=>{
        const b = el('button',null,o);
        const on = String(ST[f.key])===o;
        if(on) b.classList.add('on');
        b.setAttribute('aria-pressed', on ? 'true' : 'false');
        b.onclick = ()=>{ ST[f.key]=+o; markOn(seg, b); };
        seg.appendChild(b);
      });
      box.appendChild(seg);
    } else {
      const seg = segBox(f.label);
      const opts = (f.key==='taboo') ? ['', ...f.options.filter(x=>x)] : f.options;
      opts.forEach(o=>{
        const b = el('button',null,o===''?'无忌口':o);
        const on = (ST[f.key]??'')===o;
        if(on) b.classList.add('on');
        b.setAttribute('aria-pressed', on ? 'true' : 'false');
        b.onclick = ()=>{ ST[f.key]=o; markOn(seg, b); };
        seg.appendChild(b);
      });
      box.appendChild(seg);
    }
    form.appendChild(box);
  });

  Q = document.getElementById('q');
  SUG = document.getElementById('chips');
  SEL = document.getElementById('sel');
  NAMES = META.dishes.map(d=>d.name);
  Q.addEventListener('input', renderSug);
  renderSel(); renderSug();

  document.getElementById('go').onclick = run;
  render();
}

function render(){
  // 让表单控件与 ST 同步（换预设时用）。
  // 按 dataset.key 找字段，不靠 DOM 下标 —— 以后在表单里插任何元素都不会错位。
  document.querySelectorAll('#form .field').forEach(box=>{
    const f = META.fields.find(x=>x.key === box.dataset.key);
    if(!f) return;
    if(f.key==='available_time'){
      const r = box.querySelector('input');
      r.value = ST[f.key] ?? 45;
      box.querySelector('.big').textContent = r.value+' 分钟';
    } else {
      const seg = box.querySelector('.seg');
      const opts = (f.key==='taboo') ? ['', ...f.options.filter(x=>x)] : f.options;
      const cur = String(ST[f.key] ?? (f.key==='hunger'?3:opts[0]));
      const idx = opts.findIndex(o=>String(o)===cur);
      [...seg.children].forEach((b,i)=>{
        const on = (i === idx);
        b.classList.toggle('on', on);
        b.setAttribute('aria-pressed', on ? 'true' : 'false');
      });
    }
  });
}

// 供自动化截图/联调使用：外部页面可用 postMessage({type:'autorun'}) 直接触发一次推荐。
// 正常浏览器访问不会用到这条路径，不影响手动操作。
window.addEventListener('message', ev=>{
  const m = ev.data;
  if(m && m.type === 'autorun'){
    if(m.preset && META.presets[m.preset]) ST = {...META.presets[m.preset]};
    if(m.fill) Object.assign(ST, m.fill);
    render(); run();
  }
});

async function run(){
  const btn = document.getElementById('go');
  const out = document.getElementById('out');
  btn.disabled = true; btn.innerHTML = '<span class="spin"></span>正在算…';
  // 结果区立刻进入加载态：否则右侧一直停着上一次的结果，用户不知道点没点上
  out.innerHTML = '<div class="loading">正在按你的条件打分…</div>';
  try{
    const res = await fetch('/api/predict',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({...ST, exclude:[...EX], topk:5})});
    const d = await res.json();
    if(d.error) throw new Error(d.error);
    draw(d);
  }catch(e){
    // fetch 抛 TypeError = 压根没连上（服务没起来）；其余按后端报错处理。
    // 这两种情况用户该做的事完全不同，别糊成一句话。
    const msg = String((e && e.message) || e);
    const offline = (e instanceof TypeError) ||
                    /Failed to fetch|NetworkError|Load failed/i.test(msg);
    out.innerHTML = offline
      ? '<div class="error">连不上本地服务。确认 <code>启动.bat</code> 还开着，' +
        '或在本目录运行 <code>python server.py</code>，再刷新页面。</div>'
      : '<div class="error">算的时候出错了：' + msg +
        '<br>详细报错看服务端那个命令行窗口。</div>';
  }finally{
    btn.disabled=false; btn.textContent='给我推荐';
  }
}

function draw(d){
  const out = document.getElementById('out');
  out.innerHTML='';
  const sl = document.getElementById('stage-label');
  if (sl) sl.textContent = d.stage === 'single'
    ? '单阶段：整目录 softmax，直接在全部菜里排序'
    : '两层结构：先判断品类，再在品类内排序';
  const bar = (p)=>{const w=Math.max(6,Math.round(p*130));
    return `<span class="bar" aria-hidden="true" style="width:${w}px"></span>`;};
  const cu = el('div','cuisine');
  cu.innerHTML = (d.stage === 'single' ? '品类倾向（展示用）： ' : '第一层判断： ') + d.cuisine_top.map(c=>
    `${bar(c.prob)}<b>${c.name}</b> ${(c.prob*100).toFixed(0)}%`).join(' &nbsp;&nbsp; ');
  out.appendChild(cu);

  const t = d.items[0];
  const pick = el('div','pick');
  pick.innerHTML = `<div class="name">${t.dish_name}</div>
    <div class="meta">${t.category} · ${t.price} 元 · ${t.time_cost} 分钟 ·
      相对倾向 ${pct(t.prob)}</div>
    <div class="why">为什么：${t.reason}</div>
    <button class="nope">不想吃这道 → 换一个</button>`;
  // 排除并立刻重算：把菜名塞进 EX，再跑一次（后端本来就支持 exclude，零后端改动）
  pick.querySelector('.nope').onclick = ()=>{
    EX.add(t.dish_name); renderSel(); renderSug(); run();
  };
  out.appendChild(pick);

  d.items.slice(1).forEach(x=>{
    const a = el('div','alt');
    a.innerHTML = `<div><b>第${x.rank}名 · ${x.dish_name}</b>
        <span class="tag">· ${x.category} · ${x.price}元 · ${x.reason}</span></div>
      <span class="pct">${pct(x.prob)}</span>`;
    out.appendChild(a);
  });

  // 展开：只列头部候选的原始打分（后端只发前 20 条），不做任何美化。
  // 说明文字必须跟着模型结构走：单阶段在整目录上统一排序，分数是**跨品类可比**的；
  // 只有两层模型才只能在品类内横向比较。写死一句会让页面自相矛盾。
  // 表里不显示原始分（18.906 对用户没有意义），显示「与第 1 名的差」——
  // 这就是 P1-1：分数绝对值未校准，差值才可读；品类列是因为单阶段分数跨品类可比。
  const det = el('details');
  const catOf = new Map(META.dishes.map(x=>[x.name, x.category]));
  const totalCand = d.candidates ?? d.all_scores.length;
  const topScore = d.all_scores.length ? d.all_scores[0].score : 0;
  const detNote = d.stage === 'single'
    ? '「与第 1 名差」是模型打分之间的距离；单阶段在全部菜上统一排序，' +
      '<b>分数跨品类可比</b>，差距可以直接横着比。'
    : '「与第 1 名差」是模型打分之间的距离，<b>只能在同一品类内横向比较</b>，' +
      '跨品类没有可比性（这是两层结构的特点）。';
  det.innerHTML = '<summary>看头部 ' + d.all_scores.length + ' 名与第 1 名的差距（共 ' +
      totalCand + ' 道候选，不美化）</summary>' +
    '<div class="fine">' + detNote + '差距越小，说明这几道菜' +
    '本来就差不多，随便挑一个都行。</div>' +
    '<table class="raw"><tr><th>菜</th><th>品类</th><th>与第 1 名差</th></tr>' +
    d.all_scores.map((s,i)=>`<tr><td>${s.dish_name}</td>` +
      `<td class="cat">${catOf.get(s.dish_name) || '—'}</td>` +
      `<td>${i === 0 ? '—' : Math.max(0, topScore - s.score).toFixed(3)}</td></tr>`).join('') +
    '</table>';
  out.appendChild(det);

  if(d.truth){
    out.appendChild(el('div','truth',
      `对照：规则下的最优是「${d.truth.best_dish}」(效用 ${d.truth.best_possible})，
       本次首推效用 ${d.truth.model_top1_utility}。效用是 0~1 的连续值。`));
  }
  if(d.excluded) out.appendChild(el('div','fine',`已排除 ${d.excluded} 道菜`));
}

boot();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="今天该吃什么 —— 本地 Web 界面")
    ap.add_argument("--port", type=int, default=PORT_DEFAULT)
    ap.add_argument("--host", default=HOST_DEFAULT)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    print("加载模型中 ...")
    load_model()
    s = _MODEL["stats"]
    print(f"  就绪：{s['n_dishes']} 道菜 / {s['n_categories']} 个品类 / "
          f"{s['n_scenarios']} 个情境")

    url = f"http://{args.host}:{args.port}/"
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\n  界面地址： {url}")
    print("  按 Ctrl+C 停止\n")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
