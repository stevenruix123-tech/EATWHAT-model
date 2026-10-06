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

    # ---- 第一层：品类分布 ----
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

    # ---- 第二层：候选品类内逐道打分 ----
    cand = np.where(DISH_CAT_IDX[:, 0] == win)[0]
    ids, dc, dn = T.tile_dishes(cand[None, :], 1)
    scores = dm.forward(c, n, ids, dc, dn)[0].astype(float)
    ex_mask = np.array([DISH_NAMES[int(i)] in exclude for i in cand])
    if ex_mask.all():                      # 排除过头了，退回全库
        ex_mask = np.zeros_like(ex_mask, dtype=bool)
    scores = np.where(ex_mask, -1e9, scores)

    # 品类内相对概率。
    # 为什么不能直接 softmax：模型的原始分数没有概率校准，同一品类内分数
    # 往往挤在很窄的区间（真实效用差 0.008，分数差也很小），直接 softmax
    # 会给出「98% vs 0.5%」这种严重夸大的对比。
    # 做法：先按温度缩放软化，再映射到该品类内的相对匹配度。
    T_SCALE = 0.35
    z = (scores - scores.max()) / T_SCALE
    p = np.exp(np.clip(z, -30, 0))
    p = p / p.sum()

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
            "match": round(float(p[j]), 4),
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
        "cuisine_top": [
            {"name": CATEGORIES[int(i)], "prob": round(float(cat_p[i]), 4)}
            for i in cat_order[:3]
        ],
        "candidates": int(len(cand)),
        "excluded": int(ex_mask.sum()),
        "items": items,
        # 该品类内所有候选的原始打分（不做任何美化，供用户自己判断差距）
        "all_scores": sorted(
            [{"dish_name": DISH_NAMES[int(cand[k])], "score": round(float(scores[k]), 3)}
             for k in range(len(cand))],
            key=lambda x: -x["score"]),
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
  header{padding:28px 20px 8px;max-width:1080px;margin:0 auto}
  h1{margin:0;font-size:24px;letter-spacing:.5px}
  .sub{color:var(--muted);font-size:13px;margin-top:6px}
  .wrap{max-width:1080px;margin:0 auto;padding:12px 20px 60px;
    display:grid;grid-template-columns:1.05fr 1fr;gap:22px;align-items:start}
  @media (max-width:880px){ .wrap{grid-template-columns:1fr} }
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;
    padding:18px 18px 20px}
  .card h2{margin:0 0 14px;font-size:15px;font-weight:600;color:var(--muted);
    letter-spacing:.4px}
  .field{margin-bottom:14px}
  .field > label{display:block;font-size:13px;color:var(--muted);margin-bottom:6px}
  .seg{display:flex;flex-wrap:wrap;gap:6px}
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
  .presets{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:16px}
  .presets button{border:1px dashed var(--line);background:transparent;color:var(--muted);
    padding:5px 10px;border-radius:999px;font-size:12.5px;cursor:pointer}
  .presets button:hover{color:var(--brand);border-color:var(--brand)}
  .cuisine{font-size:13px;color:var(--muted);margin-bottom:12px}
  .bar{display:inline-block;height:7px;border-radius:4px;background:var(--bar);
    vertical-align:middle;margin:0 7px 2px 0}
  .pick{border:1px solid var(--brand);background:var(--brand-soft);border-radius:12px;
    padding:14px 16px;margin-bottom:12px}
  .pick .name{font-size:21px;font-weight:700;letter-spacing:.3px}
  .pick .meta{font-size:12.5px;color:var(--muted);margin-top:3px}
  .pick .why{font-size:13px;margin-top:8px;color:var(--ink);opacity:.85}
  .alt{display:flex;justify-content:space-between;gap:10px;padding:9px 2px;
    border-bottom:1px dashed var(--line);font-size:14px}
  .alt:last-child{border-bottom:0}
  .alt .tag{color:var(--muted);font-size:12.5px}
  table.raw{width:100%;border-collapse:collapse;margin-top:8px;font-size:12.5px;
    font-variant-numeric:tabular-nums}
  table.raw th{text-align:left;color:var(--muted);font-weight:500;padding:3px 4px;
    border-bottom:1px solid var(--line)}
  table.raw td{padding:3px 4px;border-bottom:1px dashed var(--line)}
  table.raw td:last-child{text-align:right;color:var(--muted)}
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
  .chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
  .chips button{border:1px solid var(--line);background:transparent;color:var(--muted);
    padding:4px 9px;border-radius:8px;font-size:12px;cursor:pointer}
  .chips button.rm{background:var(--brand);color:#fff;border-color:var(--brand)}
</style>
</head>
<body>
<header>
  <h1>今天该吃什么</h1>
  <div class="sub">两层决策模型 · 第一层判断品类，第二层在品类内排序 ·
    <span id="stats"></span></div>
</header>

<div class="wrap">
  <section class="card">
    <h2>你的情况</h2>
    <div class="presets" id="presets"></div>
    <div id="form"></div>
    <button class="go" id="go">给我推荐</button>
    <details>
      <summary>不想吃的东西（点一下排除，再点恢复）</summary>
      <div class="chips" id="chips"></div>
    </details>
    <div class="fine">模型只看这些情境特征和菜品属性，没有联网、没有调用大模型。</div>
  </section>

  <section class="card">
    <h2>推荐结果</h2>
    <div id="out"><div class="empty">左边点几下，然后按「给我推荐」</div></div>
  </section>
</div>

<script>
let META=null, ST={}, EX=new Set();

function el(t,c,h){const e=document.createElement(t); if(c)e.className=c;
  if(h!==undefined)e.innerHTML=h; return e;}

async function boot(){
  META = await (await fetch('/api/meta')).json();
  ST = {...META.presets['工作日午餐']};
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
      const seg = el('div','seg');
      f.options.forEach(o=>{
        const b = el('button',null,o);
        if(String(ST[f.key])===o) b.classList.add('on');
        b.onclick = ()=>{ ST[f.key]=+o;
          [...seg.children].forEach(x=>x.classList.remove('on')); b.classList.add('on'); };
        seg.appendChild(b);
      });
      box.appendChild(seg);
    } else {
      const seg = el('div','seg');
      const opts = (f.key==='taboo') ? ['', ...f.options.filter(x=>x)] : f.options;
      opts.forEach(o=>{
        const b = el('button',null,o===''?'无忌口':o);
        if((ST[f.key]??'')===o) b.classList.add('on');
        b.onclick = ()=>{ ST[f.key]=o;
          [...seg.children].forEach(x=>x.classList.remove('on')); b.classList.add('on'); };
        seg.appendChild(b);
      });
      box.appendChild(seg);
    }
    form.appendChild(box);
  });

  // 菜品排除
  const chips = document.getElementById('chips');
  META.dishes.forEach(d=>{
    const b = el('button',null,d.name);
    b.dataset.name = d.name;
    b.onclick = ()=>{
      if(EX.has(d.name)){ EX.delete(d.name); b.classList.remove('rm'); }
      else { EX.add(d.name); b.classList.add('rm'); }
    };
    chips.appendChild(b);
  });

  document.getElementById('go').onclick = run;
  render();
}

function render(){
  // 让表单控件与 ST 同步（换预设时用）
  document.querySelectorAll('#form .field').forEach((box,i)=>{
    const f = META.fields[i];
    if(f.key==='available_time'){
      const r = box.querySelector('input');
      r.value = ST[f.key] ?? 45;
      box.querySelector('.big').textContent = r.value+' 分钟';
    } else {
      const seg = box.querySelector('.seg');
      [...seg.children].forEach(b=>b.classList.remove('on'));
      const opts = (f.key==='taboo') ? ['', ...f.options.filter(x=>x)] : f.options;
      const cur = String(ST[f.key] ?? (f.key==='hunger'?3:opts[0]));
      const idx = opts.findIndex(o=>String(o)===cur);
      if(idx>=0) seg.children[idx].classList.add('on');
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
  try{
    const res = await fetch('/api/predict',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({...ST, exclude:[...EX], topk:5})});
    const d = await res.json();
    if(d.error) throw new Error(d.error);
    draw(d);
  }catch(e){
    out.innerHTML = '<div class="empty">出错了：'+e.message+'</div>';
  }finally{
    btn.disabled=false; btn.textContent='给我推荐';
  }
}

function draw(d){
  const out = document.getElementById('out');
  out.innerHTML='';
  const bar = (p)=>{const w=Math.max(6,Math.round(p*130));
    return `<span class="bar" style="width:${w}px"></span>`;};
  const cu = el('div','cuisine');
  cu.innerHTML = '第一层判断： ' + d.cuisine_top.map(c=>
    `${bar(c.prob)}<b>${c.name}</b> ${(c.prob*100).toFixed(0)}%`).join(' &nbsp;&nbsp; ');
  out.appendChild(cu);

  const t = d.items[0];
  const pick = el('div','pick');
  pick.innerHTML = `<div class="name">${t.dish_name}</div>
    <div class="meta">${t.category} · ${t.price} 元 · ${t.time_cost} 分钟</div>
    <div class="why">为什么：${t.reason}</div>`;
  out.appendChild(pick);

  d.items.slice(1).forEach(x=>{
    const a = el('div','alt');
    a.innerHTML = `<div><b>第{x.rank}名 · ${x.dish_name}</b>
        <span class="tag">· ${x.category} · ${x.price}元 · ${x.reason}</span></div>`;
    out.appendChild(a);
  });

  // 展开：把品类内所有候选的原始打分都列出来，不做任何美化
  const det = el('details');
  det.innerHTML = '<summary>看模型给每道菜的原始打分（不美化）</summary>' +
    '<div class="fine">分数是模型学出来的效用预测值，<b>只能在同一品类内横向比较</b>，' +
    '跨品类没有可比性（这是两层结构的特点）。同一品类内分数越接近，说明这几道菜' +
    '本来就差不多，随便挑一个都行。</div>' +
    '<table class="raw"><tr><th>菜</th><th>分数</th></tr>' +
    d.all_scores.map(s=>`<tr><td>${s.dish_name}</td><td>${s.score.toFixed(3)}</td></tr>`).join('') +
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
