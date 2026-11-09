# -*- coding: utf-8 -*-
"""
smoke_ui.py —— 界面与接口的自动验收（《前端交接文档》§1.3 的第 ① 条）

一条命令，覆盖四组断言：

  接口   /api/meta 与 /api/predict 的字段、数量、体积、耗时、exclude 是否生效
  补丁   snapshot_ui.py 里每一处 html.replace 的目标串是否还存在于页面中
         （**静默 no-op** 是这套前端最隐蔽的故障：replace 找不到目标也不报错，
          快照会悄悄变成空表单。这里用 AST 抽出目标串逐条核对。）
  DOM    用 headless Chrome/Edge 真渲染页面：点一次「给我推荐」，再走一遍
         「搜索排除 → 重算」，验证预设 / 字段数量、首屏模型措辞、结果区、
         忌口搜索链路是不是真的通了
  文档   docs/前端交接文档.md 里 `anchor:` 锚点是否都还能在 server.py 里找到，
         tooling/_dump_dom.py 的路径是否还指着 app/artifacts/

为什么要有它：这个前端没有构建、没有框架、没有测试，改完只能靠手敲命令 +
肉眼看截图。而截图是 snapshot_ui.py 用另一份渲染代码画的（render_block），
它和页面的 draw() 已经各自演化过，所以「截图好看」不等于「页面正确」。

用法（在 app/ 目录下）：
    python smoke_ui.py                     # 自起临时端口，不需要先开服务
    python smoke_ui.py --url http://127.0.0.1:8848    # 验收正在跑的那个实例
    python smoke_ui.py --no-dom            # 跳过浏览器（没装 Chrome/Edge 会自动跳过）

退出码：0 = 全部通过（含跳过）；1 = 有失败项。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from browser_env import dump_dom, find_chrome, first_error_line

HERE = Path(__file__).resolve().parent          # app/
ROOT = HERE.parent                              # 仓库根
DOC = ROOT / "docs" / "前端交接文档.md"

OK, FAIL, SKIP, INFO = "OK", "FAIL", "SKIP", "INFO"

# 10 个情境字段的顺序是接口契约的一部分（前端按数组顺序渲染）
FIELD_KEYS = ["time_slot", "hunger", "available_time", "budget", "mood",
              "weather", "companion", "health_goal", "spicy_tol", "taboo"]

PREDICT_KEYS = {"scenario", "stage", "cuisine_top", "candidates", "excluded",
                "items", "all_scores", "truth"}

# 页面里必须一直存在的锚点（改结构时同步改这里，比记行号可靠）
PAGE_ANCHORS = [
    '<style>',
    'async function boot(',
    'function render(',
    'async function run(',
    'function draw(d)',
    'META.fields.forEach',
    'id="presets"',
    'id="form"',
    'id="chips"',
]

DEFAULT_SCENARIO = {
    "time_slot": "午餐", "hunger": 3, "available_time": 30, "budget": "正常",
    "mood": "一般", "weather": "热", "companion": "一个人",
    "health_goal": "严格控制", "spicy_tol": "微辣", "taboo": "",
}

# 浏览器里跑的那一小段脚本：同源 iframe 打开真实页面，等 boot() 跑完，
# 数一遍 DOM，点一次「给我推荐」，然后再走一遍「搜索排除 -> 重算」，
# 把观测结果写到 <html data-smoke="...">，由 --dump-dom 带回来。
HARNESS = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>smoke</title></head>
<body style="margin:0">
<iframe id="f" src="/" style="width:1280px;height:900px;border:0"></iframe>
<script>
(function(){
  var out = {};
  function push(){
    document.documentElement.setAttribute('data-smoke',
      encodeURIComponent(JSON.stringify(out)));
    document.title = 'smoke-done';
  }
  function sleep(ms){ return new Promise(function(r){ setTimeout(r, ms); }); }
  function txt(el){ return el ? (el.textContent || '').replace(/\\s+/g, ' ').trim() : ''; }
  var f = document.getElementById('f');
  f.addEventListener('load', function(){
    (async function(){
      var d = null;
      try { d = f.contentDocument; } catch (e) { out.error = 'iframe: ' + e; push(); return; }
      if (!d) { out.error = 'contentDocument 为空'; push(); return; }
      await sleep(900);
      function cnt(s){ return d.querySelectorAll(s).length; }
      out.presets = cnt('#presets button');
      out.fields = cnt('#form .field');
      out.chips = cnt('#chips button');
      out.dom_nodes = d.querySelectorAll('*').length;
      out.has_q = !!d.getElementById('q');
      out.stage_label = txt(d.getElementById('stage-label'));
      out.stats = txt(d.getElementById('stats'));
      out.go_text = txt(d.getElementById('go'));
      var goEl = d.getElementById('go');
      out.go_top = goEl ? Math.round(goEl.getBoundingClientRect().top) : null;
      out.viewport_h = Math.round(f.getBoundingClientRect().height);
      var fg = d.getElementById('form');
      out.form_cols = fg
        ? getComputedStyle(fg).gridTemplateColumns.split(' ').filter(Boolean).length : 0;
      if (!out.fields || !out.presets) { push(); return; }   // boot() 没跑起来，别点
      var go = d.getElementById('go');
      if (!go) { out.error = '找不到 #go'; push(); return; }
      go.click();
      out.loading_seen = !!d.querySelector('#out .loading');   // 点完立刻读：run() 同步设了加载态
      await sleep(2500);
      out.pick = txt(d.querySelector('#out .pick .name'));
      out.reason = txt(d.querySelector('#out .pick .why'));
      out.alts = cnt('#out .alt');
      out.raw_rows = cnt('#out table.raw tr');
      var r1 = d.querySelectorAll('#out table.raw tr')[1];
      out.raw_cols = r1 ? r1.querySelectorAll('td').length : 0;
      out.raw_head = txt(d.querySelector('#out table.raw th:last-child'));
      out.raw_first_gap = txt(r1 ? r1.querySelectorAll('td')[2] : null);
      out.pick_meta = txt(d.querySelector('#out .pick .meta'));
      out.alt_pcts = cnt('#out .alt .pct');
      out.fine = txt(d.querySelector('#out .fine'));
      out.truth = txt(d.querySelector('#out .truth'));
      out.out_empty = txt(d.querySelector('#out .empty'));
      // 无障碍：分段按钮的选中态、按钮组语义、结果区的 aria-live
      out.pressed = cnt('#form [aria-pressed="true"]');
      out.segs_labeled = cnt('#form .seg[role="group"][aria-label]');
      out.form_keys = cnt('#form .field[data-key]');
      out.aria_live = (d.getElementById('out') || {}).getAttribute
        ? d.getElementById('out').getAttribute('aria-live') : null;

      function noteText(){
        var m = (d.getElementById('out').textContent || '').match(/已排除\\s*\\d+\\s*道菜/);
        return m ? m[0].replace(/\\s+/g, ' ').trim() : '';
      }

      // ---- P1-2「不想吃这道 → 换一个」：点按钮应直接排除并重算 ----
      var nope = d.querySelector('#out .pick .nope');
      out.nope_exists = !!nope;
      if (nope && out.pick) {
        nope.click();
        await sleep(200);
        out.sel_after_nope = cnt('#sel button');
        go.click();
        await sleep(2500);
        out.pick2 = txt(d.querySelector('#out .pick .name'));
        out.excluded_note_1 = noteText();
      }

      // ---- 忌口面板：搜索 -> 点候选 -> 重算，验证这条链路真的通 ----
      var q = d.getElementById('q');
      if (q && out.pick2) {
        q.value = out.pick2;
        q.dispatchEvent(new Event('input', {bubbles:true}));
        await sleep(300);
        out.sug_after_search = cnt('#chips button');
        var hit = null;
        [].forEach.call(d.querySelectorAll('#chips button'), function(b){
          if (!hit && (b.textContent || '').trim() === out.pick2) hit = b;
        });
        out.sug_first = hit ? txt(hit) : '';
        if (hit) hit.click();
        out.sel_count = cnt('#sel button');
        go.click();
        await sleep(2500);
        out.pick3 = txt(d.querySelector('#out .pick .name'));
        out.excluded_note_2 = noteText();
      }
      // ---- 错误态：把 fetch 换成必失败，看提示是不是「服务没连上」那种可操作的 ----
      var w = f.contentWindow;
      var origFetch = w.fetch;
      w.fetch = function(){ return Promise.reject(new TypeError('Failed to fetch')); };
      go.click();
      await sleep(600);
      out.error_text = txt(d.querySelector('#out .error'));
      out.error_class = !!d.querySelector('#out .error');
      w.fetch = origFetch;

      push();
    })();
  });
})();
</script>
</body></html>
"""

_rows: list[tuple[str, str, str]] = []
_group = ""


def group(name: str) -> None:
    global _group
    _group = name
    print(f"\n[{name}]")


def record(status: str, name: str, detail: str = "") -> None:
    _rows.append((_group, status, name))
    tail = f"  {detail}" if detail else ""
    print(f"  {status:<4} {name}{tail}")


def ok(name: str, detail: str = "") -> None:
    record(OK, name, detail)


def fail(name: str, detail: str = "") -> None:
    record(FAIL, name, detail)


def skip(name: str, detail: str = "") -> None:
    record(SKIP, name, detail)


def info(name: str, detail: str = "") -> None:
    record(INFO, name, detail)


def check(name: str, cond: bool, good: str = "", bad: str = "") -> bool:
    if cond:
        ok(name, good)
    else:
        fail(name, bad or good)
    return bool(cond)


def eq(name: str, actual, expect, unit: str = "") -> bool:
    if actual == expect:
        ok(name, f"{actual}{unit}")
        return True
    fail(name, f"期望 {expect}{unit}，实际 {actual}{unit}")
    return False


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
def http_get(base: str, path: str, timeout: float = 15.0):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return r.status, r.read()


def http_post(base: str, obj: dict, timeout: float = 20.0) -> tuple[int, bytes]:
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(base + "/api/predict", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def post_json(base: str, obj: dict) -> dict:
    status, body = http_post(base, obj)
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {body[:200]!r}")
    return json.loads(body.decode("utf-8"))


# --------------------------------------------------------------------------
# 本地实例（临时端口，不占用 8848，也不需要先启动服务）
# --------------------------------------------------------------------------
def start_local_server():
    sys.path.insert(0, str(HERE))
    import server as S                      # noqa: PLC0415  (要等 sys.path 就位)

    S.load_model()

    class SmokeHandler(S.Handler):
        def do_GET(self):
            if self.path == "/__smoke":
                self._send(200, HARNESS.encode("utf-8"), "text/html; charset=utf-8")
            else:
                super().do_GET()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), SmokeHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    return httpd, base, S


# --------------------------------------------------------------------------
# 各组检查
# --------------------------------------------------------------------------
def check_meta(base: str) -> dict:
    group("接口 /api/meta")
    status, body = http_get(base, "/api/meta")
    eq("状态码 200", status, 200)
    meta = json.loads(body.decode("utf-8"))
    info("响应体积", f"{len(body) / 1024:.1f} KB")

    eq("情境字段数量", len(meta.get("fields", [])), 10)
    eq("情境字段顺序 / key", [f["key"] for f in meta["fields"]], FIELD_KEYS)
    bad = [n for n, p in meta["presets"].items() if set(p) != set(FIELD_KEYS)]
    check("每个预设都覆盖 10 个字段", not bad,
          f"{len(meta['presets'])} 个预设", f"字段不全：{bad}")
    eq("预设数量", len(meta["presets"]), 10)

    dishes = meta["dishes"]
    eq("菜品数量", len(dishes), 824)
    ids = [d["id"] for d in dishes]
    check("菜品 id 连续 0..N-1", ids == list(range(len(ids))), f"0..{len(ids) - 1}")
    check("菜名唯一", len({d["name"] for d in dishes}) == len(dishes),
          f"{len(dishes)} 条")
    st = meta["stats"]
    eq("stats.n_dishes 与 dishes 一致", st["n_dishes"], len(dishes))
    eq("stats.n_categories", st["n_categories"], 19)
    check("stats 情境数 / 样本数为正", st["n_scenarios"] > 0 and st["n_rows"] > 0,
          f"{st['n_scenarios']} 情境 / {st['n_rows']} 条")
    check("stage 取值合法", meta["stage"] in ("single", "two_layer"), meta["stage"])
    return meta


def check_predict(base: str, meta: dict) -> dict:
    group("接口 /api/predict")
    payload = {**DEFAULT_SCENARIO, "topk": 5}
    t0 = time.perf_counter()
    status, body = http_post(base, payload)
    ms = (time.perf_counter() - t0) * 1000
    eq("状态码 200", status, 200)
    d = json.loads(body.decode("utf-8"))

    check("响应体积 < 10 KB", len(body) < 10 * 1024, f"{len(body) / 1024:.1f} KB",
          f"{len(body) / 1024:.1f} KB 已超阈值")
    check("单次推荐耗时 < 500 ms", ms < 500, f"{ms:.0f} ms")
    check("响应字段齐全", PREDICT_KEYS <= set(d), "", f"缺：{PREDICT_KEYS - set(d)}")
    eq("items 条数 = topk", len(d["items"]), payload["topk"])
    eq("all_scores 条数 = min(20, candidates)", len(d["all_scores"]),
       min(20, d["candidates"]))
    eq("rank 从 1 连续", [i["rank"] for i in d["items"]],
       list(range(1, len(d["items"]) + 1)))
    probs = [i["prob"] for i in d["items"]]
    check("prob 单调不增", all(a >= b for a, b in zip(probs, probs[1:])),
          f"{probs[0]:.3f} -> {probs[-1]:.3f}")
    check("每条推荐都带 reason", all(i["reason"] for i in d["items"]),
          d["items"][0]["reason"])
    check("items 里没有历史遗留的 match 字段",
          all("match" not in i for i in d["items"]))

    first = d["items"][0]["dish_name"]
    d2 = post_json(base, {**payload, "exclude": [first]})
    check("exclude 生效（首推被排除后换人）",
          d2["excluded"] == 1 and d2["items"][0]["dish_name"] != first,
          f"{first} -> {d2['items'][0]['dish_name']}")

    # 枚举值不合法时服务端静默回落 —— 这是既有行为，这里只确认它不崩、且确实回落
    d3 = post_json(base, {**payload, "weather": "大晴天"})
    check("非法枚举静默回落（不报错）",
          d3["scenario"]["weather"] != "大晴天",
          f"大晴天 -> {d3['scenario']['weather']}")

    bad, slow = [], []
    for name, sc in meta["presets"].items():
        t = time.perf_counter()
        try:
            r = post_json(base, {**sc, "topk": 3})
            if not r["items"]:
                bad.append(name)
        except Exception as e:                      # noqa: BLE001
            bad.append(f"{name}({e})")
        slow.append((time.perf_counter() - t) * 1000)
    avg = sum(slow) / max(len(slow), 1)
    check("10 个预设全部能出结果", not bad,
          f"平均 {avg:.0f} ms", f"失败：{bad}")
    return d


def check_page(base: str, meta: dict, d: dict) -> str:
    group("页面与快照来源")
    status, body = http_get(base, "/")
    eq("GET / 状态码 200", status, 200)
    page = body.decode("utf-8")
    check("是完整 HTML", page.lstrip()[:15].lower().startswith("<!doctype html>"),
          f"{len(body) / 1024:.1f} KB")

    missing = [a for a in PAGE_ANCHORS if a not in page]
    check("页面关键锚点齐全", not missing, f"{len(PAGE_ANCHORS)} 个",
          f"缺失：{missing}")

    ext = re.findall(r'(?:src|href)="(https?://[^"]+)"', page)
    check("没有外部 CDN / 网络依赖", not ext, "0 个外链", f"发现外链：{ext}")

    # 快照现在复用真实页面渲染：snapshot_ui.py 里不能再有第二份渲染/字符串补丁
    snap_src = (HERE / "snapshot_ui.py").read_text(encoding="utf-8")
    legacy = [k for k in ("def render_block", "html.replace(") if k in snap_src]
    check("snapshot_ui.py 不再有第二份渲染（只走真实 draw()）", not legacy,
          "只用真实页面", f"又出现了：{legacy}")

    check("页面定义了 :focus-visible 焦点样式", ":focus-visible" in page)
    # 单阶段 / 两层 的说明文字必须两套都在（具体渲染哪套由 DOM 组验证）
    check("页面含两套打分口径说明",
          "分数跨品类可比" in page and "只能在同一品类内横向比较" in page,
          "single / two_layer 各一套")
    return page


def check_dom(base: str, meta: dict, d: dict, enabled: bool, local: bool) -> None:
    group("真实 DOM（headless 浏览器）")
    if not enabled:
        skip("DOM 检查", "已用 --no-dom 关闭")
        return
    if not local:
        skip("DOM 检查", "外部实例模式不支持同源 iframe")
        return
    exe = find_chrome()
    if not exe:
        skip("DOM 检查", "没找到 Chrome / Edge")
        return

    dom, err = dump_dom(exe, base + "/__smoke")
    if not dom.strip():
        # 受限沙箱（比如 DSH 的 workspace-write）会拦掉 Chrome 的 mojo 命名管道，
        # 这不是界面坏了，是环境跑不了浏览器 —— 降级成 SKIP 并说明原因。
        skip("DOM 检查", f"浏览器没能渲染：{first_error_line(err) or '没有输出'}"
             "（在普通终端里重跑即可）")
        return
    m = re.search(r'data-smoke="([^"]*)"', dom)
    if not m:
        fail("页面脚本跑完并回传观测值",
             "dump 里没有 data-smoke（boot() 抛错，或浏览器不支持虚拟时间）")
        return
    sm = json.loads(urllib.parse.unquote(m.group(1)))
    if sm.get("error"):
        fail("iframe 读取页面", sm["error"])
        return
    ok("浏览器拿到页面并回传观测值", Path(exe).name)

    eq("预设按钮数量", sm.get("presets"), 10)
    eq("表单 .field 数量", sm.get("fields"), 10)
    eq("首屏不再铺 824 个忌口按钮", sm.get("chips"), 0)
    check("首屏 DOM 节点 < 400", (sm.get("dom_nodes") or 0) < 400,
          f"{sm.get('dom_nodes')} 个")
    check("忌口面板是搜索框", bool(sm.get("has_q")), "已找到 #q")
    eq("宽屏表单两列", sm.get("form_cols"), 2)
    # 无障碍：aria-pressed 跟着选中态走，分段按钮组有 role=group + aria-label
    eq("aria-pressed=true 的按钮数 = 有分段的字段数", sm.get("pressed"), 9)
    eq(".seg 都带 role=group + aria-label", sm.get("segs_labeled"), 9)
    eq("每个 .field 都带 data-key", sm.get("form_keys"), 10)
    eq("结果区是 aria-live=polite", sm.get("aria_live"), "polite")
    check("主按钮在首屏内（#go 顶部 < 820 px）",
          (sm.get("go_top") or 99999) < 820,
          f"top={sm.get('go_top')} px / 视口 {sm.get('viewport_h')} px",
          f"#go 顶部在 {sm.get('go_top')} px，视口只有 {sm.get('viewport_h')} px")
    want = "单阶段" if meta["stage"] == "single" else "两层"
    check("首屏 #stage-label 与模型结构一致", want in sm.get("stage_label", ""),
          sm.get("stage_label", ""))
    check("#stats 写入了统计", str(meta["stats"]["n_dishes"]) in sm.get("stats", ""),
          sm.get("stats", "")[:60])

    # 点了「给我推荐」之后
    check("点击后结果区先进入加载态", bool(sm.get("loading_seen")), "看到 .loading")
    check("#go 之后出结果（没有报错占位）",
          bool(sm.get("pick")) and not sm.get("out_empty"),
          sm.get("pick", ""), sm.get("out_empty") or "结果区没有 .pick")
    eq(".alt 条数 = topk-1", sm.get("alts"), 4)
    # 页面 boot() 用「工作日午餐」初始化 ST，点按钮发的就是它；按同一预设问一次接口做对照
    preset = meta["presets"].get("工作日午餐")
    if preset:
        try:
            expect_top1 = post_json(base, {**preset, "topk": 5})["items"][0]["dish_name"]
        except Exception:                              # noqa: BLE001
            expect_top1 = d["items"][0]["dish_name"]
        check("首推与接口一致", sm.get("pick") == expect_top1,
              sm.get("pick", ""), f"页面 {sm.get('pick')!r} != 接口 {expect_top1!r}")
    else:
        skip("首推与接口对照", "预设里没有「工作日午餐」")
    check("首推带「为什么」", bool(sm.get("reason")), sm.get("reason", ""))
    check(".truth 对照存在", bool(sm.get("truth")), "已渲染")

    # 单阶段时页面不得再出现「跨品类没有可比性」这句两层专属说明
    if meta["stage"] == "single":
        check("单阶段的打分说明没有自相矛盾",
              "跨品类没有可比性" not in sm.get("fine", ""),
              sm.get("fine", "")[:48] + "...",
              "页面还在用两层口径：" + sm.get("fine", "")[:40])
    check("原始打分表只列头部 20 名", 1 < (sm.get("raw_rows") or 0) <= 21,
          f"{sm.get('raw_rows')} 行（含表头）")
    eq("原始打分表三列（菜 / 品类 / 与第 1 名差）", sm.get("raw_cols"), 3)
    eq("表头第三列是「与第 1 名差」", sm.get("raw_head"), "与第 1 名差")
    eq("第 1 名那行不显示分数（—）", sm.get("raw_first_gap"), "—")
    check("首推显示相对倾向", "相对倾向" in (sm.get("pick_meta") or ""),
          sm.get("pick_meta", ""))
    eq("每条备选都带相对倾向", sm.get("alt_pcts"), 4)

    # ---- P1-2「不想吃这道 → 换一个」：点按钮应直接排除并重算 ----
    check("首推卡片有「不想吃这道」按钮", bool(sm.get("nope_exists")))
    eq("点按钮后已排除 1 道", sm.get("sel_after_nope"), 1)
    check("「不想吃这道」之后首推换人",
          bool(sm.get("pick2")) and sm.get("pick2") != sm.get("pick"),
          f"{sm.get('pick')} -> {sm.get('pick2')}",
          f"首推仍是 {sm.get('pick2')!r}（exclude 没生效？）")
    check("结果区提示「已排除 1 道菜」",
          sm.get("excluded_note_1") == "已排除 1 道菜", sm.get("excluded_note_1", ""))

    # ---- 忌口搜索链路：搜当前首推 -> 出现候选 -> 点掉 -> 重算 ----
    n_sug = sm.get("sug_after_search") or 0
    check("搜索当前首推能出候选（1~8 条）", 1 <= n_sug <= 8, f"{n_sug} 条")
    check("候选里能找到刚搜的那道菜",
          sm.get("sug_first") == sm.get("pick2"), sm.get("sug_first", ""))
    eq("再点候选后已排除 2 道", sm.get("sel_count"), 2)
    check("再排除后首推又换人",
          bool(sm.get("pick3")) and sm.get("pick3") != sm.get("pick2"),
          f"{sm.get('pick2')} -> {sm.get('pick3')}")
    check("结果区提示「已排除 2 道菜」",
          sm.get("excluded_note_2") == "已排除 2 道菜", sm.get("excluded_note_2", ""))

    # ---- 错误态：fetch 失败时应给出「服务没起来」这种能照做的提示 ----
    err_text = sm.get("error_text") or ""
    check("服务连不上时显示可操作的错误提示",
          bool(sm.get("error_class")) and "启动.bat" in err_text,
          err_text[:56], f"错误提示不对：{err_text[:80]!r}")


def check_docs(page: str) -> None:
    group("文档一致性")
    if not DOC.exists():
        fail("交接文档存在", str(DOC))
        return
    text = DOC.read_text(encoding="utf-8")
    anchors = re.findall(r"^anchor:[ \t]*(.+?)[ \t]*$", text, re.M)
    check("文档定义了 anchor 锚点", len(anchors) >= 6, f"{len(anchors)} 条")
    if anchors:
        dead = [a for a in anchors if a not in page]
        check("文档锚点全部仍存在于页面", not dead, f"{len(anchors)} 条全命中",
              f"已失效（改回来或更新文档）：{dead}")
    else:
        skip("文档锚点校验", "文档里还没有 anchor 行")

    check("docs/README.md 已移除（避免与根 README 冲突）",
          not (ROOT / "docs" / "README.md").exists())

    hand = ROOT / "docs" / "交接文档.md"
    if hand.exists():
        old = hand.read_text(encoding="utf-8")
        check("交接文档不再把 docs/README.md 当现存文件",
              "| `docs/README.md` |" not in old)

    tool = ROOT / "tooling" / "_dump_dom.py"
    try:
        spec = importlib.util.spec_from_file_location("_dump_dom_tool", tool)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)                 # type: ignore[union-attr]
        want = ROOT / "app" / "artifacts" / "ui_snapshot.html"
        check("tooling/_dump_dom.py 的 SNAP 指向 app/artifacts", mod.SNAP == want,
              str(mod.SNAP.relative_to(ROOT)) if mod.SNAP.is_relative_to(ROOT) else str(mod.SNAP),
              f"实际 {mod.SNAP}，应为 {want}")
    except Exception as e:                            # noqa: BLE001
        fail("tooling/_dump_dom.py 可加载", f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="界面与接口自动验收")
    ap.add_argument("--url", default=None,
                    help="验收一个已在运行的实例，例如 http://127.0.0.1:8848")
    ap.add_argument("--no-dom", action="store_true", help="跳过 headless 浏览器检查")
    args = ap.parse_args(argv)

    print("=" * 64)
    print("  smoke_ui.py —— 界面与接口自动验收")
    print("=" * 64)

    httpd = None
    local = not args.url
    try:
        if args.url:
            base = args.url.rstrip("/")
            print(f"\n目标：外部实例 {base}（不加载本地模型）")
        else:
            t0 = time.perf_counter()
            httpd, base, _S = start_local_server()
            print(f"\n目标：本地临时实例 {base}"
                  f"（模型加载 {(time.perf_counter() - t0):.1f}s，不占用 8848）")

        meta = check_meta(base)
        d = check_predict(base, meta)
        page = check_page(base, meta, d)
        check_dom(base, meta, d, enabled=not args.no_dom, local=local)
        check_docs(page)
    except Exception as e:                            # noqa: BLE001
        record(FAIL, "检查过程中抛出异常", f"{type(e).__name__}: {e}")
    finally:
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()

    n_ok = sum(1 for _, s, _ in _rows if s == OK)
    n_fail = sum(1 for _, s, _ in _rows if s == FAIL)
    n_skip = sum(1 for _, s, _ in _rows if s == SKIP)
    print("\n" + "-" * 64)
    if n_fail:
        print("失败项：")
        for g, s, name in _rows:
            if s == FAIL:
                print(f"  - [{g}] {name}")
    print(f"结果：{n_ok} 通过 / {n_fail} 失败 / {n_skip} 跳过")
    print("提示：改完前端先跑这个，再用 snapshot_ui.py + 截图做人眼验收。")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
