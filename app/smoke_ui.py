# -*- coding: utf-8 -*-
"""
smoke_ui.py —— 界面与接口的自动验收（《前端交接文档》§1.3 的第 ① 条）

一条命令，覆盖四组断言：

  接口   /api/meta 与 /api/predict 的字段、数量、体积、耗时、exclude 是否生效
  补丁   snapshot_ui.py 里每一处 html.replace 的目标串是否还存在于页面中
         （**静默 no-op** 是这套前端最隐蔽的故障：replace 找不到目标也不报错，
          快照会悄悄变成空表单。这里用 AST 抽出目标串逐条核对。）
  DOM    用 headless Chrome/Edge 真渲染页面、真点一次「给我推荐」，验证预设 /
         字段 / 忌口按钮数量、首屏模型措辞、结果区是不是真的画出来了
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
import ast
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

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
# 数一遍 DOM，再点一次「给我推荐」，等结果画完，把观测结果写到
# <html data-smoke="...">，由 --dump-dom 带回来。
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
  function txt(el){ return el ? (el.textContent || '').replace(/\\s+/g, ' ').trim() : ''; }
  var f = document.getElementById('f');
  f.addEventListener('load', function(){
    setTimeout(function(){
      var d = null;
      try { d = f.contentDocument; } catch (e) { out.error = 'iframe: ' + e; push(); return; }
      if (!d) { out.error = 'contentDocument 为空'; push(); return; }
      function cnt(s){ return d.querySelectorAll(s).length; }
      out.presets = cnt('#presets button');
      out.fields = cnt('#form .field');
      out.chips = cnt('#chips button');
      out.stage_label = txt(d.getElementById('stage-label'));
      out.stats = txt(d.getElementById('stats'));
      out.go_text = txt(d.getElementById('go'));
      if (!out.fields || !out.presets) { push(); return; }   // boot() 没跑起来，别点
      var go = d.getElementById('go');
      if (!go) { out.error = '找不到 #go'; push(); return; }
      go.click();
      setTimeout(function(){
        out.pick = txt(d.querySelector('#out .pick .name'));
        out.reason = txt(d.querySelector('#out .pick .why'));
        out.alts = cnt('#out .alt');
        out.raw_rows = cnt('#out table.raw tr');
        out.fine = txt(d.querySelector('#out .fine'));
        out.truth = txt(d.querySelector('#out .truth'));
        out.out_empty = txt(d.querySelector('#out .empty'));
        push();
      }, 2500);
    }, 900);
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

    check("响应体积 < 60 KB", len(body) < 60 * 1024, f"{len(body) / 1024:.1f} KB",
          f"{len(body) / 1024:.1f} KB 已超阈值")
    check("单次推荐耗时 < 500 ms", ms < 500, f"{ms:.0f} ms")
    check("响应字段齐全", PREDICT_KEYS <= set(d), "", f"缺：{PREDICT_KEYS - set(d)}")
    eq("items 条数 = topk", len(d["items"]), payload["topk"])
    eq("all_scores 条数 = candidates", len(d["all_scores"]), d["candidates"])
    eq("rank 从 1 连续", [i["rank"] for i in d["items"]],
       list(range(1, len(d["items"]) + 1)))
    probs = [i["prob"] for i in d["items"]]
    check("prob 单调不增", all(a >= b for a, b in zip(probs, probs[1:])),
          f"{probs[0]:.3f} -> {probs[-1]:.3f}")
    check("每条推荐都带 reason", all(i["reason"] for i in d["items"]),
          d["items"][0]["reason"])

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


def check_page(base: str, meta: dict, d: dict) -> tuple[str, list[str]]:
    group("页面与快照补丁")
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

    # ---- 关键：snapshot_ui.py 的每一处 html.replace 目标都必须还在页面里 ----
    try:
        needles = snapshot_patch_targets()
    except FileNotFoundError:
        needles = []
        skip("读取 snapshot_ui.py", "文件不存在")
    if needles:
        dead = [n for n in needles if n not in page]
        check(f"snapshot_ui.py 的 {len(needles)} 处 replace 目标全部命中",
              not dead, "没有静默 no-op",
              "失效目标（改这里会静默生成空快照）：" +
              " ｜ ".join(repr(n)[:70] for n in dead))
    else:
        skip("snapshot_ui.py 补丁目标", "没解析到 replace")

    # 单阶段 / 两层 的说明文字必须两套都在（具体渲染哪套由 DOM 组验证）
    check("页面含两套打分口径说明",
          "分数跨品类可比" in page and "只能在同一品类内横向比较" in page,
          "single / two_layer 各一套")
    return page, [n for n in needles]


def snapshot_patch_targets() -> list[str]:
    """从 snapshot_ui.py 里抽出所有 html.replace(目标, ...) 的目标串。"""
    src = (HERE / "snapshot_ui.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    out: list[str] = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "replace"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "html" and node.args):
            a0 = node.args[0]
            if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                out.append(a0.value)
    return out


def find_chrome() -> str | None:
    for p in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"):
        if Path(p).exists():
            return p
    for n in ("chrome", "msedge", "chromium"):
        w = shutil.which(n)
        if w:
            return w
    return None


def dump_dom(exe: str, url: str) -> tuple[str, str]:
    """返回 (DOM, stderr)。stdout 用管道即可；失败时 stderr 里能看出原因。"""
    prof = tempfile.mkdtemp(prefix="smoke_dom_")
    try:
        cmd = [exe, "--headless=new", "--disable-gpu", "--no-sandbox",
               "--no-first-run", "--disable-extensions", "--hide-scrollbars",
               f"--user-data-dir={prof}", "--virtual-time-budget=12000",
               "--window-size=1280,900", "--dump-dom", url]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=120)
        return r.stdout or "", r.stderr or ""
    finally:
        shutil.rmtree(prof, ignore_errors=True)


def first_error_line(stderr: str) -> str:
    for line in stderr.splitlines():
        line = line.strip()
        if line and ("FATAL" in line or "ERROR" in line or "denied" in line.lower()
                     or "拒绝" in line):
            return re.sub(r"^\[[^\]]*\]\s*", "", line)[:90]
    return (stderr.strip().splitlines() or [""])[0][:90]


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
    eq("忌口按钮数量 = 菜品数", sm.get("chips"), len(meta["dishes"]))
    want = "单阶段" if meta["stage"] == "single" else "两层"
    check("首屏 #stage-label 与模型结构一致", want in sm.get("stage_label", ""),
          sm.get("stage_label", ""))
    check("#stats 写入了统计", str(meta["stats"]["n_dishes"]) in sm.get("stats", ""),
          sm.get("stats", "")[:60])

    # 点了「给我推荐」之后
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
    info("原始打分表行数", f"{sm.get('raw_rows')} 行"
         "（P0-1 未修时 = 菜品数 + 表头）")


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
        page, _ = check_page(base, meta, d)
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
