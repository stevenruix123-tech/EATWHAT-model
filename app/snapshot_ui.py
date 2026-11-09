# -*- coding: utf-8 -*-
"""
snapshot_ui.py —— 用**真实页面**渲染出「已出结果」的静态快照（ui_snapshot.html）

为什么不再用 render_block()：
  以前这里用 Python 把 draw() 的 HTML 再写一遍，再用 5 处 html.replace 往页面里打补丁。
  两份渲染代码必然分叉 —— 2026-11-05 就出过一次「页面说单阶段、说明文字却按两层写」的
  事故，而截图恰好是对的，肉眼验收完全发现不了。现在改成：

    起临时服务（真实 Handler + 一条 /__snapshot 路由）
      -> headless Chrome 打开 /__snapshot
      -> 同源 iframe 里打开真实页面，postMessage 触发一次推荐，等结果画完
      -> 把 iframe 的 DOM 去掉 <script> 后写回顶层文档
      -> --dump-dom 抓回这份「已渲染」的 HTML

  这样快照里的每个字都来自页面的 draw()，不可能再和页面分叉。

要求本机有 Chrome / Edge（和 §1.3 的截图步骤一样）。找不到就报错退出，不静默降级。
产物：artifacts/ui_snapshot.html（.gitignore 已忽略），再用 html_shot 截成 ui_shot.png。
"""

from __future__ import annotations

import re
import threading
from http.server import ThreadingHTTPServer

from browser_env import dump_dom, find_chrome, first_error_line
from features import ART

OUT_HTML = ART / "ui_snapshot.html"
PRESET = "深夜加班"          # 截图默认展示的预设

# 跑在浏览器里的那一小段：iframe 打开真实页面 -> 触发一次推荐 -> 把渲染后的 DOM
# 去脚本后写回顶层文档，交给 --dump-dom 带回来。
HARNESS = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>snapshot</title></head>
<body style="margin:0">
<iframe id="f" src="/" style="width:1280px;height:900px;border:0"></iframe>
<script>
(function(){
  var f = document.getElementById('f');
  function fail(msg){ document.documentElement.setAttribute('data-snap-error', msg); }
  f.addEventListener('load', function(){
    setTimeout(function(){
      var d = null;
      try { d = f.contentDocument; } catch (e) { fail('iframe: ' + e); return; }
      if (!d) { fail('contentDocument 为空'); return; }
      if (!d.querySelector('#form .field')) { fail('boot() 没跑起来'); return; }
      f.contentWindow.postMessage({type:'autorun', preset: __PRESET__}, '*');
      setTimeout(function(){
        if (!d.querySelector('#out .pick .name')) { fail('推荐结果没出来'); return; }
        var html = d.documentElement.outerHTML
          .replace(/<script[\\s\\S]*?<\\/script>/gi, '');
        document.open();
        document.write('<!DOCTYPE html>\\n' + html);
        document.close();
      }, 2600);
    }, 900);
  });
})();
</script>
</body></html>"""


def serve():
    """起一个只多一条 /__snapshot 路由的真实服务。"""
    import server as S

    S.load_model()
    harness = HARNESS.replace("__PRESET__", repr(PRESET))

    class SnapshotHandler(S.Handler):
        def do_GET(self):
            if self.path == "/__snapshot":
                self._send(200, harness.encode("utf-8"), "text/html; charset=utf-8")
            else:
                super().do_GET()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), SnapshotHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def main() -> int:
    exe = find_chrome()
    if not exe:
        print("[错误] 找不到 Chrome / Edge —— 快照现在靠真实浏览器渲染，请先装一个。")
        return 2

    httpd, base = serve()
    try:
        dom, err = dump_dom(exe, base + "/__snapshot", budget_ms=12000)
    finally:
        httpd.shutdown()
        httpd.server_close()

    if not dom.strip():
        print(f"[错误] 浏览器没有输出 DOM：{first_error_line(err) or '未知原因'}")
        print("       受限沙箱会拦 Chrome 的命名管道，请在普通终端里重跑。")
        return 2
    m = re.search(r'data-snap-error="([^"]*)"', dom)
    if m:
        print(f"[错误] 快照渲染失败：{m.group(1)}")
        return 2

    OUT_HTML.write_text(dom, encoding="utf-8")
    pick = re.search(r'<div class="name">([^<]+)</div>', dom)
    print("written ->", OUT_HTML, f"({len(dom) / 1024:.0f} KB)")
    print("首推:", pick.group(1) if pick else "（没找到 .pick .name）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
