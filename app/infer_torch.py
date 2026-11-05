# -*- coding: utf-8 -*-
"""
infer_torch.py —— 用 PyTorch 模型做推理，输出与 recommend.py 同结构的结果

作用：验证「PyTorch 训练出的权重能不能直接接管线上推理」。
如果这一步不通，训练再快也没有意义 —— 服务端要换推理栈才有价值。

用法：
    python infer_torch.py --preset 工作日午餐
    python infer_torch.py --all-presets
    python infer_torch.py --json '{"time_slot":"晚餐","health_goal":"想健康"}'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import features as feat
import recommend as R
import train_torch as TT
from features import ART, DISH_CAT_IDX, DISH_NAMES, N_DISH, CATEGORIES


class TorchBundle:
    """与 numpy 版 recommend.load_models() 对应的最小接口：predict(sc) -> dict"""

    def __init__(self, path="artifacts_torch_h96/model_torch.pt", device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(HERE / path, map_location=self.device, weights_only=False)
        m = ck.get("meta", {})
        TT.DishScorer.__init__.__defaults__ = (
            int(m.get("hid_dish", TT.HID_DISH)),
            int(m.get("d_ctx", TT.D_CTX)),
            int(m.get("id_dim", TT.ID_DIM)),
        )
        TT.CuisineNet.__init__.__defaults__ = (
            int(m.get("hid_ctx", TT.HID_CTX)), int(m.get("d_ctx", TT.D_CTX)))
        self.cm = TT.CuisineNet().to(self.device); self.cm.load_state_dict(ck["cuisine"]); self.cm.eval()
        self.dm = TT.DishScorer().to(self.device); self.dm.load_state_dict(ck["dish"]); self.dm.eval()
        self.ids, self.dc, self.dn = TT.dish_batch_tensors(self.device)
        self.dish_cat = DISH_CAT_IDX[:, 0]
        self.meta = m

    @torch.no_grad()
    def predict(self, sc: dict, topk: int = 3):
        df = pd.DataFrame([sc])
        c, n = feat.encode_context(df)
        ck = torch.as_tensor(np.ascontiguousarray(c), device=self.device)
        nk = torch.as_tensor(np.array(n, dtype=np.float32, copy=True), device=self.device)
        logits = self.cm(ck, nk)[0]
        # 与 server.py 同样的温度缩放，让展示概率不过度自信
        T_SCALE = 2.0
        z = (logits - logits.max()) / T_SCALE
        p = torch.softmax(z, dim=-1).cpu().numpy()
        win = int(logits.argmax())
        cand = np.where(self.dish_cat == win)[0]
        s = self.dm(ck, nk, self.ids, self.dc, self.dn)[0].float().cpu().numpy()
        order = cand[np.argsort(-s[cand])][:topk]
        items = [{"rank": i + 1, "dish_name": DISH_NAMES[int(d)],
                  "category": CATEGORIES[int(self.dish_cat[d])],
                  "score": round(float(s[d]), 4)}
                 for i, d in enumerate(order)]
        return {"cuisine_top": [(CATEGORIES[int(j)], round(float(p[j]), 4))
                               for j in np.argsort(-p)[:3]],
                "candidates": int(len(cand)), "items": items}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts_torch_h96/model_torch.pt")
    ap.add_argument("--preset", default=None)
    ap.add_argument("--all-presets", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    b = TorchBundle(args.model)
    print(f"设备: {b.device}   结构 meta: {b.meta}")
    print()

    def show(name, sc):
        r = b.predict(sc)
        tops = "  ".join(f"{n}({p:.0%})" for n, p in r["cuisine_top"])
        print(f"### {name}")
        print(f"  第一层品类判断：{tops}")
        print(f"  第二层候选 {r['candidates']} 道：")
        for it in r["items"]:
            print(f"    {it['rank']}. {it['dish_name']:<12} [{it['category']}] 评分 {it['score']}")
        print()

    if args.json:
        show("自定义", json.loads(args.json))
    elif args.all_presets:
        for name, sc in R.PRESETS.items():
            show(name, sc)
    else:
        name = args.preset or "工作日午餐"
        show(name, R.PRESETS[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
