# -*- coding: utf-8 -*-
"""
export_npz.py —— 把 PyTorch 训练产物导出成 numpy 线上权重 model.npz（§7.3）

为什么需要：训练用 PyTorch（快），推理用 numpy（部署包 2 MB vs 150 MB+）。
两条路要打通，缺的就是这一步。导出后 `recommend.py` / `server.py`
**一行都不用改** —— 它们只认 `artifacts/model.npz`。

键名与 `train.py` 的 `np.savez` 严格一致：
    cuisine.enc.{emb0..7 | W1,b1,W2,b2}  cuisine.W  cuisine.b
    dish.enc.{emb0..7 | W1,b1,W2,b2}     dish.dish_id_emb  dish.dish_emb0..6
    dish.W1  dish.b1  dish.W2  dish.b2

两处容易错：
  1. `nn.Linear.weight` 是 (out, in)，numpy 版存的是 (in, out) —— 必须转置；
  2. `nn.Embedding.weight` 存的就是 (num, dim)，直接拷。

用法：
    python export_npz.py --model artifacts_20k/model_torch.pt \
                         --out artifacts_20k/model.npz
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch


def _enc(sd: dict, prefix: str) -> dict:
    """ContextEncoder 的 state_dict -> numpy 参数字典（不带前缀）。"""
    out = {}
    # emb0..7：Embedding 权重直接拷。
    # 键形如 "enc.embs.0.weight"，序号在第 3 段（不是第 2 段 —— 第 2 段是 "embs"，
    # 取错会让 8 个 embedding 全写进同一个键，只剩最后一个。）
    pat = re.compile(rf"^{re.escape(prefix)}embs\.(\d+)\.weight$")
    for k, v in sd.items():
        m = pat.match(k)
        if m:
            out[f"emb{m.group(1)}"] = v.detach().cpu().numpy().astype(np.float64)
    for n in ("W1", "W2"):
        out[n] = sd[f"{prefix}{n}.weight"].detach().cpu().numpy().T.astype(np.float64)
        out[f"b{n[1]}"] = sd[f"{prefix}{n}.bias"].detach().cpu().numpy().astype(np.float64)
    return out


def export(model_path: Path, out_path: Path) -> dict:
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    z = {}

    # checkpoint 里有 "cuisine" 才是两层模型；单阶段（整目录 softmax）的
    # checkpoint 只有 "dish"，导出后 npz 里也没有 cuisine.* 键 ——
    # recommend.load_models 就是靠这个判断结构的。
    c = ck.get("cuisine")
    if c is not None:
        for k, v in _enc(c, "enc.").items():
            z[f"cuisine.enc.{k}"] = v
        z["cuisine.W"] = c["head.weight"].numpy().T.astype(np.float64)
        z["cuisine.b"] = c["head.bias"].numpy().astype(np.float64)

    d = ck["dish"]
    for k, v in _enc(d, "enc.").items():
        z[f"dish.enc.{k}"] = v
    z["dish.dish_id_emb"] = d["dish_id_emb.weight"].numpy().astype(np.float64)
    for k, v in d.items():
        if k.startswith("dish_embs."):
            z[f"dish.dish_emb{k.split('.')[1]}"] = v.numpy().astype(np.float64)
    for n in ("W1", "W2"):
        z[f"dish.{n}"] = d[f"{n}.weight"].numpy().T.astype(np.float64)
        z[f"dish.b{n[1]}"] = d[f"{n}.bias"].numpy().astype(np.float64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **z)
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model_torch.pt")
    ap.add_argument("--out", required=True, help="输出的 model.npz")
    args = ap.parse_args()

    model_path, out_path = Path(args.model), Path(args.out)
    z = export(model_path, out_path)
    n_par = sum(v.size for v in z.values())
    print(f"已导出 {len(z)} 个张量 / {n_par:,} 参数 -> {out_path}")
    print(f"  大小 {out_path.stat().st_size / 1024:.0f} KB")
    print("\n务必用 eval_npz.py 与 eval_torch.py 对同一份测试集各测一次，")
    print("两个读数应当一致（差异只应来自 float32/float64 舍入）。")


if __name__ == "__main__":
    main()
