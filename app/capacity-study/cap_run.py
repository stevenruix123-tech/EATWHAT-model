# -*- coding: utf-8 -*-
"""
cap_run.py —— 容量实验的单次运行器（在 capacity-study 目录内运行，完全隔离）

用法：
    python cap_run.py --id-dim 8 --hidden 48 --epochs 60 --tag id8_h48

做三件事：
  1. 用 train.py 的同一份代码/超参训练第二层，但把 id_dim / hidden 参数化
  2. 逐 epoch 记录验证集 top-1 / top-3（对照 gold），而不是只看排序损失
  3. 记录「每轮耗时」——把 forward / backward / loss 分开计时，
     这样能回答「加容量是变慢还是变快（按达到同等质量算）」

输出：results/<tag>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ART = HERE / "artifacts"

import train as T  # noqa: E402
from features import N_DISH, N_CAT, emb_dim  # noqa: E402


def build_samples(cache: Path) -> pd.DataFrame:
    """读取本目录已生成的数据集。

    刻意不改用 build_dataset.build()：那样会把父目录的 meal-decision/build_dataset.py
    导入进来（名字同名），在错误的 artifacts 目录里找文件。数据集固定生成一次，
    所有配置跑在同一份数据上才可比。
    """
    if not cache.exists():
        raise SystemExit(
            f"缺少 {cache}\n"
            f"请先在本目录执行：python build_dataset.py"
        )
    return pd.read_csv(cache)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id-dim", type=int, required=True)
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    outdir = HERE / "results"
    outdir.mkdir(exist_ok=True)

    samples = build_samples(ART / "samples.csv")
    samples["taboo"] = samples["taboo"].fillna("").astype(str)

    # ---- 参数化 DishScorer：只改 id_dim / hidden，其余逻辑与 train.py 完全一致 ----
    OrigInit = T.DishScorer.__init__

    def patched(self, rng, d_ctx=24, hidden=48, id_dim=8):
        OrigInit(self, rng, d_ctx=d_ctx, hidden=args.hidden, id_dim=args.id_dim)

    T.DishScorer.__init__ = patched
    T.EPOCHS = args.epochs
    T.SEED = args.seed
    T._UTIL_CACHE.clear()
    if hasattr(T, "_UTIL_MATRIX"):
        T._UTIL_MATRIX = None

    # ---- 计时：包住 forward / backward，逐 epoch 累计 ----
    timing = {"fwd_s": 0.0, "bwd_s": 0.0, "other_s": 0.0, "n_batch": 0}
    _fwd, _bwd = T.DishScorer.forward, T.DishScorer.backward

    def timed_fwd(self, *a, **k):
        t0 = time.perf_counter()
        r = _fwd(self, *a, **k)
        timing["fwd_s"] += time.perf_counter() - t0
        return r

    def timed_bwd(self, *a, **k):
        t0 = time.perf_counter()
        r = _bwd(self, *a, **k)
        timing["bwd_s"] += time.perf_counter() - t0
        return r

    T.DishScorer.forward = timed_fwd
    T.DishScorer.backward = timed_bwd

    # ---- 逐 epoch 记录 top-1 / top-3 ----
    per_epoch: list[dict] = []
    OrigEvalTop1, OrigEvalTopk = T._eval_top1, T._eval_topk

    state = {"va": None}
    tr = T.attach_gold(samples[samples.split == "train"].reset_index(drop=True))
    va = T.attach_gold(samples[samples.split == "val"].reset_index(drop=True))
    state["va"] = T._pack(va)

    def spy_top1(model, P, chunk=256):
        v = OrigEvalTop1(model, P, chunk)
        per_epoch[-1]["val_top1"] = float(v)
        return v

    def spy_topk(model, P, k=3, chunk=256):
        v = OrigEvalTopk(model, P, k, chunk)
        per_epoch[-1]["val_top3"] = float(v)
        return v

    T._eval_top1 = spy_top1
    T._eval_topk = spy_topk

    # 每轮开始时插一条记录
    OrigTrainDish = T.train_dish

    t_start = time.perf_counter()
    # 手工复刻 train_dish 的外层循环以插入 per-epoch 计时
    def run():
        import numpy as _np
        trp = T._pack(tr)
        V = T._pack(va)
        rng = _np.random.default_rng(T.SEED + 1)
        model = T.DishScorer(rng)
        params = model.params()
        opt = T.Adam(params, T.LR)
        all_ids = _np.arange(N_DISH)[None, :]
        groups = {}
        for r, sid in enumerate(trp["sid"]):
            groups.setdefault(sid, []).append(r)
        group_list = list(groups.values())
        n_groups = len(group_list)
        best = (1e9, None, -1)

        for ep in range(T.EPOCHS):
            per_epoch.append({"epoch": ep})
            e0 = time.perf_counter()
            timing["fwd_s"] = timing["bwd_s"] = timing["other_s"] = 0.0
            timing["n_batch"] = 0
            perm = rng.permutation(n_groups)
            tr_rank = tr_reg = 0.0
            n_rank_rows = 0
            for i in range(0, n_groups, T.BATCH):
                batch = [r for gi in perm[i:i + T.BATCH] for r in group_list[gi]]
                b = _np.array(batch)
                B_ = len(b)
                timing["n_batch"] += 1
                ck, nk = trp["cat"][b], trp["num"][b]
                ids, dc, dn = T.tile_dishes(all_ids, B_)
                s = model.forward(ck, nk, ids, dc, dn)
                is_chosen = trp["chosen"][b]
                if is_chosen.any():
                    group = (T.DISH_CAT_IDX[:, 0][None, :] == trp["cat_id"][b][:, None])
                    s_m = _np.where(group, s, -1e9)
                    gt = T.group_targets(trp, b)
                    gt_soft = T.softmax_rows(gt)
                    loss_rank, dp = T.softmax_ce_soft(s_m[is_chosen], gt_soft[is_chosen])
                    ds = _np.zeros_like(s)
                    ds[is_chosen] = dp
                    tr_rank += loss_rank * int(is_chosen.sum())
                    n_rank_rows += int(is_chosen.sum())
                else:
                    ds = _np.zeros_like(s)
                obs = _np.zeros_like(s)
                obs[_np.arange(B_), trp["dish"][b]] = 1.0
                pred_obs = (s * obs).sum(1)
                loss_reg = float(T.huber(pred_obs, trp["y"][b], T.HUBER_DELTA).mean())
                d_reg = T.d_huber(pred_obs, trp["y"][b], T.HUBER_DELTA)
                ds[_np.arange(B_), trp["dish"][b]] += T.LAM_REG * d_reg / B_
                tr_reg += loss_reg * B_
                model.backward(ds)
                g = dict(model.grads)
                for k in list(g):
                    if k.endswith(("W1", "W2")) and not k.startswith("enc."):
                        g[k] = g[k] + T.REG * params[k]
                opt.step(params, g)
            t_train = time.perf_counter() - e0
            per_epoch[-1]["train_s"] = round(t_train, 3)
            per_epoch[-1]["fwd_s"] = round(timing["fwd_s"], 3)
            per_epoch[-1]["bwd_s"] = round(timing["bwd_s"], 3)
            per_epoch[-1]["n_batch"] = timing["n_batch"]
            per_epoch[-1]["train_rank"] = round(tr_rank / max(n_rank_rows, 1), 5)
            per_epoch[-1]["train_reg"] = round(tr_reg / max(len(trp), 1), 5)
            # 验证（这两次调用会被 spy 填进 val_top1 / val_top3）
            vl = T._eval_rank(model, V)
            T._eval_top1(model, V)
            T._eval_topk(model, V, 3)
            per_epoch[-1]["val_rank"] = round(vl, 5)
            per_epoch[-1]["epoch_total_s"] = round(time.perf_counter() - e0, 3)
            if vl < best[0]:
                best = (vl, {k: v.copy() for k, v in params.items()}, ep)
            if ep % 10 == 0 or ep == T.EPOCHS - 1:
                print(f"    ep{ep:3d} rank={per_epoch[-1]['train_rank']:.4f} "
                      f"val_rank={vl:.4f} top1={per_epoch[-1]['val_top1']:.2%} "
                      f"top3={per_epoch[-1]['val_top3']:.2%} "
                      f"train={t_train:.2f}s", flush=True)
        for k in params:
            params[k][...] = best[1][k]
        return model, best

    model, best = run()
    wall = time.perf_counter() - t_start

    n_params = sum(v.size for v in model.params().values())
    epochs_done = len(per_epoch)
    tot_epoch_s = sum(e["epoch_total_s"] for e in per_epoch)

    # 最佳轮次（按验证排序损失，与 train.py 的选轮口径一致）的命中率
    best_ep = best[2]
    rec = next((e for e in per_epoch if e["epoch"] == best_ep), per_epoch[-1])

    res = {
        "tag": args.tag,
        "id_dim": args.id_dim,
        "hidden": args.hidden,
        "seed": args.seed,
        "epochs": epochs_done,
        "n_params": int(n_params),
        "dish_id_emb_dim": int(model.dish_id_emb.shape[1]),
        "wall_s": round(wall, 1),
        "epoch_total_s_mean": round(tot_epoch_s / max(epochs_done, 1), 3),
        "train_only_s_mean": round(np.mean([e["train_s"] for e in per_epoch]), 3),
        "fwd_s_mean": round(np.mean([e["fwd_s"] for e in per_epoch]), 3),
        "bwd_s_mean": round(np.mean([e["bwd_s"] for e in per_epoch]), 3),
        "best_epoch": best_ep,
        "best_val_rank": round(best[0], 5),
        "best_epoch_val_top1": rec.get("val_top1"),
        "best_epoch_val_top3": rec.get("val_top3"),
        "final_val_top1": per_epoch[-1].get("val_top1"),
        "final_val_top3": per_epoch[-1].get("val_top3"),
        "peak_val_top1": max(e.get("val_top1", 0) for e in per_epoch),
        "peak_val_top3": max(e.get("val_top3", 0) for e in per_epoch),
        "per_epoch": per_epoch,
    }
    (outdir / f"{args.tag}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {args.tag}: 参数 {n_params:,}  最佳轮 {best_ep}  "
          f"val_top1={rec.get('val_top1')}  val_top3={rec.get('val_top3')}  "
          f"每轮 {res['epoch_total_s_mean']}s  总计 {wall:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
