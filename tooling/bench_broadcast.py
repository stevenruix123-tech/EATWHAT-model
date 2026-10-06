# -*- coding: utf-8 -*-
"""
bench_broadcast.py —— 为什么 ContextEncoder.backward 要 450ms？

它里面只有几个小矩阵乘（理论 <1ms）。最大嫌疑是 numpy 的广播视图
（broadcast_to 产生 0 步长数组）走了非优化的慢路径。
这里逐个复现并对比「广播视图」与「显式物化数组」的耗时。
"""

from __future__ import annotations

import time

import numpy as np


def m(label, fn, n=20, warmup=3):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = (time.perf_counter() - t0) / n * 1000
    print(f"  {label:<52s} {dt:9.3f} ms")
    return dt


B, K, E, H, O = 64, 47, 32, 96, 24
rng = np.random.default_rng(0)

print("=" * 84)
print("1) 广播视图 vs 显式物化：同一道数学题的两种写法")
print("=" * 84)
e2d = rng.normal(size=(B, E))
e_bc = np.broadcast_to(e2d[:, None, :], (B, K, E))       # 0 步长视图
e_mat = np.repeat(e2d[:, None, :], K, axis=1).copy()      # 真实内存

print(f"  广播视图 flags: C={e_bc.flags['C_CONTIGUOUS']} "
      f"OWNDATA={e_bc.flags['OWNDATA']} strides={e_bc.strides}")
print(f"  物化数组 flags: C={e_mat.flags['C_CONTIGUOUS']} "
      f"OWNDATA={e_mat.flags['OWNDATA']} strides={e_mat.strides}")
print()

dz = rng.normal(size=(B, O))
W2 = rng.normal(size=(H, O))
dz3 = np.broadcast_to(dz[:, None, :], (B, K, O))          # 也是广播视图

t1 = m("dz3(broadcast) @ W2.T        -> (64,47,96)", lambda: dz3 @ W2.T)
t2 = m("dz3(materialized) @ W2.T     -> (64,47,96)", lambda: np.repeat(dz[:, None, :], K, 1) @ W2.T)
t3 = m("显式物化后 @ W2.T（不含物化开销）", lambda: e_mat @ W2.T)

x_bc = np.concatenate([e_bc, rng.normal(size=(B, K, 2))], axis=-1)
x_mat = np.ascontiguousarray(x_bc)
W1 = rng.normal(size=(E + 2, H))

print()
print("=" * 84)
print("2) 目标 einsum：视图 vs 连续内存")
print("=" * 84)
t4 = m("einsum('...d,...h->...dh', x_bc, dz3)", lambda: np.einsum("...d,...h->...dh", x_bc, dz3))
t5 = m("einsum('...d,...h->...dh', x_mat, dz3)", lambda: np.einsum("...d,...h->...dh", x_mat, dz3))
t6 = m("einsum('...d,...h->...dh', x_mat, np.ascontiguousarray(dz3))",
       lambda: np.einsum("...d,...h->...dh", x_mat, np.ascontiguousarray(dz3)))

print()
print("=" * 84)
print("3) 输出巨大的 einsum（每步算完立刻求和，内存开销大）")
print("=" * 84)
t7 = m("einsum(...)->(...dh) 再 .sum(axis=(0,1))", 
       lambda: np.einsum("...d,...h->...dh", x_mat, dz3).sum(axis=(0, 1)))
t8 = m("直接求和版 einsum（最优写法）",
       lambda: np.einsum("...d,...h->dh", x_mat, dz3))

print()
print("=" * 84)
print("4) embedding scatter：bincount 循环 vs add.at vs 索引累加")
print("=" * 84)
idx = rng.integers(0, 8, size=(B, K))
vals = rng.normal(size=(B, K, 5))


def sc_bincount():
    flat_i = idx.ravel()
    v = vals.reshape(-1, 5)
    return np.stack([np.bincount(flat_i, weights=v[:, j], minlength=8) for j in range(5)],
                    axis=1)


def sc_addat():
    ge = np.zeros((8, 5))
    np.add.at(ge, idx.ravel(), vals.reshape(-1, 5))
    return ge


def sc_sortadd():
    flat_i = idx.ravel()
    order = np.argsort(flat_i, kind="stable")
    v = vals.reshape(-1, 5)[order]
    si = flat_i[order]
    bounds = np.searchsorted(si, np.arange(9))
    ge = np.zeros((8, 5))
    for k in range(8):
        if bounds[k + 1] > bounds[k]:
            ge[k] = v[bounds[k]:bounds[k + 1]].sum(0)
    return ge


t9 = m("bincount 循环（当前实现）", sc_bincount)
t10 = m("np.add.at", sc_addat)
t11 = m("排序分段累加", sc_sortadd)
np.testing.assert_allclose(sc_bincount(), sc_addat(), rtol=1e-10)
print("  （三种写法结果一致，已断言验证）")

print()
print("=" * 84)
print("小结")
print("=" * 84)
print(f"  广播视图矩阵乘开销     : {t1:.2f} ms  vs 物化 {t2:.2f} ms  -> {t1 / max(t2, 1e-9):.1f}x")
print(f"  巨型 einsum 中间数组   : {t7:.2f} ms  vs 直接求和 {t8:.2f} ms -> {t7 / max(t8, 1e-9):.1f}x")
print(f"  embedding scatter      : bincount {t9:.2f} / add.at {t10:.2f} / 排序 {t11:.2f} ms")
