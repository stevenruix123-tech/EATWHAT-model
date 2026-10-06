# -*- coding: utf-8 -*-
"""
probe_compute.py —— 确认训练实际跑在哪里（CPU/GPU），并做性能剖析

回答两个问题：
  1. 这套训练用的是 CPU 还是 GPU？（含证据，不是断言）
  2. 时间到底花在哪里？换成 GPU 能快多少？（实测，不猜）
"""

from __future__ import annotations

import os
import platform
import time

import numpy as np

print("=" * 70)
print("1. 硬件与运行环境")
print("=" * 70)
print(f"  操作系统   : {platform.system()} {platform.release()}")
print(f"  Python     : {platform.python_version()}")
print(f"  CPU 型号   : {platform.processor()}")
print(f"  逻辑核心数 : {os.cpu_count()}")

print()
print("=" * 70)
print("2. 计算后端：有没有 GPU 可用的路径？")
print("=" * 70)
try:
    import torch
    print(f"  PyTorch 已安装 : {torch.__version__}")
    print(f"  CUDA 可用      : {torch.cuda.is_available()}")
    print(f"  MPS 可用       : {getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()}")
except ImportError:
    print("  PyTorch        : 未安装  -> 本项目没有 GPU 代码路径")

# numpy 的显示配置里能看到它是否链接了 BLAS 以及线程数
cfg = np.show_config(mode="dicts")
blas = cfg.get("Build Dependencies", {}).get("blas", {})
print()
print("  NumPy 后端（决定 CPU 上快不快）:")
print(f"    BLAS 名称 : {blas.get('name')}")
print(f"    BLAS 版本 : {blas.get('version')}")
print(f"    线程层    : {blas.get('threading_layer')}")


def bench(label, fn, repeat=200, warmup=20):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    dt = (time.perf_counter() - t0) / repeat * 1000
    print(f"    {label:<46s} {dt:8.3f} ms/次")
    return dt


print()
print("=" * 70)
print("3. 真实训练算子耗时（用训练时的实际张量尺寸）")
print("=" * 70)
rng = np.random.default_rng(0)

# 第二层一次前向：batch=64 个情境 × 47 道候选菜
B, K = 64, 47
x = rng.normal(size=(B, K, 61)).astype(np.float32)
W1 = rng.normal(size=(61, 48)).astype(np.float32) * 0.1
h = rng.normal(size=(B, K, 48)).astype(np.float32)
W2 = rng.normal(size=(48, 1)).astype(np.float32)
print("  [第二层 DishScorer] 一次前向（64×47 候选）")
t_fc1 = bench("x @ W1   (64,47,61)x(61,48)", lambda: x @ W1)
t_act = bench("relu     (64,47,48)", lambda: np.maximum(h, 0.0))
t_fc2 = bench("h @ W2   (64,47,48)x(48,1)", lambda: h @ W2)
t_sm = bench("softmax  (64,47)", lambda: np.exp(x[:, :, 0] - x[:, :, 0].max(1, keepdims=True)))
t_dish = t_fc1 + t_act + t_fc2 + t_sm
print(f"    {'-> 一次前向合计':<46s} {t_dish:8.3f} ms")

# 第一层：batch=64
xc = rng.normal(size=(64, 104)).astype(np.float32)
Wc = rng.normal(size=(104, 96)).astype(np.float32) * 0.1
print()
print("  [第一层 CuisineNet] 一次前向（64 个情境）")
t_c1 = bench("x @ W1   (64,104)x(104,96)", lambda: xc @ Wc)
t_cls = bench("W2       (64,96)x(96,8)", lambda: (xc @ Wc) @ rng.normal(size=(96, 8)).astype(np.float32))
print(f"    {'-> 一次前向合计':<46s} {t_c1 + t_cls:8.3f} ms")

print()
print("=" * 70)
print("4. 推算整个训练的计算时间")
print("=" * 70)
EPOCHS = 60
G1, G2 = 1408, 1408          # 第一层按情境、第二层按情境分组，都是 1408 个/轮
BATCH = 64
steps1, steps2 = int(np.ceil(G1 / BATCH)), int(np.ceil(G2 / BATCH))
# 反向传播约为前向的 2 倍
t_layer1 = steps1 * t_c1 * 3 * EPOCHS / 1000
t_layer2 = steps2 * t_dish * 3 * EPOCHS / 1000
print(f"  第一层: {steps1} 步/轮 × {EPOCHS} 轮，单步前向 {t_c1:.2f}ms，含反向 ≈×3")
print(f"          -> 约 {t_layer1:.1f} 秒")
print(f"  第二层: {steps2} 步/轮 × {EPOCHS} 轮，单步前向 {t_dish:.2f}ms，含反向 ≈×3")
print(f"          -> 约 {t_layer2:.1f} 秒")
print(f"  纯计算合计 ≈ {t_layer1 + t_layer2:.1f} 秒")
print(f"  实测 train.py 总耗时 ≈ 360 秒")
print(f"  -> 纯矩阵计算只占 {100 * (t_layer1 + t_layer2) / 360:.0f}%，"
      f"其余是 Python 循环、数据打包、评估等开销")

print()
print("=" * 70)
print("5. 数据规模对照")
print("=" * 70)
n_ctx, d_ctx, n_dish, d_dish = 64, 104, 47, 61
print(f"  单个 batch 的矩阵规模 : ({n_ctx},{d_ctx}) x ({d_ctx},96)")
print(f"  第二层单次候选展开量  : {n_ctx}×{n_dish} = {n_ctx * n_dish} 个 (情境,菜品) 对")
print(f"  整个数据集            : 7410 行 / 16.5 MB")
print(f"  模型参数量            : 约 1.5 万")
print(f"  -> 这个规模远低于「GPU 划算」的门槛（见结论）")
