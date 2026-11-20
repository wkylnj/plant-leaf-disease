"""本地 CPU 训练吞吐实测：回答「这台机器能不能训完一个 epoch」。

只测前向+反向，不含数据加载（用随机张量），因此结果是乐观上界。
真实训练会因 JPEG 解码与数据增强更慢。

用法:
    python ml/src/bench_cpu.py
    python ml/src/bench_cpu.py --models resnet18,mobilenet_v3_large --img-size 160
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from models import build_model


def bench(model_name: str, num_classes: int, img_size: int, batch_size: int, iters: int) -> dict:
    device = torch.device("cpu")
    model, meta = build_model(model_name, num_classes, pretrained=False)
    model.to(device).train()
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 预热：让 PyTorch 完成算子选择与内存分配
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()

    n_threads = torch.get_num_threads()
    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
    secs = time.perf_counter() - t0

    imgs_per_sec = iters * batch_size / secs
    return {
        "model": model_name,
        "params_m": meta.params_m,
        "img_size": img_size,
        "batch_size": batch_size,
        "threads": n_threads,
        "imgs_per_sec": round(imgs_per_sec, 1),
        "secs_per_iter": round(secs / iters, 3),
        "min_per_epoch_39k": round(39170 / imgs_per_sec / 60, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU 训练吞吐实测")
    parser.add_argument("--models", default="resnet18,mobilenet_v3_large")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=38)
    parser.add_argument("--threads", type=int, default=0, help="0 表示用默认线程数")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    print(f"torch {torch.__version__}  默认线程数 {torch.get_num_threads()}")
    print(f"图像尺寸 {args.img_size}  batch {args.batch_size}  迭代 {args.iters}")
    print("")
    header = f"{'model':22s} {'params':>8s} {'img/s':>9s} {'min/epoch':>10s}"
    print(header)
    print("-" * len(header))

    results = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        try:
            r = bench(name, args.num_classes, args.img_size, args.batch_size, args.iters)
        except Exception as exc:  # noqa: BLE001 - 单个模型失败不影响其余
            print(f"{name:22s} 失败: {exc}")
            continue
        results.append(r)
        print(f"{name:22s} {r['params_m']:7.1f}M {r['imgs_per_sec']:9.1f} {r['min_per_epoch_39k']:10.1f}")

    if results:
        print("")
        print("注：min/epoch 按训练集 39170 张估算，且不含数据加载开销，是乐观下界。")
        fastest = min(results, key=lambda r: r["min_per_epoch_39k"])
        print(f"最快: {fastest['model']}  约 {fastest['min_per_epoch_39k']} 分钟/epoch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
