"""本地 CPU 训练速度基准：不训练，只测吞吐，用来决定"本地能跑多大"。

为什么需要单独测：
    "CPU 训练慢"是一句没有信息量的话。慢到什么程度、哪个模型慢、
    慢在数据加载还是慢在前向反向，决定了后续策略完全不同：
      - 如果慢在数据加载 → 加 num_workers、降分辨率、预先 resize 存盘
      - 如果慢在前向反向 → 只能减模型规模或减 epoch
    所以这里分开计时：纯数据加载 vs 纯计算，两者相加才能对上墙钟时间。
"""

from __future__ import annotations

import time

import torch

from data import build_dataloaders
from models import build_model


def bench(model_name: str, data_root: str, batch_size: int, num_workers: int, n_batches: int = 20):
    device = torch.device("cpu")
    torch.set_num_threads(torch.get_num_threads())

    train_loader, _, _, classes = build_dataloaders(
        data_root, img_size=224, batch_size=batch_size, num_workers=num_workers
    )

    model, meta = build_model(model_name, len(classes), pretrained=True)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # --- 纯数据加载计时：只迭代取数据，不做任何计算 ---
    t0 = time.time()
    for i, _ in enumerate(train_loader):
        if i + 1 >= n_batches:
            break
    data_secs = time.time() - t0

    # --- 纯计算计时：数据已经在内存里，只跑前向+反向 ---
    batches = []
    for i, (images, labels) in enumerate(train_loader):
        batches.append((images, labels))
        if i + 1 >= n_batches:
            break

    t0 = time.time()
    for images, labels in batches:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    compute_secs = time.time() - t0

    n = len(batches)
    per_batch = (data_secs + compute_secs) / n
    imgs_per_sec = batch_size / per_batch

    print(
        f"{model_name:22s} params={meta.params_m:6.2f}M  "
        f"data={data_secs/n:5.2f}s/b  compute={compute_secs/n:5.2f}s/b  "
        f"合计={per_batch:5.2f}s/b  {imgs_per_sec:6.1f} img/s"
    )
    return imgs_per_sec


if __name__ == "__main__":
    import sys

    data_root = sys.argv[1] if len(sys.argv) > 1 else "data/processed/plantvillage"
    print(f"threads={torch.get_num_threads()}")
    print(f"数据: {data_root}")
    print()

    for name in ["resnet18", "mobilenet_v3_large", "shufflenet_v2_x1_0", "resnet50"]:
        bench(name, data_root, batch_size=32, num_workers=4, n_batches=12)
