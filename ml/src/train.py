"""训练脚本：单模型训练 + 每 epoch 验证 + 最优权重保存 + 结果落盘。

设计取舍说明：

1. 为什么不用现成的 trainer（如 PyTorch Lightning）
   这个项目要在 Colab 上跑、要给队友看、要在报告里解释每个超参数。
   一个 300 行的自写循环比一个框架更容易讲清楚"我们到底做了什么"，
   而且 Colab 上少一个依赖就少一个版本冲突的来源。

2. 优化策略：warmup + cosine 退火
   预训练模型直接上大学习率会把预训练学到的特征冲掉，所以前几个 epoch
   线性 warmup 到目标学习率。之后 cosine 退火，末期小学习率有助于收敛到更平的极小值。

3. 分层学习率
   骨干网络用 lr * 0.1，新加的分类头用 lr。
   分类头是随机初始化的，需要快速学习；骨干是预训练的，只需要微调。
   用同一个学习率会导致骨干被破坏而头部还没学好。

4. 混合精度
   只在 CUDA 上开启。CPU 上 bfloat16 支持不完整，反而更慢。

5. 断点续训
   Colab 免费版会断连，所以每个 epoch 都存 last.pt，支持 --resume。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from data import build_dataloaders, save_class_map
from models import build_model


def set_seed(seed: int) -> None:
    """固定所有随机源。不固定的话，两次训练的结果差 0.5% 都说不清是模型差异还是噪声。"""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn 的 benchmark 会让每次选的卷积算法不同，结果不可复现，所以关掉
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_optimizer(model: nn.Module, head_names: tuple[str, ...], lr: float, wd: float, backbone_scale: float = 0.1):
    """把参数分成"骨干"和"分类头"两组，给不同学习率。

    head_names 是分类头的参数名前缀，比如 ResNet 是 ("fc.",)，ViT 是 ("heads.",)。
    用前缀匹配而不是按模块对象分组，是因为模型被 torchvision 封装过，
    直接拿 model.fc 在部分模型上会拿不到。
    """
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.startswith(h) or f".{h}" in f".{name}" for h in head_names):
            head_params.append(param)
        else:
            backbone_params.append(param)

    groups = [
        {"params": backbone_params, "lr": lr * backbone_scale},
        {"params": head_params, "lr": lr},
    ]
    return AdamW(groups, weight_decay=wd)


def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """线性 warmup + cosine 退火。返回的是按 epoch 调用的 LambdaLR。"""

    def fn(epoch: int) -> float:
        if epoch < warmup_epochs:
            # +1 避免 epoch=0 时学习率为 0，那样第一个 epoch 等于白跑
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        # 余弦从 1 降到 0.01，不完全降到 0，末期保留一点学习能力
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, fn)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int):
    """在给定 loader 上评估，返回 loss / acc / macro-F1 和全部预测。

    返回 predictions 是为了后面画混淆矩阵和算 ECE，不用再跑一遍。
    """
    model.eval()
    total_loss, n = 0.0, 0
    preds_all, labels_all, probs_all = [], [], []

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)

        probs = torch.softmax(logits.float(), dim=1)
        preds_all.append(probs.argmax(1).cpu())
        labels_all.append(labels.cpu())
        probs_all.append(probs.cpu())

    preds = torch.cat(preds_all).numpy()
    labels = torch.cat(labels_all).numpy()
    probs = torch.cat(probs_all).numpy()

    acc = float((preds == labels).mean())
    # macro-F1：每类等权。类别不均衡时它比 accuracy 更能反映真实水平
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    return {
        "loss": total_loss / max(1, n),
        "acc": acc,
        "macro_f1": macro_f1,
        "preds": preds,
        "labels": labels,
        "probs": probs,
    }


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, grad_clip: float = 1.0):
    """训练一个 epoch，返回平均 loss 和训练集上的准确率。"""
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # 混合精度只在 CUDA 上开。CPU 上 autocast 会退化甚至更慢
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            # 梯度裁剪要在 unscale 之后做，否则裁的是放大过的梯度，阈值没有意义
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += labels.size(0)

    return total_loss / max(1, n), correct / max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--data-root", default="data/processed/plantvillage")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true", help="从零训练，用于消融")
    ap.add_argument("--weak-aug", action="store_true", help="弱增强，用于消融")
    ap.add_argument("--balanced-sampling", action="store_true", help="开启类别平衡采样")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", default=None, help="从 last.pt 续训")
    ap.add_argument("--limit-batches", type=int, default=None, help="调试用，每 epoch 只跑 N 个 batch")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = args.run_name or f"{args.model}_sz{args.img_size}_seed{args.seed}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 把本次运行的超参数原样存下来。报告里的数字必须能追溯到某次运行
    (run_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[{run_name}] device={device}")

    train_loader, val_loader, test_loader, classes = build_dataloaders(
        args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        strong_aug=not args.weak_aug,
        weighted_sampling=args.balanced_sampling,
    )
    num_classes = len(classes)
    save_class_map(classes, run_dir / "classes.json")
    print(f"类别数={num_classes}  train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  test={len(test_loader.dataset)}")

    model, meta = build_model(args.model, num_classes, pretrained=not args.no_pretrained)
    model.to(device)
    print(f"模型={meta.name}  家族={meta.family}  参数量={meta.params_m:.2f}M")

    # 标签平滑：防止模型对训练集过度自信，对校准误差（ECE）有直接好处
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # 分类头参数名前缀，用于分层学习率
    head_prefixes = {
        "resnet18": ("fc.",),
        "resnet50": ("fc.",),
        "mobilenet_v3_large": ("classifier.",),
        "mobilenet_v3_small": ("classifier.",),
        "shufflenet_v2_x1_0": ("fc.",),
        "efficientnet_b0": ("classifier.",),
        "convnext_tiny": ("classifier.",),
        "vit_b_16": ("heads.",),
        "swin_t": ("head.",),
    }[args.model.lower()]

    optimizer = build_optimizer(model, head_prefixes, args.lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    start_epoch, best_acc = 0, 0.0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"从 {args.resume} 续训，起始 epoch={start_epoch}，历史最优 acc={best_acc:.4f}")

    history_path = run_dir / "history.csv"
    if not history_path.exists():
        with history_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "lr", "secs"]
            )

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # 调试模式下截断 batch 数，让本地 CPU 也能在几分钟内跑完一轮
        if args.limit_batches:
            import itertools

            train_iter = itertools.islice(train_loader, args.limit_batches)
            val_iter = itertools.islice(val_loader, max(1, args.limit_batches // 2))
        else:
            train_iter, val_iter = train_loader, val_loader

        train_loss, train_acc = train_one_epoch(
            model, train_iter, criterion, optimizer, device, scaler
        )

        val = evaluate(model, val_iter, criterion, device, num_classes)
        scheduler.step()

        secs = time.time() - t0
        lr_now = optimizer.param_groups[-1]["lr"]

        with history_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.6f}", f"{train_acc:.6f}", f"{val['loss']:.6f}",
                 f"{val['acc']:.6f}", f"{val['macro_f1']:.6f}", f"{lr_now:.8f}", f"{secs:.1f}"]
            )

        print(
            f"epoch {epoch+1}/{args.epochs}  "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f}  "
            f"val_loss={val['loss']:.4f} acc={val['acc']:.4f} f1={val['macro_f1']:.4f}  "
            f"lr={lr_now:.2e}  {secs:.0f}s"
        )

        # 保存最优。用 val acc 选模型而不是 train loss，
        # 因为我们要的是泛化能力，不是拟合能力
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            torch.save(
                {"model": model.state_dict(), "meta": meta.__dict__, "classes": classes,
                 "epoch": epoch, "val_acc": val["acc"], "args": vars(args)},
                run_dir / "best.pt",
            )
            print(f"  -> 新最优 val_acc={best_acc:.4f}，已保存 best.pt")

        # 每个 epoch 都存 last.pt，Colab 断连后可续训
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "epoch": epoch, "best_acc": best_acc},
            run_dir / "last.pt",
        )

    # 训练结束，用最优权重在测试集上评一次，结果直接落盘
    print("\n训练完成，用 best.pt 在测试集上评估……")
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])

    test = evaluate(model, test_loader, criterion, device, num_classes)
    summary = {
        "model": args.model,
        "family": meta.family,
        "params_m": round(meta.params_m, 3),
        "img_size": args.img_size,
        "seed": args.seed,
        "epochs": args.epochs,
        "pretrained": not args.no_pretrained,
        "strong_aug": not args.weak_aug,
        "balanced_sampling": args.balanced_sampling,
        "best_val_acc": round(float(ckpt["val_acc"]), 6),
        "test_loss": round(float(test["loss"]), 6),
        "test_acc": round(float(test["acc"]), 6),
        "test_macro_f1": round(float(test["macro_f1"]), 6),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 混淆矩阵和预测概率存下来，供后续 ECE / Grad-CAM 分析用，不用重跑推理
    np.savez_compressed(
        run_dir / "test_predictions.npz",
        preds=test["preds"], labels=test["labels"], probs=test["probs"], classes=np.array(classes),
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
