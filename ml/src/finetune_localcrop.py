"""E-12 局部裁剪训练：强迫模型只看叶片局部，检验能否摆脱对全局统计特征的依赖。

为什么这样设计
--------------
E-7 Grad-CAM 的结论是：模型判断时看的是"整片叶子的全局统计特征"（整体色调、
纹理能量、明暗分布），而不是"病斑的局部形态"。E-2 跨域实验和 E-10 域适应增强
都印证了这一点——只要成像条件（光照、色温、背景、构图）一变，全局统计量就漂移，
模型立刻崩到 28%。

前面五条被证伪的路线（TTA、颜色对齐、叶片裁剪推理、域适应增强、多模型集成）
有个共同点：都只改变输入或输出的表层统计，没改变模型内部依赖的特征类型。
局部裁剪是唯一一条能真正改变"模型看到什么"的路线：训练时只喂叶片的局部小块，
全局构图信息被破坏，模型如果还想学好，就只能依赖病斑的局部纹理与形状。

为什么在已有权重上微调，而不是从零训练 15 轮
--------------------------------------------
本机是纯 CPU（Ryzen 7 7840HS），ResNet18 在 224 分辨率、强增强下吞吐约 7 img/s，
39170 张 × 15 轮 ≈ 23 小时，不可行。Kaggle 的 local-crop kernel 排队 6 小时以上
仍未分配机器。所以改成在已收敛的权重上做短程微调（3 轮、弱增强），成本约 40 分钟。
这在方法上是"用局部裁剪做域适应"，与 E-14 PlantDoc 微调同属"动权重"的路线，
但不需要目标域标签——这正是它相对 E-14 的价值。

评估口径
--------
- 同域：PlantVillage test 10878 张，看局部裁剪是否损害原有精度
- 跨域：PlantDoc 全量 2920 张（零样本口径，无任何目标域标签参与）
- 基线：同一个 checkpoint 的原始零样本数字（PV test 0.9941 / PlantDoc 0.2818）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from eval_crossdomain import (  # noqa: E402
    PLANTDOC_TO_PV_FOLDER,
    PlantDocDataset,
    crop_of_folder,
    evaluate as evaluate_crossdomain,
)
from models import build_model  # noqa: E402


def build_train_transform(img_size: int, scale_min: float, scale_max: float):
    """训练 transform：局部裁剪 + 翻转，不加 RandAugment。

    弱增强是有意的：这一步要检验的是"看局部"本身的效果，如果同时叠加
    强颜色扰动，两个因素混在一起就分不清是谁起的作用（E-10 已经证明
    颜色扰动单独用是负收益）。
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                img_size,
                scale=(scale_min, scale_max),
                ratio=(0.75, 1.33),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize(
                int(img_size * 256 / 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def split_param_groups(model: nn.Module, model_name: str) -> tuple[list, list]:
    """把参数分成骨干与分类头两组，头用更大的学习率。"""
    head_keys = ("classifier", "fc", "head")
    backbone, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in head_keys):
            head.append(p)
        else:
            backbone.append(p)
    return backbone, head


@torch.no_grad()
def evaluate_same_domain(model, loader, device) -> dict:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += int((pred == y).sum())
        total += int(y.numel())
    return {"n": total, "top1_acc": correct / max(1, total)}


def evaluate_cross(model, loader, device, pv_folders: list[str]) -> dict:
    """跨域评估：在 eval_crossdomain.evaluate 之上补作物级准确率。

    evaluate() 只返回病害级指标，作物级要拿预测类别与真值类别各自的作物名比较。
    复用 crop_of_folder 保证与 crossdomain.md 里的口径完全一致。
    """
    res = evaluate_crossdomain(model, loader, device, len(pv_folders))
    pv_crop = [crop_of_folder(f) for f in pv_folders]
    pred = res["_pred"]
    true = res["_true"]
    conf = res["_conf"]
    crop_ok = sum(1 for p, t in zip(pred, true) if pv_crop[p] == pv_crop[t])
    return {
        "n": res["n"],
        "top1_acc": res["top1_acc"],
        "top3_acc": res["top3_acc"],
        "crop_acc": crop_ok / max(1, res["n"]),
        "mean_conf": float(sum(conf) / len(conf)) if conf else 0.0,
        "elapsed_s": res["elapsed_s"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="已有 checkpoint 的 run 目录")
    ap.add_argument("--data-root", default="data/processed/plantvillage")
    ap.add_argument("--plantdoc-root", default="data/raw/plantdoc")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--scale-min", type=float, default=0.15)
    ap.add_argument("--scale-max", type=float, default=0.5)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit-batches", type=int, default=0, help="冒烟用，0 表示不限")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(args.run_dir)
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    classes = list(ck["classes"])  # 38 类文件夹名，顺序与模型输出严格对应
    model_name = ck.get("args", {}).get("model") or run_dir.name.rsplit("_", 2)[0]
    model, meta = build_model(model_name, len(classes), pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(device)

    print(f"[info] 模型 {model_name}（{meta.params_m:.3f}M 参数），ckpt epoch={ck.get('epoch')}")
    print(f"[info] 类别数 {len(classes)}，设备 {device}")
    print(
        f"[info] 局部裁剪 scale=({args.scale_min}, {args.scale_max})，"
        f"{args.epochs} 轮，lr 骨干 {args.lr_backbone} / 头 {args.lr_head}"
    )

    root = Path(args.data_root)
    train_ds = ImageFolder(root / "train", transform=build_train_transform(args.img_size, args.scale_min, args.scale_max))
    val_ds = ImageFolder(root / "val", transform=build_eval_transform(args.img_size))
    test_ds = ImageFolder(root / "test", transform=build_eval_transform(args.img_size))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 微调前基线（同一个 checkpoint，保证对比严格）
    t0 = time.time()
    base_same = evaluate_same_domain(model, test_loader, device)
    print(f"[before] PV test top1 = {base_same['top1_acc']:.4f}（{time.time() - t0:.1f}s）")

    pd_root = Path(args.plantdoc_root)
    if pd_root.is_dir():
        pd_ds = PlantDocDataset(pd_root, "all", classes, transform=build_eval_transform(args.img_size))
        pd_loader = DataLoader(pd_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        base_cross = evaluate_cross(model, pd_loader, device, classes)
        print(
            f"[before] PlantDoc top1 = {base_cross['top1_acc']:.4f} "
            f"crop = {base_cross['crop_acc']:.4f}（n={base_cross['n']}）"
        )
    else:
        pd_loader = None
        base_cross = None
        print(f"[warn] PlantDoc 目录不存在：{pd_root}，跳过跨域评估")

    backbone, head = split_param_groups(model, model_name)
    print(f"[info] 优化器：骨干 {len(backbone)} 个张量 lr={args.lr_backbone}，头 {len(head)} 个张量 lr={args.lr_head}")
    opt = torch.optim.AdamW(
        [
            {"params": backbone, "lr": args.lr_backbone},
            {"params": head, "lr": args.lr_head},
        ],
        weight_decay=0.01,
    )
    total_steps = args.epochs * (args.limit_batches or len(train_loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_correct = 0
        ep_n = 0
        ep_t0 = time.time()
        for bi, (x, y) in enumerate(train_loader, 1):
            if args.limit_batches and bi > args.limit_batches:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            sched.step()
            ep_loss += float(loss.item()) * y.numel()
            ep_correct += int((out.argmax(1) == y).sum())
            ep_n += int(y.numel())
            if bi % 100 == 0:
                print(
                    f"[ep{ep} b{bi}] loss={ep_loss / ep_n:.4f} acc={ep_correct / ep_n:.4f} "
                    f"({time.time() - ep_t0:.0f}s)"
                )
        train_loss = ep_loss / max(1, ep_n)
        train_acc = ep_correct / max(1, ep_n)
        same = evaluate_same_domain(model, val_loader, device)
        rec = {
            "epoch": ep,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_top1": round(same["top1_acc"], 6),
            "secs": round(time.time() - ep_t0, 1),
        }
        history.append(rec)
        print(
            f"[ep{ep}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_top1={same['top1_acc']:.4f} {rec['secs']:.0f}s"
        )

    # 微调后评估
    after_same = evaluate_same_domain(model, test_loader, device)
    print(f"[after] PV test top1 = {after_same['top1_acc']:.4f}")
    after_cross = None
    if pd_loader is not None:
        after_cross = evaluate_cross(model, pd_loader, device, classes)
        print(
            f"[after] PlantDoc top1 = {after_cross['top1_acc']:.4f} "
            f"top3 = {after_cross['top3_acc']:.4f} crop = {after_cross['crop_acc']:.4f} "
            f"conf = {after_cross['mean_conf']:.4f}"
        )

    result = {
        "model": model_name,
        "params_m": round(meta.params_m, 3),
        "run_dir": str(run_dir),
        "ckpt_epoch": ck.get("epoch"),
        "config": {
            "img_size": args.img_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr_backbone": args.lr_backbone,
            "lr_head": args.lr_head,
            "local_crop_scale": [args.scale_min, args.scale_max],
            "strong_aug": False,
            "label_smoothing": 0.1,
            "seed": args.seed,
        },
        "before": {"same_domain": base_same, "crossdomain": base_cross},
        "after": {"same_domain": after_same, "crossdomain": after_cross},
        "delta": {
            "same_domain_pp": round((after_same["top1_acc"] - base_same["top1_acc"]) * 100, 2),
            "crossdomain_pp": (
                round((after_cross["top1_acc"] - base_cross["top1_acc"]) * 100, 2)
                if after_cross and base_cross
                else None
            ),
            "crop_pp": (
                round((after_cross["crop_acc"] - base_cross["crop_acc"]) * 100, 2)
                if after_cross and base_cross
                else None
            ),
        },
        "history": history,
    }
    out_path = Path(args.out) if args.out else run_dir / "localcrop_finetune.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] 写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
