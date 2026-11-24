#!/usr/bin/env python3
"""Colab 训练脚本（自包含，不依赖本仓库其它模块）。

为什么单独写一个而不复用 ml/src/train.py：
`colab exec -f` 只把单个文件送到远端 kernel，本地 import 的模块不会跟着过去。
所以这里把数据下载、转换、transform、模型构建、训练循环全部内联。

用法（在已存在的 Colab 会话里）：
    colab exec -s probe -f ml/src/train_colab.py
    # 或带参数
    echo 'import sys; sys.argv=["x","--model","resnet18","--epochs","15"]' | colab exec -s probe -f ...

更常用的方式是把参数写在文件末尾的 DEFAULTS 里。

产出（runs/<run_name>/）：
    history.csv            每轮 loss/acc/f1/lr/耗时
    summary.json           最终指标
    best.pt                最佳权重
    curves.png             损失与精度曲线
    confusion_matrix.png   38x38 混淆矩阵
    per_class_f1.csv       每类精确率/召回率/F1
    model.onnx             导出模型（opset 17）
    classes.json           类别清单
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- 默认配置
DEFAULTS = {
    "model": "resnet18",
    "epochs": 15,
    "img_size": 224,
    "batch_size": 64,
    "lr": 3e-4,
    "weight_decay": 0.05,
    "label_smoothing": 0.1,
    "warmup_epochs": 2,
    "num_workers": 2,
    "seed": 42,
    "strong_aug": True,
    "da_aug": False,         # 域适应增强：模拟田间拍摄的光照/色彩/背景差异
    "local_crop": False,     # 局部裁剪训练：只截叶片一小块，逼模型看病斑而非全局统计量
    "local_scale_min": 0.15,
    "local_scale_max": 0.5,
    "data_root": "/content/data/plantvillage",
    "out_dir": "/content/runs",
    "run_name": "",          # 空则自动用 模型名_时间戳
    "limit_batches": 0,      # >0 时只跑这么多 batch（冒烟用）
    "bench": 0,              # >0 时只测速这么多 batch 然后退出
    "export_onnx": True,
    "no_pretrained": False,
    "prepare_only": False,
}

# 环境变量覆盖：colab exec 没法传命令行参数，用 COLAB_<KEY> 环境变量代替。
# 例：colab exec -s probe -f ml/src/train_colab.py --env COLAB_BENCH=20
for _k in list(DEFAULTS):
    _env = os.environ.get("COLAB_" + _k.upper())
    if _env is None:
        continue
    _v = DEFAULTS[_k]
    if isinstance(_v, bool):
        DEFAULTS[_k] = _env.strip().lower() in ("1", "true", "yes", "y", "on")
    elif isinstance(_v, int):
        DEFAULTS[_k] = int(_env)
    elif isinstance(_v, float):
        DEFAULTS[_k] = float(_env)
    else:
        DEFAULTS[_k] = _env

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- 工具函数
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sanitize(name: str) -> str:
    """把 PlantVillage 的原始类名变成安全的目录名。"""
    out = name.replace("___", "_").replace("__", "_")
    for ch in " ()[],'\"-":
        out = out.replace(ch, "_")
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def setup_matplotlib():
    """配置 matplotlib：无界面后端 + 尽量找中文字体。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # Colab 上常见的中文字体路径
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    found = None
    for p in candidates:
        if Path(p).exists():
            found = p
            break
    if found:
        font_manager.fontManager.addfont(found)
        name = font_manager.FontProperties(fname=found).get_name()
        plt.rcParams["font.family"] = [name]
        plt.rcParams["axes.unicode_minus"] = False
        log(f"中文字体: {name} ({found})")
    else:
        log("未找到中文字体，图内文字用英文")
    return plt


# ---------------------------------------------------------------- 数据准备
def download_and_prepare(data_root: str, limit_per_class: int = 0) -> Path:
    """下载 HF 上的 PlantVillage 并转成 ImageFolder 结构。已有则跳过。"""
    root = Path(data_root)
    marker = root / ".prepared"
    if marker.exists():
        log(f"数据已就绪: {root}")
        return root

    from datasets import load_dataset

    log("下载数据集 GVJahnavi/PlantVillage_dataset（约 813MB）…")
    ds = load_dataset("GVJahnavi/PlantVillage_dataset")

    # 解析类别名。HF 元数据有几种可能结构，逐个试。
    # 注意 ClassLabel._int2str 是 list 不是 callable，别调它。
    label_names = None
    for split in ("train", "test"):
        feat = ds[split].features.get("label")
        if feat is None:
            continue
        for attr in ("names", "_int2str"):
            v = getattr(feat, attr, None)
            if v is None:
                continue
            if isinstance(v, dict):
                label_names = [v[i] for i in sorted(v)]
            elif isinstance(v, (list, tuple)) and v:
                label_names = list(v)
            if label_names:
                break
        if label_names:
            break
        try:
            cl = feat["class_label"]["names"]
            label_names = [cl[i] for i in sorted(cl)] if isinstance(cl, dict) else list(cl)
            break
        except Exception:
            continue

    if not label_names:
        # 最后兜底：直接从 train split 里收集 label 列的唯一值
        uniq = sorted({int(x) for x in ds["train"]["label"]})
        label_names = [f"class_{i}" for i in uniq]
        log(f"警告：未能解析原始类名，用占位名 class_*（共 {len(uniq)} 类）")

    num_classes = len(label_names)
    log(f"共 {num_classes} 类")

    # 目录名加两位索引前缀，保证 ImageFolder 的字典序 == 标签序
    class_dirs = {i: f"{i:02d}_{sanitize(n)}" for i, n in enumerate(label_names)}

    import random

    from PIL import Image

    def dump(split_name: str, out_split: str, val_ratio: float = 0.0, seed: int = 42):
        split = ds[split_name]
        idxs = list(range(len(split)))
        if val_ratio > 0:
            rng = random.Random(seed)
            rng.shuffle(idxs)
            n_val = int(len(idxs) * val_ratio)
            val_idx, tr_idx = set(idxs[:n_val]), idxs[n_val:]
        else:
            val_idx, tr_idx = set(), idxs

        counts = {out_split: 0, "val": 0}
        for i in idxs:
            row = split[i]
            label = int(row["label"])
            # datasets 版本不同，image 可能是 {"bytes":...} 也可能是已解码的 PIL Image
            raw = row["image"]
            if isinstance(raw, dict):
                import io

                img = Image.open(io.BytesIO(raw["bytes"]))
            elif isinstance(raw, (bytes, bytearray)):
                import io

                img = Image.open(io.BytesIO(raw))
            else:
                img = raw
            img = img.convert("RGB")
            which = "val" if i in val_idx else out_split
            d = root / which / class_dirs[label]
            d.mkdir(parents=True, exist_ok=True)
            img.save(d / f"{i:06d}.jpg", quality=95)
            counts[which] += 1
        return counts

    log("转换 train（含 10% 验证集）…")
    c1 = dump("train", "train", val_ratio=0.1)
    log(f"  train={c1['train']} val={c1['val']}")
    log("转换 test…")
    c2 = dump("test", "test")
    log(f"  test={c2['test']}")

    # 保存类别清单
    classes = [class_dirs[i] for i in range(num_classes)]
    (root / "classes.json").write_text(
        json.dumps(
            [{"index": i, "folder": class_dirs[i], "name": label_names[i]} for i in range(num_classes)],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    marker.write_text("ok", encoding="utf-8")
    log(f"数据就绪: {root}")
    return root


# ---------------------------------------------------------------- transform
def build_transforms(
    img_size: int,
    train: bool,
    strong: bool = True,
    da: bool = False,
    local_crop: bool = False,
    local_scale_min: float = 0.15,
    local_scale_max: float = 0.5,
):
    from torchvision import transforms as T

    if not train:
        # 验证/测试口径绝不能随 --local-crop 变化，否则评测数字没法和历史比
        return T.Compose(
            [
                T.Resize(int(img_size * 256 / 224)),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    # ---- 局部裁剪训练（E-da_aug 失败后的下一步假设）----
    # da_aug 证明：扰动像素统计分布不改变模型依赖的特征类型。
    # 这里改用很小的裁剪尺度（默认只保留 15%~50% 面积），让训练样本里经常
    # 只剩病斑/叶脉的一小块，模型无法再靠"整片叶子的全局颜色/纹理"蒙对。
    if local_crop:
        first = T.RandomResizedCrop(
            img_size, scale=(local_scale_min, local_scale_max), ratio=(0.75, 1.33)
        )
    else:
        first = T.RandomResizedCrop(img_size, scale=(0.6, 1.0) if strong else (0.8, 1.0))

    ops = [
        first,
        T.RandomHorizontalFlip(),
    ]
    if strong:
        ops += [
            T.RandomVerticalFlip(p=0.2),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ColorJitter(0.3, 0.3, 0.3, 0.05),
        ]

    # ---- 域适应增强（E2 跨域 28% 的直接对策）----
    # 跨域崩溃的根因（见 ml/artifacts/gradcam.md 与 crossdomain.md）：
    # 模型依赖的是整片叶子的全局颜色/纹理统计量，而 PlantDoc 真实照片的
    # 光照、色温、对比度、背景与 PlantVillage 实验室摆拍差异巨大。
    # 这里刻意把「成像条件」当成干扰因素去扰动，强迫模型只依赖与成像无关的
    # 形态特征（病斑形状、叶脉纹理）。四项针对性扰动：
    #   1) 强色彩抖动：模拟不同设备白平衡、阴天/晴天色温
    #   2) 随机灰度：模拟单色/低饱和拍摄，切断"靠颜色分类"的捷径
    #   3) 高斯模糊：模拟对焦不准、手机近距离拍摄
    #   4) 亮度/对比度大范围：模拟逆光、阴影、曝光不足
    if da:
        ops += [
            T.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.15),
            T.RandomGrayscale(p=0.15),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
            T.RandomAutocontrast(p=0.2),
            T.RandomEqualize(p=0.2),
        ]

    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if strong or da:
        ops.append(T.RandomErasing(p=0.25, scale=(0.02, 0.15)))
    return T.Compose(ops)


# ---------------------------------------------------------------- 模型
def build_model(name: str, num_classes: int, pretrained: bool = True):
    """返回 (model, params_m)。分类头替换位置必须写死，不能用遍历找最后一个 Linear。"""
    import torch.nn as nn
    from torchvision import models as M

    if name == "resnet18":
        w = M.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = M.resnet18(weights=w)
        m.fc = nn.Linear(512, num_classes)
        head = ["fc"]
    elif name == "resnet50":
        w = M.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = M.resnet50(weights=w)
        m.fc = nn.Linear(2048, num_classes)
        head = ["fc"]
    elif name == "mobilenet_v3_large":
        w = M.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        m = M.mobilenet_v3_large(weights=w)
        m.classifier[3] = nn.Linear(1280, num_classes)  # 4 层，替换最后一层
        head = ["classifier.3"]
    elif name == "shufflenet_v2_x1_0":
        w = M.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        m = M.shufflenet_v2_x1_0(weights=w)
        m.fc = nn.Linear(1024, num_classes)
        head = ["fc"]
    elif name == "efficientnet_b0":
        w = M.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = M.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(1280, num_classes)  # 2 层，替换最后一层
        head = ["classifier.1"]
    elif name == "swin_t":
        w = M.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        m = M.swin_t(weights=w)
        m.head = nn.Linear(768, num_classes)
        head = ["head"]
    elif name == "vit_b_16":
        w = M.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        m = M.vit_b_16(weights=w)
        m.heads.head = nn.Linear(768, num_classes)
        head = ["heads.head"]
    else:
        raise ValueError(f"未知模型: {name}")

    params_m = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
    return m, params_m, head


# ---------------------------------------------------------------- 评估
def evaluate(model, loader, criterion, device, num_classes: int):
    import numpy as np
    import torch
    from sklearn.metrics import f1_score

    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    preds_all, labels_all, probs_all = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(x)
                loss = criterion(out, y)
            prob = torch.softmax(out.float(), dim=1)
            pred = prob.argmax(dim=1)
            loss_sum += loss.item() * y.size(0)
            correct += (pred == y).sum().item()
            total += y.size(0)
            preds_all.append(pred.cpu().numpy())
            labels_all.append(y.cpu().numpy())
            probs_all.append(prob.cpu().numpy())

    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    probs = np.concatenate(probs_all)
    return {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "preds": preds,
        "labels": labels,
        "probs": probs,
    }


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k.replace('_', '-')}", action="store_true", default=v)
            ap.add_argument(f"--no-{k.replace('_', '-')}", dest=k, action="store_false")
        else:
            ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    # 用 parse_known_args：colab exec 会给 sys.argv 塞 `-f kernel-xxx.json`，
    # 严格解析会直接报 unrecognized arguments 退出。
    args, unknown = ap.parse_known_args()
    if unknown:
        log(f"忽略未知参数: {unknown}")

    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import classification_report, confusion_matrix
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} " + (f"({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # 固定随机种子。cudnn.benchmark 打开以换速度，代价是结果不完全可复现。
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    data_root = download_and_prepare(args.data_root)
    if args.prepare_only:
        log("prepare_only 模式，数据已就绪，退出")
        return
    classes = [d.name for d in sorted((data_root / "train").iterdir()) if d.is_dir()]
    num_classes = len(classes)
    log(f"类别数 {num_classes}")

    train_ds = ImageFolder(
        str(data_root / "train"),
        transform=build_transforms(
            args.img_size,
            True,
            args.strong_aug,
            args.da_aug,
            local_crop=args.local_crop,
            local_scale_min=args.local_scale_min,
            local_scale_max=args.local_scale_max,
        ),
    )
    val_ds = ImageFolder(str(data_root / "val"), transform=build_transforms(args.img_size, False))
    test_ds = ImageFolder(str(data_root / "test"), transform=build_transforms(args.img_size, False))
    assert train_ds.classes == val_ds.classes == test_ds.classes, "三个 split 的类别不一致"

    pin = device.type == "cuda"
    common = dict(num_workers=args.num_workers, pin_memory=pin, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **common)
    log(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    model, params_m, head_names = build_model(args.model, num_classes, not args.no_pretrained)
    model = model.to(device)
    log(f"模型 {args.model} 参数量 {params_m:.2f}M")

    # ---------------- 测速模式 ----------------
    if args.bench > 0:
        log(f"测速模式：跑 {args.bench} 个 batch")
        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        model.train()
        # 先热身 2 个 batch（让 cudnn 选算法、让 worker 预热）
        t_warm = time.time()
        it = iter(train_loader)
        for _ in range(2):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(train_loader)
                x, y = next(it)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        log(f"热身 {time.time() - t_warm:.1f}s")

        t0 = time.time()
        n_img = 0
        it = iter(train_loader)
        for _ in range(args.bench):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(train_loader)
                x, y = next(it)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            n_img += y.size(0)
        dt = time.time() - t0
        ips = n_img / dt
        log(f"RESULT_BENCH {json.dumps({'imgs': n_img, 'secs': round(dt, 2), 'img_per_s': round(ips, 1), 'sec_per_epoch': round(len(train_ds) / ips, 1)})}")
        return

    # ---------------- 优化器与调度器 ----------------
    head_params, backbone_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(n.startswith(h) for h in head_names):
            head_params.append(p)
        else:
            backbone_params.append(p)
    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    log(f"优化器: 骨干 {len(backbone_params)} 个张量 lr={args.lr * 0.1:.2e}, 头 {len(head_params)} 个张量 lr={args.lr:.2e}")

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    run_name = args.run_name or f"{args.model}_{time.strftime('%m%d_%H%M')}"
    out = Path(args.out_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "classes.json").write_text(
        json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"输出目录 {out}")

    # ---------------- 训练循环 ----------------
    hist = []
    best_acc = 0.0
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        loss_sum, correct, total = 0.0, 0, 0
        for bi, (x, y) in enumerate(train_loader):
            if args.limit_batches and bi >= args.limit_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out_ = model(x)
                loss = criterion(out_, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            global_step += 1

            loss_sum += loss.item() * y.size(0)
            correct += (out_.argmax(1) == y).sum().item()
            total += y.size(0)

            if bi % 50 == 0:
                log(f"  ep{epoch + 1} b{bi}/{steps_per_epoch} loss={loss.item():.4f} acc={correct / max(total, 1):.4f}")

        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)
        val = evaluate(model, val_loader, criterion, device, num_classes)
        secs = time.time() - t0
        lr_now = opt.param_groups[1]["lr"]
        hist.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val["loss"], 6),
                "val_acc": round(val["acc"], 6),
                "val_macro_f1": round(val["macro_f1"], 6),
                "lr": lr_now,
                "secs": round(secs, 1),
            }
        )
        log(
            f"epoch {epoch + 1}/{args.epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f} val_f1={val['macro_f1']:.4f} "
            f"lr={lr_now:.2e} {secs:.0f}s"
        )

        if val["acc"] > best_acc:
            best_acc = val["acc"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": classes,
                    "epoch": epoch,
                    "val_acc": val["acc"],
                    "args": vars(args),
                },
                out / "best.pt",
            )
            log(f"  保存 best.pt (val_acc={best_acc:.4f})")

    # ---------------- 测试 ----------------
    log("加载最佳权重做测试集评估…")
    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, test_loader, criterion, device, num_classes)
    log(f"TEST acc={test['acc']:.4f} macro_f1={test['macro_f1']:.4f} loss={test['loss']:.4f}")

    summary = {
        "model": args.model,
        "params_m": round(params_m, 3),
        "img_size": args.img_size,
        "seed": args.seed,
        "epochs": args.epochs,
        "pretrained": not args.no_pretrained,
        "strong_aug": args.strong_aug,
        "da_aug": args.da_aug,
        "local_crop": args.local_crop,
        "local_scale_min": args.local_scale_min,
        "local_scale_max": args.local_scale_max,
        "best_val_acc": round(best_acc, 6),
        "test_loss": round(test["loss"], 6),
        "test_acc": round(test["acc"], 6),
        "test_macro_f1": round(test["macro_f1"], 6),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # history.csv
    with open(out / "history.csv", "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,val_macro_f1,lr,secs\n")
        for h in hist:
            f.write(
                f"{h['epoch']},{h['train_loss']},{h['train_acc']},{h['val_loss']},"
                f"{h['val_acc']},{h['val_macro_f1']},{h['lr']:.8f},{h['secs']}\n"
            )

    # ---------------- 每类指标 ----------------
    rep = classification_report(
        test["labels"], test["preds"], target_names=classes, output_dict=True, zero_division=0
    )
    with open(out / "per_class_f1.csv", "w", encoding="utf-8") as f:
        f.write("class,precision,recall,f1,support\n")
        for c in classes:
            r = rep.get(c, {})
            f.write(
                f"{c},{r.get('precision', 0):.4f},{r.get('recall', 0):.4f},"
                f"{r.get('f1-score', 0):.4f},{int(r.get('support', 0))}\n"
            )

    # ---------------- 图 ----------------
    plt = setup_matplotlib()
    cjk = plt.rcParams["font.family"][0] not in ("sans-serif", "DejaVu Sans")
    L = (lambda cn, en: cn) if cjk else (lambda cn, en: en)

    # 曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ep = [h["epoch"] + 1 for h in hist]
    axes[0].plot(ep, [h["train_loss"] for h in hist], "o-", label=L("训练损失", "train loss"))
    axes[0].plot(ep, [h["val_loss"] for h in hist], "s-", label=L("验证损失", "val loss"))
    axes[0].set_xlabel(L("轮次", "epoch"))
    axes[0].set_ylabel(L("损失", "loss"))
    axes[0].set_title(L("损失曲线", "Loss"))
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, [h["train_acc"] for h in hist], "o-", label=L("训练精度", "train acc"))
    axes[1].plot(ep, [h["val_acc"] for h in hist], "s-", label=L("验证精度", "val acc"))
    axes[1].plot(ep, [h["val_macro_f1"] for h in hist], "^-", label=L("验证 macro-F1", "val macro-F1"))
    axes[1].set_xlabel(L("轮次", "epoch"))
    axes[1].set_ylabel(L("精度", "accuracy"))
    axes[1].set_title(L("精度曲线", "Accuracy"))
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"{args.model}  {args.img_size}px  " + L("测试精度", "test acc") + f"={test['acc']:.4f}")
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=140)
    plt.close(fig)
    log("已保存 curves.png")

    # 混淆矩阵（归一化）
    cm = confusion_matrix(test["labels"], test["preds"], labels=list(range(num_classes)))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel(L("预测类别", "predicted"))
    ax.set_ylabel(L("真实类别", "true"))
    ax.set_title(
        L("混淆矩阵（按真实类归一化）", "Confusion matrix (row-normalized)")
        + f"  {args.model}  acc={test['acc']:.4f}"
    )
    short = [c[3:][:22] for c in classes]
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_yticklabels(short, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    log("已保存 confusion_matrix.png")

    # ---------------- ONNX 导出 ----------------
    if args.export_onnx:
        try:
            import torch

            model.eval()
            dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)
            onnx_path = out / "model.onnx"
            torch.onnx.export(
                model,
                dummy,
                str(onnx_path),
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17,
                do_constant_folding=True,
            )
            log(f"已导出 {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

            # 用 onnxruntime 验证数值一致性（若已安装）
            try:
                import numpy as np
                import onnxruntime as ort

                sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
                with torch.no_grad():
                    ref = model(dummy).cpu().numpy()
                got = sess.run(None, {"input": dummy.cpu().numpy()})[0]
                diff = float(np.abs(ref - got).max())
                log(f"ONNX 数值校验 最大误差 {diff:.2e}")
                summary["onnx_max_diff"] = diff
            except ImportError:
                log("未装 onnxruntime，跳过数值校验")
        except Exception as e:
            log(f"ONNX 导出失败（不影响训练结果）: {e}")

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("RESULT_SUMMARY " + json.dumps(summary, ensure_ascii=False))
    log(f"全部完成，产物在 {out}")


if __name__ == "__main__":
    main()
