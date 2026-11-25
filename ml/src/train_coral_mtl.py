#!/usr/bin/env python3
"""CORAL 特征对齐 + 多任务学习（作物 / 病害）的无监督域适应训练。

要解决的问题
------------
PlantVillage（实验室摆拍）训练出的 ResNet18 直接拿到 PlantDoc（田间手机拍摄）
上只有 **28.18%** 的 top-1、51.13% 的作物级准确率（`ml/runs/resnet18_224_15ep/`）。
已有六条「不碰权重、只改输入/输出」的路线全部被证伪（见
`ml/artifacts/crossdomain_experiments.md`），结论是：**跨域掉点的主因是特征本身
不对，不是统计量漂移或决策边界不够尖锐**。所以这条路线直接在训练时改造特征空间。

方法（两条腿，必须一起看）
--------------------------
1. **CORAL 特征对齐**（Sun & Saenko, 2016；Deep CORAL）
   把源域和目标域在骨干输出层的**二阶统计量**（特征协方差）对齐。给定源特征
   S ∈ R^{n×d}、目标特征 T ∈ R^{m×d}：

       Cov(X) = (X - 1μᵀ)ᵀ (X - 1μᵀ) / (n - 1)
       L_coral = ||Cov(S) - Cov(T)||²_F / (4 d²)

   分母 4d² 是原论文的归一化项，作用是让 loss 的量级不随特征维度爆炸——d=512
   时如果不除，Frobenius 范数平方是 512² 量级的求和，会和 CE（≈1）差 5 个数量级，
   μ 就没法调了。
   **关键实现细节**：CORAL 必须在 float32 下算。autocast 会把特征降成 fp16，
   512×512 的协方差矩阵在 fp16 下既有溢出风险、又有明显的精度损失（协方差是
   二阶量，误差被平方放大），所以这里显式 `with torch.autocast(enabled=False)`。

2. **多任务头**：骨干特征同时接两个分类头
   - 病害头（38 类）：主任务，也是评估时唯一用到的头
   - 作物头（14 类）：辅任务。作物标签从 PlantVillage 文件夹名解析
     （`07_Corn_maize_...` → `Corn_maize`），按已知作物名长度降序匹配，
     避免 `Cherry` 抢在 `Cherry_including_sour` 前面。
   总损失：L = CE_病害 + λ · CE_作物 + μ · L_coral，λ 默认 0.4。

   为什么作物头可能有帮助：作物身份是比病害**跨域鲁棒得多**的语义（"这是番茄"
   在实验室和田间都成立，"这是早疫病"在田间可能只剩几个斑）。把作物作为显式
   辅任务，等于要求共享特征保留一个域无关的粗粒度语义轴，CORAL 对齐时就不容易
   把特征压成"域不变但无语义"的退化解。这也是为什么 λ 不能太小（≈0 等于没用）
   也不能太大（会盖掉病害信号）。

为什么 μ 要 ramp-up
-------------------
前几轮特征还在剧烈变化，此时强行对齐协方差等于给一个还在移动的目标画靶子，
而且 CORAL 的梯度会主导早期更新。所以前 `coral_warmup_epochs` 轮把 μ 从 0
线性升到设定值，之后恒定。默认 2 轮。

无监督域适应的定义（必须守住）
------------------------------
训练时**只用 PlantDoc 的图像，不用它的标签**。实现上目标域走
`ImageFolder`（标签会被读出来），但训练循环里 `for x_t, _ in ...` 直接丢弃，
loss 里不出现任何目标域标签。目标域标签只在最后 `--eval-tgt` 的评估阶段用，
且只用于报告数字——不参与任何梯度或模型选择。

用法
----
    # 本地冒烟（CPU，几分钟）
    python src/train_coral_mtl.py --epochs 1 --limit-batches 20 \
        --batch-size 16 --num-workers 0 --limit-val-batches 10 --no-eval-tgt \
        --out-dir runs_smoke --run-name coral_mtl_smoke

    # 全量（Kaggle GPU）
    python train_coral_mtl.py --download-src --src-root data/plantvillage \
        --tgt-root data/plantdoc/train --epochs 15 --batch-size 64 \
        --coral-weight 0.5 --crop-weight 0.4 --out-dir runs --run-name coral_mtl_mu0.5

    # 训练完的跨域评估（与基线完全同一口径）
    python src/eval_crossdomain.py --run-dir runs/coral_mtl_mu0.5 \
        --plantdoc-root data/raw/plantdoc --split all

产物（runs/<run_name>/）
------------------------
    best.pt            与 eval_crossdomain.py 完全兼容（resnet18 + fc=病害头），
                       额外带 crop_head / crop_classes，供多任务分析
    last.pt            最后一轮权重
    history.csv        每轮 病害/作物/CORAL 三项 loss 与指标
    summary.json       最终指标（源域 val/test + 目标域跨域）
    config.json        完整超参
    classes.json       38 个病害类文件夹名（顺序 == 模型输出索引）
    crop_classes.json  14 个作物类
    curves.png         损失 / CORAL / 精度曲线（无 matplotlib 时跳过）
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- 默认配置
DEFAULTS: dict[str, object] = {
    # 模型 / 数据
    "model": "resnet18",
    "img_size": 224,
    "src_root": "data/processed/plantvillage",   # 源域：有标签
    "tgt_root": "data/raw/plantdoc/train",       # 目标域：只用图像
    "download_src": False,                       # Kaggle 上没有现成 PV，从 HF 拉
    "num_workers": 2,
    "seed": 42,
    # 优化
    "epochs": 15,
    "batch_size": 64,
    "lr": 3e-4,
    "weight_decay": 0.05,
    "label_smoothing": 0.1,
    "warmup_epochs": 2,
    "grad_clip": 1.0,
    # 方法超参
    "crop_weight": 0.4,        # λ：作物 CE 的权重
    "coral_weight": 0.5,       # μ：CORAL 的权重
    "coral_warmup_epochs": 2,  # μ 从 0 线性升到 coral_weight 的轮数
    # 梯度自校准：CORAL 除以 4d² 后 loss 量级只有 ~5e-3，而病害 CE 是 ~3.6，
    # 直接乘 μ=0.5 时 CORAL 的梯度只占任务梯度的 0.19%，等于没开。开这个选项
    # 会在首个 batch 上实测两者梯度范数，把 CORAL 放大到「梯度占比 = coral_grad_ratio」。
    # 关掉时退回论文原始公式（μ 就是字面权重），便于与文献对照。
    "coral_autoscale": False,
    "coral_grad_ratio": 0.05,  # 自校准目标：CORAL 梯度占总梯度的比例
    "no_coral": False,         # 关掉 CORAL（= 只做多任务，做消融）
    "no_mtl": False,           # 关掉作物头（= 只做 CORAL，做消融）
    # 增强
    "strong_aug": True,
    "da_aug": False,           # 与 E-10 一致：默认不开（已验证负收益）
    # 输出
    "out_dir": "runs",
    "run_name": "",
    "eval_tgt": True,          # 训练结束后做一次 PlantDoc 评估（仅评估用标签）
    "export_onnx": False,
    # 规模限制（冒烟用）
    "limit_batches": 0,
    "limit_val_batches": 0,
    "limit_tgt_eval": 0,
    "log_every": 50,
}

# 环境变量覆盖：Kaggle/Colab 里不方便传长命令行时用 CORAL_<KEY>
for _k in list(DEFAULTS):
    _env = os.environ.get("CORAL_" + _k.upper())
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

# 作物名按长度降序：'Cherry_including_sour' 必须先于 'Cherry' 匹配，
# 'Corn_maize' 先于 'Corn'，'Pepper_bell' 先于 'Pepper'。
# 这份清单与 analyze_errors.CROP_KEYS 口径一致，共 14 种作物。
CROP_KEYS: list[str] = [
    "Cherry_including_sour",
    "Corn_maize",
    "Pepper_bell",
    "Strawberry",
    "Raspberry",
    "Blueberry",
    "Peach",
    "Apple",
    "Grape",
    "Orange",
    "Potato",
    "Squash",
    "Soybean",
    "Tomato",
]

# PlantDoc 类名 → PlantVillage 文件夹名。仅在 --eval-tgt 时使用（评估，不训练）。
# 与 eval_crossdomain.PLANTDOC_TO_PV_FOLDER 完全一致。
PLANTDOC_TO_PV_FOLDER: dict[str, str] = {
    "Apple_leaf": "03_Apple_healthy",
    "Apple_rust_leaf": "02_Apple_Cedar_apple_rust",
    "Apple_Scab_Leaf": "00_Apple_Apple_scab",
    "Bell_pepper_leaf": "19_Pepper_bell_healthy",
    "Bell_pepper_leaf_spot": "18_Pepper_bell_Bacterial_spot",
    "Blueberry_leaf": "04_Blueberry_healthy",
    "Cherry_leaf": "06_Cherry_including_sour_healthy",
    "grape_leaf": "14_Grape_healthy",
    "grape_leaf_black_rot": "11_Grape_Black_rot",
    "Corn_Gray_leaf_spot": "07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn_leaf_blight": "09_Corn_maize_Northern_Leaf_Blight",
    "Corn_rust_leaf": "08_Corn_maize_Common_rust",
    "Peach_leaf": "17_Peach_healthy",
    "Potato_leaf_early_blight": "20_Potato_Early_blight",
    "Potato_leaf_late_blight": "21_Potato_Late_blight",
    "Raspberry_leaf": "23_Raspberry_healthy",
    "Soyabean_leaf": "24_Soybean_healthy",
    "Squash_Powdery_mildew_leaf": "25_Squash_Powdery_mildew",
    "Strawberry_leaf": "27_Strawberry_healthy",
    "Tomato_leaf": "37_Tomato_healthy",
    "Tomato_Early_blight_leaf": "29_Tomato_Early_blight",
    "Tomato_leaf_late_blight": "30_Tomato_Late_blight",
    "Tomato_leaf_bacterial_spot": "28_Tomato_Bacterial_spot",
    "Tomato_leaf_mosaic_virus": "36_Tomato_Tomato_mosaic_virus",
    "Tomato_leaf_yellow_virus": "35_Tomato_Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mold_leaf": "31_Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot": "32_Tomato_Septoria_leaf_spot",
}


# ---------------------------------------------------------------- 工具函数
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def crop_of_folder(folder: str) -> str:
    """从 PlantVillage 文件夹名取作物名。

    '07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot' → 'Corn_maize'
    '22_Potato_healthy' → 'Potato'
    口径与 eval_crossdomain.crop_of_folder / analyze_errors.crop_of 一致。
    """
    body = re.sub(r"^\d+_", "", folder)
    for crop in sorted(CROP_KEYS, key=len, reverse=True):
        if body.startswith(crop):
            return crop
    parts = body.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else body


def setup_matplotlib():
    """无界面后端 + 尽量找中文字体；找不到也不报错。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            font_manager.fontManager.addfont(p)
            name = font_manager.FontProperties(fname=p).get_name()
            plt.rcParams["font.family"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            log(f"中文字体: {name}")
            break
    return plt


# ---------------------------------------------------------------- 数据准备
def prepare_plantvillage(data_root: str) -> Path:
    """从 HuggingFace 下载 PlantVillage 并转成 ImageFolder 结构（已有则跳过）。

    逻辑与 train_colab.download_and_prepare 一致，只是去掉无关参数。
    仅在 --download-src 时调用（本地数据已就绪，Kaggle 上没有）。
    """
    root = Path(data_root)
    marker = root / ".prepared"
    if marker.exists():
        log(f"源域数据已就绪: {root}")
        return root

    from datasets import load_dataset

    log("下载 GVJahnavi/PlantVillage_dataset（约 813MB）…")
    ds = load_dataset("GVJahnavi/PlantVillage_dataset")

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
        uniq = sorted({int(x) for x in ds["train"]["label"]})
        label_names = [f"class_{i}" for i in uniq]
        log(f"警告：未能解析原始类名，用占位名 class_*（共 {len(uniq)} 类）")

    def sanitize(name: str) -> str:
        out = name.replace("___", "_").replace("__", "_")
        for ch in " ()[],'\"-":
            out = out.replace(ch, "_")
        while "__" in out:
            out = out.replace("__", "_")
        return out.strip("_")

    class_dirs = {i: f"{i:02d}_{sanitize(n)}" for i, n in enumerate(label_names)}
    log(f"共 {len(label_names)} 类")

    import io
    import random

    from PIL import Image

    def dump(split_name: str, out_split: str, val_ratio: float = 0.0, seed: int = 42):
        split = ds[split_name]
        idxs = list(range(len(split)))
        if val_ratio > 0:
            rng = random.Random(seed)
            rng.shuffle(idxs)
            n_val = int(len(idxs) * val_ratio)
            val_idx = set(idxs[:n_val])
        else:
            val_idx = set()
        counts = {out_split: 0, "val": 0}
        for i in idxs:
            row = split[i]
            label = int(row["label"])
            raw = row["image"]
            if isinstance(raw, dict):
                img = Image.open(io.BytesIO(raw["bytes"]))
            elif isinstance(raw, (bytes, bytearray)):
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

    c1 = dump("train", "train", val_ratio=0.1)
    log(f"  train={c1['train']} val={c1['val']}")
    c2 = dump("test", "test")
    log(f"  test={c2['test']}")

    (root / "classes.json").write_text(
        json.dumps(
            [{"index": i, "folder": class_dirs[i], "name": label_names[i]} for i in range(len(label_names))],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    marker.write_text("ok", encoding="utf-8")
    return root


# ---------------------------------------------------------------- transform
def build_transforms(img_size: int, train: bool, strong: bool = True, da: bool = False):
    """与 train_colab.build_transforms 同一口径。

    验证/测试口径必须和基线完全一致（Resize(256) + CenterCrop(224)），
    否则跨域数字没法和 28.18% 比。
    """
    from torchvision import transforms as T

    if not train:
        return T.Compose(
            [
                T.Resize(int(img_size * 256 / 224)),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    ops = [
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0) if strong else (0.8, 1.0)),
        T.RandomHorizontalFlip(),
    ]
    if strong:
        ops += [
            T.RandomVerticalFlip(p=0.2),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ColorJitter(0.3, 0.3, 0.3, 0.05),
        ]
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
# 每个模型的 (骨干输出维度, 分类头在模型里的点分路径)。
# 路径必须写死：遍历找"最后一个 Linear"在 MobileNetV3 上会命中 classifier[0]，
# 在 Swin 上会命中某个 MLP，是这类脚本最常见的静默错误。
MODEL_SPECS: dict[str, tuple[int, str]] = {
    "resnet18": (512, "fc"),
    "resnet50": (2048, "fc"),
    "mobilenet_v3_large": (1280, "classifier.3"),
    "shufflenet_v2_x1_0": (1024, "fc"),
    "efficientnet_b0": (1280, "classifier.1"),
    "convnext_tiny": (768, "classifier.2"),
    "swin_t": (768, "head"),
    "vit_b_16": (768, "heads.head"),
}


def _split_head_path(model, dotted: str):
    """把 'classifier.3' 拆成 (父模块, 子键)，支持数字下标。"""
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    return parent, parts[-1]


def build_backbone(name: str, pretrained: bool = True):
    """返回 (去掉分类头的骨干, 特征维度, 分类头点分路径)。"""
    import torch.nn as nn
    from torchvision import models as M

    name = name.lower()
    if name not in MODEL_SPECS:
        raise ValueError(f"未知模型 {name}，可选 {sorted(MODEL_SPECS)}")
    feat_dim, head_key = MODEL_SPECS[name]

    if name == "resnet18":
        m = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    elif name == "resnet50":
        m = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    elif name == "mobilenet_v3_large":
        m = M.mobilenet_v3_large(weights=M.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
    elif name == "shufflenet_v2_x1_0":
        m = M.shufflenet_v2_x1_0(weights=M.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None)
    elif name == "efficientnet_b0":
        m = M.efficientnet_b0(weights=M.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    elif name == "convnext_tiny":
        m = M.convnext_tiny(weights=M.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
    elif name == "swin_t":
        m = M.swin_t(weights=M.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
    elif name == "vit_b_16":
        m = M.vit_b_16(weights=M.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
    else:  # pragma: no cover - 上面已覆盖全部键
        raise ValueError(name)

    parent, leaf = _split_head_path(m, head_key)
    if leaf.isdigit():
        parent[int(leaf)] = nn.Identity()
    else:
        setattr(parent, leaf, nn.Identity())
    return m, feat_dim, head_key


def build_mtl_model(name: str, n_dis: int, n_crop: int, pretrained: bool = True, dropout: float = 0.0):
    """构建「骨干 + 病害头 + 作物头」。

    返回一个 nn.Module，其 state_dict 的键是：
        backbone.*                     骨干权重
        fc_dis.weight / fc_dis.bias    病害头（38 类）
        fc_crop.weight / fc_crop.bias  作物头（14 类）
    导出时再把 fc_dis 写回标准模型的分类头位置，见 export_eval_state_dict()。
    """
    import torch
    import torch.nn as nn

    backbone, feat_dim, head_key = build_backbone(name, pretrained)

    class _MTL(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head_key = head_key
            self.feat_dim = feat_dim
            self.fc_dis = nn.Linear(feat_dim, n_dis)
            self.fc_crop = nn.Linear(feat_dim, n_crop)
            self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            # 两个头都从零初始化；骨干带 ImageNet 预训练
            for h in (self.fc_dis, self.fc_crop):
                nn.init.zeros_(h.bias)
                nn.init.normal_(h.weight, std=0.01)

        def forward(self, x):
            """返回 (特征, 病害 logits, 作物 logits)。特征供 CORAL 使用。"""
            f = self.backbone(x)
            f = torch.flatten(f, 1)
            f = self.drop(f)
            return f, self.fc_dis(f), self.fc_crop(f)

        def export_eval_state_dict(self) -> dict:
            """转成 models.build_model(name, n_dis) 能直接 load 的 state_dict。

            eval_crossdomain.py 走的是 `models.build_model(name, 38)`，
            分类头在 head_key 位置（resnet18 是 fc）。这里把骨干权重原样搬过去，
            再把病害头塞进 head_key，键名就完全对上了。
            """
            sd = {k: v.detach().clone() for k, v in self.backbone.state_dict().items()}
            sd[f"{head_key}.weight"] = self.fc_dis.weight.detach().clone()
            sd[f"{head_key}.bias"] = self.fc_dis.bias.detach().clone()
            return sd

    return _MTL()


# ---------------------------------------------------------------- CORAL loss
def coral_loss(src: "torch.Tensor", tgt: "torch.Tensor") -> "torch.Tensor":
    """Deep CORAL：对齐两个 batch 特征的协方差矩阵。

        Cov(X) = (X - mean)ᵀ (X - mean) / (n - 1)
        L = ||Cov(S) - Cov(T)||²_F / (4 d²)

    参数
    ----
    src : (n, d) 源域特征
    tgt : (m, d) 目标域特征（m 可以与 n 不同）

    注意
    ----
    1. 必须 fp32：调用方负责在 autocast 关闭的上下文里调用。协方差是二阶量，
       fp16 下 512×512 的累加误差会被平方放大。
    2. batch 里只有 1 个样本时 (n-1)=0，直接返回 0，避免除零产生 NaN 污染整个
       训练（NaN 一旦进入权重，后面所有轮次都是 NaN）。
    """
    import torch

    n, d = src.shape
    m = tgt.shape[0]
    if n < 2 or m < 2:
        return src.new_zeros(())

    xm = src - src.mean(dim=0, keepdim=True)
    ym = tgt - tgt.mean(dim=0, keepdim=True)
    cov_s = xm.t() @ xm / (n - 1)
    cov_t = ym.t() @ ym / (m - 1)
    return (cov_s - cov_t).pow(2).sum() / (4.0 * d * d)


# ------------------------------------------------------- CORAL 梯度自校准
def measure_coral_scale(
    model,
    x_s: "torch.Tensor",
    y_s: "torch.Tensor",
    y_crop: "torch.Tensor",
    x_t: "torch.Tensor",
    ce,
    lam: float,
    ratio: float,
    mu_max: float,
    device,
) -> float:
    """实测 CORAL 与任务损失的梯度范数比，返回放大系数。

    为什么需要这个
    --------------
    CORAL 的定义里除以 4d² 是为了让 loss 数值与 d 无关，但 ResNet18 的 d=512，
    除完只剩 ~5e-3 量级；而病害 CE 在训练初期是 ~3.6。实测梯度范数：
        病害 CE  ≈ 5.32
        作物 CE  ≈ 5.20
        CORAL    ≈ 0.020
    也就是说 μ=0.5 时 CORAL 只占总梯度的 0.19%——论文里 μ 取 0.5~1.0 是在
    「未归一化的 CORAL」下调的，直接套用会得到一个「跑了 15 轮但等于没开」的
    负结果，而这会被误读成「CORAL 对这个任务无效」。

    语义约定
    --------
    放大系数按 μ_max 归一化：μ 达到 μ_max 时，CORAL 梯度正好占总梯度的
    `ratio`。这样 ramp 期间占比从 0 线性升到 ratio，与 coral_warmup_epochs
    的意图一致；而不是在 ramp 的中间值上校准、之后又悄悄翻倍。
        系数 = ratio × ||g_task|| / (||g_coral|| × μ_max)
    有效 μ = μ × 系数，最终 μ_eff = μ_max × 系数。

    只在第一个 batch 上测一次（两个额外 backward），之后固定不变——每步重测
    会让有效权重随 batch 抖动，等价于给 CORAL 加噪声。
    """
    import torch

    model.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        f_s, logit_dis, logit_crop = model(x_s)
        loss_task = ce(logit_dis, y_s)
        if lam > 0:
            loss_task = loss_task + lam * ce(logit_crop, y_crop)
    loss_task.backward(retain_graph=True)
    g_task = torch.nn.utils.get_total_norm(
        [p.grad for p in model.parameters() if p.grad is not None]
    ).item()

    model.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        f_t = model(x_t)[0]
    with torch.amp.autocast("cuda", enabled=False):
        loss_coral = coral_loss(f_s.float(), f_t.float())
    if loss_coral.item() <= 0:
        model.zero_grad(set_to_none=True)
        return 1.0
    loss_coral.backward()
    g_coral = torch.nn.utils.get_total_norm(
        [p.grad for p in model.parameters() if p.grad is not None]
    ).item()
    model.zero_grad(set_to_none=True)

    if g_coral <= 0 or mu_max <= 0:
        return 1.0
    if not (g_task == g_task) or not (g_coral == g_coral):  # NaN 检查
        return 1.0
    scale = ratio * g_task / (g_coral * mu_max)
    # 上限防止极端 batch（CORAL 恰好在极小值）把系数放大到破坏训练
    return float(min(max(scale, 1.0), 2000.0))


# ---------------------------------------------------------------- 评估
def evaluate_disease(model, loader, device, limit_batches: int = 0) -> dict:
    """源域验证/测试：病害头 top-1 + macro-F1（与基线同口径）。"""
    import numpy as np
    import torch
    from sklearn.metrics import f1_score

    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            if limit_batches and bi >= limit_batches:
                break
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                _, logit_dis, _ = model(x)
            preds_all.append(logit_dis.float().argmax(1).cpu().numpy())
            labels_all.append(y.numpy())

    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    return {
        "n": int(len(labels)),
        "acc": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def target_diagnostics(model, loader, device, limit_batches: int = 0) -> dict:
    """目标域**无标签**诊断量，用于观察对齐是否在起作用。

    三个量都不需要标签：
      - mean_conf：平均最大 softmax 概率。跨域掉点时通常很低（基线 0.4366）。
      - entropy：平均预测熵（nats）。越低说明模型越"敢下判断"。
      - crop_dis_agree：病害头的 argmax 与作物头 argmax 推出的作物是否一致。
        这是多任务特有的自洽性指标：两个头互相印证说明特征里作物语义清晰。
    """
    import numpy as np
    import torch

    model.eval()
    conf, ent, agree, n = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for bi, (x, _y) in enumerate(loader):
            if limit_batches and bi >= limit_batches:
                break
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                _, logit_dis, logit_crop = model(x)
            p = torch.softmax(logit_dis.float(), dim=1)
            conf += p.max(dim=1).values.sum().item()
            ent += (-(p * (p + 1e-12).log()).sum(dim=1)).sum().item()
            agree += int((p.argmax(1) == logit_crop.float().argmax(1)).sum().item())
            n += x.size(0)
    if n == 0:
        return {"n": 0, "mean_conf": 0.0, "entropy": 0.0, "crop_dis_agree": 0.0}
    return {
        "n": n,
        "mean_conf": conf / n,
        "entropy": ent / n,
        # 注意：病害 38 类与作物 14 类的索引空间不同，这里的"一致"只作为
        # 粗略的稳定性信号（两个头对同一张图的判断是否落在同一个大类上），
        # 不做严格语义解释。
        "crop_dis_agree": agree / n,
    }


def eval_crossdomain(
    model,
    plantdoc_root: Path,
    pv_folders: list[str],
    img_size: int,
    device,
    batch_size: int = 32,
    num_workers: int = 0,
    limit: int = 0,
) -> dict:
    """PlantDoc 跨域评估（**仅评估**，标签不参与训练）。

    与 eval_crossdomain.py 的口径完全一致：split=all（train+test，2920 张），
    27 类可映射，输出 top-1 / top-3 / 作物级。
    """
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    folder_index = {f: i for i, f in enumerate(pv_folders)}
    pv_crop = [crop_of_folder(f) for f in pv_folders]
    items: list[tuple[Path, int]] = []
    for sp in ("train", "test"):
        sp_dir = plantdoc_root / sp
        if not sp_dir.exists():
            continue
        for cls_dir in sorted(p for p in sp_dir.iterdir() if p.is_dir()):
            pv_folder = PLANTDOC_TO_PV_FOLDER.get(cls_dir.name)
            if pv_folder is None or pv_folder not in folder_index:
                continue
            label = folder_index[pv_folder]
            for img_path in sorted(cls_dir.glob("*.jpg")):
                items.append((img_path, label))
    if limit:
        items = items[:limit]

    tf = build_transforms(img_size, train=False)

    class _DS(Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, i):
            path, label = items[i]
            return tf(Image.open(path).convert("RGB")), label

    if not items:
        return {"n": 0, "error": "PlantDoc 映射后无样本"}

    loader = DataLoader(_DS(), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    preds, trues, top3 = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                _, logit_dis, _ = model(x)
            p = torch.softmax(logit_dis.float(), dim=1)
            preds.extend(p.argmax(1).cpu().tolist())
            top3.extend(p.topk(min(3, len(pv_folders)), dim=1).indices.cpu().tolist())
            trues.extend(y.tolist())

    n = len(trues)
    correct = sum(1 for p, t in zip(preds, trues) if p == t)
    crop_ok = sum(1 for p, t in zip(preds, trues) if pv_crop[p] == pv_crop[t])
    top3_ok = sum(1 for t3, t in zip(top3, trues) if t in t3)
    return {
        "n": n,
        "top1_acc": correct / n,
        "top3_acc": top3_ok / n,
        "crop_acc": crop_ok / n,
    }


# ---------------------------------------------------------------- 训练
def main() -> int:
    ap = argparse.ArgumentParser(description="CORAL + 多任务 无监督域适应训练")
    for k, v in DEFAULTS.items():
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            ap.add_argument(flag, action="store_true", default=v)
            ap.add_argument(f"--no-{k.replace('_', '-')}", dest=k, action="store_false")
        else:
            ap.add_argument(flag, type=type(v), default=v)
    # parse_known_args：colab/kaggle 会给 argv 塞 -f kernel.json 之类的参数
    args, unknown = ap.parse_known_args()
    if unknown:
        log(f"忽略未知参数: {unknown}")

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    # ---------------- 源域 ----------------
    if args.download_src:
        prepare_plantvillage(args.src_root)
    src_root = Path(args.src_root)
    for sp in ("train", "val"):
        if not (src_root / sp).is_dir():
            log(f"[错误] 缺少 {src_root / sp}，先跑 prepare_data.py 或加 --download-src")
            return 1

    train_tf = build_transforms(args.img_size, True, args.strong_aug, args.da_aug)
    eval_tf = build_transforms(args.img_size, False)

    src_train = ImageFolder(str(src_root / "train"), transform=train_tf)
    src_val = ImageFolder(str(src_root / "val"), transform=eval_tf)
    assert src_train.classes == src_val.classes, "train/val 类别不一致"
    pv_folders = list(src_train.classes)
    n_dis = len(pv_folders)

    # ---------------- 目标域（只用图像） ----------------
    tgt_root = Path(args.tgt_root)
    if not tgt_root.is_dir():
        log(f"[错误] 目标域目录不存在: {tgt_root}")
        return 1
    # ImageFolder 会读标签，但训练循环里立刻丢弃（`for x_t, _ in ...`）。
    # 用同一个强增强：目标域的成像风格必须被同样扰动，否则 CORAL 对齐的是
    # 「源域增强后分布 vs 目标域未增强分布」，等于人为制造新的域偏移。
    tgt_train = ImageFolder(str(tgt_root), transform=train_tf)
    log(f"源域 train={len(src_train)} val={len(src_val)}  目标域(无标签)={len(tgt_train)}")

    # ---------------- 作物标签 ----------------
    crops = sorted({crop_of_folder(c) for c in pv_folders})
    crop_index = {c: i for i, c in enumerate(crops)}
    cls_crop = np.array([crop_index[crop_of_folder(c)] for c in pv_folders], dtype=np.int64)
    log(f"病害 {n_dis} 类 / 作物 {len(crops)} 类: {crops}")

    pin = device.type == "cuda"
    common = dict(num_workers=args.num_workers, pin_memory=pin, persistent_workers=args.num_workers > 0)
    src_loader = DataLoader(src_train, batch_size=args.batch_size, shuffle=True, drop_last=True, **common)
    tgt_loader = DataLoader(tgt_train, batch_size=args.batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(src_val, batch_size=args.batch_size, shuffle=False, **common)

    def _cycle(loader):
        """无限重复的 loader 迭代器（不缓存数据，比 itertools.cycle 省内存）。"""
        while True:
            yield from loader

    # ---------------- 模型 ----------------
    model = build_mtl_model(
        args.model, n_dis, len(crops), pretrained=True, dropout=0.0
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    log(f"模型 {args.model} 参数量 {n_params:.2f}M（骨干 + 病害头 + 作物头）")

    # 分层学习率：骨干 ×0.1（与基线一致）
    head_params, backbone_params = [], []
    for n_, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n_.startswith("backbone."):
            backbone_params.append(p)
        else:
            head_params.append(p)
    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    log(f"优化器: 骨干 {len(backbone_params)} 张量 lr={args.lr * 0.1:.2e}, "
        f"头 {len(head_params)} 张量 lr={args.lr:.2e}")

    steps_per_epoch = min(len(src_loader), args.limit_batches) if args.limit_batches else len(src_loader)
    total_steps = max(args.epochs * steps_per_epoch, 1)
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    lam = 0.0 if args.no_mtl else float(args.crop_weight)
    mu_max = 0.0 if args.no_coral else float(args.coral_weight)

    run_name = args.run_name or f"{args.model}_coral_mtl_{time.strftime('%m%d_%H%M')}"
    out = Path(args.out_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "classes.json").write_text(json.dumps(pv_folders, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "crop_classes.json").write_text(json.dumps(crops, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"输出目录 {out}")
    log(f"λ(作物)={lam}  μ(CORAL)={mu_max}  coral_warmup={args.coral_warmup_epochs} 轮")

    # ---------------- 训练循环 ----------------
    hist: list[dict] = []
    best_acc = -1.0
    global_step = 0
    coral_scale: float | None = None  # 首个 batch 上由 measure_coral_scale 确定

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        # μ 线性 ramp-up：前 coral_warmup_epochs 轮从 0 升到 μ_max。
        # 理由见文件头：早期特征还在剧变，强行对齐是给移动靶画圈。
        if mu_max > 0 and args.coral_warmup_epochs > 0:
            mu = mu_max * min(1.0, (epoch + 1) / args.coral_warmup_epochs)
        else:
            mu = mu_max

        sum_total = sum_dis = sum_crop = sum_coral = 0.0
        correct = total = 0
        n_coral_steps = 0

        tgt_iter = _cycle(tgt_loader)
        for bi, (x_s, y_s) in enumerate(src_loader):
            if args.limit_batches and bi >= args.limit_batches:
                break
            x_t, _y_t = next(tgt_iter)  # 目标域标签在这里被丢弃，loss 里不出现
            x_s = x_s.to(device, non_blocking=True)
            y_s = y_s.to(device, non_blocking=True)
            y_crop = torch.from_numpy(cls_crop[y_s.detach().cpu().numpy()]).to(device)
            x_t = x_t.to(device, non_blocking=True)

            # 首个有效 batch 上做一次梯度自校准，之后复用同一系数（见 measure_coral_scale）
            if mu > 0 and coral_scale is None:
                coral_scale = measure_coral_scale(
                    model, x_s, y_s, y_crop, x_t, ce, lam,
                    float(args.coral_grad_ratio), mu_max, device,
                ) if args.coral_autoscale else 1.0
                if args.coral_autoscale:
                    log(f"  CORAL 自校准：放大 ×{coral_scale:.2f}，"
                        f"μ 达到 μ_max={mu_max} 时 CORAL 梯度占总梯度 "
                        f"{args.coral_grad_ratio:.1%}（即 μ_eff_max="
                        f"{mu_max * coral_scale:.3f}）")

            # 两次前向都放在同一个 autocast 块里：源域与目标域经过完全相同的
            # 数值路径，CORAL 比较的才是「分布差异」而不是「精度差异」。
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                f_s, logit_dis, logit_crop = model(x_s)
                f_t = model(x_t)[0] if mu > 0 else None
                loss_dis = ce(logit_dis, y_s)
                loss_crop = ce(logit_crop, y_crop) if lam > 0 else logit_dis.new_zeros(())
                loss = loss_dis + lam * loss_crop

            # CORAL 的协方差计算必须在 fp32 下进行（见 coral_loss 文档）
            if mu > 0 and f_t is not None:
                with torch.amp.autocast("cuda", enabled=False):
                    loss_coral = coral_loss(f_s.float(), f_t.float())
                loss = loss + (mu * (coral_scale or 1.0)) * loss_coral
                sum_coral += loss_coral.item()
                n_coral_steps += 1

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            global_step += 1

            bs = y_s.size(0)
            sum_total += loss.item() * bs
            sum_dis += loss_dis.item() * bs
            sum_crop += float(loss_crop) * bs
            correct += int((logit_dis.detach().argmax(1) == y_s).sum().item())
            total += bs

            if args.log_every and bi % args.log_every == 0:
                coral_now = (sum_coral / max(n_coral_steps, 1)) if mu > 0 else 0.0
                eff_mu = mu * (coral_scale or 1.0)
                log(f"  ep{epoch + 1} b{bi}/{steps_per_epoch} total={loss.item():.4f} "
                    f"dis={loss_dis.item():.4f} crop={float(loss_crop):.4f} "
                    f"coral={coral_now:.4f}(μ_eff={eff_mu:.3f}) acc={correct / max(total, 1):.4f}")

        train_total = sum_total / max(total, 1)
        train_dis = sum_dis / max(total, 1)
        train_crop = sum_crop / max(total, 1)
        train_coral = sum_coral / max(n_coral_steps, 1)
        train_acc = correct / max(total, 1)

        val = evaluate_disease(model, val_loader, device, args.limit_val_batches)
        diag = target_diagnostics(model, tgt_loader, device, args.limit_batches)
        secs = time.time() - t0
        lr_now = opt.param_groups[1]["lr"]
        hist.append({
            "epoch": epoch,
            "train_loss": round(train_total, 6),
            "train_dis_loss": round(train_dis, 6),
            "train_crop_loss": round(train_crop, 6),
            "train_coral": round(train_coral, 6),
            "mu": round(mu, 6),
            "mu_eff": round(mu * (coral_scale or 1.0), 6),
            "train_dis_acc": round(train_acc, 6),
            "val_dis_acc": round(val["acc"], 6),
            "val_macro_f1": round(val["macro_f1"], 6),
            "tgt_mean_conf": round(diag["mean_conf"], 6),
            "tgt_entropy": round(diag["entropy"], 6),
            "tgt_crop_dis_agree": round(diag["crop_dis_agree"], 6),
            "lr": lr_now,
            "secs": round(secs, 1),
        })
        log(f"epoch {epoch + 1}/{args.epochs} total={train_total:.4f} dis={train_dis:.4f} "
            f"crop={train_crop:.4f} coral={train_coral:.4f} train_acc={train_acc:.4f} "
            f"val_acc={val['acc']:.4f} val_f1={val['macro_f1']:.4f} "
            f"tgt_conf={diag['mean_conf']:.3f} {secs:.0f}s")

        # 模型选择只看源域验证集：目标域没有标签，用它选模型就破坏无监督假设。
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            torch.save(
                {
                    "model": model.export_eval_state_dict(),
                    "crop_head": {
                        "weight": model.fc_crop.weight.detach().clone(),
                        "bias": model.fc_crop.bias.detach().clone(),
                    },
                    "crop_classes": crops,
                    "classes": pv_folders,
                    "epoch": epoch,
                    "val_acc": val["acc"],
                    "args": vars(args),
                },
                out / "best.pt",
            )
            log(f"  保存 best.pt (val_acc={best_acc:.4f})")

    torch.save(
        {
            "model": model.export_eval_state_dict(),
            "crop_head": {
                "weight": model.fc_crop.weight.detach().clone(),
                "bias": model.fc_crop.bias.detach().clone(),
            },
            "crop_classes": crops,
            "classes": pv_folders,
            "epoch": args.epochs - 1,
            "args": vars(args),
        },
        out / "last.pt",
    )

    # ---------------- 载入最佳权重做目标域评估 ----------------
    summary: dict[str, object] = {
        "model": args.model,
        "params_m": round(n_params, 3),
        "img_size": args.img_size,
        "seed": args.seed,
        "epochs": args.epochs,
        "crop_weight": lam,
        "coral_weight": mu_max,
        "coral_warmup_epochs": args.coral_warmup_epochs,
        "no_coral": bool(args.no_coral),
        "no_mtl": bool(args.no_mtl),
        "strong_aug": args.strong_aug,
        "da_aug": args.da_aug,
        "best_val_acc": round(best_acc, 6),
        "n_train": len(src_train),
        "n_val": len(src_val),
        "n_target_unlabeled": len(tgt_train),
        "n_disease_classes": n_dis,
        "n_crop_classes": len(crops),
    }

    if (out / "best.pt").exists():
        ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
        # 用导出的标准 state_dict 还原病害头，保证与 eval_crossdomain 同一前向路径
        head_key = MODEL_SPECS[args.model.lower()][1]
        model2 = build_mtl_model(args.model, n_dis, len(crops), pretrained=False).to(device)
        # best.pt 的 model 是标准键名（骨干 + head_key 位置的病害头），
        # 骨干部分不含 head_key.* 前缀，直接灌进 model2.backbone
        bb_sd = {k: v for k, v in ck["model"].items() if not k.startswith(f"{head_key}.")}
        missing, unexpected = model2.backbone.load_state_dict(bb_sd, strict=False)
        if missing or unexpected:
            log(f"  骨干加载 missing={missing} unexpected={unexpected}")
        model2.fc_dis.load_state_dict({"weight": ck["model"][f"{head_key}.weight"],
                                       "bias": ck["model"][f"{head_key}.bias"]})
        model2.fc_crop.load_state_dict(ck["crop_head"])
        model2.eval()
        summary["best_epoch"] = ck.get("epoch")

        if args.eval_tgt:
            pd_root = Path(args.tgt_root).parent
            log(f"跨域评估（仅评估）：{pd_root} split=all")
            cd = eval_crossdomain(
                model2, pd_root, pv_folders, args.img_size, device,
                batch_size=32, num_workers=0, limit=args.limit_tgt_eval,
            )
            log(f"TARGET top1={cd.get('top1_acc', 0):.4f} top3={cd.get('top3_acc', 0):.4f} "
                f"crop={cd.get('crop_acc', 0):.4f} n={cd.get('n', 0)}")
            summary["target"] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in cd.items()}

    (out / "history.csv").write_text(
        "epoch,train_loss,train_dis_loss,train_crop_loss,train_coral,mu,mu_eff,train_dis_acc,"
        "val_dis_acc,val_macro_f1,tgt_mean_conf,tgt_entropy,tgt_crop_dis_agree,lr,secs\n"
        + "".join(
            f"{h['epoch']},{h['train_loss']},{h['train_dis_loss']},{h['train_crop_loss']},"
            f"{h['train_coral']},{h['mu']},{h['mu_eff']},{h['train_dis_acc']},{h['val_dis_acc']},"
            f"{h['val_macro_f1']},{h['tgt_mean_conf']},{h['tgt_entropy']},"
            f"{h['tgt_crop_dis_agree']},{h['lr']:.8f},{h['secs']}\n"
            for h in hist
        ),
        encoding="utf-8",
    )

    # ---------------- 曲线（失败不影响训练结果） ----------------
    try:
        plt = setup_matplotlib()
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
        ep = [h["epoch"] + 1 for h in hist]
        axes[0].plot(ep, [h["train_dis_loss"] for h in hist], "o-", label="disease CE")
        if lam > 0:
            axes[0].plot(ep, [h["train_crop_loss"] for h in hist], "s-", label="crop CE")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].set_title("Task losses")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        if mu_max > 0:
            ax2 = axes[1]
            ax2.plot(ep, [h["train_coral"] for h in hist], "^-", color="tab:red")
            ax2.set_xlabel("epoch")
            ax2.set_ylabel("CORAL")
            ax2.set_title(f"CORAL loss (mu_max={mu_max})")
            ax2.grid(alpha=0.3)
            ax2b = ax2.twinx()
            ax2b.plot(ep, [h["mu"] for h in hist], "--", color="gray", alpha=0.7)
            ax2b.set_ylabel("mu (ramp)")
        else:
            axes[1].axis("off")
            axes[1].set_title("CORAL disabled")

        axes[2].plot(ep, [h["val_dis_acc"] for h in hist], "o-", label="source val acc")
        axes[2].plot(ep, [h["tgt_mean_conf"] for h in hist], "s-", label="target mean conf (no label)")
        axes[2].set_xlabel("epoch")
        axes[2].set_ylabel("value")
        axes[2].set_title("Source val acc / target confidence")
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        fig.suptitle(f"{args.model} CORAL+MTL  lam={lam} mu={mu_max}")
        fig.tight_layout()
        fig.savefig(out / "curves.png", dpi=140)
        plt.close(fig)
        log("已保存 curves.png")
    except Exception as e:
        log(f"绘图失败（不影响结果）: {type(e).__name__}: {e}")

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("RESULT_SUMMARY " + json.dumps(summary, ensure_ascii=False))
    log(f"全部完成，产物在 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
