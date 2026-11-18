"""数据加载：训练/验证/测试的 transform 与 DataLoader 构造。

这里单独成一个模块，是因为 transform 的选择本身就是实验的一部分：
报告里要对比"强增强 vs 弱增强"对 PlantVillage 这种单背景数据集的影响，
把 transform 集中在一处，做消融时只改一个参数，不用动训练循环。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ImageNet 的均值和标准差。
# 即使是不用预训练权重的模型，这里也统一用这组值：
# 一是让不同模型的输入分布一致，对比才公平；
# 二是后面大部分模型都会加载 ImageNet 预训练权重，混用会导致微调时
# 第一层就要重新适应，白白浪费预训练带来的收敛优势。
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int, train: bool, strong: bool = True):
    """构造 transform。

    strong=True 时用 RandAugment + RandomErasing，这是处理"过拟合到背景"的常用组合：
    PlantVillage 的图几乎都是"一片叶子居中、纯色背景"，模型很容易学到
    "看背景颜色猜类别"这种捷径。随机擦除和颜色扰动会强迫它去看叶片本身的纹理。
    代价是收敛变慢，所以报告里要做一组 strong=False 的对照来说明它到底值不值。
    """
    if not train:
        # 验证/测试只做 resize + 中心裁剪 + 归一化，必须完全确定，否则指标不可复现
        return transforms.Compose(
            [
                transforms.Resize(int(img_size * 256 / 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    ops = [
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),  # 叶片没有固定的"正着"方向，垂直翻转是合理的
    ]
    if strong:
        ops += [
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if strong:
        ops.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)))
    return transforms.Compose(ops)


def build_dataloaders(
    data_root: str | Path,
    img_size: int,
    batch_size: int,
    num_workers: int = 4,
    strong_aug: bool = True,
    weighted_sampling: bool = False,
):
    """返回 (train_loader, val_loader, test_loader, class_names)。

    weighted_sampling=True 时用 WeightedRandomSampler 做类别平衡采样。
    PlantVillage 是明显不均衡的：Tomato 有 7 个类别共 1.8 万张，
    而 Orange 只有一类 2010 张。不均衡会让模型偏向大类，
    总体 accuracy 好看但 macro-F1 拉胯——所以指标里两个都要报。
    """
    root = Path(data_root)
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    for d in (train_dir, val_dir, test_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"缺少目录 {d}，先跑 prepare_data.py")

    train_ds = datasets.ImageFolder(train_dir, build_transforms(img_size, True, strong_aug))
    val_ds = datasets.ImageFolder(val_dir, build_transforms(img_size, False))
    test_ds = datasets.ImageFolder(test_dir, build_transforms(img_size, False))

    # 三个 split 的类别顺序必须完全一致，否则评估时标签会错位。
    # ImageFolder 按目录名字典序排，只要目录名一致就是一致的，这里断言一下。
    assert train_ds.classes == val_ds.classes == test_ds.classes, "train/val/test 的类别列表不一致"

    sampler = None
    shuffle = True
    if weighted_sampling:
        counts = [0] * len(train_ds.classes)
        for _, label in train_ds.samples:
            counts[label] += 1
        # 每个样本的权重 = 1 / 该类样本数，这样每个类被抽到的期望次数相同
        weights = [1.0 / counts[label] for _, label in train_ds.samples]
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False  # 用了 sampler 就不能再 shuffle，PyTorch 会直接报错

    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)

    return train_loader, val_loader, test_loader, train_ds.classes


def save_class_map(classes: list[str], path: str | Path) -> None:
    """把类别顺序落盘。

    推理服务端必须用训练时的同一套顺序做 argmax，
    少这一步就会出现"模型没错、服务端标签错位"的经典事故。
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")


def load_class_map(path: str | Path) -> list[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
