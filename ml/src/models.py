"""模型工厂：统一构建参与对比的各个模型，并给出统一的可训练接口。

为什么需要这个文件而不是在训练脚本里 if/else：
    对比实验的公平性靠"除了模型结构，其他一切相同"来保证。如果每个模型
    的改造方式散落在训练脚本里，很容易出现"ResNet 换了分类头但 MobileNet 忘了换"
    这种低级错误，而这种错误的后果是结论完全错误且极难发现。
    所以所有模型都经过同一个 build_model()，输出保证：
      - 输入 3xHxW，输出 num_classes 维 logits
      - 分类头被替换成新的（预训练时原头是 1000 类）
      - 返回 (model, meta)，meta 里带参数量和说明，供报告直接引用

关于预训练权重：
    全部加载 ImageNet-1k 预训练权重。理由是 PlantVillage 只有 5 万张图、
    38 类，从零训练一个 ViT 会严重欠拟合（ViT 需要千万级数据），
    对比就变成了"谁更依赖数据"而不是"谁结构更合适"。
    报告里必须明确写出"使用了预训练"，否则结论会被质疑。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import models


@dataclass
class ModelMeta:
    """模型元信息，训练完直接写进结果 CSV，避免事后手抄出错。"""

    name: str
    family: str          # CNN / 轻量CNN / Transformer
    params_m: float      # 参数量（百万）
    notes: str = ""


def _count_params(model: nn.Module) -> float:
    """统计可训练参数量（百万）。只数 requires_grad 的，冻结的部分不算。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def build_model(name: str, num_classes: int, pretrained: bool = True) -> tuple[nn.Module, ModelMeta]:
    """按名字构建模型，返回 (model, meta)。

    支持的 name：
        resnet18, resnet50, mobilenet_v3_large, mobilenet_v3_small,
        shufflenet_v2_x1_0, efficientnet_b0, convnext_tiny, vit_b_16, swin_t
    """
    name = name.lower()

    # ---- ResNet 系列：经典基线，报告里的"标准答案" ----
    if name == "resnet18":
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=w)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        meta = ModelMeta(name, "CNN", 0, "残差网络，18 层，最常用的轻量基线")

    elif name == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=w)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        meta = ModelMeta(name, "CNN", 0, "残差网络，50 层，瓶颈结构，容量大")

    # ---- 轻量 CNN：实验四的重点，移动端部署的候选 ----
    elif name == "mobilenet_v3_large":
        w = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v3_large(weights=w)
        # MobileNetV3 的分类头是 Sequential(Linear, Hardswish, Dropout, Linear)
        # 只替换最后一层，保留前面的 dropout，否则等于把正则化去掉了
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        meta = ModelMeta(name, "轻量CNN", 0, "深度可分离卷积 + SE 模块 + h-swish")

    elif name == "mobilenet_v3_small":
        w = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=w)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        meta = ModelMeta(name, "轻量CNN", 0, "MobileNetV3 的小宽度版本")

    elif name == "shufflenet_v2_x1_0":
        w = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.shufflenet_v2_x1_0(weights=w)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        meta = ModelMeta(name, "轻量CNN", 0, "通道混洗 + 分组卷积，专为移动端设计")

    # ---- EfficientNet：复合缩放的代表 ----
    elif name == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=w)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        meta = ModelMeta(name, "轻量CNN", 0, "复合缩放（深度/宽度/分辨率同时放大）")

    elif name == "convnext_tiny":
        w = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=w)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        meta = ModelMeta(name, "CNN", 0, "现代化 ResNet，大卷积核 + LayerNorm")

    # ---- Transformer：实验五的重点，用来对照归纳偏置 ----
    elif name == "vit_b_16":
        w = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=w)
        # ViT 的分类头是 heads.head
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        meta = ModelMeta(name, "Transformer", 0, "纯自注意力，全局感受野，无卷积归纳偏置")

    elif name == "swin_t":
        w = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.swin_t(weights=w)
        model.head = nn.Linear(model.head.in_features, num_classes)
        meta = ModelMeta(name, "Transformer", 0, "层次化窗口注意力，兼顾局部与全局")

    else:
        raise ValueError(
            f"不认识的模型 {name}。可选：resnet18, resnet50, mobilenet_v3_large, "
            f"mobilenet_v3_small, shufflenet_v2_x1_0, efficientnet_b0, convnext_tiny, vit_b_16, swin_t"
        )

    meta.params_m = _count_params(model)
    return model, meta


# 预置的对比组合，训练脚本可以直接引用，保证所有实验用同一份清单
EXPERIMENT_MODELS = [
    "resnet18",
    "resnet50",
    "mobilenet_v3_large",
    "shufflenet_v2_x1_0",
    "efficientnet_b0",
    "swin_t",
]
