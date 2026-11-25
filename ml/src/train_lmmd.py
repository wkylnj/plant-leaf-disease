"""LMMD 局部最大均值差异子域对齐训练脚本（MSUN 论文第 ② 模块）。

## 为什么写这个脚本

项目目标是 PlantVillage（实验室摆拍）→ PlantDoc（真实田间）的跨域识别。
当前最佳零样本 37.50%（Swin-T 多分辨率平均），ResNet18 基线 28.18%。
跨域崩溃的根因（见 ml/artifacts/gradcam.md、crossdomain.md）是模型依赖整片叶子的
全局颜色/纹理统计量，而真实照片的光照、色温、背景与实验室摆拍差异巨大。

要复现的论文：
    Wu X, Fan X, Luo P, Das Choudhury S, Tjahjadi T, Hu C.
    From Laboratory to Field: Unsupervised Domain Adaptation for Plant Disease
    Recognition in the Wild. Plant Phenomics 2023;5:0038.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC10059679/

论文的 MSUN 有三个模块，本脚本只实现第 ② 个（LMMD），一次只验证一个变量：
    ① 多表示模块（Inception Adaptation，1×1/3×3/5×5 并联）—— 不做
    ② LMMD 局部最大均值差异（按类别对齐子域）—— 本脚本
    ③ 不确定性正则（熵最小化）—— 不做

## 为什么必须是"局部"（按类）而不是全局 MMD

全局 MMD 对齐的是源域和目标域的整体分布。但不同病害在特征空间里本来就是分开的
簇，全局对齐会把"番茄早疫病"的簇往"苹果黑腐病"的簇上拉——因为目标域整体均值
和源域整体均值不一样，最小化整体差异的最省力解法就是让所有簇互相靠拢。
论文明确指出：跨域植物病害数据 interclass similarity 高、intraclass variation 大，
全局对齐必然造成负迁移。所以必须按类别 c 分别对齐：源域第 c 类的均值 ↔ 目标域
"属于第 c 类"的加权均值。

## 公式（论文 Eq.5–Eq.9）

Eq.5 局部差异的期望形式：
    d_H(p, q) = E_c || E_{p(c)} φ(x^s) − E_{q(c)} φ(x^t) ||²_H

Eq.6 经验估计：
    d̂_H(p,q) = (1/C) Σ_c || Σ_i w_i^{sc} φ(x_i^s) − Σ_j w_j^{tc} φ(x_j^t) ||²_H
    约束 Σ_i w_i^{sc} = 1、Σ_j w_j^{tc} = 1

Eq.7 权重归一化：
    w_i^c = y_i^c / Σ_{(x_j,y_j)∈D} y_j^c
    源域 y 用真标签 one-hot；目标域没有标签，用网络输出的软预测概率 ŷ_i = f(x_i)。
    论文特别强调目标域用"软概率"而不是硬伪标签，就是为了缓解伪标签错误带来的负迁移。

Eq.9 核形式（本脚本的实现依据，不需要显式特征映射 φ）：
    d̂_l(p,q) = (1/C) Σ_c [ 1/n_sc² Σ_i Σ_j w_i^{sc} w_j^{sc} k(M_i^s, M_j^s)
                         + 1/n_tc² Σ_i Σ_j w_i^{tc} w_j^{tc} k(M_i^t, M_j^t)
                         − 2/(n_sc n_tc) Σ_i Σ_j w_i^{sc} w_j^{tc} k(M_i^s, M_j^t) ]

Eq.11 总损失：
    min_f (1/n_s) Σ_i J(f(x_i^s), y_i^s) + λ Σ_i^{n_r} d̂((h_i∘g)(X_s), (h_i∘g)(X_t)) + γ L_en
本脚本取 λ 项 + 源域 CE，不做 L_en（不确定性正则）。

实现上采用 Eq.7 的"权重各自归一化到 1"，此时 Eq.9 里的 1/n_sc²、1/n_tc²、2/(n_sc·n_tc)
系数全被吸收进权重里，核形式退化成干净的三项二次型：
    ||Σ_i w_i φ(x_i^s) − Σ_j w_j φ(x_j^t)||²
       = w_sᵀ K_ss w_s + w_tᵀ K_tt w_t − 2 w_sᵀ K_st w_t
这正是主流 LMMD 实现（DeepDA 等）的写法，也等价于 Eq.6。

## φ 取哪一层

论文的 φ 是分类头之前的特征（多表示模块 h_i ∘ 骨干 g 的输出）。本脚本用 forward
hook 挂在分类头模块上，抓它的输入——就是骨干输出、分类头之前的那个特征向量。
好处是不用改任何模型结构，也不依赖各模型 forward 的内部实现细节。

## 关键风险（必须读）

本项目之前做过伪标签自训练（实验 E-13），结果 0.3373 → 0.3135（−2.38pp），说明
伪标签在我们这个数据上不可靠。LMMD 的目标域权重同样来自模型的软预测概率，本质上是
"软伪标签"，有完全一样的风险：模型在目标域上本来就分不对，按错误概率加权对齐
会把源域特征往错误方向拉。文档 ml/artifacts/lmmd_plan.md 里有详细的风险分析。

## 用法

本地冒烟（CPU，几分钟）：
    python ml/src/train_lmmd.py --epochs 1 --limit-batches 20 --batch-size 8 \\
        --num-workers 0 --no-pretrained --target-eval-limit 2 \\
        --run-name smoke_lmmd

完整训练见 kaggle/lmmd/run.py。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# 保证从仓库根目录直接 `python ml/src/train_lmmd.py` 时能 import 同目录的 models
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from models import build_model  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 分类头模块名。models.build_model 不返回头名字，这里按模型家族写死，
# 和 models.py 里替换分类头的位置严格对应（改了 models.py 必须同步这里）。
HEAD_MODULE: dict[str, str] = {
    "resnet18": "fc",
    "resnet50": "fc",
    "shufflenet_v2_x1_0": "fc",
    "mobilenet_v3_large": "classifier",
    "mobilenet_v3_small": "classifier",
    "efficientnet_b0": "classifier",
    "convnext_tiny": "classifier",
    "vit_b_16": "heads",
    "swin_t": "head",
}

DEFAULTS: dict = {
    # ---- 模型与数据 ----
    "model": "resnet18",
    "img_size": 224,
    "data_root": "ml/data/processed/plantvillage",   # 源域（有标签）
    "target_root": "ml/data/raw/plantdoc",           # 目标域（无标签，训练时丢标签）
    "num_workers": 2,
    "seed": 42,
    "no_pretrained": False,
    # ---- 训练超参（对齐 train_colab.py 的默认值，保证可比）----
    "epochs": 15,
    "batch_size": 32,
    "target_batch_size": 0,      # 0 = 与 batch_size 相同
    "lr": 3e-4,
    "weight_decay": 0.05,
    "label_smoothing": 0.1,
    "warmup_epochs": 1,
    "strong_aug": True,
    "target_aug": False,         # 目标域默认不做强增强：增强会削掉域差异本身
    "limit_batches": 0,          # >0 时每轮只跑这么多 batch（冒烟用）
    # ---- LMMD 超参 ----
    # λ。论文（Eq.11）没给具体值。本机实测（ml/tmp/lmmd_lambda_probe.py，resnet18
    # 预训练 + 真实数据 6 个 batch）：CE 在骨干上的梯度范数 ≈ 7.7，LMMD ≈ 0.31，
    # 比值中位数 25.5。所以 λ=0.1/0.5/1.0 时 LMMD 的梯度贡献只有 CE 的
    # 0.4%/2%/4%，几乎等于没开这项损失。要使两项梯度同量级需要 λ≈25。
    # 因此默认值取 1.0（约 4%，保守起点），建议扫 1.0 / 5.0 / 10.0 三档，
    # 分别对应 LMMD 梯度约占 CE 的 4% / 20% / 40%。
    # 想先复现"温和对齐"可以仍用 0.1/0.5，但要知道那基本是无效剂量。
    "lmmd_weight": 1.0,
    "lmmd_kernel": "rbf",        # rbf | linear
    "lmmd_sigma": 0.0,           # >0 时用固定 σ；0 表示用中位数启发式自适应
    "lmmd_normalize_features": False,  # True 则先把 φ L2 归一化到单位球面
    "lmmd_grad_target_probs": False,   # 是否让梯度穿过目标域软概率
    "lmmd_warmup_epochs": 1,     # λ 从 0 线性升到设定值的轮数，避免早期噪声权重破坏特征
    # ---- 评估与产物 ----
    "eval_target": True,
    "target_eval_split": "test",
    "target_eval_limit": 0,      # 每类最多取几张做目标域评估（冒烟用，0=全量）
    "out_dir": "ml/runs",
    "run_name": "",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def build_transforms(img_size: int, train: bool, strong: bool = True):
    """训练/评估的图像变换。

    评估口径必须与 train_colab.py 完全一致（Resize(int(size*256/224)) + CenterCrop），
    否则跨域数字和历史结果不可比。这是硬约束，不要"顺手优化"。
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

    ops = [T.RandomResizedCrop(img_size, scale=(0.6, 1.0) if strong else (0.8, 1.0))]
    ops.append(T.RandomHorizontalFlip())
    if strong:
        ops += [
            T.RandomVerticalFlip(p=0.2),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ColorJitter(0.3, 0.3, 0.3, 0.05),
        ]
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    # RandomErasing 只吃 tensor，必须放在 ToTensor 之后（与 train_colab.py 一致）
    if strong:
        ops.append(T.RandomErasing(p=0.25, scale=(0.02, 0.15)))
    return T.Compose(ops)


class FeatureTap:
    """用 forward hook 抓分类头的输入特征，即 φ(x)。

    为什么用 hook 而不是改模型 forward：
        models.py 里 9 个模型的 forward 各不相同（ResNet 是 avgpool→flatten→fc，
        ConvNeXt 的 classifier 首层是 LayerNorm2d 吃 4D 张量，ViT 是 heads.head 吃
        CLS token）。改 forward 要写 9 份且容易错；hook 挂在分类头模块上，抓到的
        就是"分类头之前"的张量，一次覆盖全部模型。
    注意 ConvNeXt 抓到的是 (B,C,1,1)，所以统一 flatten(1) 成 (B,C)。
    """

    def __init__(self, model: nn.Module, head_name: str) -> None:
        self.head = model.get_submodule(head_name)
        self._feat: torch.Tensor | None = None
        self._handle = self.head.register_forward_pre_hook(self._hook)

    def _hook(self, module: nn.Module, inputs: tuple) -> None:
        self._feat = inputs[0].flatten(1)

    def pop(self) -> torch.Tensor:
        """取出并清空缓存的特征。取不到说明 forward 没走到分类头，直接报错。"""
        feat = self._feat
        self._feat = None
        if feat is None:
            raise RuntimeError("没抓到分类头输入特征，检查 HEAD_MODULE 映射是否与模型匹配")
        return feat

    def close(self) -> None:
        self._handle.remove()


def _sq_dists(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """成对欧氏距离平方矩阵 (n,m)。float32 下 cdist 比展开式 (a²+b²−2ab) 稳定。"""
    return torch.cdist(a, b, p=2).pow(2)


def _kernel(a: torch.Tensor, b: torch.Tensor, sigma2: torch.Tensor, kind: str) -> torch.Tensor:
    """核矩阵 k(a_i, b_j)。

    RBF: k(a,b) = exp(-||a-b||² / (2σ²))，σ² 由中位数启发式给出或固定。
    """
    if kind == "linear":
        return a @ b.t()
    if kind != "rbf":
        raise ValueError(f"不认识的核 {kind}")
    return torch.exp(-_sq_dists(a, b) / (2.0 * sigma2.clamp_min(1e-12)))


def median_sigma2(feats: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """中位数启发式带宽：σ² = median(非零成对距离平方)。

    为什么要自适应而不是固定值：不同模型的骨干特征尺度差很多（ResNet18 是 512 维
    平均池化后特征，ViT 是 768 维 CLS token，量级不同），固定 σ 会让核矩阵要么
    全是 1（几乎不区分）要么全是 0（梯度消失）。中位数启发式让核的"有效带宽"
    跟着特征的实际分布走。
    返回的是 detached 标量：带宽是超参，不该参与反传。
    """
    with torch.no_grad():
        d2 = _sq_dists(feats.detach(), feats.detach())
        n = d2.size(0)
        if n <= 1:
            return torch.tensor(1.0, device=feats.device, dtype=feats.dtype)
        mask = ~torch.eye(n, dtype=torch.bool, device=d2.device)
        off_diag = d2[mask]
        if off_diag.numel() == 0:
            return torch.tensor(1.0, device=feats.device, dtype=feats.dtype)
        return off_diag.median().clamp_min(eps)


def lmmd_loss(
    feat_s: torch.Tensor,
    logits_t: torch.Tensor,
    labels_s: torch.Tensor,
    feat_t: torch.Tensor,
    num_classes: int,
    kernel: str = "rbf",
    sigma: float = 0.0,
    normalize_features: bool = False,
    grad_target_probs: bool = False,
) -> tuple[torch.Tensor, dict]:
    """局部最大均值差异（论文 Eq.6 / Eq.7 / Eq.9）。

    参数
    ----
    feat_s : (n_s, D) 源域骨干特征 φ(x^s)
    logits_t : (n_t, C) 目标域分类头输出（未 softmax）
    labels_s : (n_s,) 源域真标签
    feat_t : (n_t, D) 目标域骨干特征 φ(x^t)
    num_classes : C
    kernel : "rbf" | "linear"
    sigma : >0 用固定 σ；0 用中位数启发式
    normalize_features : 是否先把特征 L2 归一化（把对齐限制在方向上，削弱尺度差异）
    grad_target_probs : 是否让梯度穿过目标域软概率。默认 False——让梯度穿过权重的话，
        模型可以靠"把预测变尖"来降低 LMMD，而不是真的对齐特征，属于退化解。

    返回
    ----
    (loss, stats)，stats 里有 σ²、参与对齐的类别数、各类权重质量，供日志诊断。

    实现要点
    --------
    1. 权重按 Eq.7 各自归一化到 1（源域 one-hot、目标域 softmax 概率），
       这样 Eq.9 的 1/n² 系数被吸收，只需算三项二次型。
    2. 只对"本 batch 源域里真实出现过的类别"求和。若某类在源域 batch 里一个样本都
       没有，源域该项恒为 0，但目标域项 w_tᵀ K_tt w_t 不为 0——继续累加会变成一个
       纯粹的"把目标域特征往一起挤"的惩罚项，与论文语义不符，必须跳过。
    3. 核矩阵只算一次（与类别无关），循环里只做二次型，避免 C 次 O(n²) 距离计算。
    """
    # LMMD 全程在 float32 下算：autocast 的 fp16 下 exp/cdist 容易溢出成 NaN，
    # 且这一项的数值本来就不需要半精度换来的那点速度。
    feat_s = feat_s.float()
    feat_t = feat_t.float()
    logits_t = logits_t.float()

    if feat_s.dim() != 2 or feat_t.dim() != 2:
        raise ValueError(f"特征必须是二维 (n,D)，实际 {tuple(feat_s.shape)} / {tuple(feat_t.shape)}")
    if feat_s.size(1) != feat_t.size(1):
        raise ValueError(f"源域/目标域特征维度不一致 {feat_s.size(1)} vs {feat_t.size(1)}")
    if logits_t.size(0) != feat_t.size(0):
        raise ValueError(
            f"目标域 logits 与特征样本数不一致 {logits_t.size(0)} vs {feat_t.size(0)}"
        )
    if labels_s.numel() != feat_s.size(0):
        raise ValueError(f"源域标签数与特征数不一致 {labels_s.numel()} vs {feat_s.size(0)}")

    if normalize_features:
        feat_s = torch.nn.functional.normalize(feat_s, dim=1)
        feat_t = torch.nn.functional.normalize(feat_t, dim=1)

    # ---- 带宽 ----
    if sigma and sigma > 0:
        sigma2 = torch.tensor(float(sigma) ** 2, device=feat_s.device, dtype=torch.float32)
    else:
        # 源域+目标域拼一起估带宽：只用源域会把带宽估得偏离目标域的实际尺度
        sigma2 = median_sigma2(torch.cat([feat_s.detach(), feat_t.detach()], dim=0))

    # ---- 权重（Eq.7）----
    # 源域：真标签 one-hot
    ws = torch.zeros(feat_s.size(0), num_classes, device=feat_s.device, dtype=torch.float32)
    ws.scatter_(1, labels_s.view(-1, 1), 1.0)
    # 目标域：模型软预测概率（没有标签，只能用这个）
    probs_t = torch.softmax(logits_t, dim=1)
    wt = probs_t if grad_target_probs else probs_t.detach()

    # ---- 核矩阵只算一次 ----
    k_ss = _kernel(feat_s, feat_s, sigma2, kernel)
    k_tt = _kernel(feat_t, feat_t, sigma2, kernel)
    k_st = _kernel(feat_s, feat_t, sigma2, kernel)

    eps = 1e-8
    present = (ws.sum(dim=0) > 0).nonzero(as_tuple=False).flatten().tolist()
    if not present:
        zero = feat_s.sum() * 0.0
        return zero, {"sigma2": float(sigma2), "n_classes": 0, "n_present": 0}

    total = feat_s.new_zeros(())
    mass_s_list: list[float] = []
    mass_t_list: list[float] = []
    for c in present:
        ws_c = ws[:, c]
        wt_c = wt[:, c]
        # Eq.7：各自除以该类权重和，使 Σw = 1
        ws_c = ws_c / ws_c.sum().clamp_min(eps)
        wt_c = wt_c / wt_c.sum().clamp_min(eps)
        term = (
            ws_c @ k_ss @ ws_c
            + wt_c @ k_tt @ wt_c
            - 2.0 * (ws_c @ k_st @ wt_c)
        )
        total = total + term
        mass_s_list.append(float(ws[:, c].sum()))
        mass_t_list.append(float(wt[:, c].sum()))

    loss = total / float(num_classes)
    stats = {
        "sigma2": float(sigma2),
        "n_classes": len(present),
        "n_present": len(present),
        # 目标域该类总概率质量：偏小说明模型几乎不认为目标域有这一类，
        # 此时归一化后权重会把极少数样本放大成"整类代表"，是对齐噪声的来源
        "min_target_mass": min(mass_t_list),
        "mean_target_mass": sum(mass_t_list) / len(mass_t_list),
    }
    return loss, stats


def evaluate_source(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """源域（PlantVillage val）评估：loss / acc。"""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += float(criterion(logits, y).item()) * y.size(0)
            correct += int((logits.argmax(1) == y).sum().item())
            total += y.size(0)
    model.train()
    return {"loss": loss_sum / max(total, 1), "acc": correct / max(total, 1), "n": total}


def evaluate_target(
    model: nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
) -> dict:
    """目标域（PlantDoc）评估，复用 eval_crossdomain 的类别映射与评估口径。

    为什么不自己写映射：PLANTDOC_TO_PV_FOLDER 是评估口径的唯一真源，
    复制一份必然有一天两边不同步，导致"训练时看到的目标域精度"和最终报告不一致。
    """
    import eval_crossdomain  # 延迟导入：只有开 --eval-target 才需要

    return eval_crossdomain.evaluate(model, loader, device, num_classes)


def infinite(loader):
    """无限循环取 batch。目标域只有 2670 张、源域 39170 张，两边长度差 15 倍，
    所以目标域必须循环取，否则每轮只能对齐前 1/15 的源域数据。"""
    while True:
        for batch in loader:
            yield batch


def head_prefixes(model_name: str) -> list[str]:
    """给优化器分组用：分类头 lr=lr，其余（骨干）lr=lr*0.1，与 train_colab.py 一致。"""
    name = HEAD_MODULE.get(model_name.lower())
    if name is None:
        raise ValueError(f"不认识的模型 {model_name}，可选 {sorted(HEAD_MODULE)}")
    return [name]


def main() -> int:
    ap = argparse.ArgumentParser(description="LMMD 子域对齐训练（MSUN 论文第②模块）")
    for k, v in DEFAULTS.items():
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            ap.add_argument(flag, action="store_true", default=v)
            ap.add_argument(f"--no-{k.replace('_', '-')}", dest=k, action="store_false")
        else:
            ap.add_argument(flag, type=type(v), default=v)
    # parse_known_args：colab / kaggle exec 会往 sys.argv 塞 `-f kernel-xxx.json`
    args, unknown = ap.parse_known_args()
    if unknown:
        log(f"忽略未知参数: {unknown}")

    import numpy as np
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    # ---------------- 数据 ----------------
    src_root = Path(args.data_root)
    tgt_root = Path(args.target_root)
    for root, tag in ((src_root, "源域"), (tgt_root, "目标域")):
        if not root.exists():
            log(f"错误：{tag}目录不存在 {root}")
            return 2

    classes = [d.name for d in sorted((src_root / "train").iterdir()) if d.is_dir()]
    num_classes = len(classes)
    log(f"源域类别数 {num_classes}（{classes[0]} … {classes[-1]}）")

    src_train = ImageFolder(
        str(src_root / "train"),
        transform=build_transforms(args.img_size, True, args.strong_aug),
    )
    src_val = ImageFolder(
        str(src_root / "val"), transform=build_transforms(args.img_size, False)
    )
    # 目标域用 ImageFolder 读，标签在训练里直接丢弃（无监督域适应设定）
    tgt_train = ImageFolder(
        str(tgt_root / "train"),
        transform=build_transforms(args.img_size, args.target_aug, args.strong_aug),
    )
    log(f"源域 train={len(src_train)} val={len(src_val)}；目标域 train={len(tgt_train)}（标签丢弃）")

    if len(tgt_train) == 0:
        log("错误：目标域训练集为空")
        return 2

    tgt_bs = args.target_batch_size or args.batch_size
    pin = device.type == "cuda"
    common = dict(num_workers=args.num_workers, pin_memory=pin,
                  persistent_workers=args.num_workers > 0)
    src_loader = DataLoader(src_train, batch_size=args.batch_size, shuffle=True,
                            drop_last=True, **common)
    src_val_loader = DataLoader(src_val, batch_size=args.batch_size, shuffle=False, **common)
    tgt_loader = DataLoader(tgt_train, batch_size=tgt_bs, shuffle=True, drop_last=True, **common)
    log(f"源域 batch={args.batch_size}，目标域 batch={tgt_bs}，"
        f"每轮 {len(src_loader)} 步（目标域循环取）")

    tgt_eval_loader = None
    if args.eval_target:
        import eval_crossdomain

        tgt_eval_ds = eval_crossdomain.PlantDocDataset(
            root=tgt_root,
            split=args.target_eval_split,
            pv_folders=classes,
            transform=build_transforms(args.img_size, False),
            limit_per_class=args.target_eval_limit,
        )
        if len(tgt_eval_ds) == 0:
            log("警告：目标域评估集为空，关闭目标域评估")
            args.eval_target = False
        else:
            tgt_eval_loader = DataLoader(
                tgt_eval_ds, batch_size=args.batch_size, shuffle=False, **common
            )
            log(f"目标域评估集 {len(tgt_eval_ds)} 张（split={args.target_eval_split}）")

    # ---------------- 模型 ----------------
    model, meta = build_model(args.model, num_classes, not args.no_pretrained)
    model = model.to(device)
    log(f"模型 {meta.name}（{meta.family}）参数量 {meta.params_m:.2f}M")

    head_names = head_prefixes(args.model)
    tap = FeatureTap(model, HEAD_MODULE[args.model.lower()])
    # 先跑一次前向确认 hook 真的能抓到特征（模型/映射不匹配时立刻报错，
    # 而不是训练到第一个 LMMD 才崩）
    with torch.no_grad():
        probe = torch.zeros(2, 3, args.img_size, args.img_size, device=device)
        model(probe)
        probe_feat = tap.pop()
    log(f"φ 维度 {probe_feat.shape[1]}（分类头 {HEAD_MODULE[args.model.lower()]} 的输入）")

    # ---------------- 优化器 / 调度器 ----------------
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
    log(f"优化器: 骨干 {len(backbone_params)} 个张量 lr={args.lr * 0.1:.2e}, "
        f"头 {len(head_params)} 个张量 lr={args.lr:.2e}")

    steps_per_epoch = len(src_loader)
    if args.limit_batches:
        steps_per_epoch = min(steps_per_epoch, args.limit_batches)
    total_steps = max(args.epochs * steps_per_epoch, 1)
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    amp_enabled = device.type == "cuda"

    # ---------------- 产物目录 ----------------
    run_name = args.run_name or f"lmmd_{args.model}_{time.strftime('%m%d_%H%M')}"
    out = Path(args.out_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    (out / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    log(f"输出目录 {out}")

    # ---------------- 训练 ----------------
    tgt_iter = infinite(tgt_loader)
    hist: list[dict] = []
    best_src = 0.0
    best_tgt = 0.0
    global_step = 0
    # 日志间隔：正常训练每 50 步一条，冒烟（步数很少）时至少打 5 条，否则看不出趋势
    log_every = max(1, min(50, steps_per_epoch // 5))

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        # λ 线性升到设定值：早期模型在目标域上的软概率基本是噪声，
        # 一开始就给大 λ 会把骨干特征往错误方向拽，这是 LMMD 类方法常见的崩法
        if args.lmmd_warmup_epochs > 0:
            lam = args.lmmd_weight * min(1.0, (epoch + 1) / args.lmmd_warmup_epochs)
        else:
            lam = args.lmmd_weight

        ce_sum, lm_sum, tot_sum, correct, total = 0.0, 0.0, 0.0, 0, 0
        sigma_hist: list[float] = []
        mass_hist: list[float] = []
        n_batch = 0

        for bi, (x_s, y_s) in enumerate(src_loader):
            if args.limit_batches and bi >= args.limit_batches:
                break
            x_t, _y_t_discard = next(tgt_iter)  # 目标域标签丢弃，无监督设定
            x_s = x_s.to(device, non_blocking=True)
            y_s = y_s.to(device, non_blocking=True)
            x_t = x_t.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits_s = model(x_s)
                feat_s = tap.pop()
                logits_t = model(x_t)
                feat_t = tap.pop()
                ce = criterion(logits_s, y_s)

            # LMMD 单独关掉 autocast 在 float32 下算
            with torch.autocast(device_type="cuda", enabled=False):
                lmmd, lm_stats = lmmd_loss(
                    feat_s=feat_s,
                    logits_t=logits_t,
                    labels_s=y_s,
                    feat_t=feat_t,
                    num_classes=num_classes,
                    kernel=args.lmmd_kernel,
                    sigma=args.lmmd_sigma,
                    normalize_features=args.lmmd_normalize_features,
                    grad_target_probs=args.lmmd_grad_target_probs,
                )
            loss = ce + lam * lmmd

            if not torch.isfinite(loss):
                log(f"  警告：ep{epoch + 1} b{bi} 损失非有限（ce={float(ce):.4f} "
                    f"lmmd={float(lmmd):.4f} σ²={lm_stats.get('sigma2')}），跳过该 batch")
                opt.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            global_step += 1

            bs = y_s.size(0)
            ce_sum += float(ce.item()) * bs
            lm_sum += float(lmmd.item()) * bs
            tot_sum += float(loss.item()) * bs
            correct += int((logits_s.argmax(1) == y_s).sum().item())
            total += bs
            sigma_hist.append(lm_stats.get("sigma2", float("nan")))
            mass_hist.append(lm_stats.get("mean_target_mass", float("nan")))
            n_batch += 1

            if bi % log_every == 0:
                log(f"  ep{epoch + 1} b{bi}/{steps_per_epoch} ce={float(ce):.4f} "
                    f"lmmd={float(lmmd):.4f} λ={lam:.3f} "
                    f"σ²={lm_stats.get('sigma2', 0):.3g} "
                    f"类数={lm_stats.get('n_classes', 0)} "
                    f"acc={correct / max(total, 1):.4f}")

        # ---------------- 每轮评估 ----------------
        val = evaluate_source(model, src_val_loader, criterion, device)
        tgt = {"top1_acc": float("nan"), "top3_acc": float("nan"), "n": 0}
        if tgt_eval_loader is not None:
            tgt = evaluate_target(model, tgt_eval_loader, device, num_classes)
            model.train()

        secs = time.time() - t0
        lr_now = opt.param_groups[1]["lr"]
        row = {
            "epoch": epoch,
            "lambda": round(lam, 6),
            "train_ce": round(ce_sum / max(total, 1), 6),
            "train_lmmd": round(lm_sum / max(total, 1), 6),
            "train_total": round(tot_sum / max(total, 1), 6),
            "train_acc": round(correct / max(total, 1), 6),
            "val_loss": round(val["loss"], 6),
            "val_acc": round(val["acc"], 6),
            "target_top1": round(float(tgt["top1_acc"]), 6),
            "target_top3": round(float(tgt["top3_acc"]), 6),
            "sigma2_mean": round(float(np.nanmean(sigma_hist)) if sigma_hist else 0.0, 6),
            "target_mass_mean": round(float(np.nanmean(mass_hist)) if mass_hist else 0.0, 6),
            "lr": lr_now,
            "secs": round(secs, 1),
        }
        hist.append(row)
        log(
            f"epoch {epoch + 1}/{args.epochs} λ={lam:.3f} "
            f"ce={row['train_ce']:.4f} lmmd={row['train_lmmd']:.4f} "
            f"train_acc={row['train_acc']:.4f} val_acc={row['val_acc']:.4f} "
            f"target_top1={row['target_top1']:.4f} "
            f"σ²={row['sigma2_mean']:.3g} lr={lr_now:.2e} {secs:.0f}s"
        )

        # 两个最优各存一份：按源域 val 选会挑出"实验室里最准"的权重，
        # 但我们的目标是跨域，所以按目标域 top1 选的权重同样要留，最后分别报告
        if val["acc"] > best_src:
            best_src = val["acc"]
            torch.save(
                {"model": model.state_dict(), "classes": classes, "epoch": epoch,
                 "val_acc": val["acc"], "target_top1": row["target_top1"],
                 "args": vars(args)},
                out / "best_source.pt",
            )
            log(f"  保存 best_source.pt (val_acc={best_src:.4f})")
        if tgt_eval_loader is not None and float(tgt["top1_acc"]) > best_tgt:
            best_tgt = float(tgt["top1_acc"])
            torch.save(
                {"model": model.state_dict(), "classes": classes, "epoch": epoch,
                 "val_acc": val["acc"], "target_top1": best_tgt,
                 "args": vars(args)},
                out / "best_target.pt",
            )
            log(f"  保存 best_target.pt (target_top1={best_tgt:.4f})")

    tap.close()
    torch.save({"model": model.state_dict(), "classes": classes, "epoch": args.epochs - 1,
                "args": vars(args)}, out / "last.pt")

    # ---------------- 汇总 ----------------
    summary = {
        "method": "LMMD",
        "model": args.model,
        "params_m": round(meta.params_m, 3),
        "family": meta.family,
        "img_size": args.img_size,
        "seed": args.seed,
        "epochs": args.epochs,
        "lmmd_weight": args.lmmd_weight,
        "lmmd_kernel": args.lmmd_kernel,
        "lmmd_sigma": args.lmmd_sigma,
        "lmmd_normalize_features": args.lmmd_normalize_features,
        "lmmd_grad_target_probs": args.lmmd_grad_target_probs,
        "lmmd_warmup_epochs": args.lmmd_warmup_epochs,
        "batch_size": args.batch_size,
        "target_batch_size": args.target_batch_size or args.batch_size,
        "pretrained": not args.no_pretrained,
        "strong_aug": args.strong_aug,
        "target_aug": args.target_aug,
        "n_source_train": len(src_train),
        "n_source_val": len(src_val),
        "n_target_train": len(tgt_train),
        "n_target_eval": int(tgt["n"]),
        "best_source_val_acc": round(best_src, 6),
        "best_target_top1": round(best_tgt, 6),
        "final_target_top1": round(float(tgt["top1_acc"]), 6),
        "final_target_top3": round(float(tgt["top3_acc"]), 6),
        "phi_dim": int(probe_feat.shape[1]),
        "head_module": HEAD_MODULE[args.model.lower()],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                     encoding="utf-8")

    with open(out / "history.csv", "w", encoding="utf-8") as f:
        f.write("epoch,lambda,train_ce,train_lmmd,train_total,train_acc,val_loss,val_acc,"
                "target_top1,target_top3,sigma2_mean,target_mass_mean,lr,secs\n")
        for h in hist:
            f.write(
                f"{h['epoch']},{h['lambda']},{h['train_ce']},{h['train_lmmd']},"
                f"{h['train_total']},{h['train_acc']},{h['val_loss']},{h['val_acc']},"
                f"{h['target_top1']},{h['target_top3']},{h['sigma2_mean']},"
                f"{h['target_mass_mean']},{h['lr']:.8f},{h['secs']}\n"
            )

    log("RESULT_SUMMARY " + json.dumps(summary, ensure_ascii=False))
    log(f"全部完成，产物在 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
