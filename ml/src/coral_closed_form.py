"""闭式实验：在冻结特征上直接检验 CORAL 协方差对齐是否有用。

不训练任何网络，只做矩阵运算：
  A. 源域特征训 LR → 目标域测（无对齐，作为对照）
  B. CORAL 白化后再训 LR（用目标域无标签协方差，符合无监督设定）
  C. 目标域标签训 LR（上界，仅用于定位瓶颈）

如果 B 明显优于 A，说明「对齐二阶统计量」这条思路在这些特征上成立；
如果 B ≈ A，说明 CORAL 大概率也救不了，应该把精力放在别处。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import build_transforms  # noqa: E402
from models import build_model  # noqa: E402

PV_ROOT = "data/processed/plantvillage"
PD_FEAT = "runs_smoke/pd_feats.npy"
PD_LABEL = "runs_smoke/pd_labels.npy"
CKPT = "runs/resnet18_224_15ep/best.pt"
OUT = "runs_smoke/coral_closed_form.json"


def log(m: str) -> None:
    print(f"[coral-cf] {m}", flush=True)


def extract(model, root: str, device, limit: int = 0) -> tuple[np.ndarray, np.ndarray]:
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    ds = ImageFolder(root, transform=build_transforms(224, False))
    if limit and limit < len(ds):
        # 分层抽样，保证每类都有
        rng = np.random.default_rng(42)
        targets = np.array(ds.targets)
        keep = []
        per = max(1, limit // len(ds.classes))
        for c in range(len(ds.classes)):
            idx = np.where(targets == c)[0]
            keep.extend(rng.choice(idx, size=min(per, len(idx)), replace=False).tolist())
        ds = torch.utils.data.Subset(ds, sorted(keep))
        log(f"  分层抽样 {len(ds)} 张（每类 {per}）")

    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    feats, labels = [], []
    model.eval()
    with torch.no_grad():
        for bi, (x, y) in enumerate(dl):
            x = x.to(device)
            feats.append(model(x).float().cpu().numpy())
            labels.append(y.numpy())
            if bi % 20 == 0:
                log(f"    {bi * 64}/{len(ds)}")
    return np.concatenate(feats), np.concatenate(labels)


def coral_whiten(fit_x: np.ndarray, apply_x: np.ndarray, eps: float = 1e-3):
    """用 fit_x 的协方差白化 apply_x（各自白化到单位协方差）。

    注意：这不是经典 CORAL 变换，只是「把两边都变成标准正态」。
    仅作为对照，看「抹掉全部二阶结构」的代价。
    """
    mu = fit_x.mean(0, keepdims=True)
    cov = np.cov(fit_x - mu, rowvar=False)
    d = cov.shape[0]
    cov += np.eye(d) * eps * np.trace(cov) / d
    w, v = np.linalg.eigh(cov)
    w = np.maximum(w, 1e-12)
    # 白化： (X-mu) @ V @ diag(w^-1/2) @ V^T
    whiten = v @ np.diag(w ** -0.5) @ v.T
    return (apply_x - mu) @ whiten


def _cov_sqrt(x: np.ndarray, sign: float, eps: float = 1e-3):
    """返回 Cov(x)^sign 的矩阵平方根（sign=+0.5 或 -0.5）。"""
    mu = x.mean(0, keepdims=True)
    cov = np.cov(x - mu, rowvar=False)
    d = cov.shape[0]
    cov = cov + np.eye(d) * eps * np.trace(cov) / d
    w, v = np.linalg.eigh(cov)
    w = np.maximum(w, 1e-12)
    return mu, v @ np.diag(w ** sign) @ v.T


def coral_transform(src: np.ndarray, tgt: np.ndarray, eps: float = 1e-3):
    """经典 CORAL 重着色：把目标域的二阶统计量搬成源域的。

        Xt' = (Xt - mu_t) @ Ct^{-1/2} @ Cs^{1/2} + mu_s

    这才是 CORAL 论文里真正做的事（不是各自白化）。训练时的 CORAL loss
    就是在用梯度逼近这个变换的效果，所以这个闭式版本是它的上界近似。
    """
    mu_s, cs_half = _cov_sqrt(src, 0.5, eps)
    mu_t, ct_neg_half = _cov_sqrt(tgt, -0.5, eps)
    return (tgt - mu_t) @ ct_neg_half @ cs_half + mu_s


def main() -> int:
    import json

    import torch.nn as nn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    device = torch.device("cpu")
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model, _ = build_model(ck["args"]["model"], len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["model"])
    model.fc = nn.Identity()
    model.to(device).eval()
    log(f"加载 {CKPT}，骨干输出 512 维")

    log("提取源域特征（PlantVillage val，分层抽样）")
    xs, ys = extract(model, f"{PV_ROOT}/val", device, limit=3000)
    log(f"  源域 {xs.shape}")

    log("加载目标域特征")
    xt_all = np.load(PD_FEAT)
    yt_all = np.load(PD_LABEL)
    log(f"  目标域 {xt_all.shape}")

    # 目标域一半用于拟合无标签协方差 + 上界探针，另一半用于评估
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(xt_all))
    half = len(idx) // 2
    tr_i, te_i = idx[:half], idx[half:]
    xt_fit, xt_te = xt_all[tr_i], xt_all[te_i]
    yt_fit, yt_te = yt_all[tr_i], yt_all[te_i]

    results = {}

    # A. 无对齐：源域训 → 目标域测
    lr = LogisticRegression(max_iter=3000, C=1.0)
    lr.fit(xs, ys)
    a = accuracy_score(yt_te, lr.predict(xt_te))
    results["A_source_only"] = round(float(a), 4)
    log(f"A 无对齐: {a:.4f}")

    # B. CORAL 白化：各自白化到单位协方差（抹掉全部二阶结构，对照）
    log("B 各自白化（用目标域无标签半集的协方差）")
    xs_w = coral_whiten(xs, xs)
    xt_te_w = coral_whiten(xt_fit, xt_te)
    lr2 = LogisticRegression(max_iter=3000, C=1.0)
    lr2.fit(xs_w, ys)
    b = accuracy_score(yt_te, lr2.predict(xt_te_w))
    results["B_whiten_both"] = round(float(b), 4)
    log(f"B 各自白化: {b:.4f}  ({b - a:+.4f} vs A)")

    # B2. 经典 CORAL 重着色：目标域协方差 → 源域协方差
    log("B2 经典 CORAL 重着色（目标域对齐到源域二阶矩）")
    xt_te_c = coral_transform(xs, xt_fit)
    xt_te_c = coral_transform(xs, xt_te)  # 用同一源域统计量变换测试半集
    lr2b = LogisticRegression(max_iter=3000, C=1.0)
    lr2b.fit(xs, ys)
    b2 = accuracy_score(yt_te, lr2b.predict(xt_te_c))
    results["B2_coral_recolor"] = round(float(b2), 4)
    log(f"B2 经典 CORAL: {b2:.4f}  ({b2 - a:+.4f} vs A)")

    # C. 上界：目标域标签训 → 目标域测
    lr3 = LogisticRegression(max_iter=3000, C=1.0)
    lr3.fit(xt_fit, yt_fit)
    c = accuracy_score(yt_te, lr3.predict(xt_te))
    results["C_target_upper"] = round(float(c), 4)
    log(f"C 目标域上界: {c:.4f}")

    # D. 源域 + 目标域特征拼接训练（只做诊断，需目标标签，非无监督）
    lr4 = LogisticRegression(max_iter=3000, C=1.0)
    lr4.fit(np.vstack([xs, xt_fit]), np.concatenate([ys, yt_fit]))
    d = accuracy_score(yt_te, lr4.predict(xt_te))
    results["D_joint_upper"] = round(float(d), 4)
    log(f"D 联合训练上界: {d:.4f}")

    results["delta_whiten"] = round(float(b - a), 4)
    results["delta_coral"] = round(float(b2 - a), 4)
    Path(OUT).write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"已写 {OUT}")
    log(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
