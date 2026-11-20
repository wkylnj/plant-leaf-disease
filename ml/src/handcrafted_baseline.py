"""手写特征基线：不用任何预训练权重，纯手工特征 + 线性分类器。

动机来自 iWSP 论文的视觉皮层实验（arXiv:2611.18114）：该文用 14 个显式计算
（Gabor 能量、颜色对比、轮廓曲率等）产出约 1 万维特征，配合 ridge regression
在神经科学基准上打败全部经典可解释基线，但仍略低于黑箱 AlexNet。

本项目做同样的对照，回答一个报告里绕不开的问题：
    「ImageNet 预训练那 1100 万个学到的参数，到底比人手设计的特征强多少？」

三组特征：
    1. 颜色直方图（HSV，每通道 32 bin）+ 颜色矩 —— 病斑的核心表现是变色
    2. GLCM 纹理（灰度共生矩阵：对比度/相关性/能量/同质性）—— 病斑的粗糙质地
    3. Gabor 滤波器组响应 —— 病斑边缘与叶脉方向性

用法:
    python ml/src/handcrafted_baseline.py                 # 全量（约 20-40 分钟，CPU）
    python ml/src/handcrafted_baseline.py --limit 3000    # 快速验证
    python ml/src/handcrafted_baseline.py --features color,glcm
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "ml" / "data" / "processed" / "plantvillage"
ARTIFACT_DIR = PROJECT_ROOT / "ml" / "artifacts"


# ---------------------------------------------------------------- 特征提取

def feat_color(img_bgr: np.ndarray) -> np.ndarray:
    """HSV 颜色直方图 + 一阶颜色矩。

    病斑的典型表现是局部变黄/褐/黑，与健康组织的绿色分布显著不同，
    因此颜色直方图是这类任务最强的单一线索。
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    feats = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [32], [0, 256]).ravel()
        hist = hist / (hist.sum() + 1e-8)  # 归一化成分布，消除叶片大小影响
        feats.append(hist)
        feats.append([hsv[:, :, ch].mean() / 255.0, hsv[:, :, ch].std() / 255.0])
    return np.concatenate([np.asarray(f, dtype=np.float32).ravel() for f in feats])


def feat_glcm(img_bgr: np.ndarray) -> np.ndarray:
    """灰度共生矩阵纹理特征。

    取 4 个方向（0/45/90/135 度）与 2 个距离，对 4 个属性取均值与标准差，
    描述病斑区域的粗糙度与规律性。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 量化到 16 级：级数越少，共生矩阵越稠密，对光照变化越鲁棒
    gray = (gray / 16).astype(np.uint8)
    glcm = graycomatrix(gray, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=16, symmetric=True, normed=True)
    feats = []
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
        vals = graycoprops(glcm, prop)  # shape: (n_dist, n_angle)
        feats.extend([vals.mean(), vals.std()])
    return np.asarray(feats, dtype=np.float32)


_GABOR_KERNELS: list[np.ndarray] = []


def _build_gabor_kernels() -> list[np.ndarray]:
    """4 个尺度 × 6 个方向的 Gabor 核，覆盖不同宽度的病斑边缘。"""
    global _GABOR_KERNELS
    if _GABOR_KERNELS:
        return _GABOR_KERNELS
    kernels = []
    for sigma in (1.5, 3.0, 5.0, 7.0):
        lam = sigma * 2.2
        for theta in np.arange(0, np.pi, np.pi / 6):
            kern = cv2.getGaborKernel((15, 15), sigma, theta, lam, 0.5, 0, ktype=cv2.CV_32F)
            kern -= kern.mean()  # 去直流，避免响应被整体亮度主导
            kernels.append(kern)
    _GABOR_KERNELS = kernels
    return kernels


def feat_gabor(img_bgr: np.ndarray) -> np.ndarray:
    """Gabor 响应统计：每个核的均值与标准差，共 24 核 × 2 = 48 维。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    feats = []
    for kern in _build_gabor_kernels():
        resp = cv2.filter2D(gray, cv2.CV_32F, kern)
        feats.extend([float(np.abs(resp).mean()), float(resp.std())])
    return np.asarray(feats, dtype=np.float32)


EXTRACTORS = {"color": feat_color, "glcm": feat_glcm, "gabor": feat_gabor}


# ---------------------------------------------------------------- 数据读取

def list_samples(split_dir: Path, limit_per_class: int | None, seed: int) -> tuple[list[Path], list[int], list[str]]:
    classes = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
    rng = np.random.default_rng(seed)
    paths: list[Path] = []
    labels: list[int] = []
    for idx, name in enumerate(classes):
        files = sorted(p for p in (split_dir / name).iterdir() if p.suffix.lower() in IMG_EXT)
        if limit_per_class is not None and len(files) > limit_per_class:
            picked = rng.choice(len(files), size=limit_per_class, replace=False)
            files = [files[i] for i in sorted(picked)]
        paths.extend(files)
        labels.extend([idx] * len(files))
    return paths, labels, classes


def imread_unicode(path: Path) -> np.ndarray | None:
    """读取图像，兼容 Windows 下的非 ASCII 路径。

    cv2.imread 在 Windows 上走 ANSI 路径，遇到中文目录会静默返回 None，
    因此改用 numpy 读字节流再让 OpenCV 解码。
    """
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def extract(paths: list[Path], features: list[str], img_size: int, verbose: bool = True) -> np.ndarray:
    mats = []
    t0 = time.time()
    for i, p in enumerate(paths):
        img = imread_unicode(p)
        if img is None:
            raise RuntimeError(f"无法读取图像: {p}")
        if img.shape[0] != img_size or img.shape[1] != img_size:
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        vec = np.concatenate([EXTRACTORS[f](img) for f in features])
        mats.append(vec)
        if verbose and (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"  已处理 {i + 1}/{len(paths)}  ({elapsed:.0f}s, {(i + 1) / elapsed:.0f} 张/秒)")
    return np.vstack(mats)


# ---------------------------------------------------------------- 主流程

def main() -> int:
    parser = argparse.ArgumentParser(description="手写特征基线")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--features", default="color,glcm,gabor",
                        help="逗号分隔，可选 color,glcm,gabor")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--limit-per-class", type=int, default=None,
                        help="每类最多用多少张（调试用）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--out-name", default="handcrafted_baseline")
    args = parser.parse_args()

    features = [f.strip() for f in args.features.split(",") if f.strip()]
    unknown = [f for f in features if f not in EXTRACTORS]
    if unknown:
        print(f"[错误] 未知特征: {unknown}，可选 {list(EXTRACTORS)}")
        return 1

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    data_root: Path = args.data_root
    if not data_root.is_dir():
        print(f"[错误] 数据目录不存在: {data_root}")
        return 1

    print(f"特征组合: {features}")
    print("读取 train ...")
    tr_paths, tr_labels, classes = list_samples(data_root / "train", args.limit_per_class, args.seed)
    print(f"  train {len(tr_paths)} 张")
    print("读取 test ...")
    te_paths, te_labels, _ = list_samples(data_root / "test", args.limit_per_class, args.seed)
    print(f"  test  {len(te_paths)} 张")

    t0 = time.time()
    print("提取 train 特征 ...")
    X_tr = extract(tr_paths, features, args.img_size)
    print("提取 test 特征 ...")
    X_te = extract(te_paths, features, args.img_size)
    feat_secs = time.time() - t0
    print(f"特征维度: {X_tr.shape[1]}，耗时 {feat_secs:.1f}s")

    clf = Pipeline([
        ("scaler", StandardScaler()),
        # 用 saga + L2，兼顾速度与多分类稳定性；class_weight 处理 36 倍不均衡
        ("clf", LogisticRegression(max_iter=args.max_iter, C=1.0,
                                   class_weight="balanced")),
    ])
    t1 = time.time()
    print("训练 LogisticRegression ...")
    clf.fit(X_tr, tr_labels)
    train_secs = time.time() - t1

    print("评估 ...")
    pred = clf.predict(X_te)
    acc = accuracy_score(te_labels, pred)
    macro_f1 = f1_score(te_labels, pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(te_labels, pred, average=None, zero_division=0)

    print("")
    print(f"test accuracy = {acc:.4f}")
    print(f"test macro-F1 = {macro_f1:.4f}")
    print(f"（多数类基线 accuracy = {max(np.bincount(tr_labels)) / len(tr_labels):.4f}）")

    worst = np.argsort(per_class_f1)[:5]
    print("")
    print("最难识别的 5 个类：")
    for i in worst:
        print(f"  {classes[i]:45s} F1={per_class_f1[i]:.3f}")

    report = classification_report(te_labels, pred, target_names=classes,
                                   digits=4, zero_division=0)
    cm = confusion_matrix(te_labels, pred)

    np.savez_compressed(ARTIFACT_DIR / f"{args.out_name}_preds.npz",
                        labels=np.array(te_labels), preds=pred, probs=None)
    (ARTIFACT_DIR / f"{args.out_name}_report.txt").write_text(
        f"features={features}\nimg_size={args.img_size}\n"
        f"accuracy={acc:.6f}\nmacro_f1={macro_f1:.6f}\n"
        f"feature_dim={X_tr.shape[1]}\n"
        f"feature_secs={feat_secs:.1f}\ntrain_secs={train_secs:.1f}\n\n" + report,
        encoding="utf-8",
    )

    summary = {
        "features": features,
        "img_size": args.img_size,
        "feature_dim": int(X_tr.shape[1]),
        "train_size": len(tr_paths),
        "test_size": len(te_paths),
        "accuracy": round(float(acc), 6),
        "macro_f1": round(float(macro_f1), 6),
        "feature_secs": round(feat_secs, 1),
        "train_secs": round(train_secs, 1),
        "per_class_f1": {classes[i]: round(float(per_class_f1[i]), 4) for i in range(len(classes))},
        "hardest": [
            {"class": classes[i], "f1": round(float(per_class_f1[i]), 4)} for i in worst
        ],
    }
    (ARTIFACT_DIR / f"{args.out_name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("")
    print(f"已写出 {ARTIFACT_DIR.relative_to(PROJECT_ROOT).as_posix()}/{args.out_name}_*.json|txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
