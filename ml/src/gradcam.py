"""E7 Grad-CAM：模型到底在看病斑还是背景？

为什么需要这个实验：
    PlantVillage 是实验室摆拍数据集 —— 背景统一浅色、叶片居中、光照一致。
    模型在测试集上能到 99%，但很可能学的是"背景 + 叶片轮廓"这条捷径，
    而不是真正的病斑形态。这解释了为什么真实田间照片上精度会崩。

    Grad-CAM 把模型判断时"看得最重的位置"画成热力图。如果热力图亮在
    叶片外（背景/角落），说明模型在作弊；如果亮在病斑上，说明学到了东西。

除了热力图，本脚本还给两个可量化指标（报告里比"看图说话"有力得多）：
    1. 3×3 网格 CAM 质量分布 —— 模型注意力落在九宫格哪些格子里
    2. 中心区域 vs 四角区域 的注意力占比 —— 叶片在中心、背景在四角，
       若四角占比高就是明确的捷径证据

用法：
    python src/gradcam.py --run-dir runs/resnet18_224_15ep
    python src/gradcam.py --run-dir runs/resnet18_224_15ep --per-class 1 --classes 0,7,22
    python src/gradcam.py --run-dir runs/resnet18_224_15ep --num 60

输出：
    ml/runs/<run>/gradcam/overlay_<idx>_<label>.png   单图热力图
    ml/runs/<run>/gradcam/grid.png                    总览网格
    ml/runs/<run>/gradcam/summary.json                量化指标
    ml/artifacts/gradcam.md                           分析文档（表格）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data import build_transforms
from models import build_model

ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = ROOT.parent

# 各模型最后一层卷积（或等效层）的定位方式
TARGET_LAYER = {
    "resnet18": ("layer4", -1),
    "resnet50": ("layer4", -1),
    "mobilenet_v3_large": ("features", -1),
    "mobilenet_v3_small": ("features", -1),
    "shufflenet_v2_x1_0": ("conv5", None),
    "efficientnet_b0": ("features", -1),
}


def get_target_module(model, model_name: str):
    """返回要挂 hook 的模块。"""
    if model_name not in TARGET_LAYER:
        raise SystemExit(
            f"{model_name} 暂不支持 Grad-CAM（当前支持 {sorted(TARGET_LAYER)}）。\n"
            f"Swin/ViT 类需要把 token 序列 reshape 回 2D 特征图，另行处理。"
        )
    attr, idx = TARGET_LAYER[model_name]
    mod = getattr(model, attr)
    if idx is not None:
        mod = mod[idx]
    return mod


class GradCAM:
    """标准 Grad-CAM：梯度加权 + ReLU + 归一化。"""

    def __init__(self, model, target_module):
        self.model = model
        self.acts: torch.Tensor | None = None
        self.grads: torch.Tensor | None = None
        self._fh = target_module.register_forward_hook(self._save_act)
        self._bh = target_module.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.acts = out

    def _save_grad(self, module, grad_in, grad_out):
        self.grads = grad_out[0]

    def remove(self):
        self._fh.remove()
        self._bh.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        """返回 (cam[H,W] 归一化到 0-1, 预测类别, 概率)。"""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = torch.softmax(logits.float(), dim=1)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        score = logits[0, class_idx]
        score.backward()

        assert self.acts is not None and self.grads is not None
        acts = self.acts.detach()[0]      # [C,H,W]
        grads = self.grads.detach()[0]    # [C,H,W]

        # 通道权重 = 梯度的全局平均池化
        weights = grads.mean(dim=(1, 2), keepdim=True)  # [C,1,1]
        cam = (weights * acts).sum(dim=0)               # [H,W]
        cam = F.relu(cam)

        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy(), class_idx, float(probs[0, class_idx])


def cam_grid_mass(cam: np.ndarray) -> list[float]:
    """把 CAM 按 3×3 切块，返回每格占总能量的比例（行优先，9 个数）。"""
    h, w = cam.shape
    hs = np.array_split(np.arange(h), 3)
    ws = np.array_split(np.arange(w), 3)
    total = float(cam.sum())
    if total <= 0:
        return [0.0] * 9
    out = []
    for ri in hs:
        for ci in ws:
            out.append(float(cam[np.ix_(ri, ci)].sum() / total))
    return out


def corner_vs_center(cam: np.ndarray) -> dict:
    """四角区域 vs 中心区域的注意力占比。

    叶片在 PlantVillage 里居中且占大部分面积，四角几乎全是背景。
    所以"四角占比"越高，越说明模型在看背景。
    """
    h, w = cam.shape
    # 中心 50% 区域
    h0, h1 = int(h * 0.25), int(h * 0.75)
    w0, w1 = int(w * 0.25), int(w * 0.75)
    total = float(cam.sum())
    if total <= 0:
        return {"center": 0.0, "corners": 0.0}
    center = float(cam[h0:h1, w0:w1].sum() / total)

    # 四角各取 25%×25%
    ch, cw = int(h * 0.25), int(w * 0.25)
    corners = (
        cam[:ch, :cw].sum()
        + cam[:ch, -cw:].sum()
        + cam[-ch:, :cw].sum()
        + cam[-ch:, -cw:].sum()
    )
    return {"center": center, "corners": float(corners / total)}


def overlay(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """把 CAM 用 jet 色图叠到原图上。纯 numpy 实现 jet，避免依赖 cv2。"""
    h, w = img_rgb.shape[:2]
    cam_r = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    heat = _jet(cam_r)
    out = (img_rgb.astype(np.float32) * (1 - alpha) + heat * 255 * alpha).clip(0, 255)
    return out.astype(np.uint8)


def _jet(x: np.ndarray) -> np.ndarray:
    """numpy 版 jet 色图，输入 [0,1]，输出 [H,W,3] 的 [0,1]。"""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def load_rgb(path: Path, size: int) -> tuple[np.ndarray, torch.Tensor]:
    """读图并返回 (原图 RGB 数组, 归一化后的 tensor)。PIL 支持中文路径。"""
    img = Image.open(path).convert("RGB")
    tf = build_transforms(size, train=False)
    x = tf(img).unsqueeze(0)
    arr = np.array(img.resize((size, size), Image.BILINEAR))
    return arr, x


def pick_samples(test_root: Path, per_class: int, only_classes: list[int] | None, num: int):
    """从测试集抽样。返回 [(path, label_idx, class_name)]。"""
    class_dirs = sorted([d for d in test_root.iterdir() if d.is_dir()])
    picked = []
    for idx, d in enumerate(class_dirs):
        if only_classes is not None and idx not in only_classes:
            continue
        files = sorted(d.glob("*.jpg")) + sorted(d.glob("*.JPG")) + sorted(d.glob("*.png"))
        for f in files[:per_class]:
            picked.append((f, idx, d.name))
        if num and len(picked) >= num:
            break
    return picked[:num] if num else picked


def main() -> int:
    ap = argparse.ArgumentParser(description="E7 Grad-CAM 可视化")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-root", default=str(ML_ROOT / "ml" / "data" / "processed" / "plantvillage"))
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--per-class", type=int, default=1, help="每类抽几张")
    ap.add_argument("--num", type=int, default=0, help="总张数上限，0 表示不限")
    ap.add_argument("--classes", default="", help="只测这些类别索引，逗号分隔")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"找不到 {ckpt_path}")

    out_dir = Path(args.out) if args.out else run_dir / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_name = (ck.get("args") or {}).get("model") or "resnet18"
    classes = ck["classes"]
    model, meta = build_model(model_name, len(classes), pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()

    target = get_target_module(model, model_name)
    cam_fn = GradCAM(model, target)

    only = [int(x) for x in args.classes.split(",") if x.strip()] or None
    test_root = Path(args.data_root) / "test"
    samples = pick_samples(test_root, args.per_class, only, args.num)
    print(f"[gradcam] {model_name}  {len(samples)} 张图  target={type(target).__name__}")

    records = []
    tiles = []
    for i, (path, label_idx, class_name) in enumerate(samples):
        arr, x = load_rgb(path, args.img_size)
        cam, pred_idx, prob = cam_fn(x)
        correct = int(pred_idx) == int(label_idx)
        mass = cam_grid_mass(cam)
        cc = corner_vs_center(cam)
        records.append(
            {
                "file": str(path.name),
                "class_dir": class_name,
                "label_idx": int(label_idx),
                "pred_idx": int(pred_idx),
                "pred_class": classes[int(pred_idx)] if int(pred_idx) < len(classes) else str(pred_idx),
                "prob": round(prob, 4),
                "correct": correct,
                "grid_mass": [round(v, 4) for v in mass],
                "center_ratio": round(cc["center"], 4),
                "corners_ratio": round(cc["corners"], 4),
            }
        )
        ov = overlay(arr, cam)
        Image.fromarray(ov).save(out_dir / f"overlay_{i:03d}_{classes[int(pred_idx)][:20]}.png")
        tiles.append((arr, cam, ov, classes[int(pred_idx)], prob, correct))
        if (i + 1) % 10 == 0:
            print(f"  已处理 {i + 1}/{len(samples)}")

    cam_fn.remove()

    # 汇总指标
    n = len(records)
    acc = sum(r["correct"] for r in records) / n if n else 0.0
    grid_mean = np.mean([r["grid_mass"] for r in records], axis=0).tolist() if n else [0] * 9
    center_mean = float(np.mean([r["center_ratio"] for r in records])) if n else 0.0
    corners_mean = float(np.mean([r["corners_ratio"] for r in records])) if n else 0.0

    summary = {
        "model": model_name,
        "params_m": round(meta.params_m, 3),
        "n_samples": n,
        "sample_acc": round(acc, 4),
        "cam_grid_mass_mean": [round(v, 4) for v in grid_mean],
        "center_ratio_mean": round(center_mean, 4),
        "corners_ratio_mean": round(corners_mean, 4),
        "target_layer": type(target).__name__,
        "records": records,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 拼总览网格（原图 | 热力图 | 叠加），最多 5 行 × 3 列
    make_grid(tiles, out_dir / "grid.png")

    print()
    print("=" * 62)
    print(f"  模型        {model_name}  ({meta.params_m:.3f} M)")
    print(f"  样本        {n} 张   抽样准确率 {acc * 100:.2f}%")
    print(f"  中心注意力  {center_mean * 100:.1f}%   四角注意力 {corners_mean * 100:.1f}%")
    print(f"  3×3 网格分布（行优先，%）:")
    for r in range(3):
        row = grid_mean[r * 3:(r + 1) * 3]
        print("    " + "  ".join(f"{v * 100:5.1f}" for v in row))
    print("=" * 62)
    print(f"[saved] {out_dir / 'summary.json'}")
    print(f"[saved] {out_dir / 'grid.png'}")
    return 0


def make_grid(tiles, out_path: Path, max_rows: int = 5):
    """三列布局：原图 | 纯热力图 | 叠加图。"""
    if not tiles:
        return
    from PIL import ImageDraw

    show = tiles[:max_rows]
    h = w = 224
    pad = 6
    label_h = 22
    cols = 3
    W = cols * w + (cols + 1) * pad
    H = len(show) * (h + label_h) + (len(show) + 1) * pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for ri, (arr, cam, ov, pred_name, prob, correct) in enumerate(show):
        y = pad + ri * (h + label_h + pad)
        cam_img = (_jet(np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0) * 255).astype(np.uint8)
        for ci, im in enumerate([arr, cam_img, ov]):
            x = pad + ci * (w + pad)
            canvas.paste(Image.fromarray(im), (x, y + label_h))
        mark = "OK" if correct else "WRONG"
        draw.text((pad, y + 4), f"{pred_name[:34]}  {prob * 100:.1f}%  [{mark}]", fill=(20, 20, 20))
    canvas.save(out_path)


if __name__ == "__main__":
    raise SystemExit(main())
