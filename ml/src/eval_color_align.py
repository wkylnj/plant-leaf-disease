"""推理时颜色对齐（跨域实验 E-17）。

诊断结论（见 ``ml/artifacts/gradcam.md`` 与 ``crossdomain.md``）：模型依赖的是
「整片叶子的全局颜色/纹理统计量」，而不是病斑局部形态。因此跨域掉点的直接
原因是 PlantDoc 田间照片的**色彩统计分布**与 PlantVillage 实验室照片不同。

本实验在**推理时**把 PlantDoc 图像的 LAB 色彩统计量向 PlantVillage 的统计量
对齐（Reinhard 色彩迁移），权重不动、不训练。强度 ``alpha`` 控制对齐程度：

- ``alpha=0``   等价 baseline
- ``alpha=1``   完全对齐到 PlantVillage 统计量
- ``alpha=0.5`` 折中

用法::

    python src/eval_color_align.py --run-dir runs/resnet18_224_15ep --split all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from eval_crossdomain import crop_of_folder  # noqa: E402
from models import build_model  # noqa: E402

ARTIFACT_DIR = ROOT / "artifacts"


# ---------------------------------------------------------------- 色彩空间

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB [0,1] (H,W,3) -> CIELAB，L 归一到 [0,1]，a/b 归一到 [-1,1] 量级。

    手写实现，避免依赖 skimage（Kaggle 版本差异风险）。
    """
    x = rgb.astype(np.float64)
    # sRGB -> 线性 RGB
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T
    # D65 白点
    white = np.array([0.95047, 1.00000, 1.08883])
    xyz = xyz / white
    eps = 216 / 24389
    kap = 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kap * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L / 100.0, a / 128.0, b / 128.0], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIELAB（同上归一化）-> sRGB [0,1]。"""
    L = lab[..., 0] * 100.0
    a = lab[..., 1] * 128.0
    b = lab[..., 2] * 128.0
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    eps = 216 / 24389
    kap = 24389 / 27
    def inv(f):
        return np.where(f ** 3 > eps, f ** 3, (116 * f - 16) / kap)
    xyz = np.stack([inv(fx), inv(fy), inv(fz)], axis=-1)
    white = np.array([0.95047, 1.00000, 1.08883])
    xyz = xyz * white
    m = np.array([[3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]])
    lin = xyz @ m.T
    lin = np.clip(lin, 0.0, 1.0)
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(srgb, 0.0, 1.0)


# ---------------------------------------------------------------- 统计量

def lab_stats(images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """在 LAB 空间统计每通道的均值与标准差（对整批像素）。"""
    px = np.concatenate([im.reshape(-1, 3) for im in images], axis=0)
    return px.mean(axis=0), px.std(axis=0)


def align_lab(rgb: np.ndarray, src_mean: np.ndarray, src_std: np.ndarray,
              tgt_mean: np.ndarray, tgt_std: np.ndarray, alpha: float) -> np.ndarray:
    """把 rgb 的 LAB 统计量按 alpha 向目标统计量迁移。"""
    lab = rgb_to_lab(rgb)
    cur_mean = lab.reshape(-1, 3).mean(axis=0)
    cur_std = lab.reshape(-1, 3).std(axis=0) + 1e-6
    scale = tgt_std / cur_std
    shift = tgt_mean - cur_mean * scale
    # alpha 插值：alpha=0 时 scale=1、shift=0
    scale = 1.0 + alpha * (scale - 1.0)
    shift = alpha * shift
    # 只迁移亮度与色彩均值/对比度，避免过度扭曲
    out = lab * scale + shift
    out[..., 0] = np.clip(out[..., 0], 0.0, 1.0)
    out[..., 1] = np.clip(out[..., 1], -1.5, 1.5)
    out[..., 2] = np.clip(out[..., 2], -1.5, 1.5)
    return lab_to_rgb(out)


# ---------------------------------------------------------------- 数据集

class ColorAlignDataset(Dataset):
    def __init__(self, root: Path, split: str, pv_folders: list[str],
                 img_size: int, alpha: float,
                 tgt_mean: np.ndarray, tgt_std: np.ndarray):
        from eval_crossdomain import PlantDocDataset

        base = PlantDocDataset(root, split, pv_folders, transform=None)
        self.items = base.items
        self.img_size = img_size
        self.alpha = alpha
        self.tgt_mean = tgt_mean
        self.tgt_std = tgt_std

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label, _orig = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.alpha > 0:
            rgb = np.asarray(img).astype(np.float64) / 255.0
            src_mean = rgb_to_lab(rgb).reshape(-1, 3).mean(axis=0)
            src_std = rgb_to_lab(rgb).reshape(-1, 3).std(axis=0) + 1e-6
            rgb = align_lab(rgb, src_mean, src_std, self.tgt_mean, self.tgt_std,
                            self.alpha)
            img = Image.fromarray((rgb * 255.0).round().astype(np.uint8))
        # 与 baseline 相同的几何预处理
        w, h = img.size
        short = int(round(self.img_size * 256 / 224))
        if w < h:
            nw, nh = short, int(round(h * short / w))
        else:
            nh, nw = short, int(round(w * short / h))
        img = img.resize((nw, nh), Image.BICUBIC)
        left = (nw - self.img_size) // 2
        top = (nh - self.img_size) // 2
        img = img.crop((left, top, left + self.img_size, top + self.img_size))
        x = np.asarray(img).copy()
        t = torch.from_numpy(x).float().permute(2, 0, 1) / 255.0
        t = (t - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return t, label


def pv_lab_stats(data_root: Path, n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """从 PlantVillage 训练集抽样估计 LAB 统计量。"""
    train = data_root / "train"
    files = sorted(train.rglob("*.jpg"))
    if not files:
        files = sorted(train.rglob("*.JPG")) + sorted(train.rglob("*.png"))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(files), size=min(n, len(files)), replace=False)
    imgs = []
    for i in idx:
        im = Image.open(files[int(i)]).convert("RGB")
        im = im.resize((112, 112), Image.BICUBIC)
        imgs.append(rgb_to_lab(np.asarray(im).astype(np.float64) / 255.0))
    return lab_stats(imgs)


# ---------------------------------------------------------------- 评估

def predict(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    ps, ls = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(F.softmax(model(x.to(device)), dim=1).cpu())
            ls.append(y)
    return torch.cat(ps), torch.cat(ls)


def metrics(probs: torch.Tensor, labels: torch.Tensor, pv_folders: list[str]) -> dict:
    n = labels.numel()
    top1 = probs.argmax(dim=1)
    top3 = probs.topk(min(3, probs.size(1)), dim=1).indices
    crops = [crop_of_folder(f) for f in pv_folders]
    crop_correct = sum(1 for p, t in zip(top1.tolist(), labels.tolist())
                       if crops[p] == crops[t])
    return {
        "n": n,
        "top1_acc": round(int((top1 == labels).sum()) / n, 4),
        "top3_acc": round(int((top3 == labels.unsqueeze(1)).any(dim=1).sum()) / n, 4),
        "crop_acc": round(crop_correct / n, 4),
        "mean_confidence": round(float(probs.max(dim=1).values.mean()), 4),
    }


def load_model(run_dir: Path):
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    classes = list(ck["classes"])
    name = ck.get("args", {}).get("model") or run_dir.name.rsplit("_", 2)[0]
    model, meta = build_model(name, len(classes), pretrained=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck["state_dict"])
    model.eval()
    return model, meta, classes


def main() -> int:
    p = argparse.ArgumentParser(description="推理时 LAB 颜色对齐跨域评估")
    p.add_argument("--run-dir", default="runs/resnet18_224_15ep")
    p.add_argument("--plantdoc-root", default=str(ROOT / "data" / "raw" / "plantdoc"))
    p.add_argument("--data-root", default=str(ROOT / "data" / "processed" / "plantvillage"))
    p.add_argument("--split", default="all", choices=["train", "test", "all"])
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.3, 0.6, 1.0])
    p.add_argument("--stats-samples", type=int, default=400)
    p.add_argument("--out", default="")
    args = p.parse_args()

    device = torch.device("cpu")
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (ROOT / args.run_dir).resolve()
    model, meta, pv_folders = load_model(run_dir)
    root = Path(args.plantdoc_root)

    print("估计 PlantVillage LAB 统计量 ...", flush=True)
    tgt_mean, tgt_std = pv_lab_stats(Path(args.data_root), args.stats_samples)
    print(f"  mean={np.round(tgt_mean, 4)}  std={np.round(tgt_std, 4)}", flush=True)
    print(f"模型 {meta.name}  split={args.split}  img_size={args.img_size}", flush=True)

    rows = []
    for alpha in args.alphas:
        t0 = time.time()
        ds = ColorAlignDataset(root, args.split, pv_folders, args.img_size,
                               alpha, tgt_mean, tgt_std)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        probs, labels = predict(model, loader, device)
        m = metrics(probs, labels, pv_folders)
        m.update({"alpha": alpha, "img_size": args.img_size,
                  "elapsed_s": round(time.time() - t0, 1)})
        rows.append(m)
        print(f"alpha={alpha:.1f}  top1={m['top1_acc']:.4f} top3={m['top3_acc']:.4f} "
              f"crop={m['crop_acc']:.4f} conf={m['mean_confidence']:.4f} "
              f"({m['elapsed_s']}s)", flush=True)

    payload = {
        "run": run_dir.name, "model": meta.name, "split": args.split,
        "img_size": args.img_size,
        "pv_lab_mean": [round(float(x), 5) for x in tgt_mean],
        "pv_lab_std": [round(float(x), 5) for x in tgt_std],
        "rows": rows,
    }
    out = Path(args.out) if args.out else run_dir / "color_align.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
