"""叶片自动定位 + 裁剪后再识别（跨域实验 E-16）。

假设：PlantVillage 是近景特写（叶片占画面 70%+），PlantDoc 是田间照片
（叶片只占画面 10%~30%）。模型在 224 输入下看到的「有效叶片像素」相差
数倍，病斑纹理被压缩到感受野之外。若先定位叶片并裁剪放大，跨域精度应上升。

不训练、不改权重，只改「图怎么喂进去」。作为对照，同时评估：
- ``baseline``：短边 256 → 中心裁 224（历史口径）
- ``leafcrop``：叶片包围盒（含 padding）→ 缩放到 224
- ``leafcrop_wide``：包围盒放大 1.4 倍（保留一些背景上下文）
- ``leafcrop_fail``：只统计抠图失败的样本，看是否集中在某类

用法::

    python src/eval_leafcrop.py --run-dir runs/resnet18_224_15ep --split all
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
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import IMAGENET_MEAN, IMAGENET_STD, build_transforms  # noqa: E402
from eval_crossdomain import PLANTDOC_TO_PV_FOLDER, crop_of_folder  # noqa: E402
from models import build_model  # noqa: E402

ARTIFACT_DIR = ROOT / "artifacts"


# ---------------------------------------------------------------- 叶片定位

def leaf_mask(rgb: np.ndarray) -> np.ndarray:
    """基于「叶片比土壤更绿/更饱和且更亮」的粗略前景掩码。

    不用 cv2（本机 5.0.0、Kaggle 4.x 行为有差异），全部 numpy + scipy。
    """
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    # 超绿指数：对绿色叶片响应强，对土壤（红棕）响应弱
    exg = 2.0 * g - r - b
    # 饱和度：叶片（无论健康绿还是病斑黄褐）通常比土壤更「有色」
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)

    score = 0.7 * _norm(exg) + 0.3 * _norm(sat)
    thr = _otsu(score)
    mask = score > max(thr, 0.25)

    # 形态学：先闭（填病斑空洞）再开（去噪点）
    mask = ndimage.binary_closing(mask, structure=np.ones((7, 7)), iterations=2)
    mask = ndimage.binary_opening(mask, structure=np.ones((5, 5)), iterations=1)

    # 取最大连通域
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    mask = lab == biggest
    # 填洞
    mask = ndimage.binary_fill_holes(mask)
    return mask


def _norm(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-6:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _otsu(x: np.ndarray, bins: int = 64) -> float:
    hist, edges = np.histogram(x.ravel(), bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.5
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = total - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1e-9)
    mt = (hist * centers).sum()
    m1 = (mt - np.cumsum(hist * centers)) / np.maximum(w1, 1e-9)
    var = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(np.argmax(var))])


def leaf_bbox(rgb: np.ndarray, pad: float = 0.10) -> tuple[tuple[int, int, int, int] | None, float]:
    """返回 (bbox=(x0,y0,x1,y1), 掩码占画面比例)；定位失败返回 (None, ratio)。"""
    h, w = rgb.shape[:2]
    mask = leaf_mask(rgb)
    ratio = float(mask.mean())
    if ratio < 0.01 or ratio > 0.95:
        return None, ratio
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None, ratio
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = x1 - x0, y1 - y0
    if bw < 24 or bh < 24:
        return None, ratio
    px, py = int(bw * pad), int(bh * pad)
    x0, y0 = max(0, x0 - px), max(0, y0 - py)
    x1, y1 = min(w, x1 + px), min(h, y1 + py)
    return (x0, y0, x1, y1), ratio


# ---------------------------------------------------------------- 数据集

class LeafCropDataset(Dataset):
    """在 __getitem__ 里做叶片定位；失败自动回退整图（并记录）。"""

    def __init__(self, root: Path, split: str, pv_folders: list[str],
                 img_size: int = 224, expand: float = 1.0, pad: float = 0.10):
        from eval_crossdomain import PlantDocDataset

        base = PlantDocDataset(root, split, pv_folders, transform=None)
        self.items = base.items
        self.img_size = img_size
        self.expand = expand
        self.pad = pad
        self.fallback = 0
        self.ratios: list[float] = []

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label, _orig = self.items[idx]
        img = Image.open(path).convert("RGB")
        rgb = np.asarray(img)
        box, ratio = leaf_bbox(rgb, self.pad)
        self.ratios.append(ratio)
        if box is None:
            self.fallback += 1
            crop = img
        else:
            x0, y0, x1, y1 = box
            if self.expand != 1.0:
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                hw = (x1 - x0) * self.expand / 2
                hh = (y1 - y0) * self.expand / 2
                x0 = max(0, int(cx - hw))
                y0 = max(0, int(cy - hh))
                x1 = min(rgb.shape[1], int(cx + hw))
                y1 = min(rgb.shape[0], int(cy + hh))
            crop = img.crop((x0, y0, x1, y1))
        crop = crop.resize((self.img_size, self.img_size), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(crop)).float().permute(2, 0, 1) / 255.0
        x = (x - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return x, label


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
    k = min(3, probs.size(1))
    top3 = probs.topk(k, dim=1).indices
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
    p = argparse.ArgumentParser(description="叶片定位裁剪跨域评估")
    p.add_argument("--run-dir", default="runs/resnet18_224_15ep")
    p.add_argument("--plantdoc-root", default=str(ROOT / "data" / "raw" / "plantdoc"))
    p.add_argument("--split", default="all", choices=["train", "test", "all"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--variants", nargs="+",
                   default=["baseline", "leafcrop", "leafcrop_wide"])
    p.add_argument("--out", default="")
    args = p.parse_args()

    device = torch.device("cpu")
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (ROOT / args.run_dir).resolve()
    model, meta, pv_folders = load_model(run_dir)
    root = Path(args.plantdoc_root)
    print(f"模型 {meta.name}  样本划分 {args.split}", flush=True)

    rows = []
    for vname in args.variants:
        t0 = time.time()
        if vname == "baseline":
            from eval_crossdomain import PlantDocDataset

            ds = PlantDocDataset(root, args.split, pv_folders,
                                 transform=build_transforms(224, train=False))
            fb = 0
            ratio_mean = None
        else:
            expand = 1.4 if vname == "leafcrop_wide" else 1.0
            ds = LeafCropDataset(root, args.split, pv_folders, 224, expand=expand)
            fb = ds.fallback
            ratio_mean = float(np.mean(ds.ratios)) if ds.ratios else None
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        probs, labels = predict(model, loader, device)
        m = metrics(probs, labels, pv_folders)
        m.update({"variant": vname, "fallback": fb,
                  "leaf_ratio_mean": round(ratio_mean, 4) if ratio_mean is not None else None,
                  "elapsed_s": round(time.time() - t0, 1)})
        rows.append(m)
        print(f"{vname:16s} top1={m['top1_acc']:.4f} top3={m['top3_acc']:.4f} "
              f"crop={m['crop_acc']:.4f} conf={m['mean_confidence']:.4f} "
              f"fallback={fb}/{m['n']} ({m['elapsed_s']}s)", flush=True)

    payload = {"run": run_dir.name, "model": meta.name, "split": args.split, "rows": rows}
    out = Path(args.out) if args.out else run_dir / "leafcrop.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
