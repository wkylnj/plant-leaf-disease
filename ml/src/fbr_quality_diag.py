"""FBR 合成质量诊断：定位合成过程到底破坏了什么。

背景
----
2026-11-25 的可行性诊断发现：
  - 原模型在原始 PlantVillage test 上 99.41%
  - 在 FBR 合成图上只有 48.59%
  - 把合成图背景涂灰（只留叶片）仍是 50.00%

第二行是关键：**背景涂灰也救不回来**，说明合成过程破坏了叶片本身，
而不是「背景换了导致模型不适应」。如果叶片信息被破坏，那用合成图训练
就是在教模型学噪声，FBR 这条路必然失败。

本脚本做单变量分解，找出是哪一步破坏的：
  1. none       —— 原图（对照）
  2. shrink     —— 只缩小到 70%（模拟合成时的缩放）
  3. rotate     —— 只旋转 25°（模拟合成时的随机旋转）
  4. rot_shrink —— 两者叠加
  5. jpeg       —— 只重存为 JPEG q=88（合成图会再压一次）
  6. enh        —— 只做亮度/对比度扰动 0.85~1.15

用法：
    python src/fbr_quality_diag.py --n 800
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import build_transforms  # noqa: E402
from fbr_feasibility import load_model  # noqa: E402

ROOT = SRC.parent
ARTIFACTS = ROOT / "artifacts"


class Variant(Dataset):
    """对原始 PlantVillage test 图施加单变量变换。"""

    def __init__(self, mode: str, root: Path, n: int, angle: float = 25.0,
                 scale: float = 0.70, jpeg_q: int = 88):
        base = ImageFolder(root)
        idx = np.random.RandomState(0).permutation(len(base))[:n].tolist()
        self.base = torch.utils.data.Subset(base, idx)
        self.tf = build_transforms(224, train=False)
        self.mode = mode
        self.angle = angle
        self.scale = scale
        self.jpeg_q = jpeg_q

    def __len__(self) -> int:
        return len(self.base)

    def _shrink(self, img: Image.Image) -> Image.Image:
        s = int(img.size[0] * self.scale)
        small = img.resize((s, s), Image.BICUBIC)
        canvas = Image.new("RGB", img.size, (128, 128, 128))
        canvas.paste(small, ((img.size[0] - s) // 2, (img.size[1] - s) // 2))
        return canvas

    def __getitem__(self, i: int):
        img, y = self.base[i]
        img = img.convert("RGB")
        if self.mode == "shrink":
            img = self._shrink(img)
        elif self.mode == "rotate":
            img = img.rotate(self.angle, resample=Image.BICUBIC, fillcolor=(128, 128, 128))
        elif self.mode == "rot_shrink":
            img = self._shrink(img).rotate(self.angle, resample=Image.BICUBIC,
                                           fillcolor=(128, 128, 128))
        elif self.mode == "jpeg":
            import io
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self.jpeg_q)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        elif self.mode == "enh":
            img = ImageEnhance.Brightness(img).enhance(0.9)
            img = ImageEnhance.Contrast(img).enhance(1.1)
        elif self.mode != "none":
            raise ValueError(f"未知模式 {self.mode}")
        return self.tf(img), y


@torch.no_grad()
def evaluate(model, ds, batch_size: int = 64) -> tuple[float, int, int]:
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    correct = n = 0
    for x, y in dl:
        correct += int((model(x).argmax(1) == y).sum())
        n += len(y)
    return correct / max(n, 1), correct, n


def main() -> int:
    ap = argparse.ArgumentParser(description="FBR 合成质量单变量诊断")
    ap.add_argument("--run-dir", default="runs/resnet18_224_15ep")
    ap.add_argument("--data-root", default="data/processed/plantvillage")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--out", default=str(ARTIFACTS / "fbr_quality_diag.json"))
    args = ap.parse_args()

    model, name, _classes = load_model(Path(args.run_dir))
    print(f"[diag] 模型 {name}，每变体抽样 {args.n} 张")

    root = Path(args.data_root) / "test"
    modes = ["none", "shrink", "rotate", "rot_shrink", "jpeg", "enh"]
    results = {}
    for m in modes:
        acc, c, n = evaluate(model, Variant(m, root, args.n))
        results[m] = {"acc": acc, "correct": c, "n": n}
        print(f"  {m:12s} acc={acc:.4f} ({c}/{n})")

    base = results["none"]["acc"]
    print("\n相对原图的下降：")
    for m in modes[1:]:
        print(f"  {m:12s} {results[m]['acc'] - base:+.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
