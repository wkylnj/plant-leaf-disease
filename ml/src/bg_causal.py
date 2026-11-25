"""背景替换的因果分解：模型到底依赖背景的什么属性？

动机
----
E-7 Grad-CAM 的结论是「模型不看背景」——四角注意力仅 2.7%，注意力弥散居中。
但 `fbr_ablate.py` 发现：只把 PlantVillage 的灰背景换成田间背景，
准确率就从 99.50% 掉到 83.50%（−16.0pp）。

这两件事**互相矛盾**。Grad-CAM 的注意力分布衡量的是「梯度在空间上的流向」，
而不是「把这块区域换掉模型会不会崩」。本脚本用因果实验（改输入、看输出）
直接测后者，并把「背景」拆成几个可独立操纵的属性：

  1. grey        —— 原始灰背景（对照）
  2. solid_green —— 纯绿色（只改颜色，不加纹理）
  3. solid_grey  —— 纯灰色（只改颜色亮度，不加纹理）
  4. field_bg    —— 真实田间背景（颜色 + 纹理都变）
  5. noise       —— 随机噪声（纯纹理，无颜色语义）
  6. blurred_bg  —— 田间背景重度模糊（保留颜色，抹掉纹理）

如果 solid_green 掉很多 → 模型依赖背景的**颜色统计**（这与 E-2 手工特征实验一致：
颜色直方图 102 维就有 79.41%）。
如果 field_bg 比 solid_green 掉更多 → 纹理也有贡献。
如果 noise 掉很少 → 模型对背景纹理不敏感，只对颜色敏感。

这个结论对 FBR 有直接指导意义：如果模型只依赖背景颜色，那么 FBR 合成图
不需要「真实田间照片」当背景，用颜色扰动就够了——而 E-10 的 ColorJitter
已经证明这条路走不通。反过来，如果纹理也重要，FBR 才有独立价值。

用法：
    python src/bg_causal.py --n 600
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import build_transforms  # noqa: E402
from fbr_feasibility import load_model  # noqa: E402
from make_fbr_dataset import foreground_mask  # noqa: E402

ROOT = SRC.parent
ARTIFACTS = ROOT / "artifacts"


class BgVariant(Dataset):
    """把背景替换成不同类型，叶片像素不动。"""

    def __init__(self, root: Path, mode: str, bg_paths: list[Path], n: int,
                 feather: float = 1.2):
        import random

        base = ImageFolder(root)
        idx = np.random.RandomState(0).permutation(len(base))[:n].tolist()
        self.base = torch.utils.data.Subset(base, idx)
        self.tf = build_transforms(224, train=False)
        self.mode = mode
        self.bg_paths = bg_paths
        self.rng = random.Random(42)
        self.np_rng = np.random.RandomState(42)
        self.feather = feather

    def __len__(self) -> int:
        return len(self.base)

    def _make_bg(self, w: int, h: int) -> np.ndarray:
        if self.mode == "solid_green":
            return np.full((h, w, 3), (90, 140, 70), dtype=np.uint8)
        if self.mode == "solid_grey":
            return np.full((h, w, 3), (128, 128, 128), dtype=np.uint8)
        if self.mode == "noise":
            return self.np_rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
        # field_bg / blurred_bg
        p = self.rng.choice(self.bg_paths)
        img = Image.open(p).convert("RGB")
        # 随机裁一块再缩放到目标尺寸（与 random_bg 一致）
        side = min(img.size)
        x0 = self.rng.randint(0, max(img.size[0] - side, 0))
        y0 = self.rng.randint(0, max(img.size[1] - side, 0))
        img = img.crop((x0, y0, x0 + side, y0 + side)).resize((w, h), Image.BICUBIC)
        if self.mode == "blurred_bg":
            img = img.filter(ImageFilter.GaussianBlur(radius=max(w, h) / 12))
        return np.asarray(img)

    def __getitem__(self, i: int):
        from scipy import ndimage

        img, y = self.base[i]
        img = img.convert("RGB")
        arr = np.asarray(img)

        if self.mode == "grey":
            out = arr
        else:
            m = foreground_mask(img)
            bg = self._make_bg(img.size[0], img.size[1]).astype(np.float32)
            alpha = ndimage.gaussian_filter(m.astype(np.float32), sigma=self.feather)
            alpha = np.clip(alpha, 0, 1)[..., None]
            out = arr.astype(np.float32) * alpha + bg * (1 - alpha)
            out = out.astype(np.uint8)
        return self.tf(Image.fromarray(out)), y


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
    ap = argparse.ArgumentParser(description="背景替换因果分解")
    ap.add_argument("--run-dir", default="runs/resnet18_224_15ep")
    ap.add_argument("--data-root", default="data/processed/plantvillage")
    ap.add_argument("--bg-root", default="data/raw/plantdoc_bg")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", default=str(ARTIFACTS / "bg_causal.json"))
    args = ap.parse_args()

    model, name, _ = load_model(Path(args.run_dir))
    root = Path(args.data_root) / "test"
    bg_paths = sorted(Path(args.bg_root).glob("bg_*.jpg"))
    print(f"[bgcausal] 模型 {name}，背景库 {len(bg_paths)} 张，抽样 {args.n}")

    modes = ["grey", "solid_grey", "solid_green", "blurred_bg", "field_bg", "noise"]
    results = {}
    for m in modes:
        acc, c, n = evaluate(model, BgVariant(root, m, bg_paths, args.n))
        results[m] = {"acc": acc, "correct": c, "n": n}
        print(f"  {m:12s} acc={acc:.4f} ({c}/{n})")

    base = results["grey"]["acc"]
    print("\n相对原始灰背景：")
    for m in modes[1:]:
        print(f"  {m:12s} {results[m]['acc'] - base:+.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
