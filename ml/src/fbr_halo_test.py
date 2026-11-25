"""验证合成图的「灰晕伪影」假设：bbox 内残留的 PlantVillage 灰底是否在害模型。

背景
----
`fbr_ablate.py` 的隔离测试给出三段式：
    原始 PV 99.50% → 只换背景 83.50%（−16.0pp）→ 完整合成 49.17%（−50.3pp）

`bg_causal.py` 证明那 −16pp 来自背景里的**清晰叶片结构**（模糊后只掉 2.67pp）。

但 −50.3pp 里还有 34pp 没解释。候选原因：`crop_leaf()` 只按 bbox 裁剪，
bbox 内非叶片像素占 32.6%（实测），这些灰底会跟着叶片一起贴到田间背景上，
形成一圈真实照片里绝不会出现的灰晕。

本脚本用不同腐蚀量收缩掩膜（把灰晕切掉），看合成图上的准确率能否回升：
  - erode=0（现状）
  - erode=2 / 3 / 5 px
  - erode=0 + 背景模糊（分离「灰晕」与「背景清晰度」两个因素）

如果腐蚀后准确率明显回升 → 灰晕是主因，FBR 合成流程必须修。
如果腐蚀没用 → 问题在别处（贴图尺寸/位置），需要另找。

用法：
    python src/fbr_halo_test.py --n 600
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


class Composite(Dataset):
    """按指定腐蚀量与背景清晰度重新合成，其余与 make_fbr_dataset 一致。"""

    def __init__(self, root: Path, bg_paths: list[Path], n: int,
                 erode: int = 0, blur_bg: bool = False, out_size: int = 224):
        import random

        base = ImageFolder(root)
        idx = np.random.RandomState(0).permutation(len(base))[:n].tolist()
        self.base = torch.utils.data.Subset(base, idx)
        self.tf = build_transforms(out_size, train=False)
        self.bg_paths = bg_paths
        self.erode = erode
        self.blur_bg = blur_bg
        self.out_size = out_size
        self.rng = random.Random(42)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        from scipy import ndimage

        img, y = self.base[i]
        img = img.convert("RGB")
        mask = foreground_mask(img)

        if self.erode > 0:
            k = np.ones((2 * self.erode + 1, 2 * self.erode + 1))
            mask = ndimage.binary_erosion(mask, k, iterations=1)

        ys, xs = np.nonzero(mask)
        if len(xs) < 50:
            leaf, alpha = img, np.ones(mask.shape, dtype=np.float32)
        else:
            h, w = mask.shape
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            mx, my = int((x1 - x0) * 0.04), int((y1 - y0) * 0.04)
            x0, y0 = max(0, x0 - mx), max(0, y0 - my)
            x1, y1 = min(w, x1 + mx + 1), min(h, y1 + my + 1)
            rgb = np.asarray(img)[y0:y1, x0:x1]
            m = mask[y0:y1, x0:x1]
            alpha = np.clip(ndimage.gaussian_filter(m.astype(np.float32), sigma=1.2), 0, 1)
            leaf = Image.fromarray(rgb)

        # 背景
        p = self.rng.choice(self.bg_paths)
        bg_img = Image.open(p).convert("RGB")
        side = min(bg_img.size)
        bx = self.rng.randint(0, max(bg_img.size[0] - side, 0))
        by = self.rng.randint(0, max(bg_img.size[1] - side, 0))
        bg_img = bg_img.crop((bx, by, bx + side, by + side))
        if self.blur_bg:
            bg_img = bg_img.filter(ImageFilter.GaussianBlur(radius=self.out_size / 12))
        bg = bg_img.resize((self.out_size, self.out_size), Image.BICUBIC)

        # 叶片缩放到画面 55%~85%，随机旋转 ±25°
        frac = self.rng.uniform(0.55, 0.85)
        target = int(self.out_size * frac)
        ratio = target / max(leaf.size)
        leaf = leaf.resize((max(int(leaf.size[0] * ratio), 8),
                            max(int(leaf.size[1] * ratio), 8)), Image.BICUBIC)
        alpha_img = Image.fromarray((alpha * 255).astype(np.uint8)).resize(leaf.size, Image.BICUBIC)
        # 关键：叶片与 alpha 必须用同一个角度，否则尺寸不匹配（曾报 images do not match）
        angle = self.rng.uniform(-25, 25)
        leaf = leaf.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0))
        alpha_img = alpha_img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=0)
        # rotate 后浮点取整可能有 1px 差异，强制对齐
        if alpha_img.size != leaf.size:
            alpha_img = alpha_img.resize(leaf.size, Image.BICUBIC)

        ox = self.rng.randint(0, max(self.out_size - leaf.size[0], 0))
        oy = self.rng.randint(0, max(self.out_size - leaf.size[1], 0))
        bg.paste(leaf, (ox, oy), alpha_img)
        return self.tf(bg), y


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
    ap = argparse.ArgumentParser(description="灰晕伪影验证")
    ap.add_argument("--run-dir", default="runs/resnet18_224_15ep")
    ap.add_argument("--data-root", default="data/processed/plantvillage")
    ap.add_argument("--bg-root", default="data/raw/plantdoc_bg")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", default=str(ARTIFACTS / "fbr_halo.json"))
    args = ap.parse_args()

    model, name, _ = load_model(Path(args.run_dir))
    root = Path(args.data_root) / "test"
    bg_paths = sorted(Path(args.bg_root).glob("bg_*.jpg"))
    print(f"[halo] 模型 {name}，背景库 {len(bg_paths)} 张，抽样 {args.n}")

    configs = [
        ("erode0", 0, False),
        ("erode2", 2, False),
        ("erode3", 3, False),
        ("erode5", 5, False),
        ("erode0_blur", 0, True),
        ("erode3_blur", 3, True),
    ]
    results = {}
    for tag, er, bl in configs:
        acc, c, n = evaluate(model, Composite(root, bg_paths, args.n, erode=er, blur_bg=bl))
        results[tag] = {"acc": acc, "correct": c, "n": n, "erode": er, "blur_bg": bl}
        print(f"  {tag:14s} acc={acc:.4f} ({c}/{n})")

    print("\n参照：原始 PV 0.9950 / 只换背景 0.8350 / 现有合成图 0.4917")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"产物 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
