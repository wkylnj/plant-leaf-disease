"""对比原始 PV 图与它的 FBR 合成图，并做「只换背景、不动叶片」的隔离测试。

背景
----
`fbr_feasibility` 发现原模型在合成图上只有 48.59%，而单变量扰动（缩放/旋转/
JPEG/亮度）加起来只掉 3.12pp。这说明合成流程里有某个环节破坏了叶片信息。

但先前那个「把背景涂灰」的测试是**无效的**：`foreground_mask()` 取的是画面里
最大的绿色连通域，在真实田间背景上它会锁到背景的草叶而不是贴上去的叶片，
于是把叶片涂灰、背景留下，当然测不准。

本脚本用两个更可靠的方式定位：
  1. 目视对比：原始图 | 合成图 并排（一眼看出贴图哪里坏了）
  2. 隔离测试：只把 PV 的灰背景换成田间背景（叶片不缩放、不旋转、不移动），
     测准确率。如果这样也掉很多，说明是「背景本身」在干扰；如果几乎不掉，
     说明问题出在缩放/旋转/贴图尺寸上。

用法：
    python src/fbr_ablate.py --n 600
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import build_transforms  # noqa: E402
from fbr_feasibility import load_model  # noqa: E402
from make_fbr_dataset import foreground_mask, random_bg  # noqa: E402

ROOT = SRC.parent
ARTIFACTS = ROOT / "artifacts"


class BgSwapOnly(Dataset):
    """只把背景换成田间图，叶片像素一个不动。"""

    def __init__(self, root: Path, bg_paths: list[Path], n: int, feather: float = 1.2):
        import random

        base = ImageFolder(root)
        idx = np.random.RandomState(0).permutation(len(base))[:n].tolist()
        self.base = torch.utils.data.Subset(base, idx)
        self.tf = build_transforms(224, train=False)
        self.bg_paths = bg_paths
        self.rng = random.Random(42)
        self.feather = feather

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        from scipy import ndimage

        img, y = self.base[i]
        img = img.convert("RGB")
        arr = np.asarray(img)
        m = foreground_mask(img)

        bg = random_bg(self.bg_paths, img.size[0], self.rng)
        alpha = ndimage.gaussian_filter(m.astype(np.float32), sigma=self.feather)
        alpha = np.clip(alpha, 0, 1)[..., None]

        out = arr.astype(np.float32) * alpha + np.asarray(bg).astype(np.float32) * (1 - alpha)
        return self.tf(Image.fromarray(out.astype(np.uint8))), y


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
    ap = argparse.ArgumentParser(description="FBR 隔离测试：只换背景")
    ap.add_argument("--run-dir", default="runs/resnet18_224_15ep")
    ap.add_argument("--data-root", default="data/processed/plantvillage")
    ap.add_argument("--fbr-test-root", default="data/processed/pv_fbr_test")
    ap.add_argument("--bg-root", default="data/raw/plantdoc_bg")
    ap.add_argument("--n", type=int, default=600)
    args = ap.parse_args()

    model, name, _ = load_model(Path(args.run_dir))
    root = Path(args.data_root) / "test"
    bg_paths = sorted(Path(args.bg_root).glob("bg_*.jpg"))
    print(f"[ablate] 模型 {name}，背景库 {len(bg_paths)} 张，抽样 {args.n}")

    tf = build_transforms(224, train=False)

    # 1. 原始
    base_ds = ImageFolder(root, transform=tf)
    idx = np.random.RandomState(0).permutation(len(base_ds))[: args.n].tolist()
    acc0, c0, n0 = evaluate(model, torch.utils.data.Subset(base_ds, idx))
    print(f"  原始 PV            acc={acc0:.4f} ({c0}/{n0})")

    # 2. 只换背景（叶片不动）
    acc1, c1, n1 = evaluate(model, BgSwapOnly(root, bg_paths, args.n))
    print(f"  只换背景           acc={acc1:.4f} ({c1}/{n1})   {acc1-acc0:+.4f}")

    # 3. 完整 FBR 合成图
    fbr_ds = ImageFolder(Path(args.fbr_test_root), transform=tf)
    idx2 = np.random.RandomState(0).permutation(len(fbr_ds))[: args.n].tolist()
    acc2, c2, n2 = evaluate(model, torch.utils.data.Subset(fbr_ds, idx2))
    print(f"  完整 FBR 合成      acc={acc2:.4f} ({c2}/{n2})   {acc2-acc0:+.4f}")

    # 4. 目视对比：原图 | 合成图
    cell = 224
    rows = 6
    sheet = Image.new("RGB", (cell * 2, cell * rows), (255, 255, 255))
    pv_files = sorted(root.rglob("*.jpg"))
    fbr_files = sorted(Path(args.fbr_test_root).rglob("*.jpg"))
    # 按类名+序号配对
    for r in range(rows):
        pv = pv_files[r * 37]
        cls = pv.parent.name
        cand = sorted((Path(args.fbr_test_root) / cls).glob("*.jpg"))
        if not cand:
            continue
        sheet.paste(Image.open(pv).convert("RGB").resize((cell, cell)), (0, r * cell))
        sheet.paste(Image.open(cand[0]).convert("RGB").resize((cell, cell)), (cell, r * cell))
    out = ROOT.parent / "_fbr_pair_compare.png"
    sheet.save(out)
    print(f"\n左=原始 PV，右=FBR 合成：{out}")

    import json
    res = {"original": acc0, "bg_swap_only": acc1, "fbr_full": acc2, "n": args.n}
    (ARTIFACTS / "fbr_ablate.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
