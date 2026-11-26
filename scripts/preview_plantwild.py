"""抽查 PlantWild 抽取子集的图像质量（生成拼图）。

用法：
    python scripts/preview_plantwild.py [--root data/raw/plantwild_subset] [--n 8]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png")


def main() -> int:
    ap = argparse.ArgumentParser(description="抽查 PlantWild 子集")
    ap.add_argument("--root", default="ml/data/raw/plantwild_subset")
    ap.add_argument("--out", default="_pw_sample.png")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--cell", type=int, default=224)
    args = ap.parse_args()

    root = Path(args.root)
    classes = sorted(p for p in root.iterdir() if p.is_dir())
    rng = random.Random(5)
    picked: list[Path] = []
    for c in rng.sample(classes, min(args.n, len(classes))):
        files = [f for f in sorted(c.iterdir()) if f.suffix.lower() in IMG_EXT]
        if files:
            picked.append(files[0])

    total = sum(1 for c in classes for f in c.iterdir() if f.suffix.lower() in IMG_EXT)
    print(f"类别目录 {len(classes)}，总图 {total}")

    cols = 4
    rows = (len(picked) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * args.cell, rows * args.cell), (255, 255, 255))
    for i, f in enumerate(picked):
        im = Image.open(f).convert("RGB")
        im.thumbnail((args.cell, args.cell))
        sheet.paste(im, ((i % cols) * args.cell, (i // cols) * args.cell))
    sheet.save(args.out)
    print(f"拼图 {args.out}（{len(picked)} 张）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
