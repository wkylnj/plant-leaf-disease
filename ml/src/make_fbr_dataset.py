"""FBR（Field Background Recomposition）背景重组数据增强。

动机与文献依据：
    Jeon et al., Ecological Informatics 2026 提出 FBR——把 PlantVillage 的叶片
    抠出来贴到真实田间背景上，仅这一步就把苹果叶跨域准确率从 47.1% 提到 57.0%
    （+9.9pp），再叠 CDAN+E 达到 91.1%。核心洞察：域偏移的主要载体是**背景与
    成像风格**，不是病斑本身。

    **与 E-16 叶片裁剪（−3.57pp）方向相反，这一点很关键**：
      - E-16 是「把背景丢掉」（裁到叶片 bbox）→ 训练时没有背景，测试时有 → 差距更大。
      - FBR 是「把背景换成目标域的」→ 训练时就有田间背景 → 差距变小。
    E-16 的失败恰好反证了「背景是域偏移主要载体」这个判断。

为什么 PlantVillage 的叶片好抠：
    它的图是单叶居中、纯灰背景、256×256（见 _pv_sample.png），背景均匀，
    用简单的颜色距离 + Otsu 就能抠干净，不需要 SAM。

合成流程（对每张 PlantVillage 图）：
    1. 抠出叶片前景（颜色距离法，背景是均匀灰）
    2. 从背景库随机取一张田间图，随机裁一块并缩放到目标尺寸
    3. 随机旋转 ±25°、缩放 0.75~1.0，把叶片合成到背景上
    4. 边缘羽化（高斯模糊 alpha），避免硬边
    5. 轻微亮度/对比度抖动，模拟不同光照

用法：
    python src/make_fbr_dataset.py --out data/processed/pv_fbr --per-image 1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def background_color(rgb: np.ndarray) -> np.ndarray:
    """估计 PlantVillage 的背景色（四角与边缘中位数）。

    PlantVillage 是纯色摆拍：叶子居中、四周是同一种灰/米色。
    取四条边带的中位数即可稳健估计，不受个别暗角影响。
    """
    h, w = rgb.shape[:2]
    band = max(4, min(h, w) // 20)
    strips = [
        rgb[:band, :, :].reshape(-1, 3),
        rgb[-band:, :, :].reshape(-1, 3),
        rgb[:, :band, :].reshape(-1, 3),
        rgb[:, -band:, :].reshape(-1, 3),
    ]
    edge = np.concatenate(strips, axis=0).astype(np.float32)
    return np.median(edge, axis=0)


def foreground_mask(img: Image.Image) -> np.ndarray:
    """抠出 PlantVillage 的叶片前景。

    实测对比了四种方法（见 scripts/_probe_matte.py 与 _matte_compare.png）：
      - 四角背景色距离法：背景有渐变，会残留大片灰块（第一版失败原因）
      - 纯饱和度：会在病斑/褐色枯死区内部挖洞（它们饱和度低）
      - 从边界泛洪：会把背景阴影当成叶片一起框进来
      - **超绿指数 + 饱和度加权：最好**，病斑完整、背景干净 → 采用这个

    2026-11-25 补充：全量合成图抽查发现 62% 有暗边——超绿掩膜在叶片边缘
    是渐变的，羽化后会把 PlantVillage 的灰紫背景带成一圈灰晕。因此在
    超绿掩膜之后再加一步「背景色距离」硬约束：距离背景色很近的像素一律
    判为背景，把灰晕切掉。两步取交集，病斑（颜色远离背景）不受影响。
    """
    rgb = np.asarray(img.convert("RGB"))
    a = rgb.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]

    # 超绿指数：绿叶响应强
    exg = 2.0 * g - r - b
    # 饱和度：褐色病斑/枯死区超绿指数低但饱和度不低，用它对冲
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)

    def nrm(x: np.ndarray) -> np.ndarray:
        lo, hi = float(x.min()), float(x.max())
        return np.zeros_like(x) if hi - lo < 1e-6 else (x - lo) / (hi - lo)

    score = 0.5 * nrm(exg) + 0.5 * nrm(sat)
    thr = _otsu(score)
    mask = score > max(thr, 0.20)

    mask = ndimage.binary_closing(mask, structure=np.ones((7, 7)), iterations=2)
    mask = ndimage.binary_opening(mask, structure=np.ones((5, 5)), iterations=1)

    # 背景色距离硬约束：切掉边缘灰晕
    bg = background_color(rgb) / 255.0
    dist = np.sqrt(((a - bg[None, None, :]) ** 2).sum(axis=2))
    mask &= dist > 0.10

    lab, n = ndimage.label(mask)
    if n == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    mask = lab == (int(np.argmax(sizes)) + 1)
    return ndimage.binary_fill_holes(mask)


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


def crop_leaf(img: Image.Image, mask: np.ndarray, margin: float = 0.04) -> tuple[Image.Image, np.ndarray]:
    """裁到叶片 bbox（保留一点边距），返回 (叶片RGB, alpha)。"""
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return img, np.ones(mask.shape, dtype=np.float32)
    h, w = mask.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    mx, my = int((x1 - x0) * margin), int((y1 - y0) * margin)
    x0, y0 = max(0, x0 - mx), max(0, y0 - my)
    x1, y1 = min(w, x1 + mx + 1), min(h, y1 + my + 1)

    rgb = np.asarray(img.convert("RGB"))[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1]
    alpha = ndimage.gaussian_filter(m.astype(np.float32), sigma=1.2)
    alpha = np.clip(alpha, 0.0, 1.0)
    return Image.fromarray(rgb), alpha


def random_bg(bg_paths: list[Path], size: int, rng: random.Random) -> Image.Image:
    """从背景库随机取一张，随机裁一块并缩放到 size×size。"""
    path = rng.choice(bg_paths)
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    # 随机裁剪位置
    x0 = rng.randint(0, max(0, w - side))
    y0 = rng.randint(0, max(0, h - side))
    img = img.crop((x0, y0, x0 + side, y0 + side))
    return img.resize((size, size), Image.BICUBIC)


def compose_one(pv_img: Image.Image, bg_paths: list[Path], out_size: int,
                rng: random.Random) -> Image.Image:
    """把一张 PlantVillage 叶片合成到随机田间背景上。"""
    mask = foreground_mask(pv_img)
    leaf, alpha = crop_leaf(pv_img, mask)

    bg = random_bg(bg_paths, out_size, rng)

    # 叶片缩放到画面的 55%~85%，随机旋转
    target = int(out_size * rng.uniform(0.55, 0.85))
    lw, lh = leaf.size
    scale = target / max(lw, lh)
    leaf = leaf.resize((max(1, int(lw * scale)), max(1, int(lh * scale))), Image.BICUBIC)
    a = Image.fromarray((alpha * 255).astype(np.uint8)).resize(leaf.size, Image.BILINEAR)

    angle = rng.uniform(-25, 25)
    leaf = leaf.rotate(angle, expand=True, resample=Image.BICUBIC)
    a = a.rotate(angle, expand=True, resample=Image.BICUBIC)

    lw, lh = leaf.size
    if lw > out_size or lh > out_size:
        s = min(out_size / lw, out_size / lh) * 0.95
        leaf = leaf.resize((max(1, int(lw * s)), max(1, int(lh * s))), Image.BICUBIC)
        a = a.resize(leaf.size, Image.BILINEAR)
        lw, lh = leaf.size

    ox = rng.randint(0, max(0, out_size - lw))
    oy = rng.randint(0, max(0, out_size - lh))
    bg.paste(leaf, (ox, oy), a)

    # 光照抖动
    bg = ImageEnhance.Brightness(bg).enhance(rng.uniform(0.85, 1.15))
    bg = ImageEnhance.Contrast(bg).enhance(rng.uniform(0.85, 1.15))
    return bg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pv-root", default="data/processed/plantvillage/train")
    ap.add_argument("--bg-root", default="data/raw/plantdoc_bg")
    ap.add_argument("--out", default="data/processed/pv_fbr")
    ap.add_argument("--per-image", type=int, default=1, help="每张源图生成几个变体")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-per-class", type=int, default=0)
    args = ap.parse_args()

    pv_root = ROOT / args.pv_root
    bg_root = ROOT / args.bg_root
    out_root = ROOT / args.out

    bg_paths = [p for p in sorted(bg_root.glob("bg_*.jpg"))]
    if not bg_paths:
        print(f"[fbr] 背景库为空：{bg_root}，请先运行 make_bg_library.py")
        return 1
    print(f"[fbr] 背景库 {len(bg_paths)} 张", flush=True)

    rng = random.Random(args.seed)
    classes = sorted([d for d in pv_root.iterdir() if d.is_dir()])
    print(f"[fbr] 源域 {len(classes)} 类", flush=True)

    total = 0
    fallback = 0
    for ci, cdir in enumerate(classes):
        files = sorted([p for p in cdir.iterdir() if p.suffix.lower() in IMG_EXTS])
        if args.limit_per_class > 0:
            files = files[: args.limit_per_class]
        out_dir = out_root / cdir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for fi, f in enumerate(files):
            try:
                pv = Image.open(f).convert("RGB")
            except Exception:  # noqa: BLE001
                fallback += 1
                continue
            for v in range(args.per_image):
                img = compose_one(pv, bg_paths, args.size, rng)
                img.save(out_dir / f"{f.stem}_fbr{v}.jpg", quality=92)
                total += 1
        print(f"[fbr] {ci + 1}/{len(classes)} {cdir.name} 累计 {total} 张", flush=True)

    meta = {
        "source": str(pv_root.relative_to(ROOT)).replace("\\", "/"),
        "bg_library": str(bg_root.relative_to(ROOT)).replace("\\", "/"),
        "bg_count": len(bg_paths),
        "per_image": args.per_image,
        "size": args.size,
        "classes": len(classes),
        "total": total,
        "read_errors": fallback,
        "seed": args.seed,
    }
    (out_root / "fbr_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[fbr] 完成：{total} 张，输出 {out_root}")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
