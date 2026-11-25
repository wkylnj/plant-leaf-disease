"""从 PlantDoc 真实田间照片里筛出可用的「田间背景库」，供 FBR 背景重组使用。

设计变更说明（2026-11-25）：
    第一版尝试「抠出不含叶片的干净背景块」，实测失败——PlantDoc 里大量图片是
    叶片特写（叶脉、病斑、花瓣占满画面），按「背景占比最高」挑窗口挑出来的仍然
    是叶子。改用整张照片当背景：真实田间照片本来就包含其他叶片、杂草、土壤，
    把 PlantVillage 的叶片贴上去才是真实的田间构图。这也更贴近 FBR 论文里
    「田间背景库」的原意（论文用 200+ 张田间场景图，不要求背景里没有植物）。

筛选规则（基于 2026-11-25 对 300 张的实测统计）：
    - 排除白底棚拍图（近白像素 > 30%，实测占 13.0%）
    - 排除底部压水印条的图（底部条带近白 > 10%，实测占 7.3%）
    - 排除手指占比过高的图（肤色 > 25%）
    - 排除过小/过暗的图

用法：
    python src/make_bg_library.py --plantdoc-root data/raw/plantdoc --out data/raw/plantdoc_bg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval_leafcrop import leaf_mask  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root: Path) -> list[Path]:
    return [p for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix.lower() in IMG_EXTS]


def skin_ratio(rgb: np.ndarray) -> float:
    """粗略肤色占比。手指捏着叶子拍照是 PlantDoc 里常见的一类图。"""
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    rule = ((r > 95) & (g > 40) & (b > 20) & ((mx - mn) > 15)
            & (np.abs(r - g) > 15) & (r > g) & (r > b))
    return float(rule.mean())


def reject_reason(rgb: np.ndarray, max_white: float = 0.30,
                  max_bottom_white: float = 0.10,
                  max_skin: float = 0.25,
                  max_leaf: float = 0.45,
                  min_bg_std: float = 9.0,
                  max_uniform_hue: float = 0.80) -> str | None:
    """返回拒绝原因；合格返回 None。

    max_leaf 是关键：实测 PlantDoc 250 张抽样中，叶片占比中位数 0.621，
    只有 15.2% 的图叶片占比 < 0.35（即真正的远景田间图）。不加这一条，
    抽出来的「背景」全是叶片特写——2026-11-25 第一版就是这么失败的。

    min_bg_std / max_uniform_hue 是第二道：实测背景库里混进大量**纯色布背景**
    （蓝色篷布、黑色布），这类背景不是田间环境，贴到叶片后会给模型注入
    「大片单一色块」这个假线索，必须排除。

    阈值标定（`scripts/_probe_bg_thresh.py`，800 张抽样、通过基础过滤的 111 张）：
    bg_std p25=5.84 p50=11.47 p75=19.06；hue_ratio p50=0.27 p75=0.55 p90=0.86。
    最初取 18.0 / 0.55 过严——2922 张只留下 27 张，多样性不够。现取
    9.0 / 0.80（约保留 54/111，全量约 200+ 张），既排掉纯色布又保住样本量。
    """
    h, w = rgb.shape[:2]
    if min(h, w) < 180:
        return "too_small"
    if float(rgb.mean()) < 35.0:
        return "too_dark"
    if float((rgb.min(axis=2) > 232).mean()) > max_white:
        return "white_studio"
    bot = rgb[int(h * 0.82):]
    if bot.size and float((bot.min(axis=2) > 225).mean()) > max_bottom_white:
        return "watermark"
    if skin_ratio(rgb) > max_skin:
        return "skin"

    mask = leaf_mask(rgb)
    if float(mask.mean()) > max_leaf:
        return "leaf_too_dense"

    # 非叶片区域（背景）的多样性检查
    bg_px = rgb[~mask]
    if bg_px.size < 1000:
        return "bg_too_small"
    if float(bg_px.astype(np.float32).mean(axis=0).std()) < min_bg_std:
        return "bg_uniform"
    # 单一色相主导（蓝色篷布、黑色布）
    f = bg_px.astype(np.float32) / 255.0
    mx = f.max(axis=1)
    mn = f.min(axis=1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    if float((sat > 0.35).mean()) > max_uniform_hue:
        return "bg_saturated_uniform"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plantdoc-root", default="data/raw/plantdoc")
    ap.add_argument("--out", default="data/raw/plantdoc_bg")
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--side", type=int, default=320,
                    help="背景图统一缩放到该短边，便于后续随机裁剪")
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()

    src_root = ROOT / args.plantdoc_root
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("bg_*.jpg"):
        old.unlink()

    imgs = list_images(src_root)
    if args.max_images > 0:
        imgs = imgs[: args.max_images]
    print(f"[bg] 源图 {len(imgs)} 张", flush=True)

    records: list[dict] = []
    rejects: dict[str, int] = {}

    for i, path in enumerate(imgs):
        if len(records) >= args.target:
            print(f"[bg] 已达目标 {args.target} 张，停止（扫过 {i}/{len(imgs)}）", flush=True)
            break
        try:
            img = Image.open(path).convert("RGB")
            rgb = np.asarray(img)
        except Exception as exc:  # noqa: BLE001
            rejects["read_error"] = rejects.get("read_error", 0) + 1
            print(f"[bg] 读取失败 {path.name}: {exc}", flush=True)
            continue

        why = reject_reason(rgb)
        if why is not None:
            rejects[why] = rejects.get(why, 0) + 1
            continue

        # 统一短边到 args.side（不裁剪，保留完整构图，后续合成时再随机裁）
        w, h = img.size
        scale = args.side / float(min(w, h))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                             Image.BICUBIC)

        out_name = f"bg_{len(records):04d}.jpg"
        img.save(out_dir / out_name, quality=92)
        records.append({
            "file": out_name,
            "source": str(path.relative_to(src_root)).replace("\\", "/"),
            "size": list(img.size),
        })

        if len(records) % 50 == 0:
            print(f"[bg] 已保存 {len(records)} 张（扫过 {i + 1}/{len(imgs)}）", flush=True)

    summary = {
        "source_images": len(imgs),
        "saved": len(records),
        "rejects": rejects,
        "side": args.side,
        "records": records,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[bg] 完成：保存 {len(records)} 张田间背景")
    print(f"[bg] 拒绝统计：{rejects}")
    print(f"[bg] 输出：{out_dir}")
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
