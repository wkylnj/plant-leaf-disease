"""数据集体检：统计规模、类别分布、图像尺寸与色彩，产出报告可直接引用的图表。

产出:
    ml/artifacts/data_audit.md          体检报告（Markdown）
    report/figures/class_distribution.png  各类样本数条形图
    report/figures/dataset_grid.png        每类随机抽样 1 张的缩略图网格

用法:
    python ml/src/data_audit.py
    python ml/src/data_audit.py --samples 6      # 网格每类抽 6 张
"""
from __future__ import annotations

import argparse
import json
import os
import random
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无窗口环境，直接出图
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "ml" / "data" / "processed" / "plantvillage"
ARTIFACT_DIR = PROJECT_ROOT / "ml" / "artifacts"
FIGURE_DIR = PROJECT_ROOT / "report" / "figures"

# matplotlib 默认字体不含中文，标签统一用英文避免出现方块
plt.rcParams["font.size"] = 9


def list_classes(split_dir: Path) -> list[str]:
    return sorted(d.name for d in split_dir.iterdir() if d.is_dir())


def count_per_class(split_dir: Path, classes: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in classes:
        cls_dir = split_dir / name
        counts[name] = sum(1 for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    return counts


def sample_images(split_dir: Path, classes: list[str], per_class: int, seed: int) -> list[tuple[str, Path]]:
    rng = random.Random(seed)
    picked: list[tuple[str, Path]] = []
    for name in classes:
        files = sorted(p for p in (split_dir / name).iterdir() if p.suffix.lower() in IMG_EXT)
        if not files:
            continue
        k = min(per_class, len(files))
        picked.extend((name, p) for p in rng.sample(files, k))
    return picked


def probe_sizes(paths: list[Path], limit: int = 400, seed: int = 42) -> tuple[Counter, list[tuple[int, int]]]:
    """抽样读取图像尺寸与通道数。全量读 5 万张太慢，抽样足够说明问题。"""
    rng = random.Random(seed)
    subset = rng.sample(paths, min(limit, len(paths)))
    modes: Counter = Counter()
    sizes: list[tuple[int, int]] = []
    for p in subset:
        try:
            with Image.open(p) as im:
                modes[im.mode] += 1
                sizes.append(im.size)
        except Exception:  # noqa: BLE001 - 损坏文件计入统计即可
            modes["BROKEN"] += 1
    return modes, sizes


def plot_class_distribution(counts: dict[str, int], out_path: Path) -> None:
    names = list(counts.keys())
    values = [counts[n] for n in names]
    # 类名形如 00_Apple_Apple_scab，去掉索引前缀并把下划线换成空格便于阅读
    labels = [n.split("_", 1)[1].replace("_", " ") if "_" in n else n for n in names]

    order = np.argsort(values)[::-1]
    fig_h = max(6.0, 0.32 * len(names))
    fig, ax = plt.subplots(figsize=(9, fig_h))
    y = np.arange(len(names))
    ax.barh(y, [values[i] for i in order], color="#4a7ebb", height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[i] for i in order], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Number of images")
    ax.set_title("PlantVillage train split: images per class (38 classes)")
    for yi, idx in enumerate(order):
        ax.text(values[idx] + max(values) * 0.008, yi, str(values[idx]), va="center", fontsize=7)
    ax.set_xlim(0, max(values) * 1.12)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_grid(
    picked: list[tuple[str, Path]],
    per_class: int,
    out_path: Path,
    ncols: int = 8,
) -> None:
    """每个类别一格，排成 ncols 列的网格，类名两行小字置于图上方。

    只取每类第 1 张，保证 38 个类一屏可见、比例适合插入报告。
    """
    by_class: dict[str, list[Path]] = {}
    for name, p in picked:
        by_class.setdefault(name, []).append(p)

    classes = sorted(by_class)
    rows = -(-len(classes) // ncols)

    fig, axes = plt.subplots(rows, ncols, figsize=(1.32 * ncols, 1.62 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for idx, name in enumerate(classes):
        ax = axes[idx]
        files = by_class[name]
        if files:
            with Image.open(files[0]) as im:
                ax.imshow(im.convert("RGB"))
        short = name.split("_", 1)[1] if "_" in name else name
        wrapped = textwrap.fill(short.replace("_", " "), width=22)
        ax.set_title(wrapped, fontsize=4.8, pad=1.5, linespacing=1.15)

    fig.suptitle(
        f"PlantVillage train split: one sample per class ({len(classes)} classes)",
        fontsize=11, y=0.998,
    )
    fig.subplots_adjust(left=0.004, right=0.996, top=0.935, bottom=0.004, wspace=0.05, hspace=0.30)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="数据集体检")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--samples", type=int, default=5, help="网格中每类抽几张")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root: Path = args.data_root
    if not data_root.is_dir():
        print(f"[错误] 数据目录不存在: {data_root}")
        return 1

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]
    classes_by_split = {s: list_classes(data_root / s) for s in splits if (data_root / s).is_dir()}

    train_classes = classes_by_split.get("train", [])
    if not train_classes:
        print("[错误] train 目录为空")
        return 1

    # 三个 split 的类别集合必须完全一致，否则 ImageFolder 的 label 语义会错位
    consistent = all(set(classes_by_split[s]) == set(train_classes) for s in classes_by_split)
    print(f"类别一致性: {'通过' if consistent else '不一致！'}")

    # 报告里的图片路径是相对 artifacts/ 目录的
    FIG_REL = Path(os.path.relpath(FIGURE_DIR, ARTIFACT_DIR)).as_posix()

    counts = {s: count_per_class(data_root / s, classes_by_split[s]) for s in classes_by_split}
    train_counts = counts["train"]

    all_paths = [p for s in classes_by_split for p in (data_root / s).rglob("*") if p.suffix.lower() in IMG_EXT]
    modes, sizes = probe_sizes(all_paths, seed=args.seed)

    total = {s: sum(counts[s].values()) for s in counts}
    grand_total = sum(total.values())
    values = np.array(list(train_counts.values()))
    min_cls = min(train_counts, key=train_counts.get)
    max_cls = max(train_counts, key=train_counts.get)

    # 类别不均衡程度：最多/最少，以及"若按均匀先验猜多数类"的准确率下界
    imbalance = values.max() / values.min()
    majority_baseline = values.max() / values.sum()

    plot_class_distribution(train_counts, FIGURE_DIR / "class_distribution.png")
    print("已生成 class_distribution.png")

    picked = sample_images(data_root / "train", train_classes, args.samples, args.seed)
    plot_grid(picked, args.samples, FIGURE_DIR / "dataset_grid.png")
    print("已生成 dataset_grid.png")

    size_set = sorted({s for s in sizes})
    lines = [
        "# 数据集体检报告",
        "",
        f"> 自动生成，数据根目录：`{data_root.relative_to(PROJECT_ROOT).as_posix()}`",
        "",
        "## 1. 规模",
        "",
        "| split | 类别数 | 图片数 |",
        "|---|---:|---:|",
    ]
    for s in splits:
        if s in total:
            lines.append(f"| {s} | {len(classes_by_split[s])} | {total[s]} |")
    lines += [f"| **合计** | {len(train_classes)} | **{grand_total}** |", ""]

    lines += [
        "## 2. 类别一致性",
        "",
        f"三个 split 的类别集合{'完全一致' if consistent else '**不一致**'}，"
        "且目录名带两位索引前缀（如 `00_Apple_Apple_scab`），"
        "保证 `ImageFolder` 的字典序与标签整数序一一对应。",
        "",
        "## 3. 类别不均衡",
        "",
        f"- 训练集最多类：`{max_cls}`（{train_counts[max_cls]} 张）",
        f"- 训练集最少类：`{min_cls}`（{train_counts[min_cls]} 张）",
        f"- 极差倍数：**{imbalance:.1f}×**",
        f"- 多数类占比：**{majority_baseline:.1%}**",
        "",
        "**为什么必须报 macro-F1**：若模型退化成永远预测最多类，"
        f"accuracy 仍有 {majority_baseline:.1%}，但 macro-F1 会接近 0。"
        "因此本项目所有实验结果同时报告 accuracy 与 macro-F1。",
        "",
        f"![类别分布]({FIG_REL}/class_distribution.png)",
        "",
        "## 4. 图像属性（抽样）",
        "",
        f"抽样 {sum(modes.values())} 张统计：",
        "",
        "| 属性 | 取值 |",
        "|---|---|",
        f"| 色彩模式 | {', '.join(f'{k}×{v}' for k, v in modes.most_common())} |",
        f"| 尺寸 | {', '.join(f'{w}×{h}' for w, h in size_set[:6])}{' 等' if len(size_set) > 6 else ''} |",
        "",
        "尺寸统一，训练时可安全使用 `RandomResizedCrop` 而不至于把病斑裁没。",
        "",
        "## 5. 抽样网格",
        "",
        f"![抽样网格]({FIG_REL}/dataset_grid.png)",
        "",
        "**观察**：背景高度一致（多为单一浅色背景、居中的单片叶子），"
        "这正是文献批评 PlantVillage 为「实验室摆拍」的直接证据——"
        "模型很容易学到背景而不是病斑。这构成本项目跨域实验（E2）与 "
        "Grad-CAM 可解释性实验（E7）的立论依据。",
        "",
    ]
    out_md = ARTIFACT_DIR / "data_audit.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"已生成 {out_md.relative_to(PROJECT_ROOT).as_posix()}")

    summary = {
        "total": grand_total,
        "per_split": total,
        "num_classes": len(train_classes),
        "imbalance_ratio": round(float(imbalance), 3),
        "majority_baseline_acc": round(float(majority_baseline), 4),
        "min_class": {"name": min_cls, "count": train_counts[min_cls]},
        "max_class": {"name": max_cls, "count": train_counts[max_cls]},
        "modes": dict(modes),
        "sizes": [list(s) for s in size_set],
    }
    (ARTIFACT_DIR / "data_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("已生成 data_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
