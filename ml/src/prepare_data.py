"""把 HuggingFace 的 PlantVillage parquet 转成标准 ImageFolder 结构。

为什么要转：
    原始 parquet 里图片是二进制 blob，每次训练都要在内存里解压 + 解码，
    Colab 上换一次随机种子就要重跑一遍解析。转成 jpg 目录后：
      1. torchvision.datasets.ImageFolder 直接可用，不需要自定义 Dataset；
      2. 数据集可以打包成 zip 上传网盘，队友不用重新下载 852MB；
      3. 单张图可以直接用眼睛看，调试数据问题时不用写代码。

输出结构：
    data/processed/plantvillage/
        train/<idx>_<class>/xxx.jpg
        val/<idx>_<class>/xxx.jpg
        test/<idx>_<class>/xxx.jpg
        classes.json          # 类别索引 -> 文件夹名 / 可读名
        manifest.csv          # 每张图的 split / 路径 / 类别，供后续分析用

用法：
    python src/prepare_data.py                       # 全量
    python src/prepare_data.py --limit-per-class 20  # 每类只取 20 张，本地冒烟测试
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

# 原始 parquet 里的 train / test 文件名是 HF 自动切的，这里按前缀匹配，避免写死分片数
TRAIN_GLOB = "train-*.parquet"
TEST_GLOB = "test-*.parquet"


def sanitize(name: str) -> str:
    """把类别名变成安全的文件夹名。

    PlantVillage 的原始类名长这样：
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot'
        'Pepper,_bell___Bacterial_spot'
    里面有空格、逗号、括号，直接当目录名在 Windows 上会很麻烦，
    而且 ImageFolder 按字典序排目录，顺序未必和 label 的整数一致。
    所以统一替换掉非字母数字字符，再在前面补零填充的索引。
    """
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
    return cleaned or "class"


def read_class_names(parquet_path: Path) -> list[str]:
    """从 parquet 的 HuggingFace 元数据里读出类别名，而不是自己手抄一遍。

    手抄 38 个名字早晚会错一个，而且顺序错了训练照样跑，只是标签全错位，
    这种 bug 极难发现。所以直接从文件自带的 metadata 里解析。
    """
    meta = pq.ParquetFile(parquet_path).schema_arrow.metadata or {}
    raw = meta.get(b"huggingface")
    if raw is None:
        raise RuntimeError(f"{parquet_path} 没有 huggingface 元数据，无法读出类别名")

    info = json.loads(raw.decode("utf-8"))["info"]
    label_feature = info["features"]["label"]

    # 不同版本的 datasets 库写出来的元数据格式不一致，这里把见过的几种都兼容上：
    #   {"names": [...], "_type": "ClassLabel"}          <- 本数据集实际是这种
    #   {"class_label": {"names": [...]}}                <- 老版本
    #   {"class_label": {"names": {"0": "a", "1": "b"}}} <- 更老的、names 是字典
    if "names" in label_feature:
        names = label_feature["names"]
    elif "class_label" in label_feature:
        names = label_feature["class_label"]["names"]
    else:
        raise RuntimeError(f"看不懂的 label 元数据格式：{list(label_feature)}")

    if isinstance(names, dict):
        # 键是字符串形式的整数，必须按数值排序，否则 '10' 会排在 '2' 前面
        return [names[k] for k in sorted(names, key=int)]
    return list(names)


def iter_rows(parquet_path: Path):
    """逐行读取，避免一次性把 300MB 的 blob 全读进内存。"""
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=256, columns=["image", "label"]):
        for row in batch.to_pylist():
            yield row["label"], row["image"]["bytes"], row["image"]["path"]


def convert_split(
    parquet_paths: list[Path],
    out_dir: Path,
    split: str,
    class_folders: list[str],
    limit_per_class: int | None,
    manifest_writer,
) -> dict[int, int]:
    """把一个 split 的所有 parquet 分片写成 jpg，返回每类实际写入的张数。"""
    counts: dict[int, int] = {}
    for pq_path in parquet_paths:
        for label, blob, src_name in iter_rows(pq_path):
            if limit_per_class is not None and counts.get(label, 0) >= limit_per_class:
                continue

            # 用原文件名，但去掉可能重复的前缀目录，并强制 .jpg 后缀
            stem = Path(src_name).stem
            dst = out_dir / class_folders[label] / f"{stem}.jpg"
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(io.BytesIO(blob)) as im:
                    # 统一转 RGB：万一有灰度或带 alpha 的图，训练时通道数会对不上
                    im.convert("RGB").save(dst, "JPEG", quality=95)

            counts[label] = counts.get(label, 0) + 1
            manifest_writer.writerow(
                [split, str(dst.relative_to(out_dir.parent)), label, class_folders[label]]
            )
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/plantvillage")
    ap.add_argument("--out-dir", default="data/processed/plantvillage")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="从 train 里切出的验证集比例")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-per-class", type=int, default=None, help="调试用，每类只留 N 张")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    train_parts = sorted(raw_dir.glob(TRAIN_GLOB))
    test_parts = sorted(raw_dir.glob(TEST_GLOB))
    if not train_parts:
        raise SystemExit(f"在 {raw_dir} 找不到 {TRAIN_GLOB}，先跑下载脚本")

    class_names = read_class_names(train_parts[0])
    width = len(str(len(class_names) - 1))
    # 目录名带索引前缀，保证 ImageFolder 的字典序 == label 的数值序
    class_folders = [f"{i:0{width}d}_{sanitize(n)}" for i, n in enumerate(class_names)]

    print(f"类别数：{len(class_names)}")
    print(f"train 分片：{[p.name for p in train_parts]}")
    print(f"test  分片：{[p.name for p in test_parts]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classes.json").write_text(
        json.dumps(
            [
                {"index": i, "folder": class_folders[i], "name": class_names[i]}
                for i in range(len(class_names))
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 先把 train 全部写成 jpg，再按比例切成 train / val
    # 这样可以保证同一张图不会既在 train 又在 val
    staging = out_dir / "_staging"
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "path", "label", "class_folder"])
        train_counts = convert_split(
            train_parts, staging, "train", class_folders, args.limit_per_class, writer
        )
        test_counts = convert_split(
            test_parts, out_dir / "test", "test", class_folders, args.limit_per_class, writer
        )

    # 分层切分：每个类别内部按 val_ratio 抽，保证小类别在验证集里也有样本
    rng = random.Random(args.seed)
    moved = 0
    for label, folder in enumerate(class_folders):
        src_dir = staging / folder
        if not src_dir.is_dir():
            continue
        files = sorted(src_dir.glob("*.jpg"))
        rng.shuffle(files)
        n_val = max(1, int(len(files) * args.val_ratio)) if len(files) > 1 else 0
        for fp in files[:n_val]:
            dst = out_dir / "val" / folder / fp.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            fp.rename(dst)
            moved += 1

    # 把剩余文件从 _staging 挪到 train。
    # 不能直接 staging.rename(train)：目标目录已存在时 Windows 会抛 WinError 183，
    # 而重复跑脚本时 train 目录必然是存在的，所以逐个类别目录移动。
    train_dir = out_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    for folder in class_folders:
        src_dir = staging / folder
        if not src_dir.is_dir():
            continue
        dst_dir = train_dir / folder
        dst_dir.mkdir(parents=True, exist_ok=True)
        for fp in src_dir.glob("*.jpg"):
            fp.replace(dst_dir / fp.name)
        src_dir.rmdir()
    if staging.is_dir():
        try:
            staging.rmdir()  # 只有空目录才能删掉，非空说明有类别名没被处理，留着让人看
        except OSError:
            print(f"注意：{staging} 非空，未删除")

    total = sum(train_counts.values()) + sum(test_counts.values())
    print(f"\n写入 {total} 张，其中验证集 {moved} 张")
    print(f"train {sum(train_counts.values()) - moved} / val {moved} / test {sum(test_counts.values())}")
    print(f"最少类的样本数：{min(train_counts.values())}，最多：{max(train_counts.values())}")
    print(f"输出目录：{out_dir.resolve()}")


if __name__ == "__main__":
    main()
