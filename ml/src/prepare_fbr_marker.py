#!/usr/bin/env python3
"""把 FBR 合成图整理成 train_colab.py 认得的目录结构，并写 `.prepared` 标记。

为什么需要这一步（而不是直接 `--data-root data/processed/pv_fbr`）：
    `train_colab.py:download_and_prepare()` 的判断只有一行——

        marker = root / ".prepared"
        if marker.exists(): return root

    也就是说：**只要 `.prepared` 在，就完全跳过 HuggingFace 下载**。但紧接着
    `main()` 会读 `data_root/train`、`/val`、`/test` 三个 ImageFolder 子目录
    （见 train_colab.py:435-449），而 `make_fbr_dataset.py` 输出的是**扁平**的
    `<类文件夹>/*.jpg`（见其 main() 里 out_dir = out_root / cdir.name）。
    两者对不上，所以这里补上 split 这一层，并且只写标记、不改任何既有脚本。

    这样做的好处：train_colab.py 一行都不用改，FBR 实验就是纯粹的
    「--data-root 指向别处」，与基线完全同一条代码路径，可比性最强。

split 怎么划分（关键设计）：
    FBR 合成图的源图来自 `plantvillage/train`，而 PlantVillage 的 val/test 是
    **另外**的原始图。所以：
      - FBR 的 train/val 从合成图内部按比例切（合成图内部切分，互不重叠）
      - FBR 的 test **必须**放原始 PlantVillage test 的副本/软链，不能放合成图
        否则 train_colab.py 结尾的 test 指标就成了「合成图测试」，毫无意义。
        同域数字请改用 `eval_ckpt.py --data-root data/processed/plantvillage`，
        这里的 test 只是为了让 ImageFolder 能建起来、跑通流程。

用法（在 ml/ 目录下）：
    python src/prepare_fbr_marker.py --fbr-root data/processed/pv_fbr
    python src/prepare_fbr_marker.py --fbr-root data/processed/pv_fbr --copy-test
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# 输出目录就是 fbr_root 本身，跑过一次后会多出 train/val/test。
# 再跑时必须把它们排除掉，否则会被当成「类别」参与校验，误报类别不一致。
RESERVED = {"train", "val", "test"}


def class_dirs_of(d: Path) -> list[Path]:
    """列出真正的类别文件夹，排除 split 目录与隐藏目录。"""
    return sorted(
        x for x in d.iterdir() if x.is_dir() and x.name not in RESERVED and not x.name.startswith(".")
    )


def log(msg: str) -> None:
    print(f"[fbr-prep] {msg}", flush=True)


def list_images(d: Path) -> list[Path]:
    return [p for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]


def split_synthetic(
    fbr_root: Path,
    out_root: Path,
    val_ratio: float,
    seed: int,
    copy: bool,
) -> tuple[int, int]:
    """把扁平合成图切成 train/val。用硬链接优先（省磁盘），失败退回复制。"""
    rng = random.Random(seed)
    n_train = n_val = 0
    class_dirs = class_dirs_of(fbr_root)
    for cdir in class_dirs:
        files = list_images(cdir)
        if not files:
            continue
        rng.shuffle(files)
        # 小规模冒烟时每类可能只有几张，int() 会算出 0 张验证图导致 val/ 空目录。
        # 只要该类至少 2 张，就保证留 1 张给 val。
        n_val_c = int(len(files) * val_ratio)
        if n_val_c == 0 and len(files) >= 2:
            n_val_c = 1
        val_files, train_files = files[:n_val_c], files[n_val_c:]
        for split, items in (("train", train_files), ("val", val_files)):
            dst_dir = out_root / split / cdir.name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src in items:
                dst = dst_dir / src.name
                if dst.exists():
                    continue
                if copy:
                    shutil.copy2(src, dst)
                else:
                    try:
                        dst.hardlink_to(src)
                    except (OSError, AttributeError):
                        shutil.copy2(src, dst)
        n_train += len(train_files)
        n_val += len(val_files)
    return n_train, n_val


def link_test(out_root: Path, pv_test: Path, copy: bool) -> int:
    """把原始 PlantVillage test 挂进来，只为让 ImageFolder 能建起来。

    注意：FBR 模型的同域指标请用 eval_ckpt.py 单独测，别信 train_colab.py
    结尾打的那行 test acc（这里挂的是原始 test，倒也没错，但 val 来自合成图，
    best.pt 是按合成图 val 选的，口径要写清楚）。
    """
    if not pv_test.is_dir():
        log(f"警告：找不到原始 test 目录 {pv_test}，跳过（train_colab 会因缺 test 报错）")
        return 0
    n = 0
    for cdir in sorted([d for d in pv_test.iterdir() if d.is_dir()]):
        dst_dir = out_root / "test" / cdir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in list_images(cdir):
            dst = dst_dir / src.name
            if dst.exists():
                # 已存在（重复运行）也要计入，否则日志会误报 test=0
                n += 1
                continue
            if copy:
                shutil.copy2(src, dst)
            else:
                try:
                    dst.hardlink_to(src)
                except (OSError, AttributeError):
                    shutil.copy2(src, dst)
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbr-root", default="data/processed/pv_fbr",
                    help="make_fbr_dataset.py 的输出（扁平 <类>/*.jpg）")
    ap.add_argument("--pv-root", default="data/processed/plantvillage",
                    help="原始 PlantVillage，用于提供 test split 与类别清单")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true",
                    help="复制而不是硬链接（跨盘符时硬链接会失败，脚本已自动回退）")
    ap.add_argument("--copy-test", action="store_true",
                    help="test split 也用复制（默认硬链接）")
    args = ap.parse_args()

    def resolve(p: str) -> Path:
        """相对路径按 ml/ 解析；绝对路径原样使用。

        注意不能直接写 `ROOT / p`：Kaggle 上 run.py 传的是
        `/kaggle/working/data/processed/pv_fbr` 这种绝对路径，
        而 Windows 下 `Path('D:/x/ml') / '/kaggle/y'` 会拼成
        `D:/kaggle/y`（盘符+根路径被吞），结果指向错误位置。
        """
        q = Path(p)
        return q if q.is_absolute() else ROOT / q

    fbr_root = resolve(args.fbr_root)
    pv_root = resolve(args.pv_root)
    out_root = fbr_root  # 就地组织，不另建目录

    if not fbr_root.is_dir():
        log(f"错误：FBR 目录不存在 {fbr_root}，请先运行 make_fbr_dataset.py")
        return 1

    class_dirs = class_dirs_of(fbr_root)
    if not class_dirs:
        log(f"错误：{fbr_root} 下没有类别文件夹")
        return 1
    log(f"FBR 类别 {len(class_dirs)} 个，输出到 {out_root}")

    # 类别名必须与原始 PlantVillage 一致，否则跨域评估的类别映射会错位
    pv_classes = [d.name for d in sorted((pv_root / "train").iterdir()) if d.is_dir()]
    fbr_classes = [d.name for d in class_dirs]
    if pv_classes != fbr_classes:
        missing = sorted(set(pv_classes) - set(fbr_classes))
        extra = sorted(set(fbr_classes) - set(pv_classes))
        log(f"错误：类别不一致。缺 {missing[:5]}…，多 {extra[:5]}…")
        return 1
    log("类别与原始 PlantVillage 一致 ✓")

    n_train, n_val = split_synthetic(fbr_root, out_root, args.val_ratio, args.seed, args.copy)
    log(f"train={n_train} val={n_val}（val_ratio={args.val_ratio}）")

    n_test = link_test(out_root, pv_root / "test", args.copy_test)
    log(f"test={n_test}（来自原始 PlantVillage test）")

    # 写标记文件：这是让 train_colab.py 跳过 HuggingFace 下载的唯一开关
    (out_root / ".prepared").write_text(
        json.dumps(
            {
                "source": "fbr",
                "fbr_root": str(args.fbr_root).replace("\\", "/"),
                "val_ratio": args.val_ratio,
                "seed": args.seed,
                "n_train": n_train,
                "n_val": n_val,
                "n_test": n_test,
                "note": "train/val 为 FBR 合成图；test 为原始 PlantVillage test",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"已写标记 {out_root / '.prepared'}")

    # 复核：三个 split 的类别集合一致
    for split in ("train", "val", "test"):
        d = out_root / split
        if not d.is_dir():
            log(f"警告：缺 {split}/")
            continue
        got = sorted([x.name for x in d.iterdir() if x.is_dir()])
        if got != pv_classes:
            log(f"警告：{split}/ 类别与原始不一致（{len(got)} vs {len(pv_classes)}）")
        else:
            log(f"{split}/ 类别 ✓ ({len(got)} 类, {sum(len(list_images(x)) for x in d.iterdir() if x.is_dir())} 张)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
