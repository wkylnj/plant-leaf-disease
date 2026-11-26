"""从 PlantWild v1 抽取映射到 PlantVillage 38 类的子集。

设计要点
--------
1. **只抽需要的 32 类**，不下载/解压 2.7 GB 整包里的其余 57 类。
   用 HTTP Range + zipfile 边读边抽（复用 scripts/probe_remote_zip.py 的 RangeFile），
   实测抽 7455 张约需下载 700~900 MB（只取需要的字节）。
2. **输出结构对齐 PlantDocDataset 的用法**：`<out>/<PlantVillage 文件夹名>/<原文件名>.jpg`，
   这样现有的 `eval_crossdomain.py` / `eval_preprocess_ablation.py` 只要把 root 换成新目录、
   把类别映射换成 `plantwild_map.PLANTWILD_TO_PV` 就能复用。
3. 同时写 `manifest.csv`（原路径、目标类别、字节数）便于审计。

用法：
    python src/fetch_plantwild.py --out data/raw/plantwild_subset
    python src/fetch_plantwild.py --out data/raw/plantwild_subset --limit-per-class 100
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import time
import zipfile
from pathlib import Path

SRC = Path(__file__).resolve().parent      # ml/src
ROOT = SRC.parent                          # ml
REPO = ROOT.parent                         # 仓库根
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plantwild_map import PLANTWILD_TO_PV  # noqa: E402
from probe_remote_zip import RangeFile  # noqa: E402

DEFAULT_URL = "https://huggingface.co/datasets/uqtwei2/PlantWild/resolve/main/plantwild.zip"
PREFIX = "plantwild/images/"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_plan(names: list[str], limit_per_class: int) -> list[tuple[str, str, str]]:
    """返回按 **zip 内顺序** 排好的抽取计划，保证读取是顺序的。

    早期版本按 `PLANTWILD_TO_PV` 的字典顺序遍历类别，而 zip 条目按路径
    字母序排列，两者顺序不一致导致每次读取都在远程文件里来回跳，
    1 MB 块缓存几乎每次都要重新发 Range 请求（实测 10 分钟只下 240 张）。
    这里改成单遍扫描 `names`（本身即 zip 顺序），命中才收，
    顺序访问能把速度提升一个量级。

    返回 `[(zip 内路径, PlantVillage 文件夹名, 输出文件名), ...]`。
    """
    per_class_count: dict[str, int] = {}
    plan: list[tuple[str, str, str]] = []
    # PLANTWILD_TO_PV 的方向是「PV 文件夹名 → PlantWild 目录名」，这里要反查
    pw_to_pv = {pw: pv for pv, pw in PLANTWILD_TO_PV.items() if pw is not None}

    for n in names:
        if not n.lower().endswith(IMG_EXT):
            continue
        if not n.startswith(PREFIX):
            continue
        rest = n[len(PREFIX):]
        parts = [p for p in rest.split("/") if p]
        if len(parts) < 2:
            continue
        pw_class = parts[0]
        pv_class = pw_to_pv.get(pw_class)
        if pv_class is None:
            continue
        if limit_per_class > 0:
            used = per_class_count.get(pw_class, 0)
            if used >= limit_per_class:
                continue
            per_class_count[pw_class] = used + 1
        plan.append((n, pv_class, Path(n).name))
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description="从 PlantWild v1 抽取可映射子集")
    ap.add_argument("--out", default="data/raw/plantwild_subset")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--limit-per-class", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = ROOT / out_root

    rf = RangeFile(args.url, block=8 << 20)
    print(f"[pw] 远程 {rf.size / 1e6:.1f} MB")
    zf = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=8 << 20))
    names = zf.namelist()

    plan = build_plan(names, args.limit_per_class)
    total = len(plan)
    print(f"[pw] 可映射 {len({pv for _, pv, _ in plan})} 类，共 {total} 张")

    manifest = out_root / "manifest.csv"
    out_root.mkdir(parents=True, exist_ok=True)
    written = skipped = failed = 0
    rows: list[tuple[str, str, str, int]] = []
    t0 = time.time()

    with manifest.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["plantwild_path", "pv_class", "out_name", "bytes"])
        for m, pv, out_name in plan:
            cls_dir = out_root / pv
            cls_dir.mkdir(parents=True, exist_ok=True)
            dst = cls_dir / out_name
            if args.skip_existing and dst.exists() and dst.stat().st_size > 0:
                skipped += 1
                continue
            try:
                data = zf.read(m)
            except Exception as e:  # noqa: BLE001
                failed += 1
                if failed <= 5:
                    print(f"[pw] 读取失败 {m}: {type(e).__name__} {e}")
                continue
            dst.write_bytes(data)
            written += 1
            rows.append((m, pv, out_name, len(data)))
            if written % 500 == 0:
                el = time.time() - t0
                print(f"[pw] 已写 {written}/{total} 张，用时 {el:.0f}s，"
                      f"速度 {written / max(el, 1):.1f} 张/s", flush=True)
        for r in rows:
            w.writerow(r)

    el = time.time() - t0
    print(f"[pw] 完成：新写 {written}，跳过已有 {skipped}，失败 {failed}，"
          f"用时 {el:.0f}s，输出 {out_root}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
