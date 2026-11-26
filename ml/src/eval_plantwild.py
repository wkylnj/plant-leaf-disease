"""PlantVillage → PlantWild 跨域评估。

为什么需要这个脚本
------------------
PlantDoc 上所有「健康」类 top-1 都是 0%（`06_Cherry_including_sour_healthy`、
`37_Tomato_healthy`），一直无法判断是模型缺陷还是 PlantDoc 的健康叶图太脏
（田间多叶、带土、有阴影）。PlantWild v1 含 13 个能对上的健康类、约 2600 张，
可以给出答案。

数据结构
--------
`fetch_plantwild.py` 已把 PlantWild v1 抽成
`<root>/<PlantVillage 文件夹名>/<原文件名>.jpg`，
所以这里只需按 `ck["classes"]` 的顺序建立「文件夹名 → 标签索引」映射。
注意**不能**用 `sorted(目录名)` 当标签顺序——那会和模型输出通道错位
（`eval_crossdomain.py` 早期踩过这个坑，top-1 从 33.73% 掉到 6.75%）。

用法
----
    python src/eval_plantwild.py --run-dir runs/swin_t_224_15ep
    python src/eval_plantwild.py --run-dir runs/resnet18_224_15ep --img-size 320
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import build_transforms  # noqa: E402
from eval_crossdomain import crop_of_folder, evaluate  # noqa: E402
from models import build_model  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class PlantWildDataset(Dataset):
    """按 PV 文件夹名映射标签的简单数据集。"""

    def __init__(self, root: Path, pv_folders: list[str], transform,
                 limit_per_class: int = 0):
        self.transform = transform
        self.folder_index = {f: i for i, f in enumerate(pv_folders)}
        self.items: list[tuple[Path, int, str]] = []
        self.skipped: dict[str, int] = {}
        self.missing: list[str] = []

        for pv in pv_folders:
            d = root / pv
            if not d.is_dir():
                self.missing.append(pv)
                continue
            files = [f for f in sorted(d.iterdir()) if f.suffix.lower() in IMG_EXT]
            if limit_per_class > 0:
                files = files[:limit_per_class]
            for f in files:
                self.items.append((f, self.folder_index[pv], pv))

        # 目录里存在但不属于 38 类的文件夹 → 记为跳过
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if d.name not in self.folder_index:
                n = sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
                self.skipped[d.name] = n

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label, _pv = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def is_healthy(folder: str) -> bool:
    return "healthy" in folder.lower()


def main() -> int:
    ap = argparse.ArgumentParser(description="PlantVillage → PlantWild 跨域评估")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--plantwild-root", default=str(ROOT / "data" / "raw" / "plantwild_subset"))
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--limit-per-class", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()
    ckpt = run_dir / "best.pt"
    if not ckpt.exists():
        print(f"[错误] 找不到 checkpoint: {ckpt}")
        return 1

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model_name = ck.get("args", {}).get("model") or run_dir.name.rsplit("_", 2)[0]
    pv_folders = list(ck["classes"])
    num_classes = len(pv_folders)

    print(f"模型      : {model_name} (epoch {ck.get('epoch', '?')})")
    print(f"PlantWild : {args.plantwild_root}")
    print(f"推理分辨率: {args.img_size}")

    model, _meta = build_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(ck["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    ds = PlantWildDataset(Path(args.plantwild_root), pv_folders,
                          build_transforms(args.img_size, train=False),
                          limit_per_class=args.limit_per_class)
    if len(ds) == 0:
        print("[错误] 没有可用样本，请先运行 src/fetch_plantwild.py")
        return 1

    print(f"可用样本  : {len(ds)}")
    print(f"覆盖类别  : {num_classes - len(ds.missing)}/{num_classes}")
    if ds.missing:
        print(f"缺失类别  : {len(ds.missing)} 个（PlantWild 无对应）")
        for m in ds.missing:
            print(f"    {m}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    res = evaluate(model, loader, device, num_classes)

    pv_crop = [crop_of_folder(f) for f in pv_folders]
    crop_correct = sum(1 for p, t in zip(res["_pred"], res["_true"]) if pv_crop[p] == pv_crop[t])
    crop_acc = crop_correct / res["n"] if res["n"] else 0.0
    mean_conf = sum(res["_conf"]) / max(len(res["_conf"]), 1)

    print()
    print("=" * 62)
    print(f"跨域 top-1 准确率 : {res['top1_acc']:.4f}  ({res['correct']}/{res['n']})")
    print(f"跨域 top-3 准确率 : {res['top3_acc']:.4f}")
    print(f"作物级准确率      : {crop_acc:.4f}  ({crop_correct}/{res['n']})")
    print(f"平均置信度        : {mean_conf:.4f}")
    print(f"耗时              : {res['elapsed_s']}s ({res['img_per_s']} img/s)")
    print("=" * 62)

    # ---- 健康类 vs 病害类（本脚本的核心问题）----
    per_class: dict[int, dict] = defaultdict(lambda: {"n": 0, "correct": 0, "crop_ok": 0, "conf": 0.0})
    for pr, tr, cf in zip(res["_pred"], res["_true"], res["_conf"]):
        d = per_class[tr]
        d["n"] += 1
        d["conf"] += cf
        if pr == tr:
            d["correct"] += 1
        if pv_crop[pr] == pv_crop[tr]:
            d["crop_ok"] += 1

    def group_stats(pred) -> dict:
        n = sum(v["n"] for k, v in per_class.items() if pred(pv_folders[k]))
        c = sum(v["correct"] for k, v in per_class.items() if pred(pv_folders[k]))
        cc = sum(v["crop_ok"] for k, v in per_class.items() if pred(pv_folders[k]))
        cf = sum(v["conf"] for k, v in per_class.items() if pred(pv_folders[k]))
        return {
            "n": n,
            "top1": c / n if n else 0.0,
            "crop": cc / n if n else 0.0,
            "conf": cf / n if n else 0.0,
        }

    healthy = group_stats(is_healthy)
    diseased = group_stats(lambda f: not is_healthy(f))

    print()
    print("健康类 vs 病害类（关键对比）")
    print(f"  健康类  n={healthy['n']:5d}  top1={healthy['top1']:.4f}  "
          f"作物级={healthy['crop']:.4f}  conf={healthy['conf']:.4f}")
    print(f"  病害类  n={diseased['n']:5d}  top1={diseased['top1']:.4f}  "
          f"作物级={diseased['crop']:.4f}  conf={diseased['conf']:.4f}")

    print()
    print("逐类结果（按 top-1 升序）")
    rows = []
    for idx, d in per_class.items():
        folder = pv_folders[idx]
        rows.append({
            "class": folder,
            "n": d["n"],
            "top1": d["correct"] / d["n"],
            "crop": d["crop_ok"] / d["n"],
            "conf": d["conf"] / d["n"],
            "healthy": is_healthy(folder),
        })
    rows.sort(key=lambda r: (r["top1"], -r["n"]))
    for r in rows:
        tag = " [健康]" if r["healthy"] else ""
        print(f"  {r['top1']:.4f}  n={r['n']:4d}  crop={r['crop']:.4f}  "
              f"conf={r['conf']:.4f}  {r['class']}{tag}")

    # ---- 产物 ----
    out_path = Path(args.out) if args.out else (run_dir / f"plantwild_{args.img_size}.json")
    if not out_path.is_absolute():
        out_path = (ROOT / out_path).resolve()
    payload = {
        "run_dir": str(run_dir),
        "model": model_name,
        "ckpt_epoch": ck.get("epoch"),
        "img_size": args.img_size,
        "n": res["n"],
        "top1_acc": res["top1_acc"],
        "top3_acc": res["top3_acc"],
        "crop_acc": crop_acc,
        "mean_conf": mean_conf,
        "elapsed_s": res["elapsed_s"],
        "healthy": healthy,
        "diseased": diseased,
        "per_class": rows,
        "missing_classes": ds.missing,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物 {out_path}")

    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "n", "top1", "crop", "conf", "healthy"])
        for r in rows:
            w.writerow([r["class"], r["n"], f"{r['top1']:.4f}",
                        f"{r['crop']:.4f}", f"{r['conf']:.4f}", int(r["healthy"])])
    print(f"产物 {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
