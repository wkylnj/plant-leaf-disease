"""E2 跨域实验：把在 PlantVillage 上训练的模型直接拿到 PlantDoc 上测试。

要回答的问题
------------
PlantVillage 是实验室摆拍（单一叶片、统一背景、恒定光照），PlantDoc 是真实
田间/手机拍摄（多叶片、复杂背景、光照不均、不同设备）。同域 99% 的模型在
跨域上还剩多少？

方法说明
--------
1. PlantDoc 只有 28 类，与 PlantVillage 的 38 类不是一一对应，需要手工映射。
   只保留能确定对应的类别；映射不上的样本直接剔除，并在结果里说明覆盖率。
2. **不做任何微调**，权重原样加载，只换测试集。这是"跨域泛化"的严格定义。
3. 除了 top-1 准确率，还统计两个更有信息量的指标：
   - **作物级准确率**：只要预测出的是同一种作物（如都是番茄），即使病害判错
     也算对。若作物级很高而病害级很低，说明模型认得出是什么植物，但认不出病。
   - **top-3 是否命中**：真实标签是否落在预测的前三个里。

用法
----
    python src/eval_crossdomain.py --run-dir runs/resnet18_224_15ep
    python src/eval_crossdomain.py --run-dir runs/mobilenet_v3_large_224_15ep --split all

产物
----
    <run-dir>/crossdomain.json         指标汇总
    <run-dir>/crossdomain_preds.csv    每张图的预测明细
    ml/artifacts/crossdomain.md        两个模型的对比报告
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import build_transforms  # noqa: E402
from models import build_model  # noqa: E402


# PlantDoc 类名 → PlantVillage **文件夹名**（classes.json 的 folder 字段）。
#
# 为什么用文件夹名而不是正式类名：PlantVillage 的正式类名分隔符不统一
# （'Apple__Apple_scab' 是双下划线，'Tomato___Early_blight' 是三下划线），
# 而文件夹名统一是 '<两位索引>_<作物>_<病害>'，用它做键更不容易出错。
#
# 只映射能确定对应的类别；PlantDoc 有 28 类，其中
# 'Tomato_two_spotted_spider_mites_leaf' 只有 2 张且病名对不上（PlantVillage 是
# Two-spotted_spider_mite），为保守起见剔除。
PLANTDOC_TO_PV_FOLDER: dict[str, str] = {
    # 苹果
    "Apple_leaf": "03_Apple_healthy",
    "Apple_rust_leaf": "02_Apple_Cedar_apple_rust",
    "Apple_Scab_Leaf": "00_Apple_Apple_scab",
    # 甜椒
    "Bell_pepper_leaf": "19_Pepper_bell_healthy",
    "Bell_pepper_leaf_spot": "18_Pepper_bell_Bacterial_spot",
    # 蓝莓 / 樱桃 / 葡萄
    "Blueberry_leaf": "04_Blueberry_healthy",
    "Cherry_leaf": "06_Cherry_including_sour_healthy",
    "grape_leaf": "14_Grape_healthy",
    "grape_leaf_black_rot": "11_Grape_Black_rot",
    # 玉米
    "Corn_Gray_leaf_spot": "07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn_leaf_blight": "09_Corn_maize_Northern_Leaf_Blight",
    "Corn_rust_leaf": "08_Corn_maize_Common_rust",
    # 桃 / 土豆
    "Peach_leaf": "17_Peach_healthy",
    "Potato_leaf_early_blight": "20_Potato_Early_blight",
    "Potato_leaf_late_blight": "21_Potato_Late_blight",
    # 其他
    "Raspberry_leaf": "23_Raspberry_healthy",
    "Soyabean_leaf": "24_Soybean_healthy",
    "Squash_Powdery_mildew_leaf": "25_Squash_Powdery_mildew",
    "Strawberry_leaf": "27_Strawberry_healthy",
    # 番茄
    "Tomato_leaf": "37_Tomato_healthy",
    "Tomato_Early_blight_leaf": "29_Tomato_Early_blight",
    "Tomato_leaf_late_blight": "30_Tomato_Late_blight",
    "Tomato_leaf_bacterial_spot": "28_Tomato_Bacterial_spot",
    "Tomato_leaf_mosaic_virus": "36_Tomato_Tomato_mosaic_virus",
    "Tomato_leaf_yellow_virus": "35_Tomato_Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mold_leaf": "31_Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot": "32_Tomato_Septoria_leaf_spot",
}

# 作物名按长度降序排列，保证 'Cherry_including_sour' 先于 'Cherry' 匹配
_CROP_KEYS = [
    "Cherry_including_sour",
    "Corn_maize",
    "Pepper_bell",
    "Strawberry",
    "Blueberry",
    "Raspberry",
    "Soybean",
    "Squash",
    "Potato",
    "Tomato",
    "Orange",
    "Apple",
    "Grape",
    "Peach",
]


def crop_of_folder(folder: str) -> str:
    """从 PlantVillage 文件夹名里取作物名。

    文件夹名格式：'<两位索引>_<作物>_<病害>'，如
    '07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot'。
    去掉索引前缀后按已知作物名做前缀匹配。
    """
    body = re.sub(r"^\d+_", "", folder)
    for crop in _CROP_KEYS:
        if body.startswith(crop):
            return crop
    # 兜底：取前两段
    parts = body.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else body


class PlantDocDataset(Dataset):
    """PlantDoc 目录 → (PIL 图, PlantVillage 标签索引)。

    只保留能映射到 PlantVillage 的类别。原图常是长条形（多叶片横排），
    这里按 ImageNet 的 resize 流程走，不做额外裁剪，保持与训练时验证集处理一致。
    """

    def __init__(
        self,
        root: Path,
        split: str,
        pv_folders: list[str],
        transform,
        limit_per_class: int = 0,
    ) -> None:
        self.transform = transform
        self.folder_index = {f: i for i, f in enumerate(pv_folders)}
        self.folders = pv_folders
        self.items: list[tuple[Path, int, str]] = []
        self.skipped: Counter[str] = Counter()

        splits = ["train", "test"] if split == "all" else [split]
        per_class_count: Counter[str] = Counter()

        for sp in splits:
            sp_dir = root / sp
            if not sp_dir.exists():
                continue
            for cls_dir in sorted(p for p in sp_dir.iterdir() if p.is_dir()):
                pv_folder = PLANTDOC_TO_PV_FOLDER.get(cls_dir.name)
                if pv_folder is None:
                    self.skipped[cls_dir.name] = len(list(cls_dir.glob("*.jpg")))
                    continue
                if pv_folder not in self.folder_index:
                    self.skipped[cls_dir.name] = len(list(cls_dir.glob("*.jpg")))
                    continue
                label = self.folder_index[pv_folder]
                for img_path in sorted(cls_dir.glob("*.jpg")):
                    if limit_per_class and per_class_count[cls_dir.name] >= limit_per_class:
                        break
                    self.items.append((img_path, label, cls_dir.name))
                    per_class_count[cls_dir.name] += 1

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label, _orig = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict:
    model.eval()
    all_pred: list[int] = []
    all_true: list[int] = []
    all_top3: list[list[int]] = []
    all_conf: list[float] = []

    t0 = time.time()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            top3 = probs.topk(min(3, num_classes), dim=1)
            all_pred.extend(top3.indices[:, 0].cpu().tolist())
            all_top3.extend(top3.indices.cpu().tolist())
            all_conf.extend(top3.values[:, 0].cpu().tolist())
            all_true.extend(y.tolist())
    elapsed = time.time() - t0

    n = len(all_true)
    correct = sum(1 for p, t in zip(all_pred, all_true) if p == t)
    top3_hit = sum(1 for p3, t in zip(all_top3, all_true) if t in p3)

    return {
        "n": n,
        "correct": correct,
        "top1_acc": correct / n if n else 0.0,
        "top3_acc": top3_hit / n if n else 0.0,
        "elapsed_s": round(elapsed, 1),
        "img_per_s": round(n / elapsed, 1) if elapsed > 0 else 0.0,
        "_pred": all_pred,
        "_true": all_true,
        "_top3": all_top3,
        "_conf": all_conf,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="PlantVillage → PlantDoc 跨域测试")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--plantdoc-root", default=str(ROOT / "data" / "raw" / "plantdoc"))
    p.add_argument("--split", default="all", choices=["train", "test", "all"],
                   help="用 PlantDoc 的哪个划分做测试（默认 all，因为不做微调，全部可用）")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--limit-per-class", type=int, default=0)
    args = p.parse_args()

    run_dir = (ROOT / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        print(f"[错误] 找不到 checkpoint: {ckpt_path}")
        return 1

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_name = ck.get("args", {}).get("model") or run_dir.name.rsplit("_", 2)[0]

    # ck["classes"] 是训练时 ImageFolder 的文件夹名列表，顺序与模型输出索引严格对应。
    # 直接用它作为标签索引，避免 classes.json 两种格式（列表 / 字典列表）带来的歧义。
    pv_folders = list(ck["classes"])
    num_classes = len(pv_folders)

    print(f"模型        : {model_name} (epoch {ck.get('epoch', '?')})")
    print(f"PlantDoc    : {args.plantdoc_root} [split={args.split}]")

    model, _meta = build_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(ck["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    transform = build_transforms(args.img_size, train=False)
    ds = PlantDocDataset(
        Path(args.plantdoc_root), args.split, pv_folders, transform,
        limit_per_class=args.limit_per_class,
    )

    if len(ds) == 0:
        print("[错误] 映射后没有可用样本，请检查 PlantDoc 目录结构")
        return 1

    print(f"可用样本    : {len(ds)}")
    if ds.skipped:
        print(f"跳过类别    : {len(ds.skipped)} 个共 {sum(ds.skipped.values())} 张")
        for k, v in ds.skipped.most_common():
            print(f"    {k}: {v}")

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )

    res = evaluate(model, loader, device, num_classes)

    # ---- 作物级准确率：预测与真实是否同一作物 ----
    pv_crop = [crop_of_folder(f) for f in pv_folders]
    crop_correct = sum(
        1 for pr, tr in zip(res["_pred"], res["_true"]) if pv_crop[pr] == pv_crop[tr]
    )
    crop_acc = crop_correct / res["n"] if res["n"] else 0.0

    print()
    print("=" * 62)
    print(f"跨域 top-1 准确率  : {res['top1_acc']:.4f}  ({res['correct']}/{res['n']})")
    print(f"跨域 top-3 准确率  : {res['top3_acc']:.4f}")
    print(f"作物级准确率       : {crop_acc:.4f}  ({crop_correct}/{res['n']})")
    print(f"平均置信度         : {sum(res['_conf']) / max(len(res['_conf']), 1):.4f}")
    print(f"耗时               : {res['elapsed_s']}s ({res['img_per_s']} img/s)")
    print("=" * 62)

    # ---- 逐类统计 ----
    per_class: dict[int, dict] = defaultdict(lambda: {"n": 0, "correct": 0, "crop_ok": 0})
    for pr, tr in zip(res["_pred"], res["_true"]):
        per_class[tr]["n"] += 1
        if pr == tr:
            per_class[tr]["correct"] += 1
        if pv_crop[pr] == pv_crop[tr]:
            per_class[tr]["crop_ok"] += 1

    rows = []
    for idx in sorted(per_class):
        d = per_class[idx]
        rows.append({
            "pv_class": pv_folders[idx],
            "n": d["n"],
            "correct": d["correct"],
            "acc": d["correct"] / d["n"],
            "crop_acc": d["crop_ok"] / d["n"],
        })
    rows.sort(key=lambda r: r["acc"])

    print("\n逐类准确率（最差 10 类）")
    print(f"{'类别':<52}{'n':>5}{'acc':>8}{'crop_acc':>10}")
    for r in rows[:10]:
        print(f"{r['pv_class']:<52}{r['n']:>5}{r['acc']:>8.2f}{r['crop_acc']:>10.2f}")

    # ---- 最常见的混淆：真实 → 预测 ----
    confusions = Counter(
        (pv_folders[t], pv_folders[p]) for t, p in zip(res["_true"], res["_pred"]) if t != p
    )
    print("\n最常见的 8 个错误（真实 → 预测）")
    for (t, p), c in confusions.most_common(8):
        print(f"  {c:>4}  {t}  →  {p}")

    # ---- 落盘 ----
    summary = {
        "model": model_name,
        "ckpt_epoch": ck.get("epoch"),
        "ckpt_val_acc": ck.get("val_acc"),
        "target_dataset": "PlantDoc",
        "split": args.split,
        "n_samples": res["n"],
        "top1_acc": round(res["top1_acc"], 4),
        "top3_acc": round(res["top3_acc"], 4),
        "crop_acc": round(crop_acc, 4),
        "mean_confidence": round(sum(res["_conf"]) / max(len(res["_conf"]), 1), 4),
        "img_per_s": res["img_per_s"],
        "mapped_classes": len(per_class),
        "skipped_classes": dict(ds.skipped),
        "per_class": rows,
        "top_confusions": [
            {"true": t, "pred": p, "n": c} for (t, p), c in confusions.most_common(20)
        ],
    }
    (run_dir / "crossdomain.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with open(run_dir / "crossdomain_preds.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "conf", "top3", "correct", "crop_correct"])
        for (path, label, _orig), pr, t3, cf in zip(
            ds.items, res["_pred"], res["_top3"], res["_conf"]
        ):
            w.writerow([
                path.as_posix(), pv_folders[label], pv_folders[pr],
                f"{cf:.4f}", "|".join(pv_folders[i] for i in t3),
                int(pr == label), int(pv_crop[pr] == pv_crop[label]),
            ])

    print(f"\n已保存: {run_dir / 'crossdomain.json'}")
    print(f"已保存: {run_dir / 'crossdomain_preds.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())