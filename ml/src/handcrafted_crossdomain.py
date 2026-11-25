"""E-02 补充：手写特征的跨域表现（PlantVillage → PlantDoc）。

动机
----
E-02 主实验只测了同域。但项目最关心的问题是域偏移，所以必须回答：

    手工特征和 CNN，谁更抗域偏移？

这个对照有明确的理论预期。手工特征（颜色直方图、GLCM、Gabor）是**固定的、
与成像条件强相关的统计量**——光照变了直方图就变。CNN 至少还有机会在训练中
学到一些不变性。所以预期是：手工特征跨域掉得比 CNN 更狠。

如果实测确实如此，那就补全了一条论证链：
    判别信息在颜色统计（E-02）→ 颜色统计对成像条件敏感（本实验）→
    所以换数据集必崩（E-02 主实验 28.18%）

实现
----
复用 `handcrafted_cache` 里已缓存的 PlantVillage 特征（不重提），只提取
PlantDoc 的 2920 张特征（约 75 秒），然后用同样的逻辑回归训练+评估。

评估口径与 `eval_crossdomain.py` 完全一致：27 类映射、top-1/top-3/作物级。

用法
----
    python ml/src/handcrafted_crossdomain.py
    python ml/src/handcrafted_crossdomain.py --combos color "color+gabor"
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eval_crossdomain import (
    PLANTDOC_TO_PV_FOLDER,
    PlantDocDataset,
    crop_of_folder,
)
from handcrafted_baseline import ARTIFACT_DIR, DEFAULT_DATA_ROOT, PROJECT_ROOT, extract
from handcrafted_ablation import ALL_FEATURES, columns_for

DEFAULT_PLANTDOC_ROOT = PROJECT_ROOT / "ml" / "data" / "raw" / "plantdoc"


def load_pv_cache(out_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    path = out_dir / f"handcrafted_feats_{split}_full.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少缓存 {path}，请先运行 `python src/handcrafted_ablation.py` 生成"
        )
    blob = np.load(path, allow_pickle=True)
    return blob["X"], blob["y"]


def extract_plantdoc(root: Path, split: str, img_size: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """提取 PlantDoc 的手写特征。

    注意：这里不走 DataLoader（手写特征需要原始 BGR 图，而 DataLoader 出来的是
    已归一化的张量），直接遍历 PlantDocDataset 的文件列表。
    """
    # 用一个恒等 transform，只借它做类别映射与文件枚举
    ds = PlantDocDataset(root, split=split, pv_folders=list(PLANTDOC_TO_PV_FOLDER.values()),
                         transform=None, limit_per_class=0)
    items = ds.items  # [(path, label, orig_name), ...]
    print(f"  PlantDoc {split}: {len(items)} 张，跳过 {ds.skipped}")

    paths = [Path(it[0]) for it in items]
    labels = np.asarray([it[1] for it in items], dtype=np.int64)

    t0 = time.time()
    X = extract(paths, ALL_FEATURES, img_size, verbose=True)
    print(f"  特征提取 {time.time() - t0:.0f}s")
    return X, labels, [it[2] for it in items]


def main() -> int:
    parser = argparse.ArgumentParser(description="手写特征跨域评估")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--plantdoc-root", type=Path, default=DEFAULT_PLANTDOC_ROOT)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--combos", nargs="*", default=None,
                        help="要测的组合，默认全量 color+glcm+gabor 与 color 单组")
    parser.add_argument("--out-name", default="handcrafted_crossdomain")
    args = parser.parse_args()

    cache_dir = ARTIFACT_DIR / "handcrafted_cache"
    print("读取 PlantVillage 特征缓存 ...")
    X_tr, y_tr = load_pv_cache(cache_dir, "train")
    print(f"  train {X_tr.shape}")
    dims = {"color": 102, "glcm": 12, "gabor": 48}
    dims_path = cache_dir / "handcrafted_feats_train_full_dims.json"
    if dims_path.exists():
        dims = json.loads(dims_path.read_text(encoding="utf-8"))

    print("提取 PlantDoc 特征 ...")
    X_pd, y_pd, orig_names = extract_plantdoc(args.plantdoc_root, "all", args.img_size)

    combos = args.combos or ["color+glcm+gabor", "color", "color+gabor"]
    rows = []
    for combo_str in combos:
        combo = tuple(combo_str.split("+"))
        cols = columns_for(combo, dims)
        print(f"\n--- {combo_str} ({len(cols)} 维) ---")

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=args.max_iter, C=1.0,
                                       class_weight="balanced")),
        ])
        clf.fit(X_tr[:, cols], y_tr)

        proba = clf.predict_proba(X_pd[:, cols])
        pred = proba.argmax(axis=1)

        # top-3
        top3_idx = np.argsort(-proba, axis=1)[:, :3]
        top3_acc = float((top3_idx == y_pd[:, None]).any(axis=1).mean())
        # 作物级：预测类与真实类属于同一作物即算对
        # 用 PlantVillage 文件夹名（classes 顺序）做映射
        classes = np.load(cache_dir / "handcrafted_feats_train_full.npz",
                          allow_pickle=True)["classes"]
        pred_crop = np.asarray([crop_of_folder(str(classes[p])) for p in pred])
        true_crop = np.asarray([crop_of_folder(str(classes[t])) for t in y_pd])
        crop_acc = float((pred_crop == true_crop).mean())

        top1 = float(accuracy_score(y_pd, pred))
        macro_f1 = float(f1_score(y_pd, pred, average="macro", zero_division=0))
        conf = float(proba.max(axis=1).mean())

        print(f"  top1={top1:.4f}  top3={top3_acc:.4f}  crop={crop_acc:.4f}  "
              f"conf={conf:.4f}  macroF1={macro_f1:.4f}")
        rows.append({
            "combo": list(combo),
            "dim": len(cols),
            "top1": top1,
            "top3": top3_acc,
            "crop": crop_acc,
            "mean_conf": conf,
            "macro_f1": macro_f1,
        })

    payload = {
        "plantdoc_root": str(args.plantdoc_root),
        "img_size": args.img_size,
        "n_plantdoc": int(len(y_pd)),
        "results": rows,
    }
    (ARTIFACT_DIR / f"{args.out_name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[done] {ARTIFACT_DIR.relative_to(PROJECT_ROOT).as_posix()}/{args.out_name}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
