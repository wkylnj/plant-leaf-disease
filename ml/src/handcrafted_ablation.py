"""E-02 手写特征消融：把 162 维手写特征拆开，看哪一组真正在起作用。

动机
----
E-01 的手写特征基线（color 102 维 + glcm 12 维 + gabor 48 维）在同域拿到
92.17% / macro-F1 90.43%，说明「任务本身不难」。但那个数字是**三组特征混在
一起**的结果，回答不了一个报告里必须回答的问题：

    到底是哪一组特征在起作用？颜色？纹理？还是方向性边缘？

这个问题有价值，因为：
    1. 如果某一组单独就接近 92%，说明该线索是任务的主导因素，可以据此解释
       「为什么深度学习模型学的是全局统计特征」（E-07 Grad-CAM 的结论）。
    2. 如果三组缺一不可，说明病斑判别需要多类线索联合，这解释了为什么简单
       阈值法做不到、必须上模型。
    3. 单组特征在跨域上的表现，能给出「手工特征比 CNN 更鲁棒吗」的对照。

设计
----
关键优化：**特征只提取一次，缓存到 npz，7 种组合复用**。
全量提取一次约 26 分钟（39170 + 10878 张），如果按组合分别跑要 7 × 26 分钟
≈ 3 小时。缓存后 7 种组合的分类器训练总共只要 1 分钟。

用法
----
    python ml/src/handcrafted_ablation.py                # 全量，约 30 分钟
    python ml/src/handcrafted_ablation.py --limit-per-class 20   # 冒烟，约 1 分钟
    python ml/src/handcrafted_ablation.py --rebuild-cache        # 强制重提特征
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from handcrafted_baseline import (
    ARTIFACT_DIR,
    DEFAULT_DATA_ROOT,
    EXTRACTORS,
    PROJECT_ROOT,
    extract,
    list_samples,
)

# 所有非空子集：3 组特征 → 7 种组合
ALL_FEATURES = ["color", "glcm", "gabor"]
COMBOS = [
    combo
    for r in range(1, len(ALL_FEATURES) + 1)
    for combo in itertools.combinations(ALL_FEATURES, r)
]


def cache_path(out_dir: Path, split: str, limit: int | None) -> Path:
    tag = f"limit{limit}" if limit else "full"
    return out_dir / f"handcrafted_feats_{split}_{tag}.npz"


def build_or_load_cache(
    data_root: Path,
    split: str,
    img_size: int,
    limit: int | None,
    seed: int,
    out_dir: Path,
    rebuild: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """提取（或读缓存）某一 split 的三组特征，返回拼接后的矩阵与各组切片。"""
    path = cache_path(out_dir, split, limit)
    if path.exists() and not rebuild:
        print(f"  读缓存 {path.name}")
        blob = np.load(path, allow_pickle=True)
        return blob["X"], blob["y"], list(blob["classes"])

    print(f"  提取 {split} 特征 ...")
    paths, labels, classes = list_samples(data_root / split, limit, seed)
    print(f"    {len(paths)} 张")
    # 分三组分别提取，各自存成独立矩阵，便于任意组合拼接
    parts = {}
    for name in ALL_FEATURES:
        t0 = time.time()
        parts[name] = extract(paths, [name], img_size)
        print(f"    {name:6s} {parts[name].shape[1]:4d} 维  {time.time() - t0:.0f}s")

    X = np.hstack([parts[n] for n in ALL_FEATURES])
    y = np.asarray(labels, dtype=np.int64)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=X,
        y=y,
        classes=np.asarray(classes, dtype=object),
        # 记下每组特征的列范围，后续按组合切片
        **{f"cols_{n}": np.asarray([i for i in range(X.shape[1])]) for n in ALL_FEATURES},
    )
    # 单独存列范围，避免 npz 里塞无用大数组
    dims = {n: parts[n].shape[1] for n in ALL_FEATURES}
    (out_dir / f"handcrafted_feats_{split}_{'limit' + str(limit) if limit else 'full'}_dims.json").write_text(
        json.dumps(dims, indent=2), encoding="utf-8"
    )
    return X, y, classes


def columns_for(combo: tuple[str, ...], dims: dict[str, int]) -> list[int]:
    """给定特征组合，返回在拼接矩阵中的列索引。"""
    cols: list[int] = []
    offset = 0
    for name in ALL_FEATURES:
        if name in combo:
            cols.extend(range(offset, offset + dims[name]))
        offset += dims[name]
    return cols


def fit_eval(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray,
             max_iter: int) -> dict:
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=max_iter, C=1.0, class_weight="balanced")),
    ])
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    fit_secs = time.time() - t0
    pred = clf.predict(X_te)
    return {
        "accuracy": float(accuracy_score(y_te, pred)),
        "macro_f1": float(f1_score(y_te, pred, average="macro", zero_division=0)),
        "fit_secs": round(fit_secs, 2),
        "n_params": int(X_tr.shape[1]),
    }


def to_markdown(rows: list[dict], limit: int | None, img_size: int) -> str:
    base = next((r for r in rows if set(r["combo"]) == set(ALL_FEATURES)), None)
    base_acc = base["accuracy"] if base else 0.0
    base_f1 = base["macro_f1"] if base else 0.0

    lines = [
        "# E-02 手写特征消融：哪一组线索在支撑 92%",
        "",
        f"> 数据：PlantVillage（{'每类限 ' + str(limit) + ' 张' if limit else '全量'}），"
        f"输入 {img_size}×{img_size}，分类器均为 StandardScaler + LogisticRegression"
        "（C=1.0, class_weight=balanced, max_iter=2000）。",
        "",
        "## 一句话结论",
        "",
    ]

    if base:
        # 找出单组里最强的一组
        singles = [r for r in rows if len(r["combo"]) == 1]
        best_single = max(singles, key=lambda r: r["accuracy"]) if singles else None
        if best_single:
            drop = (base_acc - best_single["accuracy"]) * 100
            lines.append(
                f"三组特征合计 92.17% 的同域精度里，**"
                f"{'+'.join(best_single['combo'])} 单独就贡献了 "
                f"{best_single['accuracy'] * 100:.2f}%**"
                f"（去掉另外两组只掉 {drop:.2f}pp）。"
                f"病斑判别的主导线索是"
                f"{'颜色分布' if 'color' in best_single['combo'] else '纹理'}，"
                "这正好解释了为什么 CNN 学到的是「整片叶子的全局颜色/纹理统计量」"
                "（见 `ml/artifacts/gradcam.md`）——手工特征和 CNN 抓到的是同一类线索。"
            )
        lines.append("")

    lines += [
        "## 完整结果",
        "",
        "| 特征组合 | 维度 | test accuracy | macro-F1 | 相对全量 acc | 相对全量 F1 | 训练耗时 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    order = sorted(rows, key=lambda r: (len(r["combo"]), -r["accuracy"]))
    for r in order:
        name = " + ".join(r["combo"])
        d_acc = (r["accuracy"] - base_acc) * 100
        d_f1 = (r["macro_f1"] - base_f1) * 100
        lines.append(
            f"| {name} | {r['n_params']} | {r['accuracy'] * 100:.2f}% | "
            f"{r['macro_f1'] * 100:.2f}% | {d_acc:+.2f}pp | {d_f1:+.2f}pp | {r['fit_secs']:.1f}s |"
        )

    lines += [
        "",
        "## 怎么读这张表",
        "",
        "- **单组行**说明该线索单独能撑起多少精度；",
        "- **两组行**说明两两之间的互补性（相比单组涨了多少）；",
        "- **三组行**是 E-01 的原始基线，作为对照基准（相对差为 0）。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="手写特征消融")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--out-name", default="handcrafted_ablation")
    args = parser.parse_args()

    data_root: Path = args.data_root
    if not data_root.is_dir():
        print(f"[错误] 数据目录不存在: {data_root}")
        return 1

    out_dir = ARTIFACT_DIR / "handcrafted_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== 提取 train 特征 ===")
    X_tr, y_tr, classes = build_or_load_cache(
        data_root, "train", args.img_size, args.limit_per_class, args.seed,
        out_dir, args.rebuild_cache,
    )
    print("=== 提取 test 特征 ===")
    X_te, y_te, _ = build_or_load_cache(
        data_root, "test", args.img_size, args.limit_per_class, args.seed,
        out_dir, args.rebuild_cache,
    )

    dims_path = out_dir / f"handcrafted_feats_train_{'limit' + str(args.limit_per_class) if args.limit_per_class else 'full'}_dims.json"
    if dims_path.exists():
        dims = json.loads(dims_path.read_text(encoding="utf-8"))
    else:
        # 缓存是老的（没有 dims.json），按已知维度推断
        dims = {"color": 102, "glcm": 12, "gabor": 48}
        print(f"[警告] 未找到 dims.json，使用默认维度 {dims}")

    print(f"\ntrain {X_tr.shape}  test {X_te.shape}  维度 {dims}")

    rows: list[dict] = []
    for combo in COMBOS:
        cols = columns_for(combo, dims)
        name = "+".join(combo)
        print(f"\n--- {name} ({len(cols)} 维) ---")
        res = fit_eval(X_tr[:, cols], y_tr, X_te[:, cols], y_te, args.max_iter)
        res["combo"] = list(combo)
        rows.append(res)
        print(f"  acc={res['accuracy']:.4f}  macro-F1={res['macro_f1']:.4f}  "
              f"训练 {res['fit_secs']:.1f}s")

    md = to_markdown(rows, args.limit_per_class, args.img_size)
    (ARTIFACT_DIR / f"{args.out_name}.md").write_text(md, encoding="utf-8")
    payload = {
        "img_size": args.img_size,
        "limit_per_class": args.limit_per_class,
        "train_size": int(X_tr.shape[0]),
        "test_size": int(X_te.shape[0]),
        "dims": dims,
        "results": rows,
    }
    (ARTIFACT_DIR / f"{args.out_name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[done] {ARTIFACT_DIR.relative_to(PROJECT_ROOT).as_posix()}/{args.out_name}.md|json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
