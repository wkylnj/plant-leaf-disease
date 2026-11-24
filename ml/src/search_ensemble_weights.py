"""离线集成权重搜索（读 eval_ensemble.py --save-probs 导出的 probs.npz）。

为什么需要：三模型等权集成（0.3449）低于 Swin-T 单独（0.3709），因为
ResNet18/MobileNetV3 明显更弱。等权把弱模型的信息强行掺进来反而拉低。
本脚本在概率矩阵上做权重搜索，找让跨域 top-1 最高的权重组合。

注意：**在评估集上搜权重会过拟合**。因此本脚本同时输出
「留一折交叉验证」的泛化估计：把 2920 张按固定种子分 5 折，
每折用其余 4 折选权重、在该折上评估，最后报告 5 折平均。
只有 CV 平均也优于单模型，加权集成才算真的有效。

用法::

    python src/search_ensemble_weights.py --probs artifacts/probs_3m/probs.npz
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval_crossdomain import crop_of_folder  # noqa: E402


def top1(probs: np.ndarray, labels: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == labels).mean())


def crop_acc(probs: np.ndarray, labels: np.ndarray, crops: list[str]) -> float:
    pred = probs.argmax(axis=1)
    return float(np.mean([crops[p] == crops[t] for p, t in zip(pred, labels)]))


def grid(n: int, step: float = 0.1, max_w: float = 1.0) -> list[tuple[float, ...]]:
    vals = np.round(np.arange(0.0, max_w + 1e-9, step), 2)
    out = []
    for combo in itertools.product(vals, repeat=n):
        if sum(combo) <= 0:
            continue
        out.append(tuple(float(x) for x in combo))
    return out


def blend(probs: np.ndarray, w: tuple[float, ...]) -> np.ndarray:
    wv = np.asarray(w, dtype=np.float64).reshape(-1, 1, 1)
    return (probs * wv).sum(axis=0) / sum(w)


def best_on(probs: np.ndarray, labels: np.ndarray, cands) -> tuple[tuple[float, ...], float]:
    best_w, best_a = None, -1.0
    for w in cands:
        a = top1(blend(probs, w), labels)
        if a > best_a:
            best_w, best_a = w, a
    return best_w, best_a


def main() -> int:
    p = argparse.ArgumentParser(description="集成权重搜索 + 交叉验证")
    p.add_argument("--probs", default=str(ROOT / "artifacts" / "probs_3m" / "probs.npz"))
    p.add_argument("--step", type=float, default=0.1)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="")
    args = p.parse_args()

    d = np.load(args.probs, allow_pickle=True)
    probs = d["probs"].astype(np.float64)          # [M, N, C]
    labels = d["labels"].astype(np.int64)
    members = [str(x) for x in d["members"]]
    pv_folders = [str(x) for x in d["pv_folders"]]
    crops = [crop_of_folder(f) for f in pv_folders]
    n_members, n, n_cls = probs.shape
    print(f"成员 {n_members}: {members}")
    print(f"样本 {n}  类别 {n_cls}")

    # 各成员单独
    singles = []
    for i, name in enumerate(members):
        a = top1(probs[i], labels)
        c = crop_acc(probs[i], labels, crops)
        singles.append({"member": name, "top1": round(a, 4), "crop": round(c, 4)})
        print(f"  单独 {name:34s} top1={a:.4f} crop={c:.4f}")

    cands = grid(n_members, args.step)
    print(f"\n权重候选 {len(cands)} 组，全量搜索 ...", flush=True)
    best_w, best_a = best_on(probs, labels, cands)
    best_blend = blend(probs, best_w)
    best_crop = crop_acc(best_blend, labels, crops)
    best_single = max(s["top1"] for s in singles)
    print(f"全量最优权重 {best_w}  top1={best_a:.4f} crop={best_crop:.4f} "
          f"（最好单模型 {best_single:.4f}，差 {best_a - best_single:+.4f}）")

    # 留一折 CV：每折用其余折选权重
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, args.folds)
    cv_accs, cv_weights = [], []
    for k, te in enumerate(folds):
        tr = np.concatenate([f for j, f in enumerate(folds) if j != k])
        w, _ = best_on(probs[:, tr, :], labels[tr], cands)
        a = top1(blend(probs[:, te, :], w), labels[te])
        cv_accs.append(a)
        cv_weights.append(w)
        print(f"  折 {k}: 选权重 {w} → 该折 top1={a:.4f}", flush=True)
    cv_mean = float(np.mean(cv_accs))
    # 单模型同样做 CV（公平对比）
    single_cv = []
    for i, name in enumerate(members):
        accs = [top1(probs[i][te], labels[te]) for te in folds]
        single_cv.append({"member": name, "cv_mean": round(float(np.mean(accs)), 4)})
        print(f"  单模型 CV {name:34s} {np.mean(accs):.4f}")
    best_single_cv = max(s["cv_mean"] for s in single_cv)
    print(f"\nCV 集成 {cv_mean:.4f}  vs  CV 最好单模型 {best_single_cv:.4f} "
          f"（差 {cv_mean - best_single_cv:+.4f}）")

    payload = {
        "members": members,
        "n_samples": int(n),
        "singles": singles,
        "best_weights": list(best_w),
        "best_top1_full": round(best_a, 4),
        "best_crop_full": round(best_crop, 4),
        "cv_mean": round(cv_mean, 4),
        "cv_weights": [list(w) for w in cv_weights],
        "cv_singles": single_cv,
        "cv_best_single": best_single_cv,
        "verdict": ("加权集成有效" if cv_mean > best_single_cv + 0.005
                    else "加权集成无实质收益（CV 差异 < 0.5pp）"),
    }
    out = Path(args.out) if args.out else ROOT / "artifacts" / "ensemble_weight_search.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n判定：{payload['verdict']}")
    print(f"写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
