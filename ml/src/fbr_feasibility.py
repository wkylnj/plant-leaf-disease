"""FBR 可行性诊断：合成图在特征空间里到底有没有向 PlantDoc 靠近？

要回答的问题
------------
FBR（把 PlantVillage 叶片贴到田间背景上）的前提是：
**合成图在特征空间里应该比原始 PlantVillage 图更接近 PlantDoc。**

如果这个前提不成立，那么「用合成图训练能让模型更适应田间」这个推论就没有
基础，不必浪费 GPU 去跑 15 轮训练。

为什么这个诊断重要
------------------
前面已经有六条路线被证伪，其中 E-12（局部裁剪训练）是唯一动了权重还失败的。
CORAL 闭式实验给出了解释：判别信息在特征里，缺的是「特征→病害」的映射。
但 FBR 与它们不同——它改变的是**训练数据的成像条件**，这是唯一对症的方向。
所以值得先花 10 分钟 CPU 验证前提，而不是直接排队等 GPU。

做法
----
冻结一个已训练好的 ResNet18 骨干，提取三组特征：
  1. 原始 PlantVillage test 图（源域）
  2. FBR 合成图（源域叶片 + 目标域背景）
  3. PlantDoc test 图（目标域）

然后比较：
  - 各组特征到 PlantDoc 特征中心的距离（越近越好）
  - 用最近邻分类器在三组特征上的跨域表现

关键设计：合成图必须来自 PlantVillage **test** 而不是 train，
否则合成图里的叶片本身就是模型训练时见过的，距离会被人为压低。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import build_transforms  # noqa: E402
from eval_crossdomain import PlantDocDataset, PLANTDOC_TO_PV_FOLDER  # noqa: E402
from linear_probe import extract_features, get_feature_extractor  # noqa: E402
from models import build_model  # noqa: E402

ROOT = SRC.parent
ARTIFACTS = ROOT / "artifacts"


def load_model(run_dir: Path) -> tuple[nn.Module, str, list[str]]:
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model_name = ck.get("args", {}).get("model") or run_dir.name.rsplit("_", 2)[0]
    classes = list(ck["classes"])
    model, _meta = build_model(model_name, len(classes), pretrained=False)
    model.load_state_dict(ck["model"])
    return model, model_name, classes


def domain_stats(feats: np.ndarray, ref: np.ndarray) -> dict:
    """衡量一批特征到参照域（PlantDoc）的距离。

    用两个指标：
      - 到参照域中心的欧氏距离（归一化后）
      - 参照域上的 5 折中心距离（作为「域内自然离散度」的基线）
    """
    f = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    r = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8)
    center = r.mean(axis=0)
    center = center / (np.linalg.norm(center) + 1e-8)

    d_to_ref = np.linalg.norm(f - center[None, :], axis=1)

    # 参照域内部离散度：留一折算到其余折中心的距离
    idx = np.random.RandomState(0).permutation(len(r))
    folds = np.array_split(idx, 5)
    inner = []
    for k in range(5):
        hold = folds[k]
        rest = np.concatenate([folds[j] for j in range(5) if j != k])
        c = r[rest].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-8)
        inner.append(np.linalg.norm(r[hold] - c[None, :], axis=1))
    inner = np.concatenate(inner)

    return {
        "n": int(len(f)),
        "dist_to_ref_mean": float(d_to_ref.mean()),
        "dist_to_ref_median": float(np.median(d_to_ref)),
        "ref_inner_dist_mean": float(inner.mean()),
        "ratio_vs_ref_inner": float(d_to_ref.mean() / max(inner.mean(), 1e-8)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="FBR 可行性诊断")
    ap.add_argument("--run-dir", default="runs/resnet18_224_15ep")
    ap.add_argument("--data-root", default="data/processed/plantvillage")
    ap.add_argument("--fbr-test-root", default="data/processed/pv_fbr_test")
    ap.add_argument("--plantdoc-root", default="data/raw/plantdoc")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=600, help="每域抽样上限，0 表示不限")
    ap.add_argument("--out", default=str(ARTIFACTS / "fbr_feasibility.json"))
    args = ap.parse_args()

    device = torch.device("cpu")
    tf = build_transforms(args.img_size, train=False)

    model, model_name, classes = load_model(Path(args.run_dir))
    model = get_feature_extractor(model, model_name)
    model.eval()
    print(f"[feas] 模型 {model_name}，{len(classes)} 类，骨干已冻结")

    def subset(ds):
        if args.limit and len(ds) > args.limit:
            idx = np.random.RandomState(42).permutation(len(ds))[: args.limit]
            return torch.utils.data.Subset(ds, idx.tolist())
        return ds

    def feats_of(ds, tag):
        loader = DataLoader(subset(ds), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
        f, y = extract_features(model, loader, device)
        print(f"[feas] {tag}: {f.shape}")
        return f, y

    # 1. 原始 PlantVillage test（源域）
    f_pv, y_pv = feats_of(ImageFolder(Path(args.data_root) / "test", transform=tf), "PV test")

    # 2. FBR 合成图（源域叶片 + 目标域背景，来自 PV test，无泄漏）
    f_fbr, y_fbr = feats_of(ImageFolder(Path(args.fbr_test_root), transform=tf), "FBR test")

    # 3. PlantDoc test（目标域）
    pd_ds = PlantDocDataset(Path(args.plantdoc_root), split="test",
                            pv_folders=classes, transform=tf)
    f_pd, y_pd = feats_of(pd_ds, "PlantDoc test")

    result = {
        "model": model_name,
        "feature_dim": int(f_pv.shape[1]),
        "domains": {
            "pv_test": domain_stats(f_pv, f_pd),
            "fbr_test": domain_stats(f_fbr, f_pd),
            "plantdoc_test": domain_stats(f_pd, f_pd),
        },
    }

    # 关键对比：FBR 是否比原始 PV 更接近 PlantDoc
    pv_d = result["domains"]["pv_test"]["dist_to_ref_mean"]
    fbr_d = result["domains"]["fbr_test"]["dist_to_ref_mean"]
    pd_inner = result["domains"]["plantdoc_test"]["ref_inner_dist_mean"]
    result["fbr_closer"] = bool(fbr_d < pv_d)
    result["improvement_pct"] = float((pv_d - fbr_d) / max(pv_d, 1e-8) * 100)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 结果 ===")
    print(f"PlantDoc 域内自然离散度（基线）: {pd_inner:.4f}")
    for name, d in result["domains"].items():
        print(f"{name:16s} 到 PlantDoc 中心距离 {d['dist_to_ref_mean']:.4f} "
              f"（是域内离散度的 {d['ratio_vs_ref_inner']:.2f}×）")
    print(f"\nFBR 比原始 PV 更接近 PlantDoc？{'是' if result['fbr_closer'] else '否'}"
          f"（{result['improvement_pct']:+.1f}%）")
    print(f"产物 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
