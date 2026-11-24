"""E-03 线性探针：把「表示质量」和「分类头质量」分开看。

要回答的问题
------------
E-2 说跨域只有 28%，E-7 说模型看全局统计特征，E-12 说强迫看局部反而更差。
但这些实验都没能回答一个更基础的问题：

> 掉点是因为**骨干学到的特征本身就不对**，还是因为**最后那层分类头过拟合了 PlantVillage**？

这两者的修法完全不同。如果是分类头的问题，换个头、加正则、做校准就能救；
如果是特征本身的问题，只能动骨干（微调、换预训练目标）。

线性探针（linear probe）就是分离这两者的标准做法：**冻结骨干、只训练一个线性分类器**。
如果线性探针在跨域上也崩，说明特征本身已经不可分了，跟头无关。

设计
----
- 骨干：ResNet18 / Swin-T / MobileNetV3-Large，全部冻结（eval 模式 + requires_grad=False）
- 特征：全局平均池化后的向量（ResNet18/MobileNetV3 是 1280 维，Swin-T 是 768 维）
- 探针：LogisticRegression（sklearn），在 PlantVillage train 上训练，类别权重 balanced
- 评估：PlantVillage test（同域）+ PlantDoc 全量（跨域，27 类映射）
- 对照：同一个 checkpoint 的完整模型（端到端）指标，直接从 crossdomain.json / summary 读

一个额外发现
------------
探针还有个副产品：**特征的可分性上限**。如果线性探针在同域只比端到端低一点点，
说明这个骨干的线性可分性已经足够，分类头没有额外贡献——这本身就是个有意思的结论。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import build_transforms  # noqa: E402
from eval_crossdomain import (  # noqa: E402
    PlantDocDataset,
    crop_of_folder,
    evaluate as evaluate_crossdomain,
)
from models import build_model  # noqa: E402

ROOT = SRC.parent
ARTIFACTS = ROOT / "artifacts"


def get_feature_extractor(model: nn.Module, model_name: str) -> nn.Module:
    """把分类头换成 Identity，保留骨干。

    三种模型的分类头位置不同（ResNet18 是 fc，MobileNetV3 是 classifier[3]，
    Swin-T 是 head），统一处理成「骨干 + Identity」。
    """
    if model_name.startswith("resnet"):
        model.fc = nn.Identity()
    elif model_name.startswith("mobilenet"):
        model.classifier[3] = nn.Identity()
    elif model_name.startswith("efficientnet"):
        model.classifier[1] = nn.Identity()
    elif model_name.startswith("swin"):
        model.head = nn.Identity()
    elif model_name.startswith("shufflenet"):
        model.fc = nn.Identity()
    else:
        raise ValueError(f"未支持的模型：{model_name}")
    return model


@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats: list[np.ndarray] = []
    labels: list[int] = []
    for x, y in loader:
        x = x.to(device)
        f = model(x)
        if f.dim() > 2:
            f = f.mean(dim=tuple(range(2, f.dim())))
        feats.append(f.cpu().numpy().astype(np.float32))
        labels.extend(y.tolist())
    return np.concatenate(feats, axis=0), np.array(labels)


def main() -> int:
    ap = argparse.ArgumentParser(description="E-03 线性探针")
    ap.add_argument("--runs", nargs="+", default=[
        "resnet18_224_15ep",
        "swin_t_224_15ep",
        "mobilenet_v3_large_224_15ep",
    ])
    ap.add_argument("--data-root", default="data/processed/plantvillage")
    ap.add_argument("--plantdoc-root", default="data/raw/plantdoc")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--limit-train", type=int, default=0, help="冒烟用，0 表示不限")
    ap.add_argument("--out", default=str(ARTIFACTS / "linear_probe.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.data_root)
    tf = build_transforms(args.img_size, train=False)

    # PlantVillage train（探针的训练集）与 test（同域评估）
    train_ds = ImageFolder(root / "train", transform=tf)
    test_ds = ImageFolder(root / "test", transform=tf)
    if args.limit_train:
        idx = np.random.RandomState(42).permutation(len(train_ds))[: args.limit_train]
        train_ds = torch.utils.data.Subset(train_ds, idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    results = {}
    for run in args.runs:
        run_dir = ROOT / "runs" / run
        ck_path = run_dir / "best.pt"
        if not ck_path.exists():
            print(f"[skip] {ck_path} 不存在")
            continue
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        classes = list(ck["classes"])
        model_name = ck.get("args", {}).get("model") or run.rsplit("_", 2)[0]

        print(f"\n=== {run}（{model_name}）===")
        model, meta = build_model(model_name, len(classes), pretrained=False)
        model.load_state_dict(ck["model"])
        model = get_feature_extractor(model, model_name).to(device)
        for p in model.parameters():
            p.requires_grad_(False)

        t0 = time.time()
        print("  提取 PlantVillage train 特征 ...")
        X_tr, y_tr = extract_features(model, train_loader, device)
        print("  提取 PlantVillage test 特征 ...")
        X_te, y_te = extract_features(model, test_loader, device)
        feat_secs = time.time() - t0
        print(f"  特征维度 {X_tr.shape[1]}，耗时 {feat_secs:.1f}s")

        # 探针训练
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=args.max_iter, C=1.0, class_weight="balanced")),
        ])
        t1 = time.time()
        clf.fit(X_tr, y_tr)
        probe_secs = time.time() - t1
        pred = clf.predict(X_te)
        same_acc = accuracy_score(y_te, pred)
        same_f1 = f1_score(y_te, pred, average="macro", zero_division=0)
        print(f"  [同域] 线性探针 acc={same_acc:.4f} macro-F1={same_f1:.4f}（训练 {probe_secs:.1f}s）")

        # 跨域
        pd_root = Path(args.plantdoc_root)
        cross = None
        if pd_root.is_dir():
            pd_ds = PlantDocDataset(pd_root, "all", classes, transform=tf)
            pd_loader = DataLoader(pd_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            print(f"  提取 PlantDoc 特征（{len(pd_ds)} 张）...")
            X_pd, y_pd = extract_features(model, pd_loader, device)
            pred_pd = clf.predict(X_pd)
            cross_top1 = accuracy_score(y_pd, pred_pd)
            # 作物级
            pv_crop = [crop_of_folder(f) for f in classes]
            crop_ok = sum(1 for p, t in zip(pred_pd, y_pd) if pv_crop[int(p)] == pv_crop[int(t)])
            cross_crop = crop_ok / max(1, len(y_pd))
            # top-3
            proba = clf.predict_proba(X_pd)
            top3 = np.argsort(-proba, axis=1)[:, :3]
            top3_hit = sum(1 for i, t in enumerate(y_pd) if int(t) in top3[i])
            cross_top3 = top3_hit / max(1, len(y_pd))
            cross = {
                "n": int(len(y_pd)),
                "top1_acc": float(cross_top1),
                "top3_acc": float(cross_top3),
                "crop_acc": float(cross_crop),
            }
            print(
                f"  [跨域] 线性探针 top1={cross_top1:.4f} top3={cross_top3:.4f} crop={cross_crop:.4f}"
            )

        # 端到端对照（从已有 json 读，不重复推理）
        e2e = {}
        sj = run_dir / "summary.json"
        ej = run_dir / "summary_eval.json"
        for p in (sj, ej):
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                e2e["same_top1"] = d.get("test_acc")
                e2e["same_macro_f1"] = d.get("test_macro_f1")
                break
        cj = run_dir / "crossdomain.json"
        if cj.exists():
            d = json.loads(cj.read_text(encoding="utf-8"))
            e2e["cross_top1"] = d.get("top1_acc")
            e2e["cross_top3"] = d.get("top3_acc")
            e2e["cross_crop"] = d.get("crop_acc")

        results[run] = {
            "model": model_name,
            "params_m": round(meta.params_m, 3),
            "feature_dim": int(X_tr.shape[1]),
            "probe": {
                "same_domain": {"acc": float(same_acc), "macro_f1": float(same_f1)},
                "crossdomain": cross,
                "train_secs": round(probe_secs, 1),
                "feat_secs": round(feat_secs, 1),
            },
            "end_to_end": e2e,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
