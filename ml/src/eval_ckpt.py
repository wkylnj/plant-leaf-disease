"""对已训练的 checkpoint 做测试集评估，不需要重新训练。

用途：
- Colab 会话被回收、只来得及拉回 best.pt 时，用本脚本在本地补出 test_acc / macro-F1
- 报告需要 per-class F1、混淆矩阵、错例清单时

用法：
    python src/eval_ckpt.py --run-dir ../runs/mobilenet_v3_large_224_15ep \
        --data-root ../data/processed/plantvillage --img-size 224
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data import build_dataloaders
from models import build_model


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    preds, labels, probs = [], [], []
    t0 = time.time()
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = torch.softmax(logits.float(), dim=1)
        preds.append(p.argmax(1).cpu().numpy())
        labels.append(y.numpy())
        probs.append(p.cpu().numpy())
        n += x.size(0)
    secs = time.time() - t0
    return (
        np.concatenate(preds),
        np.concatenate(labels),
        np.concatenate(probs),
        secs,
        n,
    )


def per_class_f1(y_true, y_pred, num_classes):
    """手写 per-class precision / recall / f1，避免额外依赖。"""
    rows = []
    for c in range(num_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        support = int((y_true == c).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append((c, support, prec, rec, f1))
    return rows


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="包含 best.pt 的 run 目录")
    ap.add_argument("--data-root", default="../data/processed/plantvillage")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"找不到 {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ck.get("args") or {}
    model_name = ckpt_args.get("model") or "resnet18"
    classes = ck["classes"]
    num_classes = len(classes)

    print(f"[load] {ckpt_path}")
    print(f"[load] model={model_name} num_classes={num_classes} "
          f"ckpt_epoch={ck.get('epoch')} ckpt_val_acc={ck.get('val_acc')}")

    model, meta = build_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(device)

    train_loader, val_loader, test_loader, loader_classes = build_dataloaders(
        args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        strong_aug=False,
    )
    if list(loader_classes) != list(classes):
        print("[warn] 数据目录类别与 checkpoint 类别不一致！")
        print("  ckpt:", classes[:3], "...")
        print("  data:", list(loader_classes)[:3], "...")

    loader = test_loader if args.split == "test" else val_loader
    print(f"[eval] split={args.split} batches={len(loader)} device={device}")

    preds, labels, probs, secs, n = predict_all(model, loader, device)
    acc = float((preds == labels).mean())
    rows = per_class_f1(labels, preds, num_classes)
    macro_f1 = float(np.mean([r[4] for r in rows]))
    cm = confusion_matrix(labels, preds, num_classes)

    print()
    print("=" * 62)
    print(f"  模型      {model_name}  ({meta.params_m:.3f} M 参数)")
    print(f"  split     {args.split}   样本 {n}")
    print(f"  accuracy  {acc:.6f}  ({acc * 100:.2f}%)")
    print(f"  macro-F1  {macro_f1:.6f}  ({macro_f1 * 100:.2f}%)")
    print(f"  耗时      {secs:.1f} s   ({n / secs:.1f} img/s, CPU)")
    print("=" * 62)

    # 保存结果
    out = {
        "model": model_name,
        "params_m": round(meta.params_m, 3),
        "img_size": args.img_size,
        "split": args.split,
        "n_samples": n,
        "ckpt_epoch": ck.get("epoch"),
        "ckpt_val_acc": ck.get("val_acc"),
        "test_acc": acc,
        "test_macro_f1": macro_f1,
        "eval_secs": round(secs, 1),
        "img_per_s": round(n / secs, 1),
        "note": "由 eval_ckpt.py 从 best.pt 补测，非训练时自动产出",
    }
    (run_dir / "summary_eval.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # per-class f1（按 f1 升序）
    rows_sorted = sorted(rows, key=lambda r: r[4])
    with (run_dir / "per_class_f1.csv").open("w", encoding="utf-8", newline="") as f:
        f.write("label,support,precision,recall,f1\n")
        for c, sup, p, r, f1 in rows_sorted:
            name = classes[c] if c < len(classes) else f"class_{c}"
            f.write(f"{name},{sup},{p:.6f},{r:.6f},{f1:.6f}\n")

    np.savez_compressed(
        run_dir / f"{args.split}_preds.npz",
        preds=preds,
        labels=labels,
        probs=probs.astype(np.float32),
        classes=np.array(classes, dtype=object),
    )
    np.save(run_dir / "confusion_matrix.npy", cm)

    print()
    print("最难识别的 8 类（per-class F1 升序）:")
    for c, sup, p, r, f1 in rows_sorted[:8]:
        name = classes[c] if c < len(classes) else f"class_{c}"
        print(f"  {f1:.4f}  {name:<58s} support={sup}")

    print()
    print(f"[saved] {run_dir / 'summary_eval.json'}")
    print(f"[saved] {run_dir / 'per_class_f1.csv'}")
    print(f"[saved] {run_dir / 'confusion_matrix.npy'}")
    print(f"[saved] {run_dir / (args.split + '_preds.npz')}")


if __name__ == "__main__":
    main()
