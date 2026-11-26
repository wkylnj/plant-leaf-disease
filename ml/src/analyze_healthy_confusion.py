"""分析 PlantWild 上「健康叶被误判成什么」——解释健康类准确率为何低于病害类。

背景
----
`eval_plantwild.py` 发现 ResNet18 在 PlantWild 上：
    健康类 top-1 16.01%（2860 张）
    病害类 top-1 22.58%（4593 张）
健康类比病害类还差 6.6pp，且 Peach / Cherry / Soybean / Potato 四个健康类的
**作物级准确率也只有 2~6%**（连「这是桃叶」都认不出）。

本脚本回答两个问题：
  1. 健康叶最常被预测成哪些类？是否集中在少数几个「吸铁石」类？
  2. 那些作物级都认错的类（Peach/Cherry/Soybean healthy），是被判成了别的作物
     还是别的病害？如果跨作物乱跳，说明模型对这类图像完全没有可用特征。

用法：
    python src/analyze_healthy_confusion.py --run-dir runs/resnet18_224_15ep
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import build_transforms  # noqa: E402
from eval_crossdomain import crop_of_folder  # noqa: E402
from eval_plantwild import PlantWildDataset, is_healthy  # noqa: E402
from models import build_model  # noqa: E402


@torch.no_grad()
def predict(model, loader, device) -> tuple[list[int], list[int]]:
    preds: list[int] = []
    trues: list[int] = []
    model.eval()
    for x, y in loader:
        x = x.to(device)
        preds.extend(model(x).argmax(1).cpu().tolist())
        trues.extend(y.tolist())
    return preds, trues


def main() -> int:
    ap = argparse.ArgumentParser(description="健康叶误判去向分析")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--plantwild-root", default=str(ROOT / "data" / "raw" / "plantwild_subset"))
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model_name = ck.get("args", {}).get("model") or run_dir.name.rsplit("_", 2)[0]
    pv_folders = list(ck["classes"])

    model, _ = build_model(model_name, len(pv_folders), pretrained=False)
    model.load_state_dict(ck["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ds = PlantWildDataset(Path(args.plantwild_root), pv_folders,
                          build_transforms(args.img_size, train=False))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    preds, trues = predict(model, loader, device)
    print(f"模型 {model_name} @{args.img_size}，样本 {len(trues)}")

    # ---- 1. 健康叶整体被预测成什么 ----
    healthy_idx = [i for i, f in enumerate(pv_folders) if is_healthy(f)]
    healthy_set = set(healthy_idx)
    counter = Counter(p for p, t in zip(preds, trues) if t in healthy_set)
    n_healthy = sum(1 for t in trues if t in healthy_set)

    print()
    print(f"健康叶（{n_healthy} 张）最常被判成的类：")
    for cls, cnt in counter.most_common(args.top):
        flag = "  ← 也是健康类" if cls in healthy_set else ""
        print(f"  {cnt:4d} 次 ({cnt / n_healthy:5.1%})  {pv_folders[cls]}{flag}")

    # 预测落在健康类 vs 病害类
    into_healthy = sum(c for k, c in counter.items() if k in healthy_set)
    print(f"\n  其中判成某个「健康」类：{into_healthy} 次（{into_healthy / n_healthy:.1%}）")
    print(f"  判成某个「病害」类：{n_healthy - into_healthy} 次 "
          f"（{(n_healthy - into_healthy) / n_healthy:.1%}）")

    # ---- 2. 逐健康类的去向 ----
    per_true: dict[int, Counter] = defaultdict(Counter)
    for p, t in zip(preds, trues):
        if t in healthy_set:
            per_true[t][p] += 1

    print()
    print("逐健康类的 top-3 去向（★ = 判对）")
    detail = {}
    for t in sorted(per_true, key=lambda i: -sum(per_true[i].values())):
        total = sum(per_true[t].values())
        rows = []
        print(f"  {pv_folders[t]}  n={total}")
        for cls, cnt in per_true[t].most_common(3):
            mark = "★" if cls == t else " "
            same_crop = crop_of_folder(pv_folders[cls]) == crop_of_folder(pv_folders[t])
            tag = "同作物" if same_crop else "跨作物"
            print(f"     {mark} {cnt:4d} ({cnt / total:5.1%}) [{tag}] {pv_folders[cls]}")
            rows.append({"class": pv_folders[cls], "count": cnt,
                         "ratio": cnt / total, "same_crop": same_crop,
                         "correct": cls == t})
        detail[pv_folders[t]] = {"n": total, "top3": rows}

    # 跨作物乱跳的比例（对「作物级也认不出」的类特别说明）
    print()
    print("作物级也认不出的类，其错误是否跨作物：")
    for t in sorted(per_true, key=lambda i: -sum(per_true[i].values())):
        total = sum(per_true[t].values())
        correct = per_true[t][t]
        if correct / total > 0.10:
            continue
        cross = sum(c for cls, c in per_true[t].items()
                    if crop_of_folder(pv_folders[cls]) != crop_of_folder(pv_folders[t]))
        print(f"  {pv_folders[t]:42s} n={total:4d} 判对 {correct/total:5.1%}  "
              f"跨作物 {cross/total:5.1%}")

    out = Path(args.out) if args.out else (run_dir / f"plantwild_healthy_confusion_{args.img_size}.json")
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    payload = {
        "model": model_name,
        "img_size": args.img_size,
        "n_healthy": n_healthy,
        "into_healthy": into_healthy,
        "top_predicted": [{"class": pv_folders[c], "count": n, "ratio": n / n_healthy}
                          for c, n in counter.most_common(args.top)],
        "per_healthy_class": detail,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
