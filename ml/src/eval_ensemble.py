"""多模型 / 多分辨率集成评估（跨域 + 可选同域）。

动机：E-15 预处理消融发现「提高推理分辨率」是唯一稳定正收益（ResNet18 224->320
跨域 +2.20pp）；E-11 发现「双模型集成」也有效（跨域 +0.93pp）。本脚本把两者叠加，
并同时检查提高分辨率是否损害同域精度。

用法示例::

    # 三模型各自最优分辨率做集成
    python src/eval_ensemble.py \
        --member runs/swin_t_224_15ep@288 \
        --member runs/resnet18_224_15ep@320 \
        --member runs/mobilenet_v3_large_224_15ep@320 \
        --same-domain runs/resnet18_224_15ep

    # 单成员（等价于单模型该分辨率的评测）
    python src/eval_ensemble.py --member runs/resnet18_224_15ep@320
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import build_transforms  # noqa: E402
from eval_crossdomain import PLANTDOC_TO_PV_FOLDER  # noqa: E402
from eval_preprocess_ablation import (  # noqa: E402
    PlantDocItems,
    build_variant_transform,
    metrics,
    predict_probs,
)
from models import build_model  # noqa: E402

ARTIFACT_DIR = ROOT / "artifacts"


def parse_member(spec: str) -> tuple[Path, int]:
    """``runs/xxx@320`` -> (Path, 320)。省略 ``@size`` 时默认 224。"""
    if "@" in spec:
        run_part, size_part = spec.rsplit("@", 1)
        size = int(size_part)
    else:
        run_part, size = spec, 224
    run_dir = Path(run_part)
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_part).resolve()
    return run_dir, size


def load_member(run_dir: Path):
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    classes = list(ck["classes"])
    model_name = ck.get("args", {}).get("model") or run_dir.name.rsplit("_", 2)[0]
    model, meta = build_model(model_name, len(classes), pretrained=False)
    state = ck["model"] if "model" in ck else ck["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model, meta, classes


def crossdomain_probs(model, size: int, args) -> tuple[torch.Tensor, torch.Tensor]:
    pv_folders = sorted(PLANTDOC_TO_PV_FOLDER.values())
    ds = PlantDocItems(Path(args.plantdoc_root), args.split, pv_folders,
                       build_variant_transform("baseline", size))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    return predict_probs(model, loader, args.device, flip=args.flip)


def samedomain_probs(model, size: int, args) -> tuple[torch.Tensor, torch.Tensor]:
    from torchvision.datasets import ImageFolder

    test_root = Path(args.data_root) / "test"
    ds = ImageFolder(str(test_root), transform=build_transforms(size, train=False))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    return predict_probs(model, loader, args.device, flip=False)


def same_metrics(probs: torch.Tensor, labels: torch.Tensor) -> dict:
    n = labels.numel()
    pred = probs.argmax(dim=1)
    acc = float((pred == labels).sum()) / n
    f1s = []
    for c in range(probs.size(1)):
        tp = int(((pred == c) & (labels == c)).sum())
        fp = int(((pred == c) & (labels != c)).sum())
        fn = int(((pred != c) & (labels == c)).sum())
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return {
        "n": n,
        "top1_acc": round(acc, 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="多模型/多分辨率集成评估")
    p.add_argument("--member", action="append", required=True,
                   help="run_dir 或 run_dir@img_size，可重复")
    p.add_argument("--plantdoc-root", default=str(ROOT / "data" / "raw" / "plantdoc"))
    p.add_argument("--data-root", default=str(ROOT / "data" / "processed" / "plantvillage"))
    p.add_argument("--split", default="all", choices=["train", "test", "all"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--flip", action="store_true", help="每个成员额外做水平翻转平均")
    p.add_argument("--same-domain", nargs="*", default=None,
                   help="需要额外评同域 test 的成员 run_dir（会慢，Swin 不建议）")
    p.add_argument("--out", default="")
    args = p.parse_args()

    args.device = torch.device("cpu")
    members = [parse_member(s) for s in args.member]
    same_set = set()
    if args.same_domain:
        for s in args.same_domain:
            same_set.add(str(parse_member(s)[0]))

    pv_folders = sorted(PLANTDOC_TO_PV_FOLDER.values())
    records: list[dict] = []
    cd_probs_sum: torch.Tensor | None = None
    cd_labels: torch.Tensor | None = None
    sd_probs_sum: torch.Tensor | None = None
    sd_labels: torch.Tensor | None = None
    all_same = bool(same_set) and all(str(m[0]) in same_set for m in members)

    for run_dir, size in members:
        tag = f"{run_dir.name}@{size}"
        print(f"\n=== {tag} ===", flush=True)
        t0 = time.time()
        model, meta, classes = load_member(run_dir)
        probs, labels = crossdomain_probs(model, size, args)
        if args.flip:
            probs2, _ = predict_probs(
                model,
                DataLoader(PlantDocItems(Path(args.plantdoc_root), args.split, pv_folders,
                                         build_variant_transform("baseline", size)),
                           batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers),
                args.device, flip=True)
            probs = (probs + probs2) / 2
        m = metrics(probs, labels, pv_folders)
        rec = {"member": tag, "model": meta.name, "img_size": size,
               "params_m": meta.params_m, "cross": m,
               "elapsed_s": round(time.time() - t0, 1)}
        print(f"跨域 top1={m['top1_acc']:.4f} top3={m['top3_acc']:.4f} "
              f"crop={m['crop_acc']:.4f} conf={m['mean_confidence']:.4f}", flush=True)

        if sd_labels is not None:
            assert torch.equal(labels, cd_labels), "成员间样本顺序不一致"
        cd_probs_sum = probs if cd_probs_sum is None else cd_probs_sum + probs
        cd_labels = labels

        if str(run_dir) in same_set:
            sd, sd_lab = samedomain_probs(model, size, args)
            sm = same_metrics(sd, sd_lab)
            rec["same"] = sm
            print(f"同域 top1={sm['top1_acc']:.4f} macroF1={sm['macro_f1']:.4f}", flush=True)
            if all_same:
                assert torch.equal(sd_lab, sd_labels) if sd_labels is not None else True
                sd_probs_sum = sd if sd_probs_sum is None else sd_probs_sum + sd
                sd_labels = sd_lab
        records.append(rec)

    n_members = len(members)
    ens_cd = metrics(cd_probs_sum / n_members, cd_labels, pv_folders)
    print(f"\n=== 集成（{n_members} 成员）===")
    print(f"跨域 top1={ens_cd['top1_acc']:.4f} top3={ens_cd['top3_acc']:.4f} "
          f"crop={ens_cd['crop_acc']:.4f} conf={ens_cd['mean_confidence']:.4f}")
    ens_sd = None
    if sd_probs_sum is not None:
        ens_sd = same_metrics(sd_probs_sum / n_members, sd_labels)
        print(f"同域 top1={ens_sd['top1_acc']:.4f} macroF1={ens_sd['macro_f1']:.4f}")

    payload = {
        "split": args.split,
        "members": records,
        "ensemble_cross": ens_cd,
        "ensemble_same": ens_sd,
        "config": {"flip": args.flip, "batch_size": args.batch_size},
    }
    out = Path(args.out) if args.out else ARTIFACT_DIR / "ensemble_res.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
