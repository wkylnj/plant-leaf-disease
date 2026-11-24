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

import numpy as np
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


def parse_member(spec: str) -> tuple[Path, list[int]]:
    """``runs/xxx@320`` -> (Path, [320])；``@224,288,320`` 表示多分辨率平均。"""
    if "@" in spec:
        run_part, size_part = spec.rsplit("@", 1)
        sizes = [int(s) for s in size_part.split(",") if s.strip()]
    else:
        run_part, sizes = spec, [224]
    run_dir = Path(run_part)
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_part).resolve()
    return run_dir, sizes


def load_member(run_dir: Path):
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    classes = list(ck["classes"])
    model_name = ck.get("args", {}).get("model") or run_dir.name.rsplit("_", 2)[0]
    model, meta = build_model(model_name, len(classes), pretrained=False)
    state = ck["model"] if "model" in ck else ck["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model, meta, classes


def crossdomain_loader(size: int, args, pv_folders: list[str]) -> DataLoader:
    tf, _actual = build_variant_transform("baseline", size)
    ds = PlantDocItems(Path(args.plantdoc_root), args.split, pv_folders, tf)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers)


def crossdomain_probs(model, size: int, args,
                      pv_folders: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    loader = crossdomain_loader(size, args, pv_folders)
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
    p.add_argument("--weights", default="",
                   help="逗号分隔的成员权重，如 1,0.3,0.2；空则等权")
    p.add_argument("--save-probs", default="",
                   help="把各成员概率矩阵存到该目录（.npz），便于离线权重搜索")
    p.add_argument("--out", default="")
    args = p.parse_args()

    args.device = torch.device("cpu")
    members = [parse_member(s) for s in args.member]
    same_set = set()
    if args.same_domain:
        for s in args.same_domain:
            same_set.add(str(parse_member(s)[0]))

    # 标签空间必须与模型输出通道顺序严格一致：取 ck["classes"]（38 类文件夹名）。
    # 不能对映射目标排序，那样通道会错位（实测 top-1 从 33.7% 掉到 6.8%）。
    loaded: list[tuple[Path, list[int], object, object, list[str]]] = []
    pv_folders: list[str] | None = None
    for run_dir, sizes in members:
        model, meta, classes = load_member(run_dir)
        if pv_folders is None:
            pv_folders = classes
        elif classes != pv_folders:
            raise SystemExit(f"成员 {run_dir.name} 的类别顺序与其他成员不一致")
        loaded.append((run_dir, sizes, model, meta, classes))
    assert pv_folders is not None

    records: list[dict] = []
    member_probs: list[torch.Tensor] = []
    cd_labels: torch.Tensor | None = None
    sd_probs_sum: torch.Tensor | None = None
    sd_labels: torch.Tensor | None = None
    all_same = bool(same_set) and all(str(m[0]) in same_set for m in members)

    for run_dir, sizes, model, meta, classes in loaded:
        tag = f"{run_dir.name}@{','.join(str(s) for s in sizes)}"
        print(f"\n=== {tag} ===", flush=True)
        t0 = time.time()
        probs: torch.Tensor | None = None
        labels: torch.Tensor | None = None
        per_size: dict[str, dict] = {}
        for size in sizes:
            p_i, labels = crossdomain_probs(model, size, args, pv_folders)
            if args.flip:
                p2, _ = predict_probs(model, crossdomain_loader(size, args, pv_folders),
                                      args.device, flip=True)
                p_i = (p_i + p2) / 2
            per_size[str(size)] = metrics(p_i, labels, pv_folders)
            probs = p_i if probs is None else probs + p_i
        assert probs is not None and labels is not None
        probs = probs / len(sizes)
        m = metrics(probs, labels, pv_folders)
        rec = {"member": tag, "model": meta.name, "img_sizes": sizes,
               "params_m": meta.params_m, "cross": m, "per_size": per_size,
               "elapsed_s": round(time.time() - t0, 1)}
        if len(sizes) > 1:
            for s, pm in per_size.items():
                print(f"  单独 @{s}: top1={pm['top1_acc']:.4f} crop={pm['crop_acc']:.4f}",
                      flush=True)
        print(f"跨域 top1={m['top1_acc']:.4f} top3={m['top3_acc']:.4f} "
              f"crop={m['crop_acc']:.4f} conf={m['mean_confidence']:.4f}", flush=True)

        if cd_labels is not None:
            assert torch.equal(labels, cd_labels), "成员间样本顺序不一致"
        cd_labels = labels

        if str(run_dir) in same_set:
            sd, sd_lab = samedomain_probs(model, sizes[0], args)
            sm = same_metrics(sd, sd_lab)
            rec["same"] = sm
            print(f"同域 top1={sm['top1_acc']:.4f} macroF1={sm['macro_f1']:.4f}", flush=True)
            if all_same:
                sd_probs_sum = sd if sd_probs_sum is None else sd_probs_sum + sd
                sd_labels = sd_lab
        member_probs.append(probs)
        records.append(rec)

    n_members = len(members)
    if args.weights.strip():
        w = [float(x) for x in args.weights.split(",")]
        if len(w) != n_members:
            raise SystemExit(f"--weights 需要 {n_members} 个值，收到 {len(w)} 个")
    else:
        w = [1.0] * n_members
    wsum = sum(w)
    if wsum <= 0:
        raise SystemExit("--weights 之和必须 > 0")

    stacked = torch.stack(member_probs, dim=0)          # [M, N, C]
    w_t = torch.tensor(w, dtype=torch.float32).view(-1, 1, 1)
    ens_cd = metrics((stacked * w_t).sum(dim=0) / wsum, cd_labels, pv_folders)
    print(f"\n=== 集成（{n_members} 成员，权重 {w}）===")
    print(f"跨域 top1={ens_cd['top1_acc']:.4f} top3={ens_cd['top3_acc']:.4f} "
          f"crop={ens_cd['crop_acc']:.4f} conf={ens_cd['mean_confidence']:.4f}")
    ens_sd = None
    if sd_probs_sum is not None:
        ens_sd = same_metrics(sd_probs_sum / n_members, sd_labels)
        print(f"同域 top1={ens_sd['top1_acc']:.4f} macroF1={ens_sd['macro_f1']:.4f}")

    if args.save_probs:
        sp = Path(args.save_probs)
        sp.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            sp / "probs.npz",
            probs=stacked.numpy(),
            labels=cd_labels.numpy(),
            members=np.array([r["member"] for r in records]),
            pv_folders=np.array(pv_folders),
        )
        print(f"概率矩阵已存 {sp / 'probs.npz'}  shape={tuple(stacked.shape)}")

    payload = {
        "split": args.split,
        "members": records,
        "weights": w,
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
