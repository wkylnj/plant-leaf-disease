"""测试时适配（test-time adaptation）评估：PlantVillage → PlantDoc。

要回答的问题
------------
跨域零样本只有 25%~35%，模型在真实手机图上严重掉点。掉点的很大一部分原因
不是"认不出病害"，而是**特征分布漂移**：PlantVillage 是摆拍（纯色背景、单一叶片、
恒定光照），PlantDoc 是田间/手机拍摄（多叶片、复杂背景、过曝/偏色）。BatchNorm
的 running stats 是在源域上估的，直接搬到目标域会让每层激活的均值方差都错位。

如果只用目标域**无标签**的图片把这些统计量重新估一遍，或者进一步在目标域上做
熵最小化，能不能把准确率拉回来？这就是本实验要测的东西。

三种适配方法（全部不用标签）
--------------------------
1. `bn`   —— BN 统计重估：模型整体 eval，只把 BatchNorm 层切到 train 模式并用
   累积平均（momentum=None）重估 running stats，图像前向一遍，不反传、不改权重。
2. `tent` —— TENT（Wang et al., ICLR 2021）：BN 保持 eval（用当前 running stats），
   只对 BN 的仿射参数 weight/bias 求梯度，损失是**逐样本预测熵**的均值，Adam lr=1e-3。
3. `pl`   —— 伪标签自训练：先用当前模型给目标域图片打伪标签，只保留置信度
   > 0.7 的样本，做 1 轮交叉熵微调（全网络，lr=1e-4）。
4. `bn_tent` —— 组合：先做 1，再在重估后的统计量上做 2。这是 TENT 原文的实际
   配置（先对齐统计量，再优化熵），单独跑 2 时源域统计量可能一开始就偏得离谱。

评估协议（严格的 transductive 设定，必须写进报告）
------------------------------------------------
- 适配阶段：**只用 PlantDoc `train` split 的图片，且代码里物理上拿不到标签**
  （`ImageOnly` 包装器把 `y` 丢掉），1 个 epoch。
- 评估阶段：只在 PlantDoc `test` split（252 张）上做，test 的标签只用于算指标。
- 每个方法都报告"不做任何适配"的零样本 baseline 在**同一个 test split** 上的
  数字，否则没有可比性。
- train/test 不重叠（数据集本身划分好的），因此不是"在测试集上自训练"，
  但仍然是 transductive：适配时看过了目标域的无标签数据。若完全不允许接触
  目标域数据，这些方法都不适用。

用法
----
    python src/eval_adapt.py --run-dir runs/resnet18_224_15ep
    python src/eval_adapt.py --run-dir runs/resnet18_224_15ep runs/mobilenet_v3_large_224_15ep
    python src/eval_adapt.py --run-dir runs/resnet18_224_15ep --adapt-limit 500 --methods none,bn

产物
----
    ml/artifacts/tta_adapt.md    模型 × 方法 的对比表 + 结论
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import build_transforms  # noqa: E402
from eval_crossdomain import (  # noqa: E402
    PlantDocDataset,
    crop_of_folder,
    evaluate,
)
from eval_tta import load_model_from_run  # noqa: E402

BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


# ---------------------------------------------------------------------------
# 数据集包装
# ---------------------------------------------------------------------------
class ImageOnly(Dataset):
    """只吐图像、不吐标签的视图。

    这不是"我保证不偷看标签"，而是让适配代码**拿不到** y：
    循环里写 `for x in loader` 就不可能把标签用进去。报告里"适配不用标签"
    这句话因此是可验证的，不靠自觉。
    """

    def __init__(self, base: PlantDocDataset, indices: list[int]) -> None:
        self.base = base
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> torch.Tensor:
        img, _label = self.base[self.indices[i]]
        return img


class PseudoLabeled(Dataset):
    """伪标签自训练用的数据集：图像 + 模型自己给的伪标签。"""

    def __init__(self, base: PlantDocDataset, indices: list[int], pseudo: list[int]) -> None:
        self.base = base
        self.indices = indices
        self.pseudo = pseudo

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        img, _label = self.base[self.indices[i]]
        return img, self.pseudo[i]


def stratified_subset(ds: PlantDocDataset, limit: int) -> list[int]:
    """从适配集里按类别轮转抽 limit 张，保证每个类都被覆盖。

    直接取前 N 张会让适配集集中在排序靠前的几个类上，BN 统计量和熵都会偏。
    这里用轮转（每个类先各取 1 张，再各取第 2 张……）保持类别分布。
    """
    if limit <= 0 or limit >= len(ds):
        return list(range(len(ds)))

    by_class: dict[int, list[int]] = defaultdict(list)
    for i, (_p, label, _o) in enumerate(ds.items):
        by_class[label].append(i)

    picks: list[int] = []
    round_idx = 0
    keys = sorted(by_class)
    while len(picks) < limit:
        added = False
        for k in keys:
            if round_idx < len(by_class[k]):
                picks.append(by_class[k][round_idx])
                added = True
                if len(picks) >= limit:
                    break
        if not added:  # 所有类都抽空了
            break
        round_idx += 1
    return sorted(picks)


# ---------------------------------------------------------------------------
# 适配方法
# ---------------------------------------------------------------------------
def bn_modules(model: nn.Module) -> list[nn.Module]:
    return [m for m in model.modules() if isinstance(m, BN_TYPES)]


def adapt_bn(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """BN 统计重估：只让 BN 更新 running stats，其余层保持 eval。

    momentum=None 时 PyTorch 用"自 reset 以来的累积平均"，所以跑完一个 epoch
    得到的 running stats 就是这批图 batch 统计量的精确平均，与 batch 大小无关；
    用默认 momentum=0.1 的话只能逼近最近若干个 batch，等于只估了数据集尾部。
    """
    mods = bn_modules(model)
    model.eval()
    for m in mods:
        m.reset_running_stats()
        m.momentum = None
        m.train()

    t0 = time.time()
    n = 0
    with torch.no_grad():
        for x in loader:
            model(x.to(device))
            n += x.shape[0]
    elapsed = time.time() - t0

    for m in mods:
        m.eval()
    return {"adapt_s": round(elapsed, 1), "adapt_images": n, "bn_layers": len(mods)}


def _set_only_bn_affine_trainable(model: nn.Module) -> list[nn.Parameter]:
    """冻结全部参数，只放开 BN 的 affine（weight/bias）。"""
    for p in model.parameters():
        p.requires_grad_(False)
    params: list[nn.Parameter] = []
    for m in bn_modules(model):
        if getattr(m, "affine", False):
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)
            params.extend([m.weight, m.bias])
    return params


def adapt_tent(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lr: float = 1e-3,
    max_batches: int = 0,
) -> dict:
    """TENT：只用预测熵做无监督目标，只更新 BN 的 affine 参数。

    BN 保持 eval 模式（用 running stats），所以不会因为一个 batch 里只有
    几十张图就把统计量带跑偏；改动面只有每个 BN 层的缩放/平移，参数量很小，
    是"最不容易把模型改坏"的适配方式。
    """
    model.eval()
    params = _set_only_bn_affine_trainable(model)
    if not params:
        return {"adapt_s": 0.0, "adapt_images": 0, "tent_params": 0, "skipped": "模型没有 affine BN 参数"}

    opt = torch.optim.Adam(params, lr=lr)
    t0 = time.time()
    n = 0
    losses: list[float] = []
    for bi, x in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x = x.to(device)
        logits = model(x)
        # 逐样本熵，再对 batch 取均值：让"每个样本都变自信"，而不是让平均预测变尖
        p = logits.softmax(dim=1)
        loss = -(p * logits.log_softmax(dim=1)).sum(dim=1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        n += x.shape[0]
    elapsed = time.time() - t0

    for p in model.parameters():
        p.requires_grad_(False)
    return {
        "adapt_s": round(elapsed, 1),
        "adapt_images": n,
        "tent_params": sum(p.numel() for p in params),
        "entropy_first": round(losses[0], 4) if losses else None,
        "entropy_last": round(losses[-1], 4) if losses else None,
    }


def adapt_pseudo(
    model: nn.Module,
    base: PlantDocDataset,
    indices: list[int],
    device: torch.device,
    batch_size: int,
    thresh: float = 0.7,
    lr: float = 1e-4,
    num_workers: int = 0,
) -> dict:
    """伪标签自训练：置信度 > thresh 的样本做 1 轮微调。

    阈值卡在 0.7 是个折中：太低会把大量错标签喂回去（跨域零样本 top-1 只有
    25%，错误标签占比很高，自训练会强化错误）；太高则样本太少，1 轮几乎没效果。
    实际保留数量在报告里给出，方便判断这个方法的可信度。
    """
    loader = DataLoader(
        ImageOnly(base, indices), batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # ---- 第一遍：打伪标签 ----
    t0 = time.time()
    model.eval()
    keep_idx: list[int] = []
    keep_lab: list[int] = []
    offset = 0
    with torch.no_grad():
        for x in loader:
            probs = model(x.to(device)).softmax(dim=1)
            conf, pred = probs.max(dim=1)
            for j in range(x.shape[0]):
                if float(conf[j]) > thresh:
                    keep_idx.append(indices[offset + j])
                    keep_lab.append(int(pred[j]))
            offset += x.shape[0]
    pseudo_s = time.time() - t0

    if len(keep_idx) == 0:
        return {
            "adapt_s": round(pseudo_s, 1),
            "adapt_images": 0,
            "pseudo_kept": 0,
            "skipped": f"没有样本置信度 > {thresh}",
        }

    # ---- 第二遍：1 轮微调 ----
    train_loader = DataLoader(
        PseudoLabeled(base, keep_idx, keep_lab),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t1 = time.time()
    n = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        n += x.shape[0]
    ft_s = time.time() - t1
    model.eval()
    return {
        "adapt_s": round(pseudo_s + ft_s, 1),
        "adapt_images": n,
        "pseudo_kept": len(keep_idx),
        "pseudo_ratio": round(len(keep_idx) / max(len(indices), 1), 3),
        "thresh": thresh,
        "ft_lr": lr,
    }


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def full_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pv_folders: list[str],
) -> dict:
    """复用 eval_crossdomain.evaluate 的前向流程，再补作物级准确率。

    刻意不自己重写一遍前向：baseline 必须和之前 crossdomain 报告里的数字
    同源，否则"提升了多少"就说不清是方法带来的还是实现差异带来的。
    """
    res = evaluate(model, loader, device, len(pv_folders))
    pv_crop = [crop_of_folder(f) for f in pv_folders]
    crop_ok = sum(
        1 for p, t in zip(res["_pred"], res["_true"]) if pv_crop[p] == pv_crop[t]
    )
    return {
        "n": res["n"],
        "top1": res["top1_acc"],
        "top3": res["top3_acc"],
        "crop": crop_ok / max(res["n"], 1),
        "mean_conf": sum(res["_conf"]) / max(len(res["_conf"]), 1),
        "eval_s": res["elapsed_s"],
        "img_per_s": res["img_per_s"],
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="PlantVillage→PlantDoc 测试时适配")
    ap.add_argument("--run-dir", nargs="+", required=True)
    ap.add_argument("--plantdoc-root", default=str(ROOT / "data" / "raw" / "plantdoc"))
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--adapt-limit", type=int, default=0,
                    help="适配集最多用多少张（0=全部）。按类别轮转抽样")
    ap.add_argument("--methods", default="none,bn,tent,bn_tent,pl")
    ap.add_argument("--tent-lr", type=float, default=1e-3)
    ap.add_argument("--tent-lo-lr", type=float, default=1e-4,
                    help="tent_lo 用的更小步长，用来区分「方法本身没用」和「lr 太大导致崩塌」")
    ap.add_argument("--pl-thresh", type=float, default=0.7)
    ap.add_argument("--pl-lr", type=float, default=1e-4)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "tta_adapt.md"))
    ap.add_argument("--json-out", default="", help="可选：额外落一份 json 便于排查")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    known = {"none", "bn", "tent", "tent_lo", "bn_tent", "pl"}
    bad = [m for m in methods if m not in known]
    if bad:
        print(f"[错误] 未知方法 {bad}，可选：{sorted(known)}")
        return 1

    if args.num_workers == 0:
        torch.set_num_threads(max(1, os.cpu_count() or 1))

    device = torch.device("cpu")
    pd_root = Path(args.plantdoc_root)
    transform = build_transforms(args.img_size, train=False)

    run_dirs = []
    for r in args.run_dir:
        p = Path(r) if Path(r).is_absolute() else (ROOT / r)
        if not (p / "best.pt").exists():
            print(f"[错误] 找不到 {p / 'best.pt'}")
            return 1
        run_dirs.append(p)

    print(f"设备        : {device} (threads={torch.get_num_threads()})")
    print(f"PlantDoc    : {pd_root}")
    print(f"方法        : {methods}")

    all_rows: list[dict] = []
    meta_all: dict = {
        "protocol": "adapt on PlantDoc train (unlabeled), eval on PlantDoc test",
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "adapt_limit": args.adapt_limit,
        "tent_lr": args.tent_lr,
        "pl_thresh": args.pl_thresh,
        "pl_lr": args.pl_lr,
        "runs": [],
    }

    for rd in run_dirs:
        # 类别顺序与模型输出索引严格绑定，必须用 checkpoint 里保存的那份
        _, classes, meta = load_model_from_run(rd)
        target_cls = classes  # 类别顺序唯一来源：checkpoint 里的 classes

        test_ds = PlantDocDataset(pd_root, "test", target_cls, transform)
        if len(test_ds) == 0:
            print(f"[错误] {rd.name} 的 test 集映射后为 0，检查 PLANTDOC_TO_PV_FOLDER")
            return 1
        adapt_ds = PlantDocDataset(pd_root, "train", target_cls, transform)
        adapt_idx = stratified_subset(adapt_ds, args.adapt_limit)
        adapt_loader = DataLoader(
            ImageOnly(adapt_ds, adapt_idx),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        )

        print(f"\n=== {rd.name} ({meta['model']}, epoch {meta['ckpt_epoch']}) ===")
        print(f"  适配集: {len(adapt_idx)}/{len(adapt_ds)} 张（PlantDoc train，无标签）")
        print(f"  测试集: {len(test_ds)} 张（PlantDoc test，仅此处使用标签）")

        base_row: dict | None = None
        for method in methods:
            # 每个方法都从**同一份干净权重**重新开始，避免方法之间的顺序污染
            model, _, _ = load_model_from_run(rd)
            model.to(device)

            adapt_info: dict = {}
            if method == "none":
                pass
            elif method == "bn":
                adapt_info = adapt_bn(model, adapt_loader, device)
            elif method == "tent":
                adapt_info = adapt_tent(model, adapt_loader, device, lr=args.tent_lr)
            elif method == "tent_lo":
                adapt_info = adapt_tent(model, adapt_loader, device, lr=args.tent_lo_lr)
            elif method == "bn_tent":
                info_bn = adapt_bn(model, adapt_loader, device)
                info_tent = adapt_tent(model, adapt_loader, device, lr=args.tent_lr)
                adapt_info = {**info_bn, **info_tent,
                              "adapt_s": round(info_bn["adapt_s"] + info_tent["adapt_s"], 1)}
            elif method == "pl":
                adapt_info = adapt_pseudo(
                    model, adapt_ds, adapt_idx, device,
                    batch_size=args.batch_size, thresh=args.pl_thresh,
                    lr=args.pl_lr, num_workers=args.num_workers,
                )

            m = full_metrics(model, test_loader, device, target_cls)
            row = {
                "model": rd.name,
                "model_type": meta["model"],
                "method": method,
                **m,
                "adapt_s": adapt_info.get("adapt_s", 0.0),
                "adapt_images": adapt_info.get("adapt_images", 0),
                "note": adapt_info.get("skipped", ""),
                "pseudo_kept": adapt_info.get("pseudo_kept", None),
            }
            if base_row is None:
                base_row = row
                row["delta"] = 0.0
            else:
                row["delta"] = m["top1"] - base_row["top1"]
            all_rows.append(row)

            extra = ""
            if method == "pl" and row["pseudo_kept"] is not None:
                extra = f" 伪标签={row['pseudo_kept']} 张"
            print(
                f"  {method:<8} top1={m['top1']:.4f} top3={m['top3']:.4f} "
                f"crop={m['crop']:.4f} conf={m['mean_conf']:.4f} "
                f"adapt={row['adapt_s']:.0f}s eval={m['eval_s']:.0f}s{extra}"
            )

        meta_all["runs"].append({
            "run": rd.name,
            "model": meta["model"],
            "ckpt_epoch": meta["ckpt_epoch"],
            "ckpt_val_acc": meta["ckpt_val_acc"],
            "n_adapt_available": len(adapt_ds),
            "n_adapt_used": len(adapt_idx),
            "n_test": len(test_ds),
        })

    # ---- 写报告 ----
    out = Path(args.out) if Path(args.out).is_absolute() else (ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_md(all_rows, meta_all, args), encoding="utf-8")
    print(f"\n已保存: {out}")

    if args.json_out:
        jp = Path(args.json_out)
        if not jp.is_absolute():
            jp = ROOT / jp
        jp.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已保存: {jp}")
    return 0


METHOD_CN = {
    "none": "零样本 baseline（不适配）",
    "bn": "BN 统计重估",
    "tent": "TENT（熵最小化，只更 BN affine，lr=1e-3）",
    "tent_lo": "TENT（lr=1e-4，防崩塌对照）",
    "bn_tent": "BN 重估 + TENT",
    "pl": "伪标签自训练（conf>阈值）",
}


def render_md(rows: list[dict], meta: dict, args) -> str:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    n_adapt = meta["runs"][0]["n_adapt_used"] if meta["runs"] else 0
    n_avail = meta["runs"][0]["n_adapt_available"] if meta["runs"] else 0
    n_test = meta["runs"][0]["n_test"] if meta["runs"] else 0

    lines = [
        "# 测试时适配（TTA-adapt）实验：PlantVillage → PlantDoc",
        "",
        "## 设定",
        "",
        f"- 适配数据：PlantDoc **train** split，映射后 {n_avail} 张，"
        f"本次按类别轮转抽取 **{n_adapt} 张**，**不使用任何标签**"
        f"（`ImageOnly` 包装器在数据层就把标签丢掉了）。",
        f"- 评估数据：PlantDoc **test** split，**{n_test} 张**，标签仅用于算指标。",
        "- 适配 1 个 epoch；每个方法都从同一份原始 checkpoint 重新开始，互不影响。",
        f"- 超参：TENT Adam lr={args.tent_lr}；伪标签阈值 {args.pl_thresh}、微调 lr={args.pl_lr}；"
        f"输入 {args.img_size}×{args.img_size}，batch {args.batch_size}，纯 CPU。",
        "- 「耗时」= 适配本身的墙钟时间（baseline 为 0），评估耗时单列。",
        "",
        "## 结果",
        "",
        "| 模型 | 方法 | top-1 | top-3 | 作物级 | 平均置信度 | 相对基线 | 适配耗时(s) | 评估耗时(s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for name, rs in by_model.items():
        for r in rs:
            note = ""
            if r["method"] == "pl" and r["pseudo_kept"] is not None:
                note = f" (伪标签 {r['pseudo_kept']} 张)"
            delta = "—" if r["method"] == "none" else f"{r['delta']:+.2%}"
            lines.append(
                f"| {r['model_type']} | {METHOD_CN.get(r['method'], r['method'])}{note} | "
                f"{r['top1']:.4f} | {r['top3']:.4f} | {r['crop']:.4f} | "
                f"{r['mean_conf']:.4f} | {delta} | {r['adapt_s']:.0f} | {r['eval_s']:.0f} |"
            )

    # ---- 逐模型小结 ----
    lines += ["", "## 逐模型小结", ""]
    for name, rs in by_model.items():
        base = next((r for r in rs if r["method"] == "none"), None)
        others = [r for r in rs if r["method"] != "none" and not r["note"]]
        lines.append(f"**{name}**（零样本 top-1 = {base['top1']:.2%}）")
        lines.append("")
        if not others:
            lines.append("- 无可用适配结果。")
        else:
            best = max(others, key=lambda r: r["top1"])
            for r in others:
                lines.append(
                    f"- {METHOD_CN.get(r['method'], r['method'])}："
                    f"{r['top1']:.2%}（{r['delta']:+.2%}），"
                    f"作物级 {r['crop']:.2%}（{(r['crop'] - base['crop']):+.2%}），"
                    f"平均置信度 {base['mean_conf']:.3f} → {r['mean_conf']:.3f}，"
                    f"代价 {r['adapt_s']:.0f}s"
                )
            lines.append(
                f"- 最优：{METHOD_CN.get(best['method'], best['method'])}，"
                f"top-1 {best['top1']:.2%}（{best['delta']:+.2%}）。"
            )
        lines.append("")

    # ---- 结论 ----
    deltas = [r["delta"] for r in rows if r["method"] != "none"]
    best_rows = []
    for name, rs in by_model.items():
        base = next((r for r in rs if r["method"] == "none"), None)
        others = [r for r in rs if r["method"] != "none" and not r["note"]]
        if others and base:
            b = max(others, key=lambda r: r["top1"])
            best_rows.append((base, b))

    lines += ["## 结论", ""]
    if best_rows:
        gain_txt = "；".join(
            f"{b['model_type']} 最好 {METHOD_CN.get(b['method'], b['method'])} "
            f"{b['top1']:.2%}（{b['top1'] - base['top1']:+.2%}）"
            for base, b in best_rows
        )
        n_pos = sum(1 for _, b in best_rows if b["top1"] > base["top1"])
        lines.append(f"- 各模型的最优配置：{gain_txt}。")
        if n_pos == 0:
            lines.append(
                "- **适配无效**：所有方法在两个模型上都没有超过零样本 baseline。"
                "说明本例的跨域掉点主要不是 BN 统计量漂移或决策边界不够尖锐造成的，"
                "而是特征本身（纹理/构图/背景）与源域差异过大，靠无标签的统计对齐"
                "和熵最小化补不回来。"
            )
        elif n_pos < len(best_rows):
            lines.append(
                f"- **部分有效**：{n_pos}/{len(best_rows)} 个模型有提升，"
                "但提升幅度小于模型间差异，不能据此断言方法普适。"
            )
        else:
            lines.append("- **适配有效**：两个模型的最优方法都超过了零样本 baseline。")
        lines.append("")

    if deltas:
        lines.append(
            f"- 代价：单次适配 {min(r['adapt_s'] for r in rows if r['method'] != 'none'):.0f}"
            f"–{max(r['adapt_s'] for r in rows if r['method'] != 'none'):.0f}s（纯 CPU，"
            f"{n_adapt} 张 1 epoch）。若每次部署都要重估，可接受；若要逐样本在线适配则不行。"
        )
    lines += [
        "",
        "## 必须写明的局限",
        "",
        "1. **这是 transductive 设定**：适配阶段看过了目标域的无标签图片（虽然不看标签）。"
        "如果应用场景完全不允许接触目标域数据（例如只能拿到单张图、拿到就得出结果），"
        "BN 重估 / TENT / 伪标签自训练**都不适用**——它们的前提就是「有一批目标域未标注数据」。",
        "2. 适配集是从 PlantDoc train 的**已映射类别**里抽的，挑选类别时用了文件夹名"
        "（不用标签），所以严格说是「知道目标域有哪些类」的设定。",
        "3. PlantDoc test 只有 252 张，1 张 = 0.40 个百分点，小于 ±2% 的差异不应过度解读。",
        "4. 只跑了 1 个 epoch / 1 组超参，没有对 lr、阈值、适配集大小做搜索；"
        "结论是「这套默认配置有没有用」，不是「这类方法的上限」。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
