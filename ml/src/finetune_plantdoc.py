"""E3 有监督域适应：PlantDoc 五折微调实验。

要回答的问题
------------
eval_crossdomain.py 测的是**零样本泛化**：PlantVillage 权重原样搬到 PlantDoc，
top-1 28.18%。那是在完全不给目标域标签的前提下测的，是泛化能力的下界。

本实验换一个设定：**有监督域适应（supervised domain adaptation）**——允许用
PlantDoc 的标签做微调，问"少量目标域标注 + 预训练权重，能把跨域准确率提到多少"。

这两个设定回答的是不同问题，报告里必须分开写，不能拿"微调后 60%"去否定
"零样本 28%"，也不能用微调结果冒充泛化能力。

为什么用 5 折
-------------
映射后只有 2920 张图、27 类，单次 80/20 划分的验证集只有约 584 张：一张图的
涨跌就是 0.17 个百分点，折间波动可能比方法本身的差异还大。5 折交叉验证让每张
图恰好当一次验证样本，报告"均值 ± 标准差"，并且可以把"微调到底有没有用"做成
同一折上的配对比较（ft − zs），比两个孤立数字可靠得多。

划分方式
--------
按 **PlantDoc 类别**分层，把 train 与 test 的图片合并后重新分折——这里不再区分
官方 train/test，因为两者都是"目标域带标签数据"，全部可用于交叉验证，合并后
样本量才够 5 折。具体做法：每个类别内部用固定种子打乱，再按 i % 5 轮流分配，
每折拿到该类约 1/5 的图，类间比例在 5 折里基本一致。样本数不足 5 的类
（本数据集没有）会退化为部分折缺失。

微调配置（任务给定）
--------------------
    backbone lr 1e-4、分类头 lr 1e-3、AdamW、cosine 退火、3~5 轮、
    label smoothing 0.1、batch 64、输入 224。

两个实现上的取舍，报告里要写清楚：

1. **分类头保持 38 类**（PlantVillage 原始输出维度），不换成 27 类。
   这样零样本 baseline 和微调后的模型是同一个分类头、同一套评估代码，
   比较才严格可比；只有 27 个类别会收到正梯度，其余 11 类被交叉熵压下去。
   代价是微调后的模型仍可能把概率分给用不上的类，这是该设定的真实情况。

2. **不做早停、不按验证折选轮次**。验证折只用来报告指标，不参与任何选择，
   否则"验证集"就变成了模型选择集，指标会虚高。固定跑满 --epochs 轮取最后一轮。

用法
----
    # 完整 5 折（GPU 上约 20~30 分钟/模型）
    python src/finetune_plantdoc.py --run-dir runs/resnet18_224_15ep

    # CPU 冒烟：1 折 1 轮，每类只取 4 张，只看逻辑跑不跑得通
    python src/finetune_plantdoc.py --run-dir runs/resnet18_224_15ep \
        --folds-to-run 1 --epochs 1 --limit-per-class 4 --limit-batches 2 \
        --batch-size 8 --num-workers 0

产物
----
    <run-dir>/plantdoc_ft_cv.json   每折指标 + 均值/标准差 + 与零样本的配对差
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import build_transforms  # noqa: E402
from eval_crossdomain import (  # noqa: E402
    PLANTDOC_TO_PV_FOLDER,
    PlantDocDataset,
    crop_of_folder,
)
from models import build_model  # noqa: E402


# 分类头所在模块名。微调时分类头用更大的学习率（随机初始化 / 需要快速适配），
# backbone 用小学习率（保护预训练特征）。这里按模块前缀切分参数，
# 不同模型系列的分类头名字不一样，必须显式列出，不能靠"最后一个参数"猜。
HEAD_PREFIX = {
    "resnet18": "fc",
    "resnet50": "fc",
    "shufflenet_v2_x1_0": "fc",
    "mobilenet_v3_large": "classifier",
    "mobilenet_v3_small": "classifier",
    "efficientnet_b0": "classifier",
    "convnext_tiny": "classifier",
    "vit_b_16": "heads",
    "swin_t": "head",
}


class ItemsDataset(Dataset):
    """按索引取 (PIL 图, PlantVillage 标签)，transform 由外部指定。

    为什么要这一层：交叉验证里同一批图既要当训练集（强增强）又要当验证集
    （确定性处理），而 PlantDocDataset 的 transform 在构造时就固定了。
    这里只做索引 + 打开图片，原图的扫描与类别映射仍完全复用 PlantDocDataset，
    不重复实现映射逻辑。
    """

    def __init__(self, items, indices: list[int], transform) -> None:
        self.items = items
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        path, label, _orig = self.items[self.indices[k]]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean_std(vals: list[float]) -> tuple[float, float]:
    """均值与样本标准差（ddof=1）。5 折就是 5 个数，用它估计折间波动。

    折数 < 2 时（冒烟只跑 1 折）标准差无定义，返回 0 并在 JSON 里注明。
    """
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def validate_items(items) -> tuple[list, list[str]]:
    """剔除打不开的坏图，避免训练中途才崩。

    PlantDoc 来自网络抓取，个别文件可能损坏。5 折跑一半崩掉的代价远高于
    一次性扫一遍（2920 张约几秒）。
    """
    good: list = []
    bad: list[str] = []
    for it in items:
        try:
            with Image.open(it[0]) as im:
                im.verify()
            good.append(it)
        except Exception:
            bad.append(str(it[0]))
    return good, bad


def make_stratified_folds(items, n_folds: int, seed: int) -> list[int]:
    """按类别分层分折，返回每张图的折号。

    每类内部打乱后 round-robin 分配：第 k 张给第 (k + start) % n_folds 折，
    start 是每类的随机起点。加随机起点是为了让"类样本数除不尽 5 时多出来的
    那 1 张"分散到不同折——否则每个类都从折 0 开始，折 0 会系统性偏大
    （实测 596 张 vs 折 4 的 573 张），折大小和折难度就混在一起了。
    固定 seed → 折划分可复现，换机器重跑得到同一套折（否则数字对不上）。
    """
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, (_path, label, _orig) in enumerate(items):
        by_label[label].append(idx)

    fold_of = [0] * len(items)
    rng = random.Random(seed)
    for label in sorted(by_label):
        idxs = list(by_label[label])
        rng.shuffle(idxs)
        start = rng.randrange(n_folds)
        for k, idx in enumerate(idxs):
            fold_of[idx] = (k + start) % n_folds
    return fold_of


def split_param_groups(model: nn.Module, model_name: str) -> tuple[list, list]:
    """把参数分成 (backbone, head) 两组，供不同学习率使用。"""
    prefix = HEAD_PREFIX.get(model_name.lower())
    if prefix is None:
        raise ValueError(f"模型 {model_name} 没有登记分类头模块名，请补 HEAD_PREFIX")

    backbone, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name == prefix or name.startswith(prefix + "."):
            head.append(p)
        else:
            backbone.append(p)

    assert head, f"没找到分类头参数（prefix={prefix}），检查 HEAD_PREFIX"
    assert backbone, "backbone 参数为空，检查模型构建"
    return backbone, head


@torch.no_grad()
def evaluate(model, loader, device, pv_crop: list[str]) -> dict:
    """在验证折上算 top-1 / top-3 / 作物级准确率。"""
    model.eval()
    preds: list[int] = []
    trues: list[int] = []
    top3_hit = 0
    conf_sum = 0.0

    t0 = time.time()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        top3 = probs.topk(min(3, probs.shape[1]), dim=1)
        preds.extend(top3.indices[:, 0].cpu().tolist())
        trues.extend(y.tolist())
        top3_hit += sum(
            1 for row, t in zip(top3.indices.cpu().tolist(), y.tolist()) if t in row
        )
        conf_sum += float(top3.values[:, 0].sum())
    elapsed = time.time() - t0

    n = len(trues)
    correct = sum(1 for p, t in zip(preds, trues) if p == t)
    crop_correct = sum(
        1 for p, t in zip(preds, trues) if pv_crop[p] == pv_crop[t]
    )
    return {
        "n": n,
        "correct": correct,
        "top1_acc": correct / n if n else 0.0,
        "top3_acc": top3_hit / n if n else 0.0,
        "crop_correct": crop_correct,
        "crop_acc": crop_correct / n if n else 0.0,
        "mean_conf": conf_sum / n if n else 0.0,
        "elapsed_s": round(elapsed, 1),
    }


def finetune_one_fold(
    model_name: str,
    state_dict: dict,
    num_classes: int,
    train_ds: ItemsDataset,
    val_ds: ItemsDataset,
    val_loader: DataLoader,
    pv_crop: list[str],
    cfg: dict,
    device: torch.device,
    fold: int,
    fold_log: list[str],
) -> dict:
    """在 4 折上微调、在第 5 折上评估，返回该折指标。"""
    seed = cfg["seed"] + fold
    seed_everything(seed)

    model, _meta = build_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)

    # 是否冻结 BN 统计量：域偏移下让 BN 用目标域数据重新估均值/方差通常是
    # 有益的（这正是域适应的一部分），所以默认允许更新；想复现"只调权重不动
    # 统计量"的保守做法时可以打开。
    if cfg["freeze_bn"]:
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

    backbone_params, head_params = split_param_groups(model, model_name)
    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg["backbone_lr"]},
            {"params": head_params, "lr": cfg["head_lr"]},
        ],
        weight_decay=cfg["weight_decay"],
    )
    # 每轮 step 一次的 cosine：轮数很少（3~5），按 epoch 退火足够平滑
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["epochs"], eta_min=cfg["backbone_lr"] * 0.01
    )
    crit = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    # batch=1 会让 BN 在 train 模式下报错，仅当样本数除 batch 余 1 时才丢弃尾批
    drop_last = len(train_ds) > cfg["batch_size"] and len(train_ds) % cfg["batch_size"] == 1
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=drop_last,
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
        generator=gen,
        persistent_workers=cfg["num_workers"] > 0,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict] = []
    t0 = time.time()

    for epoch in range(cfg["epochs"]):
        model.train()
        if cfg["freeze_bn"]:
            # train() 会把 BN 重新设回训练模式，冻结时每轮都要再压回去
            for m in model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()

        loss_sum = 0.0
        n_seen = 0
        for step, (x, y) in enumerate(train_loader):
            if cfg["limit_batches"] and step >= cfg["limit_batches"]:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = y.size(0)
            loss_sum += float(loss.detach()) * bs
            n_seen += bs

        sched.step()
        train_loss = loss_sum / max(n_seen, 1)

        # 每轮在有标注的验证折上算一次指标，仅用于观察收敛；不参与选轮次
        m = evaluate(model, val_loader, device, pv_crop)
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_top1": round(m["top1_acc"], 4),
            "val_crop": round(m["crop_acc"], 4),
            "lr_backbone": round(opt.param_groups[0]["lr"], 8),
            "lr_head": round(opt.param_groups[1]["lr"], 8),
        })
        fold_log.append(
            f"    轮 {epoch + 1}/{cfg['epochs']}  loss={train_loss:.4f}  "
            f"val_top1={m['top1_acc']:.4f}  val_crop={m['crop_acc']:.4f}"
        )

    elapsed = time.time() - t0
    final = evaluate(model, val_loader, device, pv_crop)
    result = {
        "fold": fold,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "epochs": cfg["epochs"],
        "train_loss_final": history[-1]["train_loss"] if history else None,
        "top1_acc": round(final["top1_acc"], 4),
        "top3_acc": round(final["top3_acc"], 4),
        "crop_acc": round(final["crop_acc"], 4),
        "mean_conf": round(final["mean_conf"], 4),
        "correct": final["correct"],
        "crop_correct": final["crop_correct"],
        "train_s": round(elapsed, 1),
        "history": history,
    }

    del model, opt, sched, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> int:
    p = argparse.ArgumentParser(description="PlantDoc 有监督域适应：5 折微调")
    p.add_argument("--run-dir", default="", help="PlantVillage 训练产物目录（含 best.pt）")
    p.add_argument("--ckpt", default="", help="直接指定 checkpoint 路径（Kaggle 上更直接）")
    p.add_argument("--model", default="", help="模型名，默认从 checkpoint 的 args 里读")
    p.add_argument("--plantdoc-root", default=str(ROOT / "data" / "raw" / "plantdoc"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--folds-to-run", type=int, default=0, help="只跑前 N 折（冒烟用，0=全部）")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--backbone-lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--limit-per-class", type=int, default=0, help="每类最多取 N 张（冒烟用）")
    p.add_argument("--limit-batches", type=int, default=0, help="每轮最多 N 个 batch（冒烟用）")
    p.add_argument("--strong-aug", action="store_true",
                   help="用 RandAugment+RandomErasing（默认只做裁剪与翻转，目标域样本少时更稳）")
    p.add_argument("--freeze-bn", action="store_true", help="冻结 BN 统计量")
    p.add_argument("--out", default="", help="结果 json 路径，默认写到 run-dir")
    args = p.parse_args()

    if args.ckpt:
        ckpt_path = Path(args.ckpt).resolve()
    elif args.run_dir:
        ckpt_path = (ROOT / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
        ckpt_path = ckpt_path / "best.pt"
    else:
        print("[错误] 需要 --run-dir 或 --ckpt")
        return 1

    if not ckpt_path.exists():
        print(f"[错误] 找不到 checkpoint: {ckpt_path}")
        return 1

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_name = (args.model or ck.get("args", {}).get("model") or "").lower()
    if not model_name:
        model_name = ckpt_path.parent.name.rsplit("_", 2)[0]

    # ck["classes"] 是训练时 ImageFolder 的文件夹名，顺序与输出索引严格对应
    pv_folders = list(ck["classes"])
    num_classes = len(pv_folders)
    pv_crop = [crop_of_folder(f) for f in pv_folders]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 76)
    print("设定：有监督域适应（supervised DA）——允许使用 PlantDoc 标签微调")
    print("注意：这与 eval_crossdomain.py 的「零样本跨域泛化」是两个不同设定，结果不可互相替代")
    print("=" * 76)
    print(f"checkpoint : {ckpt_path}")
    print(f"模型       : {model_name}（epoch {ck.get('epoch', '?')}，PV val_acc {ck.get('val_acc', '?')}）")
    print(f"设备       : {device}   torch {torch.__version__}")
    print(f"PlantDoc   : {args.plantdoc_root}")

    eval_tf = build_transforms(args.img_size, train=False)
    catalog = PlantDocDataset(
        Path(args.plantdoc_root), "all", pv_folders, eval_tf,
        limit_per_class=args.limit_per_class,
    )
    items, bad = validate_items(catalog.items)
    if bad:
        print(f"[警告] 剔除 {len(bad)} 张无法读取的图，例如 {bad[:3]}")
    if not items:
        print("[错误] 映射后没有可用样本，检查 PlantDoc 目录")
        return 1

    n_unmapped = sum(catalog.skipped.values())
    label_set = sorted({lab for _p, lab, _c in items})
    print(f"可用样本   : {len(items)} 张 / {len(label_set)} 个 PlantDoc 类（映射到 PV 的 {len(label_set)} 类）")
    print(f"未映射剔除 : {n_unmapped} 张（{list(catalog.skipped)}）")

    fold_of = make_stratified_folds(items, args.folds, args.seed)
    origin = [p.parent.parent.name for p, _l, _c in items]  # train / test

    cfg = {
        "model": model_name,
        "folds": args.folds,
        "epochs": args.epochs,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR(per-epoch)",
        "seed": args.seed,
        "strong_aug": bool(args.strong_aug),
        "freeze_bn": bool(args.freeze_bn),
        "limit_per_class": args.limit_per_class,
        "limit_batches": args.limit_batches,
        "head_classes": num_classes,
    }
    print(f"配置       : epochs={args.epochs} bs={args.batch_size} "
          f"backbone_lr={args.backbone_lr} head_lr={args.head_lr} "
          f"ls={args.label_smoothing} strong_aug={args.strong_aug}")

    # 折的组成先打印出来：折是否真的均衡、类别覆盖是否完整，一眼能看出来
    print("\n折划分（按类别分层；train/test 是 PlantDoc 原始来源，仅作说明）")
    print(f"{'折':<4}{'训练':>6}{'验证':>6}{'训练类数':>9}{'验证类数':>9}{'来源(train/test)':>18}")
    for f in range(args.folds):
        tr_idx = [i for i, k in enumerate(fold_of) if k != f]
        va_idx = [i for i, k in enumerate(fold_of) if k == f]
        tr_cls = len({items[i][1] for i in tr_idx})
        va_cls = len({items[i][1] for i in va_idx})
        n_tr = sum(1 for i in tr_idx if origin[i] == "train")
        n_te = len(tr_idx) - n_tr
        print(f"{f:<4}{len(tr_idx):>6}{len(va_idx):>6}{tr_cls:>9}{va_cls:>9}{f'{n_tr}/{n_te}':>18}")

    n_folds_run = args.folds if args.folds_to_run <= 0 else min(args.folds, args.folds_to_run)

    # ---------------- 零样本 baseline：同一套折，不微调 ----------------
    print("\n[1/2] 零样本 baseline（权重原样加载，不微调）")
    zs_model, _meta = build_model(model_name, num_classes, pretrained=False)
    zs_model.load_state_dict(ck["model"])
    zs_model.to(device).eval()

    zs_folds: list[dict] = []
    zs_preds_all: list[int] = []
    zs_true_all: list[int] = []
    for f in range(n_folds_run):
        va_idx = [i for i, k in enumerate(fold_of) if k == f]
        if not va_idx:
            # 只会在 --limit-per-class 很小时出现（冒烟），跳过而不是记 0 分
            print(f"  折 {f} 没有验证样本，跳过（冒烟时请少跑几折）")
            continue
        ds = ItemsDataset(items, va_idx, eval_tf)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
        m = evaluate(zs_model, loader, device, pv_crop)
        zs_folds.append({
            "fold": f, "n": m["n"],
            "top1_acc": round(m["top1_acc"], 4),
            "top3_acc": round(m["top3_acc"], 4),
            "crop_acc": round(m["crop_acc"], 4),
            "correct": m["correct"], "crop_correct": m["crop_correct"],
        })
        print(f"  折 {f}  零样本 top-1={m['top1_acc']:.4f}  作物级={m['crop_acc']:.4f}  (n={m['n']})")
    del zs_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 折划分覆盖全部样本，所以各折 correct 之和就是全量口径，可与 28.18% 对照
    zs_n = sum(d["n"] for d in zs_folds)
    zs_correct = sum(d["correct"] for d in zs_folds)
    zs_overall = zs_correct / zs_n if zs_n else 0.0
    if n_folds_run == args.folds:
        print(f"  全量零样本 top-1 = {zs_overall:.4f}（{zs_correct}/{zs_n}），"
              f"应与 crossdomain.json 的 0.2818 一致")
    zs_top1_mean, zs_top1_std = mean_std([d["top1_acc"] for d in zs_folds])
    zs_crop_mean, zs_crop_std = mean_std([d["crop_acc"] for d in zs_folds])

    # ---------------- 微调：每折独立从 PV 权重重新开始 ----------------
    print(f"\n[2/2] 微调 {n_folds_run} 折（每折重新从 PlantVillage 权重初始化）")
    train_tf = build_transforms(args.img_size, train=True, strong=args.strong_aug)
    ft_folds: list[dict] = []
    all_fold_log: list[str] = []

    for f in range(n_folds_run):
        tr_idx = [i for i, k in enumerate(fold_of) if k != f]
        va_idx = [i for i, k in enumerate(fold_of) if k == f]
        if not va_idx or not tr_idx:
            print(f"  折 {f} 训练或验证为空，跳过")
            continue
        train_ds = ItemsDataset(items, tr_idx, train_tf)
        val_ds = ItemsDataset(items, va_idx, eval_tf)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=device.type == "cuda")

        print(f"\n  ── 折 {f}/{n_folds_run - 1}：训练 {len(train_ds)}，验证 {len(val_ds)}")
        fold_log: list[str] = []
        res = finetune_one_fold(
            model_name, ck["model"], num_classes, train_ds, val_ds, val_loader,
            pv_crop, {**cfg, "epochs": args.epochs, "num_workers": args.num_workers},
            device, f, fold_log,
        )
        print("\n".join(fold_log))
        print(f"    折 {f} 结果  top-1={res['top1_acc']:.4f}  作物级={res['crop_acc']:.4f}  "
              f"top-3={res['top3_acc']:.4f}  用时 {res['train_s']}s")
        ft_folds.append(res)
        all_fold_log.extend(fold_log)

    ft_top1_mean, ft_top1_std = mean_std([d["top1_acc"] for d in ft_folds])
    ft_crop_mean, ft_crop_std = mean_std([d["crop_acc"] for d in ft_folds])

    # 配对差：同一折上 ft − zs。比"两个均值相减"更有信息量，
    # 因为它把折间难度差异消掉了（同一折的验证集完全相同）。
    deltas = [d["top1_acc"] - z["top1_acc"] for d, z in zip(ft_folds, zs_folds)]
    delta_mean, delta_std = mean_std(deltas)
    crop_deltas = [d["crop_acc"] - z["crop_acc"] for d, z in zip(ft_folds, zs_folds)]
    crop_delta_mean, crop_delta_std = mean_std(crop_deltas)

    print("\n" + "=" * 76)
    print("跨折汇总（mean ± std，样本标准差 ddof=1）")
    print(f"{'折':<4}{'n_val':>7}{'零样本 top-1':>14}{'微调 top-1':>13}{'零样本 作物':>14}{'微调 作物':>12}")
    for z, d in zip(zs_folds, ft_folds):
        print(f"{d['fold']:<4}{d['n_val']:>7}{z['top1_acc']:>14.4f}{d['top1_acc']:>13.4f}"
              f"{z['crop_acc']:>14.4f}{d['crop_acc']:>12.4f}")
    print("-" * 76)
    print(f"{'mean':<4}{'':>7}{zs_top1_mean:>14.4f}{ft_top1_mean:>13.4f}"
          f"{zs_crop_mean:>14.4f}{ft_crop_mean:>12.4f}")
    print(f"{'std':<4}{'':>7}{zs_top1_std:>14.4f}{ft_top1_std:>13.4f}"
          f"{zs_crop_std:>14.4f}{ft_crop_std:>12.4f}")
    print("-" * 76)
    print(f"配对差（微调 − 零样本）：top-1 {delta_mean:+.4f} ± {delta_std:.4f}；"
          f"作物级 {crop_delta_mean:+.4f} ± {crop_delta_std:.4f}")
    print(f"逐折更好的折数：{sum(1 for x in deltas if x > 0)}/{len(deltas)}")
    print("=" * 76)

    summary = {
        "experiment": "supervised_domain_adaptation",
        "setting_note": (
            "有监督域适应：使用 PlantDoc 标签微调。与 eval_crossdomain.py 的零样本"
            "跨域泛化（不碰目标域标签）是不同设定，数字不可互换。"
        ),
        "model": model_name,
        "ckpt": str(ckpt_path),
        "ckpt_epoch": ck.get("epoch"),
        "ckpt_pv_val_acc": ck.get("val_acc"),
        "dataset": {
            "name": "PlantDoc",
            "n_mapped": len(items),
            "n_unmapped_skipped": n_unmapped,
            "n_plantdoc_classes": len(label_set),
            "source_counts": dict(Counter(origin)),
            "n_bad_images": len(bad),
            "split": "train+test 合并后按类别重新分层 5 折（不保留官方划分）",
        },
        "config": cfg,
        "folds": [
            {
                "fold": f,
                "n_train": sum(1 for k in fold_of if k != f),
                "n_val": sum(1 for k in fold_of if k == f),
                "class_coverage_val": len({
                    items[i][1] for i, k in enumerate(fold_of) if k == f
                }),
            }
            for f in range(args.folds)
        ] if args.limit_per_class == 0 else None,
        "zeroshot": {
            "per_fold": zs_folds,
            "top1_mean": round(zs_top1_mean, 4),
            "top1_std": round(zs_top1_std, 4),
            "crop_mean": round(zs_crop_mean, 4),
            "crop_std": round(zs_crop_std, 4),
            "overall_top1_all_folds": round(zs_overall, 4),
        },
        "finetuned": {
            "per_fold": ft_folds,
            "top1_mean": round(ft_top1_mean, 4),
            "top1_std": round(ft_top1_std, 4),
            "crop_mean": round(ft_crop_mean, 4),
            "crop_std": round(ft_crop_std, 4),
        },
        "paired_delta": {
            "top1_mean": round(delta_mean, 4),
            "top1_std": round(delta_std, 4),
            "crop_mean": round(crop_delta_mean, 4),
            "crop_std": round(crop_delta_std, 4),
            "folds_better": sum(1 for x in deltas if x > 0),
            "n_folds": len(deltas),
        },
        "epochs_ran": args.epochs,
        "folds_ran": n_folds_run,
    }

    if args.out:
        out_path = Path(args.out)
    else:
        # 冒烟结果不能写进真实实验的目录，否则很容易被当成正式结果引用
        smoke = bool(args.limit_per_class or args.limit_batches or args.folds_to_run or args.epochs < 3)
        name = "plantdoc_ft_cv_smoke.json" if smoke else "plantdoc_ft_cv.json"
        out_path = (ckpt_path.parent if args.run_dir else Path.cwd()) / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存: {out_path}")

    # Kaggle 上直接抓这段即可，不用去翻日志
    print("RESULT_JSON_BEGIN")
    print(json.dumps(summary, ensure_ascii=False))
    print("RESULT_JSON_END")
    return 0


if __name__ == "__main__":
    sys.exit(main())
