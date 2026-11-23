"""TTA（多视图测试时增强）与模型集成评估。

为什么要做这个
-------------
同域 test 已经到 99.48%（10878 张只错 57 张），继续训练收益只有千分之几；
但跨域只有 28.18%，这才是真实短板。TTA 和集成是**不重新训练**就能提分的手段，
且对跨域这种"输入分布偏移"的场景往往比同域更有效（多个视图平均能压低
单视图在异常视角下的极端错误）。

做了什么
--------
1. 多视图 TTA：对同一张图取 原图 / 水平翻转 / 垂直翻转 / 双翻转 四个视图，
   把 softmax 概率求平均再 argmax。叶片没有固定朝向，翻转视图是合理的不变性。
2. 模型集成：把 ResNet18 与 MobileNetV3 的概率平均。两个模型结构不同、
   归纳偏置不同，错误相关性较低，平均通常能互补。

评估两套数据
-----------
- 同域：PlantVillage test（10878 张）
- 跨域：PlantDoc（2920 张，严格不做微调）

用法
----
    python src/eval_tta.py --run-dir runs/mobilenet_v3_large_224_15ep
    python src/eval_tta.py --run-dir runs/mobilenet_v3_large_224_15ep runs/resnet18_224_15ep
    python src/eval_tta.py --run-dir runs/mobilenet_v3_large_224_15ep --views hflip
    python src/eval_tta.py --run-dir ... --no-crossdomain   只测同域

产物
----
    ml/artifacts/tta_ensemble.md     对比报告（基线 vs TTA vs 集成）
    <run-dir>/tta_eval.json          每个 run 的 TTA 指标
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from eval_crossdomain import PlantDocDataset, crop_of_folder  # noqa: E402
from models import build_model  # noqa: E402


# ---------------------------------------------------------------------------
# TTA 视图
# ---------------------------------------------------------------------------
# 每个视图是一个纯函数：输入已归一化的张量，输出同样形状的张量。
# 用张量操作而不是 PIL 变换，是为了避免重复解码图片（10878 张重复解码很慢）。
VIEWS: dict[str, object] = {
    "none": lambda x: x,
    "hflip": lambda x: torch.flip(x, dims=[3]),
    "vflip": lambda x: torch.flip(x, dims=[2]),
    "hvflip": lambda x: torch.flip(x, dims=[2, 3]),
}


def build_eval_transform(img_size: int):
    """与训练时验证集完全一致：resize → centercrop → normalize。"""
    return transforms.Compose(
        [
            transforms.Resize(
                int(img_size * 256 / 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class TransformOverride(Dataset):
    """给已有 Dataset 换一套 transform，同时保留它的标签逻辑。

    ImageFolder 和 PlantDocDataset 都需要换 transform 重跑，
    用一层包装比各写一份重复代码更省事。
    """

    def __init__(self, base: Dataset, transform) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        if hasattr(self.base, "samples"):  # ImageFolder
            path, label = self.base.samples[idx]
            img = Image.open(path).convert("RGB")
            return self.transform(img), label
        # PlantDocDataset
        path, label, _orig = self.base.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def predict_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    views: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (平均后的概率 [N, C], 真实标签 [N])。"""
    model.eval()
    probs_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            acc = None
            for v in views:
                xv = VIEWS[v](x)
                p = torch.softmax(model(xv), dim=1)
                acc = p if acc is None else acc + p
            probs_chunks.append((acc / len(views)).cpu())
            label_chunks.append(y.cpu())

    return torch.cat(probs_chunks), torch.cat(label_chunks)


def metrics(probs: torch.Tensor, labels: torch.Tensor, pv_crop: list[str] | None = None) -> dict:
    """算 top-1 / top-3 / 作物级准确率。"""
    top3 = probs.topk(min(3, probs.shape[1]), dim=1)
    pred = top3.indices[:, 0]
    n = labels.numel()
    correct = int((pred == labels).sum())
    top3_hit = int(sum(1 for i in range(n) if labels[i] in top3.indices[i]))

    out = {
        "n": n,
        "top1_acc": correct / n if n else 0.0,
        "top3_acc": top3_hit / n if n else 0.0,
        "mean_conf": float(top3.values[:, 0].mean()),
    }
    if pv_crop is not None:
        crop_ok = sum(1 for i in range(n) if pv_crop[pred[i]] == pv_crop[labels[i]])
        out["crop_acc"] = crop_ok / n if n else 0.0
    return out


def load_model_from_run(run_dir: Path) -> tuple[torch.nn.Module, list[str], dict]:
    """从 run 目录加载模型与类别顺序。"""
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model_name = ck.get("args", {}).get("model") or run_dir.name.rsplit("_", 2)[0]
    classes = list(ck["classes"])
    model, _meta = build_model(model_name, len(classes), pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, classes, {"model": model_name, "ckpt_epoch": ck.get("epoch"), "ckpt_val_acc": ck.get("val_acc")}


def main() -> int:
    ap = argparse.ArgumentParser(description="TTA 与模型集成评估")
    ap.add_argument("--run-dir", nargs="+", required=True, help="一个或多个 run 目录（多个则做集成）")
    ap.add_argument("--data-root", default=str(ROOT / "data" / "processed" / "plantvillage"))
    ap.add_argument("--plantdoc-root", default=str(ROOT / "data" / "raw" / "plantdoc"))
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--views", default="none,hflip,vflip,hvflip", help="逗号分隔的视图名")
    ap.add_argument("--no-crossdomain", action="store_true", help="跳过跨域评估，只测同域")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    args = ap.parse_args()

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    for v in views:
        if v not in VIEWS:
            print(f"[错误] 未知视图 {v}，可选：{list(VIEWS)}")
            return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dirs = [Path(r) if Path(r).is_absolute() else (ROOT / r) for r in args.run_dir]
    for rd in run_dirs:
        if not (rd / "best.pt").exists():
            print(f"[错误] 找不到 {rd / 'best.pt'}")
            return 1

    transform = build_eval_transform(args.img_size)

    # ---- 同域 PlantVillage ----
    pv_root = Path(args.data_root) / args.split
    pv_ds = TransformOverride(datasets.ImageFolder(pv_root, transform), transform)
    pv_loader = DataLoader(pv_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"设备        : {device}")
    print(f"视图        : {views}")
    print(f"同域样本    : {len(pv_ds)} ({pv_root})")

    results: dict = {"views": views, "runs": [], "same_domain": {}, "cross_domain": {}}

    # ---- 跨域 PlantDoc（用第一个 run 的类别顺序，所有 run 必须一致） ----
    pd_loader = None
    pv_crop = None
    if not args.no_crossdomain:
        _, classes0, _ = load_model_from_run(run_dirs[0])
        pd_ds = PlantDocDataset(Path(args.plantdoc_root), "all", classes0, transform)
        if len(pd_ds) > 0:
            pd_loader = DataLoader(pd_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            pv_crop = [crop_of_folder(f) for f in classes0]
            print(f"跨域样本    : {len(pd_ds)} (PlantDoc)")
        else:
            print("[警告] 跨域样本为 0，跳过跨域评估")

    # 每个 run 单独测（基线单视图 + TTA）
    per_run_probs_pv: list[torch.Tensor] = []
    per_run_probs_pd: list[torch.Tensor] = []
    labels_pv: torch.Tensor | None = None
    labels_pd: torch.Tensor | None = None

    for rd in run_dirs:
        model, classes, meta = load_model_from_run(rd)
        model.to(device)
        print(f"\n--- {rd.name} ({meta['model']}, epoch {meta['ckpt_epoch']}) ---")

        # 基线：单视图
        t0 = time.time()
        p_pv_base, y_pv = predict_probs(model, pv_loader, device, ["none"])
        labels_pv = y_pv
        m_base = metrics(p_pv_base, y_pv)
        print(f"  同域 基线       : top1={m_base['top1_acc']:.4f} top3={m_base['top3_acc']:.4f}  ({time.time()-t0:.0f}s)")

        # TTA
        t0 = time.time()
        p_pv_tta, _ = predict_probs(model, pv_loader, device, views)
        m_tta = metrics(p_pv_tta, y_pv)
        print(f"  同域 TTA        : top1={m_tta['top1_acc']:.4f} top3={m_tta['top3_acc']:.4f}  ({time.time()-t0:.0f}s)")

        entry = {
            "run": rd.name,
            **meta,
            "same_domain_baseline": m_base,
            "same_domain_tta": m_tta,
        }

        if pd_loader is not None:
            p_pd_base, y_pd = predict_probs(model, pd_loader, device, ["none"])
            labels_pd = y_pd
            m_pd_base = metrics(p_pd_base, y_pd, pv_crop)
            p_pd_tta, _ = predict_probs(model, pd_loader, device, views)
            m_pd_tta = metrics(p_pd_tta, y_pd, pv_crop)
            print(f"  跨域 基线       : top1={m_pd_base['top1_acc']:.4f} crop={m_pd_base['crop_acc']:.4f}")
            print(f"  跨域 TTA        : top1={m_pd_tta['top1_acc']:.4f} crop={m_pd_tta['crop_acc']:.4f}")
            entry["cross_domain_baseline"] = m_pd_base
            entry["cross_domain_tta"] = m_pd_tta
            per_run_probs_pd.append(p_pd_tta)
        else:
            p_pd_base = p_pd_tta = None

        # 保存单 run 结果
        (rd / "tta_eval.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
        results["runs"].append(entry)
        per_run_probs_pv.append(p_pv_tta)

    # ---- 集成（两个及以上 run） ----
    if len(run_dirs) >= 2:
        print(f"\n--- 集成 {len(run_dirs)} 个模型 ---")
        ens_pv = torch.stack(per_run_probs_pv).mean(dim=0)
        m_ens = metrics(ens_pv, labels_pv)
        print(f"  同域 集成       : top1={m_ens['top1_acc']:.4f} top3={m_ens['top3_acc']:.4f}")
        results["same_domain"]["ensemble"] = m_ens

        if per_run_probs_pd:
            ens_pd = torch.stack(per_run_probs_pd).mean(dim=0)
            m_ens_pd = metrics(ens_pd, labels_pd, pv_crop)
            print(f"  跨域 集成       : top1={m_ens_pd['top1_acc']:.4f} crop={m_ens_pd['crop_acc']:.4f}")
            results["cross_domain"]["ensemble"] = m_ens_pd

    # ---- 汇总表 ----
    lines = [
        "# TTA 与模型集成评估",
        "",
        f"- 视图组合：`{'+'.join(views)}`（{len(views)} 个）",
        f"- 同域数据：PlantVillage {args.split}",
        f"- 跨域数据：PlantDoc all" + ("（已跳过）" if args.no_crossdomain else ""),
        "",
        "## 同域 PlantVillage",
        "",
        "| 配置 | top-1 | top-3 | 相对基线 |",
        "|---|---|---|---|",
    ]
    for e in results["runs"]:
        b = e["same_domain_baseline"]["top1_acc"]
        t = e["same_domain_tta"]["top1_acc"]
        lines.append(f"| {e['run']} 基线 | {b:.4f} | {e['same_domain_baseline']['top3_acc']:.4f} | — |")
        lines.append(f"| {e['run']} + TTA | {t:.4f} | {e['same_domain_tta']['top3_acc']:.4f} | {t-b:+.4f} |")
    if "ensemble" in results["same_domain"]:
        m = results["same_domain"]["ensemble"]
        best = max(e["same_domain_tta"]["top1_acc"] for e in results["runs"])
        lines.append(f"| 集成 + TTA | {m['top1_acc']:.4f} | {m['top3_acc']:.4f} | {m['top1_acc']-best:+.4f} |")

    if results["cross_domain"] or any("cross_domain_tta" in e for e in results["runs"]):
        lines += [
            "",
            "## 跨域 PlantDoc",
            "",
            "| 配置 | top-1 | 作物级 | top-3 | 相对基线 |",
            "|---|---|---|---|---|",
        ]
        for e in results["runs"]:
            if "cross_domain_tta" not in e:
                continue
            b = e["cross_domain_baseline"]
            t = e["cross_domain_tta"]
            lines.append(f"| {e['run']} 基线 | {b['top1_acc']:.4f} | {b['crop_acc']:.4f} | {b['top3_acc']:.4f} | — |")
            lines.append(f"| {e['run']} + TTA | {t['top1_acc']:.4f} | {t['crop_acc']:.4f} | {t['top3_acc']:.4f} | {t['top1_acc']-b['top1_acc']:+.4f} |")
        if "ensemble" in results["cross_domain"]:
            m = results["cross_domain"]["ensemble"]
            best = max(e["cross_domain_tta"]["top1_acc"] for e in results["runs"] if "cross_domain_tta" in e)
            lines.append(f"| 集成 + TTA | {m['top1_acc']:.4f} | {m['crop_acc']:.4f} | {m['top3_acc']:.4f} | {m['top1_acc']-best:+.4f} |")

    out_md = ROOT / "artifacts" / "tta_ensemble.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "artifacts" / "tta_ensemble.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n已保存: {out_md}")
    print(f"已保存: {ROOT / 'artifacts' / 'tta_ensemble.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
