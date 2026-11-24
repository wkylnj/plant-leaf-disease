"""E-15 预处理消融：跨域准确率低，有多少是"怎么把图喂进去"造成的？

要回答的问题
-----------
之前所有跨域实验（E2）都用**训练时验证集的同一套预处理**：短边缩到 256，
再中心裁 224。这个流程对 PlantVillage 那种"整片叶子居中、纯背景"的图很合适，
但 PlantDoc 是田间照片：叶片可能偏在角落、一图多叶、长条形横排。
中心裁 224 可能直接把主体裁掉了 —— 那么 28% 里有多少是模型不行、
有多少只是"图没喂好"？

这是最容易忽略也最便宜的一问：不改模型、不重训、只换预处理。

对比的 6 种喂法
--------------
1. baseline    短边 256 → 中心裁 224（历史口径，用来对齐 28.18%）
2. squash      整图直接拉伸成 224×224（不裁，但会被拉变形）
3. short224    短边缩到 224 → 中心裁 224（裁掉更多，对照组）
4. letterbox   整图等比缩放进 224×224 画布，空处填 ImageNet 均值灰（不裁不变形）
5. center320   短边 356 → 中心裁 320，模型直接在 320 上前向（更高分辨率）
6. multiscale  三种喂法（baseline / squash / letterbox）× 水平翻转，6 个视图 softmax 平均

用法
----
    python src/eval_preprocess_ablation.py                      # 跑全部 6 种
    python src/eval_preprocess_ablation.py --variants baseline squash letterbox
    python src/eval_preprocess_ablation.py --run-dir runs/mobilenet_v3_large_224_15ep

产物
----
    ml/artifacts/preprocess_ablation.md   对比表 + 结论
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import IMAGENET_MEAN, IMAGENET_STD, build_transforms  # noqa: E402
from eval_crossdomain import (  # noqa: E402
    PLANTDOC_TO_PV_FOLDER,
    crop_of_folder,
)
from models import build_model  # noqa: E402


# ---------------------------------------------------------------- 各种预处理

class Squash:
    """整图直接拉伸到 size×size。"""

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, img: Image.Image) -> torch.Tensor:
        import torchvision.transforms as T

        t = T.Compose([
            T.Resize((self.size, self.size), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return t(img)


class Letterbox:
    """等比缩放 + 居中填边，保证整图都在，且不变形。"""

    def __init__(self, size: int, fill=(124, 116, 104)) -> None:
        self.size = size
        # 用 ImageNet 均值换算成 0-255 的灰，保证填充区域接近"中性"
        self.fill = fill

    def __call__(self, img: Image.Image) -> torch.Tensor:
        import torchvision.transforms as T

        w, h = img.size
        s = self.size / max(w, h)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        img = img.resize((nw, nh), Image.BICUBIC)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(img, ((self.size - nw) // 2, (self.size - nh) // 2))
        t = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        return t(canvas)


def build_variant_transform(name: str, img_size: int):
    """返回 (transform, 实际前向分辨率)。

    支持 center<N> 形式，如 center384：短边缩到 N/224*256 再中心裁 N，
    用于检验「训练 224、推理更高分辨率」能否提升跨域（FixRes 现象）。
    """
    import re as _re
    import torchvision.transforms as T

    if name == "baseline":
        return build_transforms(img_size, train=False), img_size
    if name == "short224":
        t = T.Compose([
            T.Resize(224, T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return t, 224
    m = _re.fullmatch(r"center(\d+)", name)
    if m:
        size = int(m.group(1))
        short = int(round(size * 256 / 224))
        t = T.Compose([
            T.Resize(short, T.InterpolationMode.BICUBIC),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return t, size
    if name == "squash":
        return Squash(img_size), img_size
    if name == "letterbox":
        return Letterbox(img_size), img_size
    raise ValueError(f"未知 variant: {name}")


# multiscale = 多个基础 variant × 水平翻转
MULTISCALE_VIEWS = [
    ("baseline", False), ("baseline", True),
    ("squash", False), ("squash", True),
    ("letterbox", False), ("letterbox", True),
]


# ---------------------------------------------------------------- 数据

class PlantDocItems(Dataset):
    """复用 eval_crossdomain 的目录扫描逻辑，但允许每张图用不同 transform。"""

    def __init__(self, root: Path, split: str, pv_folders: list[str], transform) -> None:
        from eval_crossdomain import PlantDocDataset

        base = PlantDocDataset(root, split, pv_folders, transform=None)
        self.items = base.items
        self.transform = transform
        self.skipped = base.skipped

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label, _orig = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def predict_probs(model, loader, device, flip: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    probs_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if flip:
                x = torch.flip(x, dims=[3])
            probs_all.append(F.softmax(model(x), dim=1).cpu())
            labels_all.append(y)
    return torch.cat(probs_all, dim=0), torch.cat(labels_all, dim=0)


def metrics(probs: torch.Tensor, labels: torch.Tensor, pv_folders: list[str]) -> dict:
    n = labels.numel()
    top1 = probs.argmax(dim=1)
    k = min(3, probs.size(1))
    top3 = probs.topk(k, dim=1).indices
    correct = int((top1 == labels).sum())
    top3_hit = int((top3 == labels.unsqueeze(1)).any(dim=1).sum())

    crops = [crop_of_folder(f) for f in pv_folders]
    crop_of: dict[int, str] = {}
    for idx, f in enumerate(pv_folders):
        crop_of[idx] = crops[idx]
    crop_correct = sum(
        1 for p, t in zip(top1.tolist(), labels.tolist())
        if crop_of.get(p) == crop_of.get(t)
    )
    return {
        "n": n,
        "top1_acc": round(correct / n, 4),
        "top3_acc": round(top3_hit / n, 4),
        "crop_acc": round(crop_correct / n, 4),
        "mean_confidence": round(float(probs.max(dim=1).values.mean()), 4),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="跨域预处理消融")
    p.add_argument("--run-dir", default="runs/resnet18_224_15ep")
    p.add_argument("--plantdoc-root", default=str(ROOT / "data" / "raw" / "plantdoc"))
    p.add_argument("--split", default="all", choices=["train", "test", "all"])
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--variants", nargs="+",
                   default=["baseline", "squash", "short224", "letterbox", "multiscale"])
    p.add_argument("--out", default="")
    args = p.parse_args()

    run_dir = (ROOT / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        print(f"[错误] 找不到 checkpoint: {ckpt_path}")
        return 1

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_name = ck.get("args", {}).get("model") or run_dir.name.rsplit("_", 2)[0]
    pv_folders = list(ck["classes"])
    num_classes = len(pv_folders)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"模型     : {model_name} (epoch {ck.get('epoch', '?')})  run={run_dir.name}")
    print(f"PlantDoc : {args.plantdoc_root} [split={args.split}]  device={device}")
    print(f"variants : {args.variants}")

    # 数据只扫描一次，后续每个 variant 复用同一批路径
    probe_root = Path(args.plantdoc_root)
    probe = PlantDocItems(probe_root, args.split, pv_folders, transform=build_transforms(args.img_size, train=False))
    items = probe.items
    print(f"可用样本 : {len(items)}（跳过 {sum(probe.skipped.values())} 张：{dict(probe.skipped)}）")

    results: dict[str, dict] = {}
    t_all = time.time()

    def run_variant(vname: str, flip: bool = False, prev: torch.Tensor | None = None):
        transform, fwd_size = build_variant_transform(vname, args.img_size)
        ds = PlantDocItems(probe_root, args.split, pv_folders, transform=transform)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=False)
        model, _meta = build_model(model_name, num_classes, pretrained=False)
        model.load_state_dict(ck["model"])
        model.to(device).eval()
        if fwd_size != args.img_size:
            print(f"  [提示] {vname} 前向分辨率 {fwd_size}")
        t0 = time.time()
        probs, labels = predict_probs(model, loader, device, flip=flip)
        dt = time.time() - t0
        return probs, labels, dt

    for vname in args.variants:
        if vname == "multiscale":
            acc_probs = None
            labels = None
            dt_total = 0.0
            for base, flip in MULTISCALE_VIEWS:
                probs, labels, dt = run_variant(base, flip=flip)
                dt_total += dt
                acc_probs = probs if acc_probs is None else acc_probs + probs
            assert acc_probs is not None and labels is not None
            probs = acc_probs / len(MULTISCALE_VIEWS)
            m = metrics(probs, labels, pv_folders)
            m["elapsed_s"] = round(dt_total, 1)
            m["img_per_s"] = round(len(items) / dt_total, 1) if dt_total else 0.0
            m["note"] = f"{len(MULTISCALE_VIEWS)} 视图平均"
        else:
            probs, labels, dt = run_variant(vname)
            m = metrics(probs, labels, pv_folders)
            m["elapsed_s"] = round(dt, 1)
            m["img_per_s"] = round(len(items) / dt, 1) if dt else 0.0
            m["note"] = ""
        results[vname] = m
        print(f"  {vname:<12} top1={m['top1_acc']:.4f} top3={m['top3_acc']:.4f} "
              f"crop={m['crop_acc']:.4f} conf={m['mean_confidence']:.4f} "
              f"({m['elapsed_s']}s)")

    total = time.time() - t_all
    print(f"\n总耗时 {total:.1f}s")

    # 保存
    out_path = Path(args.out) if args.out else (run_dir / "preprocess_ablation.json")
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    payload = {
        "run": run_dir.name,
        "model": model_name,
        "n_samples": len(items),
        "split": args.split,
        "variants": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存: {out_path}")

    # 生成 markdown
    md = ["# E-15 跨域预处理消融", "",
          f"- 模型：`{run_dir.name}`（{model_name}）",
          f"- 样本：PlantDoc `{args.split}` 划分，{len(items)} 张",
          "- 口径：模型权重不动，只换「图怎么喂进去」", "",
          "| 喂法 | 说明 | top-1 | top-3 | 作物级 | 平均置信度 | 耗时(s) |",
          "|---|---|---|---|---|---|---|"]
    desc = {
        "baseline": "短边256→中心裁224（历史口径）",
        "squash": "整图拉伸到224×224",
        "short224": "短边224→中心裁224",
        "letterbox": "整图等比缩放+填边",
        "center320": "短边356→中心裁320，224转320前向",
        "multiscale": "baseline/squash/letterbox × 翻转，6视图平均",
    }
    for vname, m in results.items():
        md.append(f"| `{vname}` | {desc.get(vname, '')} | {m['top1_acc']:.4f} | "
                  f"{m['top3_acc']:.4f} | {m['crop_acc']:.4f} | {m['mean_confidence']:.4f} | "
                  f"{m['elapsed_s']} |")
    md_path = ROOT / "artifacts" / "preprocess_ablation.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"已保存: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
